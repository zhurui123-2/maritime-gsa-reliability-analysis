"""Leakage-safe trajectory prediction with validation-selected adaptive GSA."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t, wilcoxon
from sklearn.model_selection import GroupShuffleSplit
from torch.nn import functional as F

from .gsa import GradientSpectrumAlignment
from .trajectory_prediction import (
    TrajectoryModel, apply_delta, flat_parameters, haversine_km,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ["ade_5_km", "fde_5_km", "ade_10_km", "fde_10_km"]


def _clone_state(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _split(data, seed: int):
    raw_x, raw_y = data["x"], data["y"]
    groups = data["mmsi"]
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_validation, test = next(outer.split(raw_x, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 10_000)
    train_rel, validation_rel = next(
        inner.split(raw_x[train_validation], groups=groups[train_validation])
    )
    return train_validation[train_rel], train_validation[validation_rel], test


def _normalize(data, train_index):
    raw_x, raw_y = data["x"], data["y"]
    input_mean = raw_x[train_index].mean(axis=(0, 1), keepdims=True)
    input_std = raw_x[train_index].std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    x = (raw_x - input_mean) / input_std
    last_position = raw_x[:, -1, :2]
    target_delta = raw_y - last_position[:, None, :]
    target_delta[:, :, 1] = (target_delta[:, :, 1] + 180.0) % 360.0 - 180.0
    target_mean = target_delta[train_index].mean(axis=0, keepdims=True)
    target_std = target_delta[train_index].std(axis=0, keepdims=True).clip(min=1e-6)
    y = (target_delta - target_mean) / target_std
    return x, y, last_position, target_mean, target_std


@torch.no_grad()
def _evaluate(
    model, x, raw_y, last_position, target_mean, target_std, index, device,
):
    model.eval()
    prediction, _, _ = model(torch.as_tensor(x[index], dtype=torch.float32, device=device))
    scale = torch.as_tensor(target_std, dtype=torch.float32, device=device)
    center = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    origin = torch.as_tensor(
        last_position[index, None, :], dtype=torch.float32, device=device
    )
    prediction = origin + prediction * scale + center
    prediction[:, :, 0].clamp_(-90.0, 90.0)
    prediction[:, :, 1] = torch.remainder(prediction[:, :, 1] + 180.0, 360.0) - 180.0
    truth = torch.as_tensor(raw_y[index], dtype=torch.float32, device=device)
    distance = haversine_km(prediction, truth).cpu().numpy()
    return {
        "ade_5_km": float(distance[:, :5].mean()),
        "fde_5_km": float(distance[:, 4].mean()),
        "ade_10_km": float(distance[:, :10].mean()),
        "fde_10_km": float(distance[:, 9].mean()),
    }


def run_one(
    data, config: dict, method: str, seed: int, device: torch.device,
    evaluate_test: bool = True,
) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)

    train_index, validation_index, test_index = _split(data, seed)
    x, y, last_position, target_mean, target_std = _normalize(data, train_index)
    raw_y = data["y"]
    nodes = data["node"].astype(str)
    horizon = int(raw_y.shape[1])
    model = TrajectoryModel(horizon, int(config["hidden_dim"])).to(device)
    ais_parameters = list(model.ais.parameters())
    weather_parameters = list(model.weather.parameters())
    head_parameters = list(model.head.parameters())
    use_gsa = method != "A"
    structured = (
        int(config["trajectory_structured_dim"])
        if config.get("trajectory_structured_dim") else None
    )
    gsa = None
    if use_gsa:
        gsa = GradientSpectrumAlignment(
            {
                "ais": sum(parameter.numel() for parameter in ais_parameters),
                "weather": sum(parameter.numel() for parameter in weather_parameters),
            },
            int(config["trajectory_spectrum_dim"]),
            structured_dim=structured,
        ).to(device)
    gsa_optimizer = (
        None if gsa is None else torch.optim.AdamW(
            gsa.parameters(),
            lr=float(config["gsa_learning_rate"]),
            weight_decay=float(config.get("gsa_weight_decay", 0.0)),
        )
    )
    rng = np.random.default_rng(seed)
    best_score = math.inf
    best_round = 0
    best_state = None
    patience = 0
    history = []
    rounds = int(config.get("trajectory_rounds_v2", 80))
    for round_index in range(1, rounds + 1):
        client_updates = []
        for node in sorted(set(nodes[train_index])):
            local = copy.deepcopy(model)
            local.ais.flatten_parameters()
            local_optimizer = torch.optim.AdamW(
                local.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config.get("weight_decay", 0.0)),
            )
            candidates = train_index[nodes[train_index] == node]
            local.train()
            for _ in range(int(config.get("trajectory_local_epochs", 1))):
                shuffled = rng.permutation(candidates)
                batch_size = int(config.get("trajectory_batch_size", 256))
                for start in range(0, len(shuffled), batch_size):
                    batch = shuffled[start:start + batch_size]
                    xb = torch.as_tensor(x[batch], dtype=torch.float32, device=device)
                    yb = torch.as_tensor(y[batch], dtype=torch.float32, device=device)
                    local_optimizer.zero_grad(set_to_none=True)
                    prediction, ais_embedding, weather_embedding = local(xb)
                    task_loss = F.smooth_l1_loss(prediction, yb)
                    contrastive = (
                        1.0
                        - F.cosine_similarity(
                            ais_embedding, weather_embedding, dim=1
                        ).mean()
                    )
                    weight = (
                        0.0 if method == "A"
                        else float(config["trajectory_contrastive_weight"])
                    )
                    loss = task_loss + weight * contrastive
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        local.parameters(),
                        float(config.get("gradient_clip_norm", 5.0)),
                    )
                    local_optimizer.step()
            client_updates.append((
                len(candidates),
                flat_parameters(ais_parameters)
                - flat_parameters(list(local.ais.parameters())),
                flat_parameters(weather_parameters)
                - flat_parameters(list(local.weather.parameters())),
                flat_parameters(head_parameters)
                - flat_parameters(list(local.head.parameters())),
            ))
        total = sum(item[0] for item in client_updates)
        ais_delta = sum(count * delta for count, delta, _, _ in client_updates) / total
        weather_delta = sum(count * delta for count, _, delta, _ in client_updates) / total
        head_delta = sum(count * delta for count, _, _, delta in client_updates) / total
        effective_strength = 0.0
        projected_cosine = float("nan")
        if gsa is not None and gsa_optimizer is not None:
            warmup = int(config.get("trajectory_gsa_warmup_rounds", 0))
            ramp = int(config.get("trajectory_gsa_ramp_rounds", 0))
            if round_index <= warmup:
                schedule = 0.0
            elif ramp:
                schedule = min(1.0, (round_index - warmup) / ramp)
            else:
                schedule = 1.0
            projected_probe = {
                "ais": gsa.projectors["ais"].project(ais_delta),
                "weather": gsa.projectors["weather"].project(weather_delta),
            }
            projected_cosine = float(
                F.cosine_similarity(
                    projected_probe["ais"], projected_probe["weather"], dim=0
                ).detach()
            )
            gate = 1.0
            if bool(config.get("trajectory_conflict_gate", True)):
                target = float(config.get("trajectory_conflict_target", 0.5))
                gate = float(np.clip((target - projected_cosine) / (target + 1.0), 0.0, 1.0))
            effective_strength = (
                float(config["trajectory_gsa_correction_strength"])
                * schedule * gate
            )
            corrected, projected = gsa.align(
                {"ais": ais_delta, "weather": weather_delta},
                effective_strength,
            )
            gsa_loss, _ = gsa.alignment_loss(projected)
            gsa_optimizer.zero_grad(set_to_none=True)
            gsa_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                gsa.parameters(), float(config.get("gradient_clip_norm", 5.0))
            )
            gsa_optimizer.step()
            ais_delta = corrected["ais"].detach()
            weather_delta = corrected["weather"].detach()
        apply_delta(ais_parameters, ais_delta)
        apply_delta(weather_parameters, weather_delta)
        apply_delta(head_parameters, head_delta)

        validation = _evaluate(
            model, x, raw_y, last_position, target_mean, target_std,
            validation_index, device,
        )
        validation_score = 0.5 * (
            validation["ade_5_km"] + validation["ade_10_km"]
        )
        history.append({
            "round": round_index,
            "validation_score": validation_score,
            **{f"validation_{key}": value for key, value in validation.items()},
            "projected_cosine": projected_cosine,
            "effective_correction_strength": effective_strength,
        })
        min_delta = float(config.get("trajectory_early_stopping_min_delta", 1e-4))
        if validation_score < best_score - min_delta:
            best_score = validation_score
            best_round = round_index
            best_state = {
                "model": _clone_state(model),
                "gsa": None if gsa is None else _clone_state(gsa),
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(config.get("trajectory_early_stopping_patience", 15)):
                break
    if best_state is None:
        raise RuntimeError("No trajectory checkpoint was produced.")
    model.load_state_dict(best_state["model"])
    if gsa is not None and best_state["gsa"] is not None:
        gsa.load_state_dict(best_state["gsa"])
    validation = _evaluate(
        model, x, raw_y, last_position, target_mean, target_std,
        validation_index, device,
    )
    row: dict[str, float | int | str] = {
        "method": method,
        "seed": seed,
        "n_train_windows": int(len(train_index)),
        "n_validation_windows": int(len(validation_index)),
        "n_test_windows": int(len(test_index)),
        "best_round": best_round,
        "rounds_executed": len(history),
        **{f"validation_{key}": value for key, value in validation.items()},
    }
    if evaluate_test:
        row.update(_evaluate(
            model, x, raw_y, last_position, target_mean, target_std,
            test_index, device,
        ))
    row["_history"] = history
    return row


def summarize(frame: pd.DataFrame, output: Path) -> None:
    available = [metric for metric in METRICS if metric in frame]
    summaries = []
    for method, group in frame.groupby("method"):
        row = {"method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in available:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            margin = (
                float(t.ppf(0.975, len(values) - 1) * std / len(values) ** 0.5)
                if len(values) > 1 else float("nan")
            )
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_ci95_low": mean - margin,
                f"{metric}_ci95_high": mean + margin,
            })
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(
        output / "trajectory_v2_summary.csv", index=False, encoding="utf-8-sig"
    )
    tests = []
    methods = sorted(set(frame["method"]))
    if "A" in methods:
        for method in [value for value in methods if value != "A"]:
            for metric in available:
                pivot = frame[frame["method"].isin(["A", method])].pivot(
                    index="seed", columns="method", values=metric
                ).dropna()
                if len(pivot) and not np.allclose(pivot[method], pivot["A"]):
                    test = wilcoxon(pivot[method], pivot["A"])
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                else:
                    statistic, p_value = float("nan"), float("nan")
                tests.append({
                    "treatment": method,
                    "control": "A",
                    "metric": metric,
                    "n_paired_seeds": len(pivot),
                    "treatment_mean": float(pivot[method].mean()) if len(pivot) else float("nan"),
                    "control_mean": float(pivot["A"].mean()) if len(pivot) else float("nan"),
                    "relative_change_percent": (
                        100.0 * (float(pivot[method].mean()) / float(pivot["A"].mean()) - 1.0)
                        if len(pivot) else float("nan")
                    ),
                    "wilcoxon_statistic": statistic,
                    "p_value_raw": p_value,
                })
    tests_frame = pd.DataFrame(tests)
    if len(tests_frame):
        tests_frame["p_value_holm"] = float("nan")
        valid = tests_frame["p_value_raw"].dropna().sort_values()
        running = 0.0
        total = len(valid)
        for rank, (index, value) in enumerate(valid.items()):
            running = max(running, (total - rank) * float(value))
            tests_frame.loc[index, "p_value_holm"] = min(1.0, running)
    tests_frame.to_csv(
        output / "trajectory_v2_paired_tests.csv",
        index=False, encoding="utf-8-sig",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences", type=Path,
        default=PROJECT_ROOT / "trajectory_sequences_h10.npz",
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).with_name("configs") / "experiment.json",
    )
    parser.add_argument("--methods", default="A,D_ADAPTIVE")
    parser.add_argument("--seeds", default="45,46,47,48,49,50,51")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-name", default="trajectory_prediction_v2")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--contrastive-weight", type=float)
    parser.add_argument("--correction-strength", type=float)
    parser.add_argument("--spectrum-dim", type=int)
    parser.add_argument("--structured-dim", type=int)
    parser.add_argument("--warmup-rounds", type=int)
    parser.add_argument("--ramp-rounds", type=int)
    parser.add_argument("--conflict-target", type=float)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    defaults = {
        "trajectory_contrastive_weight": 0.01,
        "trajectory_gsa_correction_strength": 0.03,
        "trajectory_spectrum_dim": 16,
        "trajectory_structured_dim": 128,
        "trajectory_gsa_warmup_rounds": 5,
        "trajectory_gsa_ramp_rounds": 10,
        "trajectory_conflict_target": 0.5,
        "trajectory_conflict_gate": True,
        "trajectory_rounds_v2": 80,
        "trajectory_early_stopping_patience": 15,
        "trajectory_early_stopping_min_delta": 1e-4,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    overrides = {
        "trajectory_contrastive_weight": args.contrastive_weight,
        "trajectory_gsa_correction_strength": args.correction_strength,
        "trajectory_spectrum_dim": args.spectrum_dim,
        "trajectory_structured_dim": args.structured_dim,
        "trajectory_gsa_warmup_rounds": args.warmup_rounds,
        "trajectory_gsa_ramp_rounds": args.ramp_rounds,
        "trajectory_conflict_target": args.conflict_target,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    data = np.load(args.sequences, allow_pickle=False)
    methods = [value.strip().upper() for value in args.methods.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output = PROJECT_ROOT / config["output_dir"] / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "trajectory_v2_runs.partial.csv"
    rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    completed = {(str(row["method"]), int(row["seed"])) for row in rows}
    for method in methods:
        for seed in seeds:
            if (method, seed) in completed:
                print(f"method={method} seed={seed} resumed", flush=True)
                continue
            print(f"method={method} seed={seed} training", flush=True)
            result = run_one(
                data, config, method, seed, torch.device(args.device),
                evaluate_test=not args.skip_test,
            )
            history = result.pop("_history")
            pd.DataFrame(history).to_csv(
                output / f"{method}_seed_{seed}_convergence.csv", index=False
            )
            rows.append(result)
            completed.add((method, seed))
            pd.DataFrame(rows).to_csv(partial, index=False, encoding="utf-8-sig")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "trajectory_v2_runs.csv", index=False, encoding="utf-8-sig")
    summarize(frame, output)
    (output / "protocol.json").write_text(json.dumps({
        "task": "AIS history plus weather to future 5/10 positions",
        "split": "MMSI-disjoint train/validation/test = approximately 70/10/20",
        "selection": "minimum mean of validation ADE@5 and ADE@10",
        "test_access": not args.skip_test,
        "development_seeds": seeds if args.skip_test else [],
        "formal_test_seeds": [] if args.skip_test else seeds,
        "methods": methods,
        "config": {
            key: config[key] for key in defaults
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
