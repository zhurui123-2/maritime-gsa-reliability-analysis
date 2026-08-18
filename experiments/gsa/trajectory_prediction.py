"""Federated AIS+weather 5/10-point trajectory prediction with ADE/FDE."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import wilcoxon
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.gsa import GradientSpectrumAlignment
else:
    from .gsa import GradientSpectrumAlignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrajectoryModel(nn.Module):
    def __init__(self, horizon: int, hidden: int = 64) -> None:
        super().__init__()
        self.horizon = horizon
        self.ais = nn.GRU(5, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.weather = nn.Sequential(
            nn.Conv1d(5, hidden // 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden // 2, hidden, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, horizon * 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, state = self.ais(x[:, :, :5])
        ais_embedding = state[-1]
        weather_embedding = self.weather(x[:, :, 5:].transpose(1, 2))
        prediction = self.head(torch.cat([ais_embedding, weather_embedding], dim=1))
        prediction = prediction.reshape(-1, self.horizon, 2)
        return prediction, F.normalize(ais_embedding, dim=1), F.normalize(weather_embedding, dim=1)


def flat_parameters(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


@torch.no_grad()
def apply_delta(parameters: list[nn.Parameter], delta: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.sub_(delta[offset:offset + count].view_as(parameter))
        offset += count


@torch.no_grad()
def haversine_km(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lat1, lon1 = torch.deg2rad(prediction[..., 0]), torch.deg2rad(prediction[..., 1])
    lat2, lon2 = torch.deg2rad(target[..., 0]), torch.deg2rad(target[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2).square() + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).square()
    return 6371.0088 * 2 * torch.atan2(torch.sqrt(a), torch.sqrt((1 - a).clamp_min(0)))


def run_one(data, config: dict, ablation: str, seed: int, device: torch.device) -> dict[str, float | int | str]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    raw_x, raw_y = data["x"], data["y"]
    groups, nodes = data["mmsi"], data["node"].astype(str)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_index, test_index = next(outer.split(raw_x, groups=groups))
    # Fit every normalization statistic on training ships only.  Predict a
    # future displacement from the last observed point, which is both
    # numerically stable and leakage-safe for held-out vessels.
    input_mean = raw_x[train_index].mean(axis=(0, 1), keepdims=True)
    input_std = raw_x[train_index].std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    x = (raw_x - input_mean) / input_std
    last_position = raw_x[:, -1, :2]
    if raw_y.ndim != 3 or raw_y.shape[1] < 10 or raw_y.shape[2] != 2:
        raise ValueError(f"Expected future-position sequences shaped [N, >=10, 2], got {raw_y.shape}")
    horizon = int(raw_y.shape[1])
    target_delta = raw_y - last_position[:, None, :]
    target_delta[:, :, 1] = (target_delta[:, :, 1] + 180.0) % 360.0 - 180.0
    target_mean = target_delta[train_index].mean(axis=0, keepdims=True)
    target_std = target_delta[train_index].std(axis=0, keepdims=True).clip(min=1e-6)
    y = (target_delta - target_mean) / target_std
    model = TrajectoryModel(horizon, int(config["hidden_dim"])).to(device)
    ais_parameters, weather_parameters = list(model.ais.parameters()), list(model.weather.parameters())
    head_parameters = list(model.head.parameters())
    nonlinear = ablation == "D_NONLINEAR"
    gsa = None if ablation == "A" else GradientSpectrumAlignment(
        {"ais": sum(p.numel() for p in ais_parameters), "weather": sum(p.numel() for p in weather_parameters)},
        int(config["spectrum_dim"]), nonlinear=nonlinear,
    ).to(device)
    gsa_optimizer = None if gsa is None else torch.optim.AdamW(gsa.parameters(), lr=float(config["gsa_learning_rate"]))
    rng = np.random.default_rng(seed)
    for round_index in range(int(config.get("trajectory_rounds", 50))):
        client_updates = []
        for node in sorted(set(nodes[train_index])):
            local = copy.deepcopy(model)
            local.ais.flatten_parameters()
            local_optimizer = torch.optim.AdamW(local.parameters(), lr=float(config["learning_rate"]))
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
                    contrastive = 1.0 - F.cosine_similarity(ais_embedding, weather_embedding, dim=1).mean()
                    loss = task_loss + (float(config["contrastive_weight"]) * contrastive if ablation != "A" else 0.0)
                    loss.backward()
                    local_optimizer.step()
            client_updates.append((
                len(candidates),
                flat_parameters(ais_parameters) - flat_parameters(list(local.ais.parameters())),
                flat_parameters(weather_parameters) - flat_parameters(list(local.weather.parameters())),
                flat_parameters(head_parameters) - flat_parameters(list(local.head.parameters())),
            ))
        total = sum(item[0] for item in client_updates)
        ais_delta = sum(count * delta for count, delta, _, _ in client_updates) / total
        weather_delta = sum(count * delta for count, _, delta, _ in client_updates) / total
        head_delta = sum(count * delta for count, _, _, delta in client_updates) / total
        if gsa is not None and gsa_optimizer is not None:
            corrected, projected = gsa.align({"ais": ais_delta, "weather": weather_delta}, float(config["gsa_correction_strength"]))
            gsa_loss, _ = gsa.alignment_loss(projected)
            gsa_optimizer.zero_grad(set_to_none=True)
            gsa_loss.backward()
            gsa_optimizer.step()
            ais_delta, weather_delta = corrected["ais"].detach(), corrected["weather"].detach()
        apply_delta(ais_parameters, ais_delta)
        apply_delta(weather_parameters, weather_delta)
        apply_delta(head_parameters, head_delta)
        if (round_index + 1) % 10 == 0:
            print(
                f"ablation={ablation} seed={seed} round={round_index + 1}/"
                f"{int(config.get('trajectory_rounds', 50))}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        prediction, _, _ = model(torch.as_tensor(x[test_index], dtype=torch.float32, device=device))
        scale = torch.as_tensor(target_std, dtype=torch.float32, device=device)
        center = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
        origin = torch.as_tensor(last_position[test_index, None, :], dtype=torch.float32, device=device)
        prediction = origin + prediction * scale + center
        prediction[:, :, 0].clamp_(-90.0, 90.0)
        prediction[:, :, 1] = torch.remainder(prediction[:, :, 1] + 180.0, 360.0) - 180.0
        truth = torch.as_tensor(raw_y[test_index], dtype=torch.float32, device=device)
        distance = haversine_km(prediction, truth).cpu().numpy()
    return {
        "ablation": ablation, "seed": seed, "n_test_windows": len(test_index),
        "ade_5_km": float(distance[:, :5].mean()),
        "fde_5_km": float(distance[:, 4].mean()),
        "ade_10_km": float(distance[:, :10].mean()),
        "fde_10_km": float(distance[:, 9].mean()),
    }


def _write_summary(frame: pd.DataFrame, output: Path) -> None:
    metrics = ["ade_5_km", "fde_5_km", "ade_10_km", "fde_10_km"]
    summary = frame.groupby("ablation")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.reset_index().to_csv(output / "trajectory_summary.csv", index=False, encoding="utf-8-sig")
    tests = []
    if {"A", "D"}.issubset(set(frame["ablation"])):
        for metric in metrics:
            pivot = frame[frame["ablation"].isin(["A", "D"])].pivot(
                index="seed", columns="ablation", values=metric
            ).dropna()
            test = wilcoxon(pivot["D"], pivot["A"], alternative="two-sided")
            tests.append({
                "comparison": "D_vs_A",
                "metric": metric,
                "n_paired_seeds": len(pivot),
                "A_mean": float(pivot["A"].mean()),
                "D_mean": float(pivot["D"].mean()),
                "D_minus_A": float((pivot["D"] - pivot["A"]).mean()),
                "relative_change_percent": float(
                    100.0 * (pivot["D"].mean() - pivot["A"].mean()) / pivot["A"].mean()
                ),
                "wilcoxon_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            })
    test_frame = pd.DataFrame(tests)
    if len(test_frame):
        order = test_frame["p_value"].sort_values().index
        adjusted = pd.Series(index=test_frame.index, dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (len(test_frame) - rank) * float(test_frame.loc[index, "p_value"]))
            adjusted.loc[index] = min(1.0, running)
        test_frame["p_value_holm"] = adjusted
    test_frame.to_csv(output / "trajectory_paired_tests.csv", index=False, encoding="utf-8-sig")
    plot_summary = frame.groupby("ablation")[metrics].agg(["mean", "std"])
    variants = [value for value in ("A", "D") if value in plot_summary.index]
    x = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    colors = {"A": "#4c78a8", "D": "#e45756"}
    for offset, ablation in enumerate(variants):
        means = plot_summary.loc[ablation].xs("mean", level=1).to_numpy()
        standard_deviations = plot_summary.loc[ablation].xs("std", level=1).to_numpy()
        axis.bar(
            x + (offset - (len(variants) - 1) / 2) * width,
            means, width, yerr=standard_deviations, capsize=3,
            label=ablation, color=colors[ablation],
        )
    axis.set_xticks(x, ["ADE@5", "FDE@5", "ADE@10", "FDE@10"])
    axis.set_ylabel("Geodesic error (km; mean ± SD)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="Variant", frameon=False)
    figure.savefig(output / "trajectory_ade_fde.pdf", bbox_inches="tight")
    figure.savefig(output / "trajectory_ade_fde.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=Path, default=PROJECT_ROOT / "trajectory_sequences_h10.npz")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--ablations", default="A,D")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    data = np.load(args.sequences, allow_pickle=False)
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds.strip()
        else list(map(int, config["seeds"]))
    )
    output = PROJECT_ROOT / config["output_dir"] / "trajectory_prediction"
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "trajectory_runs.partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {(str(row["ablation"]), int(row["seed"])) for row in rows}
    for ablation in [x.strip().upper() for x in args.ablations.split(",")]:
        for seed in seeds:
            if (ablation, seed) in completed:
                print(f"ablation={ablation} seed={seed} resumed", flush=True)
                continue
            print(f"ablation={ablation} seed={seed} training", flush=True)
            rows.append(run_one(data, config, ablation, seed, torch.device(args.device)))
            completed.add((ablation, seed))
            pd.DataFrame(rows).to_csv(partial_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "trajectory_runs.csv", index=False, encoding="utf-8-sig")
    _write_summary(pd.DataFrame(rows), output)
    (output / "protocol.json").write_text(json.dumps({
        "task": "AIS + pointwise HRRR weather -> future vessel position",
        "split": "MMSI-disjoint GroupShuffleSplit",
        "normalization": "training ships only; target is normalized displacement from the final history point",
        "history": int(data["history"]), "horizon": int(data["horizon"]),
        "windows": int(len(data["x"])), "unique_mmsi": int(len(np.unique(data["mmsi"]))),
        "trajectory_rounds": int(config.get("trajectory_rounds", 50)),
        "trajectory_local_epochs": int(config.get("trajectory_local_epochs", 1)),
        "trajectory_batch_size": int(config.get("trajectory_batch_size", 256)),
        "seeds": seeds,
        "ablations": [x.strip().upper() for x in args.ablations.split(",")],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
