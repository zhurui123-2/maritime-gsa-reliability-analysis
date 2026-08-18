"""Teacher-prioritized robustness: AIS missingness, AIS drift, client dropout."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import load_complete_entities, prepare_data
    from experiments.gsa.federated import FederatedTrainer
    from experiments.gsa.metrics import evaluate_all_pairs
else:
    from .data import load_complete_entities, prepare_data
    from .federated import FederatedTrainer
    from .metrics import evaluate_all_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "ais_missing_fraction": [0.0, 0.1, 0.2, 0.3],
    "ais_drift_km": [0.0, 1.0, 5.0, 10.0],
    "weather_missing_fraction": [0.0, 0.1, 0.3, 0.5],
    "client_dropout_rate": [0.0, 0.2, 0.4],
}


def _load_checkpoint(trainer: FederatedTrainer, path: Path, ablation: str) -> None:
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload["ablation"] != ablation:
        raise ValueError(f"Checkpoint mismatch: expected {ablation}, got {payload['ablation']}")
    trainer.model.load_state_dict(payload["states"]["model"])
    trainer.classifier.load_state_dict(payload["states"]["classifier"])
    if trainer.gsa is not None and payload["states"]["gsa"] is not None:
        trainer.gsa.load_state_dict(payload["states"]["gsa"])


def _corrupt_evaluation_features(
    prepared, frame: pd.DataFrame, condition: str, value: float, seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 2_607_230 + round(value * 10_000))
    evaluation = np.concatenate([prepared.splits.validation, prepared.splits.test])
    detail: dict[str, Any] = {"evaluation_entities": int(len(evaluation))}
    if condition == "ais_missing_fraction":
        count = int(round(value * len(evaluation)))
        missing = (
            rng.choice(evaluation, size=count, replace=False)
            if count else np.asarray([], dtype=int)
        )
        prepared.features["ais"][missing] = 0.0
        detail.update({
            "missing_entities": int(count),
            "imputation_at_evaluation": "clean-training standardized mean",
            "scope": "complete AIS feature vector for the sampled entities",
        })
    elif condition == "ais_drift_km":
        columns = prepared.preprocessing["ais"]["columns"]
        lat_index, lon_index = columns.index("avg_lat"), columns.index("avg_lon")
        scales = np.asarray(prepared.preprocessing["ais"]["scale"], dtype=float)
        north_km = rng.normal(0.0, value, len(evaluation))
        east_km = rng.normal(0.0, value, len(evaluation))
        latitude = pd.to_numeric(frame.iloc[evaluation]["avg_lat"], errors="coerce").to_numpy()
        latitude_delta = north_km / 111.32
        longitude_delta = east_km / (
            111.32 * np.clip(np.cos(np.deg2rad(latitude)), 0.15, None)
        )
        prepared.features["ais"][evaluation, lat_index] += latitude_delta / scales[lat_index]
        prepared.features["ais"][evaluation, lon_index] += longitude_delta / scales[lon_index]
        detail.update({
            "distribution": "independent zero-mean Gaussian north/east displacement",
            "sigma_km_per_axis": value,
        })
    elif condition == "weather_missing_fraction":
        count = int(round(value * len(evaluation)))
        missing = rng.choice(evaluation, size=count, replace=False) if count else np.asarray([], dtype=int)
        # Zero is the clean-training mean after StandardScaler. This evaluates
        # unavailable weather without refitting preprocessing on corrupted data.
        prepared.features["weather"][missing] = 0.0
        detail.update({
            "missing_entities": int(count),
            "imputation_at_evaluation": "clean-training standardized mean",
        })
    else:
        raise ValueError(condition)
    return detail


def _evaluate_checkpoint(
    frame: pd.DataFrame, base: dict[str, Any], ablation: str, seed: int,
    condition: str, value: float, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = prepare_data(
        frame, seed, float(base["test_fraction"]), float(base["validation_fraction"])
    )
    detail = _corrupt_evaluation_features(prepared, frame, condition, value, seed)
    trainer = FederatedTrainer(prepared, base, ablation, seed, device)
    checkpoint = (
        PROJECT_ROOT / base["output_dir"] / f"seed_{seed}" / ablation / "best_checkpoint.pt"
    )
    _load_checkpoint(trainer, checkpoint, ablation)
    results = evaluate_all_pairs(trainer.embeddings(prepared.splits.test))
    return [
        {"condition": condition, "value": value, "ablation": ablation, "seed": seed, **row}
        for row in results
    ], detail


def _summarize(frame: pd.DataFrame, output: Path) -> None:
    seed_scores = frame.groupby(
        ["condition", "value", "ablation", "seed"], as_index=False
    )[["mrr", "hits_at_1", "hits_at_5", "median_rank"]].mean()
    summary = seed_scores.groupby(
        ["condition", "value", "ablation"], as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", "std"),
        hits_at_1_mean=("hits_at_1", "mean"),
        hits_at_5_mean=("hits_at_5", "mean"),
        median_rank_mean=("median_rank", "mean"),
    )
    baselines = summary[summary["value"] == 0.0][
        ["condition", "ablation", "mrr_mean"]
    ].rename(columns={"mrr_mean": "clean_mrr_mean"})
    summary = summary.merge(baselines, on=["condition", "ablation"], how="left")
    summary["mrr_change_from_clean_percent"] = (
        100.0 * (summary["mrr_mean"] - summary["clean_mrr_mean"]) / summary["clean_mrr_mean"]
    )
    summary.to_csv(output / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    tests = []
    for (condition, value), group in seed_scores.groupby(["condition", "value"]):
        pivot = group[group["ablation"].isin(["A", "D"])].pivot(
            index="seed", columns="ablation", values="mrr"
        ).dropna()
        if len(pivot) and not np.allclose(pivot["D"] - pivot["A"], 0.0):
            test = wilcoxon(pivot["D"], pivot["A"], alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            statistic, p_value = np.nan, np.nan
        tests.append({
            "condition": condition,
            "value": value,
            "n_paired_seeds": len(pivot),
            "A_mean_mrr": float(pivot["A"].mean()) if len(pivot) else np.nan,
            "D_mean_mrr": float(pivot["D"].mean()) if len(pivot) else np.nan,
            "D_minus_A_mean_mrr": float((pivot["D"] - pivot["A"]).mean()) if len(pivot) else np.nan,
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })
    pd.DataFrame(tests).to_csv(
        output / "robustness_paired_D_vs_A.csv", index=False, encoding="utf-8-sig"
    )
    present_conditions = [
        condition for condition in CONDITIONS
        if condition in set(summary["condition"])
    ]
    figure, axes = plt.subplots(
        1, len(present_conditions), figsize=(3.7 * len(present_conditions), 3.6),
        constrained_layout=True, squeeze=False,
    )
    names = {
        "ais_missing_fraction": "AIS-missing fraction",
        "ais_drift_km": "AIS drift σ (km)",
        "weather_missing_fraction": "Weather-missing fraction",
        "client_dropout_rate": "Client-dropout rate",
    }
    colors = {"A": "#4c78a8", "D": "#e45756"}
    for axis, condition in zip(axes.flat, present_conditions):
        subset = summary[summary["condition"] == condition]
        for ablation in ("A", "D"):
            values = subset[subset["ablation"] == ablation].sort_values("value")
            axis.errorbar(
                values["value"], values["mrr_mean"], yerr=values["mrr_std"],
                marker="o", capsize=3, label=ablation, color=colors[ablation],
            )
        axis.set_xlabel(names[condition])
        axis.set_ylabel("Mean retrieval MRR")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, title="Variant", frameon=False, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, 1.08),
    )
    figure.savefig(output / "robustness_three_conditions.pdf", bbox_inches="tight")
    figure.savefig(output / "robustness_three_conditions.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ablations", default="A,D")
    parser.add_argument("--conditions", default="ais_missing_fraction,ais_drift_km,client_dropout_rate")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--output-name", default="robustness_teacher")
    parser.add_argument("--reuse-from", type=Path)
    args = parser.parse_args()
    base = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    frame = load_complete_entities(PROJECT_ROOT / base["input"])
    device = torch.device(args.device)
    result_root = PROJECT_ROOT / base["output_dir"]
    baseline = pd.read_csv(result_root / "retrieval_runs.csv")
    output = result_root / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "robustness_runs.partial.csv"
    if partial_path.exists():
        rows = pd.read_csv(partial_path).to_dict("records")
    elif args.reuse_from is not None and args.reuse_from.exists():
        reused = pd.read_csv(args.reuse_from)
        requested = {
            value.strip() for value in args.conditions.split(",") if value.strip()
        }
        rows = reused[reused["condition"].isin(requested)].to_dict("records")
    else:
        rows = []
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds.strip()
        else list(map(int, base["seeds"]))
    )
    requested_conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    unknown = sorted(set(requested_conditions).difference(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown robustness conditions: {unknown}")
    completed = {
        (str(row["ablation"]), str(row["condition"]), float(row["value"]), int(row["seed"]))
        for row in rows
    }
    protocol_details: list[dict[str, Any]] = []
    for ablation in [value.strip().upper() for value in args.ablations.split(",")]:
        for condition in requested_conditions:
            values = CONDITIONS[condition]
            for value in values:
                for seed in seeds:
                    key = (ablation, condition, value, seed)
                    if key in completed:
                        print(f"{key} resumed", flush=True)
                        continue
                    if value == 0.0:
                        clean = baseline[
                            (baseline["ablation"] == ablation) & (baseline["seed"] == seed)
                        ].to_dict("records")
                        rows.extend(
                            {"condition": condition, "value": value, **record}
                            for record in clean
                        )
                    elif condition in {
                        "ais_missing_fraction", "ais_drift_km",
                        "weather_missing_fraction",
                    }:
                        evaluated, detail = _evaluate_checkpoint(
                            frame, base, ablation, seed, condition, value, device
                        )
                        rows.extend(evaluated)
                        protocol_details.append({
                            "condition": condition, "value": value, "seed": seed, **detail
                        })
                    else:
                        config = copy.deepcopy(base)
                        config["client_dropout_rate"] = value
                        prepared = prepare_data(
                            frame, seed, float(config["test_fraction"]), float(config["validation_fraction"])
                        )
                        artifact = FederatedTrainer(prepared, config, ablation, seed, device).train()
                        rows.extend({
                            "condition": condition, "value": value, **record
                        } for record in artifact.test_rows)
                    completed.add(key)
                    pd.DataFrame(rows).to_csv(partial_path, index=False, encoding="utf-8-sig")
                    print(f"{key} completed", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "robustness_runs.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(json.dumps({
        "ablations": [value.strip().upper() for value in args.ablations.split(",")],
        "seeds": seeds,
        "conditions": {name: CONDITIONS[name] for name in requested_conditions},
        "feature_corruption_design": "clean trained checkpoint; corruption only on validation/test entities",
        "client_dropout_design": "independent client unavailability sampled each federated round, followed by retraining",
        "zero_condition": "identical deterministic formal baseline reused",
        "reused_from": str(args.reuse_from.resolve()) if args.reuse_from else None,
        "details": protocol_details,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _summarize(result, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
