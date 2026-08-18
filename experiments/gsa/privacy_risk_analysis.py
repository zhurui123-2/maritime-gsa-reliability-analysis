"""Reframe completed MIA runs as an honest privacy-risk analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["loss", "confidence", "entropy", "embedding_norm", "gradient_norm"]
ABLATIONS = ["A", "D", "D_NONLINEAR"]


def _operating_points(roc: pd.DataFrame, ablation: str) -> list[dict[str, float | str]]:
    rows = []
    for limit in (0.01, 0.05, 0.10):
        eligible = roc[roc["fpr"] <= limit]
        rows.append({
            "ablation": ablation,
            "maximum_fpr": limit,
            "maximum_tpr_at_or_below_fpr": float(eligible["tpr"].max()) if len(eligible) else 0.0,
        })
    rows.append({
        "ablation": ablation,
        "maximum_fpr": np.nan,
        "maximum_tpr_at_or_below_fpr": float((roc["tpr"] - roc["fpr"]).max()),
    })
    return rows


def _standardized_mean_differences(samples: pd.DataFrame, ablation: str) -> list[dict[str, float | str]]:
    members = samples[samples["membership"] == 1]
    nonmembers = samples[samples["membership"] == 0]
    rows = []
    for feature in FEATURES:
        pooled = np.sqrt(
            ((len(members) - 1) * members[feature].var(ddof=1)
             + (len(nonmembers) - 1) * nonmembers[feature].var(ddof=1))
            / max(len(samples) - 2, 1)
        )
        rows.append({
            "ablation": ablation,
            "feature": feature,
            "member_mean": float(members[feature].mean()),
            "nonmember_mean": float(nonmembers[feature].mean()),
            "standardized_mean_difference": float(
                (members[feature].mean() - nonmembers[feature].mean()) / max(pooled, 1e-12)
            ),
        })
    return rows


def _paired_auc_difference(
    left: pd.DataFrame, right: pd.DataFrame, seed: int = 20260723, repeats: int = 5000,
) -> dict[str, float]:
    if not np.array_equal(left["membership"].to_numpy(), right["membership"].to_numpy()):
        raise ValueError("Target membership labels are not paired across A and D.")
    y = left["membership"].to_numpy()
    left_score = left["attack_probability"].to_numpy()
    right_score = right["attack_probability"].to_numpy()
    observed = roc_auc_score(y, right_score) - roc_auc_score(y, left_score)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            differences.append(
                roc_auc_score(y[index], right_score[index])
                - roc_auc_score(y[index], left_score[index])
            )
    return {
        "D_minus_A_auc": float(observed),
        "bootstrap_ci95_low": float(np.quantile(differences, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_repeats": repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "gsa_if" / "mia",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output = input_dir.parent / "privacy_risk_analysis"
    output.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_dir / "mia_summary.csv")
    samples: dict[str, pd.DataFrame] = {}
    operating_rows, effect_rows = [], []
    figure, axis = plt.subplots(figsize=(6.0, 4.8), constrained_layout=True)
    colors = {"A": "#4c78a8", "D": "#e45756", "D_NONLINEAR": "#72b7b2"}
    labels = {"A": "FedAvg (A)", "D": "Linear GSA (D)", "D_NONLINEAR": "Nonlinear GSA"}
    for ablation in ABLATIONS:
        roc = pd.read_csv(input_dir / f"roc_{ablation}.csv")
        sample = pd.read_csv(input_dir / f"attack_samples_{ablation}.csv")
        samples[ablation] = sample
        operating_rows.extend(_operating_points(roc, ablation))
        effect_rows.extend(_standardized_mean_differences(sample, ablation))
        auc = float(summary.loc[summary["ablation"] == ablation, "attack_auc"].iloc[0])
        axis.plot(roc["fpr"], roc["tpr"], color=colors[ablation], linewidth=2, label=f"{labels[ablation]}: AUC={auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Privacy risk analysis: membership inference")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    figure.savefig(output / "privacy_risk_roc.pdf", bbox_inches="tight")
    figure.savefig(output / "privacy_risk_roc.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    operating = pd.DataFrame(operating_rows)
    effects = pd.DataFrame(effect_rows)
    comparison = _paired_auc_difference(samples["A"], samples["D"])
    summary.to_csv(output / "privacy_risk_summary.csv", index=False, encoding="utf-8-sig")
    operating.to_csv(output / "privacy_risk_operating_points.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(output / "privacy_feature_separation.csv", index=False, encoding="utf-8-sig")
    (output / "privacy_risk_conclusion.json").write_text(
        json.dumps({
            "title": "Privacy risk analysis",
            "paired_auc_comparison": comparison,
            "observed_interpretation": (
                "GSA is not a privacy defense in the present implementation. "
                "Its more coordinated representations are substantially more separable under membership inference, "
                "so privacy-preserving aggregation or explicit regularization is future work."
            ),
            "claim_prohibited": "Do not claim that GSA prevents membership inference or gradient inversion.",
            "shadow_models_per_variant": 50,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "summary": summary.to_dict("records"),
        "D_minus_A": comparison,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
