"""Entity-retrieval metrics and seed-level statistical summaries."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def retrieval_metrics(query: np.ndarray, candidates: np.ndarray) -> dict[str, float]:
    query = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
    candidates = candidates / np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12, None)
    # ``@`` dispatches to the environment's BLAS runtime.  The Windows CUDA
    # environment used for the formal runs can terminate natively while loading
    # that runtime after a CUDA forward pass (0xc06d007f), without a Python
    # traceback.  Unoptimized einsum computes the identical dot products via
    # NumPy's internal loop and keeps evaluation deterministic and portable.
    similarity = np.einsum("id,jd->ij", query, candidates, optimize=False)
    correct = np.diag(similarity)
    # Average rank for exact ties; ranks are one-based.
    greater = (similarity > correct[:, None]).sum(axis=1)
    equal = (similarity == correct[:, None]).sum(axis=1) - 1
    ranks = 1.0 + greater + 0.5 * equal
    return {
        "n_test": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "hits_at_1": float(np.mean(ranks <= 1)),
        "hits_at_5": float(np.mean(ranks <= 5)),
        "median_rank": float(np.median(ranks)),
    }


def evaluate_all_pairs(embeddings: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    for left, right in combinations(sorted(embeddings), 2):
        for source, target in ((left, right), (right, left)):
            rows.append({
                "pair": f"{left}-{right}",
                "direction": f"{source}->{target}",
                **retrieval_metrics(embeddings[source], embeddings[target]),
            })
    return rows


def summarize_runs(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["ablation", "pair", "direction"]
    records = []
    for keys, group in raw.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["runs"] = int(group["seed"].nunique())
        for metric in ("mrr", "hits_at_1", "hits_at_5", "median_rank"):
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = float(stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width
            row[f"{metric}_ci95_high"] = mean + half_width
        records.append(row)
    return pd.DataFrame(records)


def paired_ablation_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered paired tests for the paper's three core comparisons."""
    comparisons = [
        ("D", "A", "full method vs bare FedAvg"),
        ("D", "B", "GSA contribution"),
        ("D", "C", "GSA vs BN"),
        ("D", "E", "contrastive complement"),
        ("D_NONLINEAR", "D", "nonlinear vs linear GSA"),
    ]
    seed_scores = raw.groupby(["ablation", "seed"], as_index=False)["mrr"].mean()
    rows = []
    for treatment, control, claim in comparisons:
        left = seed_scores[seed_scores["ablation"] == treatment].set_index("seed")["mrr"]
        right = seed_scores[seed_scores["ablation"] == control].set_index("seed")["mrr"]
        common = left.index.intersection(right.index)
        differences = (left.loc[common] - right.loc[common]).to_numpy()
        if len(differences) < 2 or np.allclose(differences, 0):
            statistic, p_value = np.nan, 1.0 if len(differences) else np.nan
        else:
            statistic, p_value = stats.wilcoxon(differences, alternative="two-sided")
        rows.append({
            "treatment": treatment, "control": control, "claim": claim,
            "n_paired_seeds": len(common), "mean_mrr_difference": float(differences.mean()) if len(differences) else np.nan,
            "wilcoxon_statistic": statistic, "p_value_raw": p_value,
        })
    result = pd.DataFrame(rows)
    # Holm correction across the pre-registered comparisons.
    # Partial formal runs (for example the priority A-vs-D run) leave the other
    # pre-registered comparisons unavailable.  Keep those adjusted p-values
    # missing rather than propagating the last finite Holm value into NaN rows.
    order = result["p_value_raw"].dropna().sort_values().index.tolist()
    adjusted = pd.Series(np.nan, index=result.index, dtype=float)
    running = 0.0
    # Retain the full pre-registered family size for conservative correction.
    m = len(comparisons)
    for rank, index in enumerate(order):
        running = max(running, float(result.loc[index, "p_value_raw"]) * (m - rank))
        adjusted.loc[index] = min(1.0, running)
    result["p_value_holm"] = adjusted
    return result
