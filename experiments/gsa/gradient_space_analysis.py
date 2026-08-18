"""Formal gradient-space evidence for heterogeneous multimodal federation.

This analysis never pads unequal parameter vectors and calls the result a raw
gradient cosine. Direct cross-modal coordinate-wise cosine/FedAvg is undefined
when dimensions differ. Instead, it compares:

1. cosine/distance in the same learned shared space, before and after the
   GSA correction; and
2. PCA/t-SNE/SVD on raw deltas before and after correction, using the same
   dimension-invariant, fixed-length normalized log-power-spectrum descriptor.

The script reuses the official ten-seed D checkpoints and generates auditable
per-seed caches, pairwise matrices, PCA/t-SNE coordinates, SVD spectra,
paired tests, and publication figures.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import MODALITIES, NODES, load_complete_entities, prepare_data
    from experiments.gsa.federated import FederatedTrainer
else:
    from .data import MODALITIES, NODES, load_complete_entities, prepare_data
    from .federated import FederatedTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("raw_fedavg_spectrum", "gsa_shared_spectrum")
CACHE_VERSION = 3


def _load_official_checkpoint(trainer: FederatedTrainer, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload.get("ablation") != "D":
        raise ValueError(f"Expected a D checkpoint, got {payload.get('ablation')!r}: {path}")
    states = payload["states"]
    trainer.model.load_state_dict(states["model"])
    trainer.classifier.load_state_dict(states["classifier"])
    if trainer.gsa is None or states["gsa"] is None:
        raise ValueError(f"Checkpoint has no GSA state: {path}")
    trainer.gsa.load_state_dict(states["gsa"])
    trainer.model.train()
    trainer.classifier.train()
    trainer.gsa.eval()
    return payload


def _spectrum_descriptor(vector: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Dimension-invariant normalized log-power-spectrum descriptor."""
    value = vector.detach().float().flatten()
    spectrum = torch.fft.rfft(value, norm="ortho").abs().square()
    spectrum = torch.log1p(spectrum)
    spectrum = F.interpolate(
        spectrum.reshape(1, 1, -1), size=output_dim, mode="linear", align_corners=False
    ).reshape(-1)
    return F.normalize(spectrum, dim=0)


def _pairwise_rows(
    descriptors: np.ndarray, seed: int, stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(MODALITIES):
        for right_index, right in enumerate(MODALITIES):
            left_value = descriptors[left_index]
            right_value = descriptors[right_index]
            cosine = float(np.dot(left_value, right_value) / (
                max(np.linalg.norm(left_value), 1e-12) * max(np.linalg.norm(right_value), 1e-12)
            ))
            rows.append({
                "seed": seed,
                "stage": stage,
                "left": left,
                "right": right,
                "cosine": cosine,
                "euclidean": float(np.linalg.norm(left_value - right_value)),
            })
    return rows


def _seed_summary(descriptors: np.ndarray, seed: int, stage: str) -> dict[str, Any]:
    cosines, distances = [], []
    consensus = descriptors.mean(axis=0)
    consensus /= max(np.linalg.norm(consensus), 1e-12)
    consensus_distances = []
    for left in range(len(MODALITIES)):
        consensus_distances.append(float(np.linalg.norm(descriptors[left] - consensus)))
        for right in range(left + 1, len(MODALITIES)):
            cosines.append(float(np.dot(descriptors[left], descriptors[right]) / (
                max(np.linalg.norm(descriptors[left]), 1e-12)
                * max(np.linalg.norm(descriptors[right]), 1e-12)
            )))
            distances.append(float(np.linalg.norm(descriptors[left] - descriptors[right])))
    return {
        "seed": seed,
        "stage": stage,
        "mean_pairwise_cosine": float(np.mean(cosines)),
        "mean_pairwise_euclidean": float(np.mean(distances)),
        "mean_distance_to_consensus": float(np.mean(consensus_distances)),
    }


def _bootstrap_mean_ci(values: np.ndarray, seed: int, repeats: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _holm(raw_p: list[float]) -> list[float]:
    order = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p), dtype=np.float64)
    running = 0.0
    count = len(raw_p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * raw_p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_tests(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("mean_pairwise_cosine", "higher_after"),
        ("mean_pairwise_euclidean", "lower_after"),
        ("mean_distance_to_consensus", "lower_after"),
    ]
    rows = []
    for metric, expected_direction in metrics:
        pivot = summary.pivot(index="seed", columns="stage", values=metric).dropna()
        before = pivot["raw_fedavg_spectrum"].to_numpy(dtype=float)
        after = pivot["gsa_shared_spectrum"].to_numpy(dtype=float)
        difference = after - before
        if np.allclose(difference, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            test = wilcoxon(after, before, alternative="two-sided", zero_method="wilcox")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append({
            "metric": metric,
            "expected_direction": expected_direction,
            "n_paired_seeds": len(pivot),
            "before_mean": float(before.mean()),
            "after_mean": float(after.mean()),
            "paired_mean_difference_after_minus_before": float(difference.mean()),
            "wilcoxon_statistic": statistic,
            "p_value_raw": p_value,
        })
    adjusted = _holm([float(row["p_value_raw"]) for row in rows])
    for row, p_value in zip(rows, adjusted):
        row["p_value_holm"] = p_value
    return pd.DataFrame(rows)


def _embedding_coordinates(descriptor_rows: pd.DataFrame, descriptors: np.ndarray, seed: int) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(descriptors)
    pca = PCA(n_components=2, random_state=seed).fit_transform(scaled)
    perplexity = min(10, max(2, len(scaled) // 4), len(scaled) - 1)
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        max_iter=2000,
    ).fit_transform(scaled)
    result = descriptor_rows.copy()
    result["pca_1"] = pca[:, 0]
    result["pca_2"] = pca[:, 1]
    result["tsne_1"] = tsne[:, 0]
    result["tsne_2"] = tsne[:, 1]
    return result


def _plot_cosine(pairwise: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
    for axis, stage, title in zip(
        axes,
        STAGES,
        ("Raw FedAvg gradient spectra", "GSA shared gradient spectra"),
    ):
        matrix = (
            pairwise[pairwise["stage"] == stage]
            .groupby(["left", "right"])["cosine"]
            .mean()
            .unstack()
            .reindex(index=MODALITIES, columns=MODALITIES)
        )
        image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(MODALITIES)), MODALITIES, rotation=35, ha="right")
        axis.set_yticks(range(len(MODALITIES)), MODALITIES)
        axis.set_title(title)
        for row in range(len(MODALITIES)):
            for column in range(len(MODALITIES)):
                axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axes, shrink=0.82, label="Cosine similarity")
    figure.savefig(output / "gradient_cosine_matrices.pdf", bbox_inches="tight")
    figure.savefig(output / "gradient_cosine_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_embeddings(coordinates: pd.DataFrame, output: Path) -> None:
    colors = {
        "ais": "#1f77b4",
        "weather": "#ff7f0e",
        "registry": "#2ca02c",
        "emissions": "#d62728",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), constrained_layout=True)
    panels = [
        ("pca_1", "pca_2", "raw_fedavg_spectrum", "PCA: Raw FedAvg spectra"),
        ("pca_1", "pca_2", "gsa_shared_spectrum", "PCA: GSA shared spectra"),
        ("tsne_1", "tsne_2", "raw_fedavg_spectrum", "t-SNE: Raw FedAvg spectra"),
        ("tsne_1", "tsne_2", "gsa_shared_spectrum", "t-SNE: GSA shared spectra"),
    ]
    for axis, (x_name, y_name, stage, title) in zip(axes.flat, panels):
        subset = coordinates[coordinates["stage"] == stage]
        for modality in MODALITIES:
            values = subset[subset["modality"] == modality]
            axis.scatter(values[x_name], values[y_name], s=34, alpha=0.82, label=modality, color=colors[modality])
        axis.set_title(title)
        axis.set_xlabel(x_name.replace("_", " ").upper())
        axis.set_ylabel(y_name.replace("_", " ").upper())
        axis.grid(alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.025),
        ncol=4, frameon=False,
    )
    figure.savefig(output / "gradient_spectrum_pca_tsne.pdf", bbox_inches="tight")
    figure.savefig(output / "gradient_spectrum_pca_tsne.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_before_after(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("mean_pairwise_cosine", "Mean pairwise cosine"),
        ("mean_pairwise_euclidean", "Mean pairwise Euclidean distance"),
        ("mean_distance_to_consensus", "Mean distance to consensus"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics):
        pivot = summary.pivot(index="seed", columns="stage", values=metric)
        for _, row in pivot.iterrows():
            axis.plot([0, 1], [row["raw_fedavg_spectrum"], row["gsa_shared_spectrum"]], color="#7f8c8d", alpha=0.45)
        means = [pivot["raw_fedavg_spectrum"].mean(), pivot["gsa_shared_spectrum"].mean()]
        axis.plot([0, 1], means, color="#b22222", marker="o", linewidth=2.3, label="Mean")
        axis.set_xticks([0, 1], ["Raw FedAvg", "GSA shared"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.savefig(output / "fedavg_vs_gsa_gradient_distance.pdf", bbox_inches="tight")
    figure.savefig(output / "fedavg_vs_gsa_gradient_distance.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_fedavg_effect(effect: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("mean_pairwise_cosine", "Mean cross-modal spectral cosine"),
        ("mean_pairwise_euclidean", "Mean cross-modal spectral distance"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics):
        pivot = effect.pivot(index="seed", columns="stage", values=metric)
        for _, row in pivot.iterrows():
            axis.plot(
                [0, 1], [row["client_local_pre_fedavg"], row["post_fedavg"]],
                color="#7f8c8d", alpha=0.45,
            )
        axis.plot(
            [0, 1],
            [pivot["client_local_pre_fedavg"].mean(), pivot["post_fedavg"].mean()],
            color="#b22222", marker="o", linewidth=2.3,
        )
        axis.set_xticks([0, 1], ["Client-local", "Post-FedAvg"])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "fedavg_before_after_unresolved_heterogeneity.pdf", bbox_inches="tight")
    figure.savefig(output / "fedavg_before_after_unresolved_heterogeneity.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_singular_spectrum(singular: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    labels = {
        "raw_fedavg_spectrum": "Raw FedAvg spectra",
        "gsa_shared_spectrum": "GSA shared spectra",
    }
    colors = {"raw_fedavg_spectrum": "#4c78a8", "gsa_shared_spectrum": "#e45756"}
    for stage in STAGES:
        subset = singular[singular["stage"] == stage]
        grouped = subset.groupby("component")["normalized_singular_value"]
        mean = grouped.mean()
        std = grouped.std(ddof=1)
        x = mean.index.to_numpy()
        axis.plot(x, mean.to_numpy(), marker="o", label=labels[stage], color=colors[stage])
        axis.fill_between(x, (mean - std).to_numpy(), (mean + std).to_numpy(), alpha=0.18, color=colors[stage])
    axis.set_xlabel("Singular-value index")
    axis.set_ylabel("Normalized singular value")
    axis.set_xticks(range(1, len(MODALITIES) + 1))
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output / "gradient_singular_spectrum.pdf", bbox_inches="tight")
    figure.savefig(output / "gradient_singular_spectrum.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _load_alignment_convergence(result_root: Path, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = result_root / f"seed_{seed}" / "D" / "gradient_spectrum_diagnostics.csv"
        frame = pd.read_csv(path)
        required = {"round", "projected_pairwise_cosine", "gsa_direction_loss", "gsa_spectrum_loss"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing convergence columns {sorted(missing)}: {path}")
        maximum = max(int(frame["round"].max()), 1)
        for row in frame.itertuples(index=False):
            rows.append({
                "seed": seed,
                "round": int(row.round),
                "normalized_progress": float(row.round) / maximum,
                "projected_pairwise_cosine": float(row.projected_pairwise_cosine),
                "gsa_direction_loss": float(row.gsa_direction_loss),
                "gsa_spectrum_loss": float(row.gsa_spectrum_loss),
            })
    return pd.DataFrame(rows)


def _plot_alignment_convergence(convergence: pd.DataFrame, output: Path) -> None:
    grid = np.linspace(0.0, 1.0, 101)
    interpolated = []
    figure, axis = plt.subplots(figsize=(6.5, 4.1), constrained_layout=True)
    for _, group in convergence.groupby("seed"):
        ordered = group.sort_values("normalized_progress")
        values = np.interp(
            grid,
            ordered["normalized_progress"].to_numpy(),
            ordered["projected_pairwise_cosine"].to_numpy(),
        )
        interpolated.append(values)
        axis.plot(grid, values, color="#4c78a8", alpha=0.15, linewidth=0.9)
    matrix = np.stack(interpolated)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
    axis.plot(grid, mean, color="#b22222", linewidth=2.4, label="10-seed mean")
    axis.fill_between(grid, mean - std, mean + std, color="#b22222", alpha=0.16, label="±1 SD")
    axis.set_xlabel("Normalized federated training progress")
    axis.set_ylabel("Projected pairwise cosine")
    axis.set_ylim(-0.15, 1.05)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(output / "gsa_alignment_convergence.pdf", bbox_inches="tight")
    figure.savefig(output / "gsa_alignment_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _alignment_convergence_summary(convergence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed, group in convergence.groupby("seed"):
        ordered = group.sort_values("round")
        rows.append({
            "seed": int(seed),
            "initial_projected_pairwise_cosine": float(ordered.iloc[0]["projected_pairwise_cosine"]),
            "final_projected_pairwise_cosine": float(ordered.iloc[-1]["projected_pairwise_cosine"]),
            "peak_projected_pairwise_cosine": float(ordered["projected_pairwise_cosine"].max()),
        })
    seed_frame = pd.DataFrame(rows)
    test = wilcoxon(
        seed_frame["final_projected_pairwise_cosine"],
        seed_frame["initial_projected_pairwise_cosine"],
        alternative="two-sided",
    )
    aggregate = pd.DataFrame([{
        "n_seeds": len(seed_frame),
        "initial_mean": float(seed_frame["initial_projected_pairwise_cosine"].mean()),
        "initial_std": float(seed_frame["initial_projected_pairwise_cosine"].std(ddof=1)),
        "final_mean": float(seed_frame["final_projected_pairwise_cosine"].mean()),
        "final_std": float(seed_frame["final_projected_pairwise_cosine"].std(ddof=1)),
        "peak_mean": float(seed_frame["peak_projected_pairwise_cosine"].mean()),
        "peak_std": float(seed_frame["peak_projected_pairwise_cosine"].std(ddof=1)),
        "wilcoxon_initial_vs_final_statistic": float(test.statistic),
        "wilcoxon_initial_vs_final_p_value": float(test.pvalue),
    }])
    return seed_frame, aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--analysis-dim", type=int, default=32)
    parser.add_argument("--probe-round-offset", type=int, default=1)
    parser.add_argument("--seeds", default="")
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds.strip()
        else [int(value) for value in config["seeds"]]
    )
    if len(seeds) < 2:
        raise ValueError("Formal paired gradient-space analysis requires at least two seeds.")
    frame = load_complete_entities(PROJECT_ROOT / config["input"])
    output = PROJECT_ROOT / config["output_dir"] / "gradient_space_analysis"
    cache = output / "seed_cache"
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(exist_ok=True)
    device = torch.device(args.device)

    all_raw_spectral, all_gsa_spectral = [], []
    pairwise_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    fedavg_effect_rows: list[dict[str, Any]] = []

    for seed in seeds:
        cache_path = cache / f"seed_{seed}.npz"
        cache_is_current = False
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            cache_is_current = (
                "cache_version" in cached.files
                and int(cached["cache_version"]) == CACHE_VERSION
            )
        if cache_is_current:
            pre_shared = cached["pre_shared"]
            post_shared = cached["post_shared"]
            pre_spectral = cached["pre_spectral"]
            post_spectral = cached["post_spectral"]
            client_spectral = cached["client_spectral"]
            dimensions = cached["dimensions"]
            best_round = int(cached["best_round"])
            print(f"seed={seed} resumed", flush=True)
        else:
            prepared = prepare_data(
                frame,
                seed,
                float(config["test_fraction"]),
                float(config["validation_fraction"]),
            )
            trainer = FederatedTrainer(prepared, config, "D", seed, device)
            checkpoint_path = PROJECT_ROOT / config["output_dir"] / f"seed_{seed}" / "D" / "best_checkpoint.pt"
            checkpoint = _load_official_checkpoint(trainer, checkpoint_path)
            best_round = int(checkpoint["best_round"])
            probe_round = best_round + int(args.probe_round_offset)
            modality_items: dict[str, list[tuple[float, torch.Tensor]]] = {name: [] for name in MODALITIES}
            client_spectral_parts = []
            for node in NODES:
                deltas, _, _, count = trainer._client_update(node, probe_round)
                for modality in MODALITIES:
                    modality_items[modality].append((float(count), deltas[modality].detach()))
                client_spectral_parts.append(torch.stack([
                    _spectrum_descriptor(deltas[name], int(args.analysis_dim))
                    for name in MODALITIES
                ]).cpu().numpy())
                print(f"seed={seed} node={node} probed", flush=True)
            aggregated = {
                name: trainer._weighted_average(values).detach()
                for name, values in modality_items.items()
            }
            dimensions = np.asarray([aggregated[name].numel() for name in MODALITIES], dtype=np.int64)
            with torch.no_grad():
                corrected, projected = trainer.gsa.align(
                    aggregated, float(config["gsa_correction_strength"])
                )
                corrected_projected = {
                    name: trainer.gsa.projectors[name].project(corrected[name])
                    for name in MODALITIES
                }
                pre_shared = torch.stack([
                    F.normalize(projected[name], dim=0)
                    for name in MODALITIES
                ]).cpu().numpy()
                post_shared = torch.stack([
                    F.normalize(corrected_projected[name], dim=0)
                    for name in MODALITIES
                ]).cpu().numpy()
                pre_spectral = torch.stack([
                    _spectrum_descriptor(aggregated[name], int(args.analysis_dim))
                    for name in MODALITIES
                ]).cpu().numpy()
                post_spectral = torch.stack([
                    _spectrum_descriptor(corrected[name], int(args.analysis_dim))
                    for name in MODALITIES
                ]).cpu().numpy()
                client_spectral = np.stack(client_spectral_parts)
            np.savez_compressed(
                cache_path,
                cache_version=np.asarray(CACHE_VERSION, dtype=np.int64),
                pre_shared=pre_shared,
                post_shared=post_shared,
                pre_spectral=pre_spectral,
                post_spectral=post_spectral,
                client_spectral=client_spectral,
                dimensions=dimensions,
                best_round=np.asarray(best_round, dtype=np.int64),
            )
            print(f"seed={seed} cached", flush=True)

        gsa_spectral = torch.stack([
            _spectrum_descriptor(torch.as_tensor(pre_shared[index]), int(args.analysis_dim))
            for index in range(len(MODALITIES))
        ]).cpu().numpy()
        all_raw_spectral.append(pre_spectral)
        all_gsa_spectral.append(gsa_spectral)
        local_summaries = [
            _seed_summary(client_spectral[index], seed, "client_local_pre_fedavg")
            for index in range(len(NODES))
        ]
        for metric in ("mean_pairwise_cosine", "mean_pairwise_euclidean", "mean_distance_to_consensus"):
            fedavg_effect_rows.append({
                "seed": seed,
                "stage": "client_local_pre_fedavg",
                metric: float(np.mean([row[metric] for row in local_summaries])),
            })
            fedavg_effect_rows.append({
                "seed": seed,
                "stage": "post_fedavg",
                metric: float(_seed_summary(pre_spectral, seed, "post_fedavg")[metric]),
            })
        for modality, dimension in zip(MODALITIES, dimensions):
            dimension_rows.append({
                "seed": seed,
                "modality": modality,
                "raw_gradient_dimension": int(dimension),
                "direct_cross_modal_coordinate_average_defined": False,
            })
        for stage, matrix, spectral_matrix in zip(
            STAGES,
            (pre_spectral, gsa_spectral),
            (pre_spectral, gsa_spectral),
        ):
            pairwise_rows.extend(_pairwise_rows(matrix, seed, stage))
            summary_rows.append(_seed_summary(matrix, seed, stage))
            singular = np.linalg.svd(spectral_matrix, compute_uv=False)
            normalized = singular / max(singular.sum(), 1e-12)
            for component, (raw_value, normalized_value) in enumerate(zip(singular, normalized), start=1):
                singular_rows.append({
                    "seed": seed,
                    "stage": stage,
                    "component": component,
                    "singular_value": float(raw_value),
                    "normalized_singular_value": float(normalized_value),
                })
    descriptors = np.concatenate([
        np.stack(all_raw_spectral, axis=0).reshape(-1, int(args.analysis_dim)),
        np.stack(all_gsa_spectral, axis=0).reshape(-1, int(args.analysis_dim)),
    ])
    # descriptor_rows were appended seed-by-seed / stage-by-stage. Rebuild the
    # row order to match the concatenated pre-then-post matrix exactly.
    ordered_rows = []
    for stage in STAGES:
        for seed in seeds:
            for modality in MODALITIES:
                ordered_rows.append({"seed": seed, "stage": stage, "modality": modality})
    descriptor_frame = pd.DataFrame(ordered_rows)
    coordinates = _embedding_coordinates(descriptor_frame, descriptors, seed=seeds[0] + 77_000)

    pairwise = pd.DataFrame(pairwise_rows)
    summary = pd.DataFrame(summary_rows)
    singular = pd.DataFrame(singular_rows)
    dimensions = pd.DataFrame(dimension_rows)
    tests = _paired_tests(summary)
    convergence = _load_alignment_convergence(PROJECT_ROOT / config["output_dir"], seeds)
    convergence_by_seed, convergence_summary = _alignment_convergence_summary(convergence)
    fedavg_effect_wide = (
        pd.DataFrame(fedavg_effect_rows)
        .groupby(["seed", "stage"], as_index=False)
        .first()
    )

    pairwise.to_csv(output / "gradient_cosine_and_distance_matrices.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "gradient_distance_by_seed.csv", index=False, encoding="utf-8-sig")
    singular.to_csv(output / "gradient_singular_spectrum.csv", index=False, encoding="utf-8-sig")
    dimensions.to_csv(output / "raw_gradient_dimensions.csv", index=False, encoding="utf-8-sig")
    coordinates.to_csv(output / "gradient_pca_tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(output / "paired_gradient_space_tests.csv", index=False, encoding="utf-8-sig")
    convergence.to_csv(output / "gsa_alignment_convergence.csv", index=False, encoding="utf-8-sig")
    convergence_by_seed.to_csv(output / "gsa_alignment_convergence_by_seed.csv", index=False, encoding="utf-8-sig")
    convergence_summary.to_csv(output / "gsa_alignment_convergence_summary.csv", index=False, encoding="utf-8-sig")
    fedavg_effect_wide.to_csv(
        output / "fedavg_before_after_cross_modal_distance.csv", index=False, encoding="utf-8-sig"
    )

    aggregate_rows = []
    for stage in STAGES:
        subset = summary[summary["stage"] == stage]
        for metric in ("mean_pairwise_cosine", "mean_pairwise_euclidean", "mean_distance_to_consensus"):
            values = subset[metric].to_numpy(dtype=float)
            low, high = _bootstrap_mean_ci(values, seed=seeds[0] + len(aggregate_rows))
            aggregate_rows.append({
                "stage": stage,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
            })
    pd.DataFrame(aggregate_rows).to_csv(output / "gradient_space_summary.csv", index=False, encoding="utf-8-sig")

    _plot_cosine(pairwise, output)
    _plot_embeddings(coordinates, output)
    _plot_before_after(summary, output)
    _plot_singular_spectrum(singular, output)
    _plot_alignment_convergence(convergence, output)
    _plot_fedavg_effect(fedavg_effect_wide, output)

    protocol = {
        "analysis": "formal gradient-space heterogeneity evidence",
        "checkpoint_ablation": "D",
        "seeds": seeds,
        "clients": list(NODES),
        "modalities": list(MODALITIES),
        "analysis_dimension": int(args.analysis_dim),
        "probe_round": "official best round plus fixed offset",
        "raw_coordinate_average": {
            "defined": False,
            "reason": "modality parameter-gradient dimensions differ",
            "dimensions": {
                modality: int(dimensions.loc[dimensions["modality"] == modality, "raw_gradient_dimension"].iloc[0])
                for modality in MODALITIES
            },
        },
        "raw_fedavg_representation": "normalized log-power spectrum of each unequal-dimensional raw FedAvg modality delta, linearly resampled to a fixed analysis dimension",
        "gsa_shared_representation": "the identical normalized log-power-spectrum descriptor applied to each learned shared-space projected gradient",
        "pca_tsne_svd": "joint analysis on comparable fixed-length spectral descriptors; t-SNE is visualization-only",
        "alignment_convergence": "projected pairwise cosine from every official D training round; the ten variable-length runs are interpolated only for the normalized-progress visualization",
        "fedavg_before_after": "cross-modal spectral dispersion is averaged over five client-local updates and compared with modality-wise FedAvg aggregates; this tests whether FedAvg alone resolves cross-modal heterogeneity",
        "paired_statistics": "two-sided paired Wilcoxon across official seeds with Holm correction",
        "claim_boundary": "The analysis tests cross-space coordination; it does not claim that modality-internal FedAvg is undefined.",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "seeds": len(seeds),
        "paired_tests": tests.to_dict("records"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
