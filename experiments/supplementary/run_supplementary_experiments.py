"""Supplementary experiments for the RESS maritime reliability study.

This script provides three additional analyses stored separately from the
primary experiments:

1. Local-only evaluation audit under a common vessel-level test set.
2. A safety-oriented high/low operational-exposure proxy with strict target/input
   separation and fixed-representation linear probing.
3. Low-frequency GSA under moderate stale-update stress, reporting task damage,
   gradient-spectrum control, worst-client performance, calls, and runtime.

The script uses only three new seeds by default and writes all row-level results,
protocol settings, and summaries to a separate output directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "multimodal_dataset.csv"
DEFAULT_OUT = ROOT / "results" / "supplementary"
CLIENTS = ["bulk", "container", "general_cargo", "passenger", "tanker"]

# Safety-proxy target variables are deliberately absent from every input group.
TARGET_ONLY_RISK_COLUMNS = [
    "max_speed",
    "speed_std",
    "surface_pressure",
    "u_wind_10m",
    "v_wind_10m",
]

GROUPS = {
    "ais": ["traj_points", "avg_speed", "avg_cog", "avg_lat", "avg_lon"],
    "weather": [
        "temperature_2m",
        "relative_humidity_2m",
        "weather_time_diff_hours",
        "sample_lat",
        "sample_lon",
    ],
    "registry": ["imo_observations", "vessel_type_ais"],
    "emissions": [
        "total_fuel_consumption_tonnes",
        "time_at_sea_hours",
        "fuel_per_distance_kg_per_nm",
    ],
}


@dataclass(frozen=True)
class Protocol:
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    hidden_dim: int = 32
    embedding_dim: int = 16
    gsa_alpha: float = 0.35
    gsa_direction_floor: float = 0.90
    staleness_decay: float = 0.70
    stress_start: int = 11
    stress_end: int = 20
    moderate_lags: tuple[int, ...] = (0, 1, 2, 2, 3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


class MultimodalKnowledgeModel(nn.Module):
    """Four separated encoders, gated fusion, and a five-class knowledge head."""

    def __init__(self, dims: dict[str, int], hidden: int, embedding: int) -> None:
        super().__init__()
        self.names = list(GROUPS)
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dims[name], hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, embedding),
                )
                for name in self.names
            }
        )
        self.gate = nn.Linear(embedding * len(self.names), len(self.names))
        self.head = nn.Linear(embedding, 5)

    def forward(self, features: dict[str, torch.Tensor], return_embedding: bool = False):
        encoded = [self.encoders[name](features[name]) for name in self.names]
        joined = torch.cat(encoded, dim=1)
        weights = torch.softmax(self.gate(joined), dim=1)
        fused = sum(weights[:, i : i + 1] * encoded[i] for i in range(len(encoded)))
        logits = self.head(fused)
        return (logits, fused) if return_embedding else logits


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = (
        ["MMSI", "federated_node", "total_co2_emissions_tonnes"]
        + sum(GROUPS.values(), [])
        + TARGET_ONLY_RISK_COLUMNS
    )
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[df["federated_node"].isin(CLIENTS)].copy()
    if df["MMSI"].duplicated().any():
        raise ValueError("MMSI is not vessel-unique; entity leakage audit failed.")
    for column in sum(GROUPS.values(), []) + TARGET_ONLY_RISK_COLUMNS + ["total_co2_emissions_tonnes"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.reset_index(drop=True)


def make_split(df: pd.DataFrame, seed: int):
    # A provisional global quintile only stabilizes split composition. The final
    # task thresholds are estimated again from the training subset alone.
    provisional = pd.qcut(
        df["total_co2_emissions_tonnes"].rank(method="first"), 5, labels=False
    )
    train_val, test = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=seed, stratify=provisional
    )
    provisional_tv = provisional.iloc[train_val]
    train, val = train_test_split(
        train_val,
        test_size=0.1875,
        random_state=seed + 1000,
        stratify=provisional_tv,
    )
    train = np.asarray(sorted(train))
    val = np.asarray(sorted(val))
    test = np.asarray(sorted(test))
    thresholds = np.quantile(
        df.loc[train, "total_co2_emissions_tonnes"].to_numpy(float), [0.2, 0.4, 0.6, 0.8]
    )
    labels = np.digitize(df["total_co2_emissions_tonnes"].to_numpy(float), thresholds).astype(np.int64)
    return train, val, test, thresholds, labels


def prepare_features(df: pd.DataFrame, train_idx: np.ndarray):
    arrays: dict[str, np.ndarray] = {}
    scalers: dict[str, RobustScaler] = {}
    for name, columns in GROUPS.items():
        block = df[columns].copy()
        medians = block.iloc[train_idx].median(numeric_only=True)
        block = block.fillna(medians).fillna(0.0)
        scaler = RobustScaler().fit(block.iloc[train_idx].to_numpy(float))
        arrays[name] = scaler.transform(block.to_numpy(float)).astype(np.float32)
        scalers[name] = scaler
    return arrays, scalers


def subset_tensors(arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device):
    return {
        name: torch.tensor(values[indices], dtype=torch.float32, device=device)
        for name, values in arrays.items()
    }


def model_state(model: nn.Module):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.detach().clone() for key, value in state.items()})


def train_local(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    indices: np.ndarray,
    protocol: Protocol,
    device: torch.device,
    epochs: int | None = None,
) -> None:
    epochs = protocol.local_epochs if epochs is None else epochs
    x = np.concatenate([arrays[name][indices] for name in GROUPS], axis=1)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(labels[indices], dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=protocol.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    offsets = np.cumsum([0] + [arrays[name].shape[1] for name in GROUPS])
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            features = {
                name: xb[:, offsets[i] : offsets[i + 1]]
                for i, name in enumerate(GROUPS)
            }
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.no_grad()
def predict(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    return_embedding: bool = False,
):
    model.eval()
    features = subset_tensors(arrays, indices, device)
    if return_embedding:
        logits, embedding = model(features, return_embedding=True)
        return torch.softmax(logits, 1).cpu().numpy(), embedding.cpu().numpy()
    return torch.softmax(model(features), 1).cpu().numpy()


def multiclass_metrics(y: np.ndarray, probabilities: np.ndarray):
    probabilities = np.clip(probabilities.astype(float), 1e-8, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    pred = probabilities.argmax(1)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y, probabilities, labels=[0, 1, 2, 3, 4])),
    }


def worst_client_f1(y: np.ndarray, probabilities: np.ndarray, nodes: np.ndarray) -> float:
    values = []
    for client in CLIENTS:
        mask = nodes == client
        if mask.any():
            values.append(
                f1_score(y[mask], probabilities[mask].argmax(1), average="macro", zero_division=0)
            )
    return float(min(values))


def registered_matrix_keys(state: dict[str, torch.Tensor]) -> list[str]:
    names = [
        "encoders.ais.2.weight",
        "encoders.weather.2.weight",
        "encoders.registry.2.weight",
        "encoders.emissions.2.weight",
        "gate.weight",
        "head.weight",
    ]
    return [key for key in names if key in state and state[key].ndim == 2]


def normalized_spectrum(matrix: torch.Tensor) -> np.ndarray:
    singular = torch.linalg.svdvals(matrix.float()).detach().cpu().numpy()
    singular = np.maximum(singular, 1e-12)
    return singular / singular.sum()


def spectrum_discrepancy(deltas: list[dict[str, torch.Tensor]], keys: list[str]) -> float:
    block_values = []
    for key in keys:
        spectra = [normalized_spectrum(delta[key]) for delta in deltas]
        pairs = []
        for i in range(len(spectra)):
            for j in range(i + 1, len(spectra)):
                pairs.append(float(jensenshannon(spectra[i], spectra[j], base=2.0) ** 2))
        if pairs:
            block_values.append(float(np.mean(pairs)))
    return float(np.mean(block_values)) if block_values else 0.0


def align_gsa(
    deltas: list[dict[str, torch.Tensor]],
    keys: list[str],
    alpha: float,
    direction_floor: float,
):
    aligned = [{key: value.clone() for key, value in delta.items()} for delta in deltas]
    used_alphas = []
    for key in keys:
        decomposed = [torch.linalg.svd(delta[key].float(), full_matrices=False) for delta in deltas]
        spectra = [s / (s.sum() + 1e-12) for _, s, _ in decomposed]
        consensus = torch.median(torch.stack(spectra), dim=0).values
        consensus = consensus / (consensus.sum() + 1e-12)
        for index, (u, singular, vh) in enumerate(decomposed):
            original = deltas[index][key].float()
            current_alpha = alpha
            accepted = original
            while current_alpha >= alpha / 16.0:
                target = (1.0 - current_alpha) * singular + current_alpha * consensus * singular.sum()
                candidate = (u * target.unsqueeze(0)) @ vh
                candidate = candidate * (
                    torch.linalg.norm(original) / (torch.linalg.norm(candidate) + 1e-12)
                )
                cosine = nn.functional.cosine_similarity(original.flatten(), candidate.flatten(), dim=0)
                if float(cosine) >= direction_floor:
                    accepted = candidate.to(original.dtype)
                    break
                current_alpha *= 0.5
            aligned[index][key] = accepted.to(deltas[index][key].dtype)
            used_alphas.append(current_alpha if current_alpha >= alpha / 16.0 else 0.0)
    return aligned, float(np.mean(used_alphas)) if used_alphas else 0.0


def aggregate_deltas(
    deltas: list[dict[str, torch.Tensor]],
    sample_sizes: list[int],
    lags: list[int],
    protocol: Protocol,
):
    raw = np.asarray(sample_sizes, dtype=float) * np.exp(-protocol.staleness_decay * np.asarray(lags))
    weights = raw / raw.sum()
    result = {}
    for key in deltas[0]:
        result[key] = sum(float(weight) * delta[key] for weight, delta in zip(weights, deltas))
    return result


def run_federated(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    protocol: Protocol,
    device: torch.device,
    stress: bool,
    gsa_frequency: int | None,
):
    set_seed(seed)
    dims = {name: arrays[name].shape[1] for name in GROUPS}
    model = MultimodalKnowledgeModel(dims, protocol.hidden_dim, protocol.embedding_dim).to(device)
    history_states = [model_state(model)]
    val_nll, dspec_pre, dspec_post, alpha_used = [], [], [], []
    gsa_calls = 0
    start = time.perf_counter()
    train_nodes = df.loc[train_idx, "federated_node"].to_numpy()
    client_indices = {
        client: train_idx[train_nodes == client]
        for client in CLIENTS
        if np.any(train_nodes == client)
    }
    for round_index in range(protocol.rounds):
        current = model_state(model)
        in_stress = stress and protocol.stress_start <= round_index + 1 <= protocol.stress_end
        lags = list(protocol.moderate_lags if in_stress else (0,) * len(CLIENTS))
        deltas, sizes, used_lags = [], [], []
        for client_position, client in enumerate(CLIENTS):
            indices = client_indices[client]
            lag = min(lags[client_position], len(history_states) - 1)
            source_state = history_states[-1 - lag]
            local = copy.deepcopy(model)
            load_state(local, source_state)
            train_local(local, arrays, labels, indices, protocol, device)
            local_state = model_state(local)
            deltas.append({key: local_state[key] - current[key] for key in current})
            sizes.append(len(indices))
            used_lags.append(lag)
        keys = registered_matrix_keys(current)
        before = spectrum_discrepancy(deltas, keys)
        apply_now = gsa_frequency is not None and (round_index + 1) % gsa_frequency == 0
        if apply_now:
            deltas, actual_alpha = align_gsa(
                deltas, keys, protocol.gsa_alpha, protocol.gsa_direction_floor
            )
            gsa_calls += 1
        else:
            actual_alpha = 0.0
        after = spectrum_discrepancy(deltas, keys)
        aggregated = aggregate_deltas(deltas, sizes, used_lags, protocol)
        updated = {key: current[key] + aggregated[key] for key in current}
        load_state(model, updated)
        history_states.append(model_state(model))
        val_prob = predict(model, arrays, val_idx, device)
        val_nll.append(multiclass_metrics(labels[val_idx], val_prob)["nll"])
        dspec_pre.append(before)
        dspec_post.append(after)
        alpha_used.append(actual_alpha)
    runtime = time.perf_counter() - start
    test_prob, test_embedding = predict(model, arrays, test_idx, device, return_embedding=True)
    test_metric = multiclass_metrics(labels[test_idx], test_prob)
    test_metric["worst_client_f1"] = worst_client_f1(
        labels[test_idx], test_prob, df.loc[test_idx, "federated_node"].to_numpy()
    )
    return {
        "model": model,
        "val_nll": np.asarray(val_nll),
        "dspec_pre": np.asarray(dspec_pre),
        "dspec_post": np.asarray(dspec_post),
        "alpha_used": np.asarray(alpha_used),
        "gsa_calls": gsa_calls,
        "runtime_sec": runtime,
        "test_embedding": test_embedding,
        **test_metric,
    }


def run_local_audit(
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    protocol: Protocol,
    device: torch.device,
):
    set_seed(seed)
    dims = {name: arrays[name].shape[1] for name in GROUPS}
    train_nodes = df.loc[train_idx, "federated_node"].to_numpy()
    test_nodes = df.loc[test_idx, "federated_node"].to_numpy()
    local_models = {}
    cross_rows = []
    for client in CLIENTS:
        local_train = train_idx[train_nodes == client]
        model = MultimodalKnowledgeModel(dims, protocol.hidden_dim, protocol.embedding_dim).to(device)
        train_local(model, arrays, labels, local_train, protocol, device, epochs=protocol.rounds)
        local_models[client] = model
        for test_client in CLIENTS:
            local_test = test_idx[test_nodes == test_client]
            prob = predict(model, arrays, local_test, device)
            metric = multiclass_metrics(labels[local_test], prob)
            cross_rows.append(
                {
                    "seed": seed,
                    "trained_client": client,
                    "tested_client": test_client,
                    "n_test": len(local_test),
                    **metric,
                }
            )
    personalized_prob = np.zeros((len(test_idx), 5), dtype=float)
    for client in CLIENTS:
        mask = test_nodes == client
        personalized_prob[mask] = predict(local_models[client], arrays, test_idx[mask], device)
    personalized = multiclass_metrics(labels[test_idx], personalized_prob)
    personalized["worst_client_f1"] = worst_client_f1(
        labels[test_idx], personalized_prob, test_nodes
    )
    global_metrics = []
    for client, model in local_models.items():
        metric = multiclass_metrics(labels[test_idx], predict(model, arrays, test_idx, device))
        global_metrics.append({"client": client, **metric})
    return personalized, global_metrics, cross_rows


def build_risk_proxy(df: pd.DataFrame, train_idx: np.ndarray):
    target = df[TARGET_ONLY_RISK_COLUMNS].copy()
    medians = target.iloc[train_idx].median(numeric_only=True)
    target = target.fillna(medians).fillna(0.0)
    wind = np.sqrt(target["u_wind_10m"] ** 2 + target["v_wind_10m"] ** 2)
    constructed = pd.DataFrame(
        {
            "max_speed": target["max_speed"],
            "speed_std": target["speed_std"],
            "low_pressure": -target["surface_pressure"],
            "wind_speed": wind,
        }
    )
    scaler = RobustScaler().fit(constructed.iloc[train_idx].to_numpy(float))
    z = scaler.transform(constructed.to_numpy(float))
    score = 0.30 * z[:, 0] + 0.30 * z[:, 1] + 0.20 * z[:, 2] + 0.20 * z[:, 3]
    threshold = float(np.quantile(score[train_idx], 0.75))
    return (score >= threshold).astype(int), score, threshold


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    total = len(y)
    return float(
        sum(
            len(chunk) / total * abs(float(y[chunk].mean()) - float(p[chunk].mean()))
            for chunk in chunks
            if len(chunk)
        )
    )


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.linspace(0.10, 0.90, 161)
    scores = [f1_score(y, p >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def safety_probe(
    model: nn.Module,
    arrays: dict[str, np.ndarray],
    risk_labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
):
    _, train_embedding = predict(model, arrays, train_idx, device, return_embedding=True)
    _, val_embedding = predict(model, arrays, val_idx, device, return_embedding=True)
    _, test_embedding = predict(model, arrays, test_idx, device, return_embedding=True)
    probe = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    probe.fit(train_embedding, risk_labels[train_idx])
    val_prob = probe.predict_proba(val_embedding)[:, 1]
    threshold = best_f1_threshold(risk_labels[val_idx], val_prob)
    test_prob = probe.predict_proba(test_embedding)[:, 1]
    y = risk_labels[test_idx]
    return {
        "auc": float(roc_auc_score(y, test_prob)),
        "f1": float(f1_score(y, test_prob >= threshold, zero_division=0)),
        "brier": float(brier_score_loss(y, test_prob)),
        "ece_10bin": expected_calibration_error(y, test_prob, bins=10),
        "threshold_from_validation": threshold,
        "test_positive_rate": float(y.mean()),
    }


def summarize(frame: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(by, key))
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run(args) -> None:
    protocol = Protocol(rounds=args.rounds)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    df = load_data()
    all_lowfreq, all_safety, all_local, all_cross, all_global_local, audits = [], [], [], [], [], []

    frequencies = {"SA_only": None, "SA_GSA_h1": 1, "SA_GSA_h3": 3, "SA_GSA_h5": 5}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        train_idx, val_idx, test_idx, thresholds, labels = make_split(df, seed)
        arrays, _ = prepare_features(df, train_idx)
        risk_labels, _, risk_threshold = build_risk_proxy(df, train_idx)
        audits.append(
            {
                "seed": seed,
                "n_total": len(df),
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "n_test": len(test_idx),
                "mmsi_unique": int(df["MMSI"].nunique()),
                "co2_thresholds": ";".join(f"{value:.8g}" for value in thresholds),
                "risk_threshold_train_only": risk_threshold,
                "risk_positive_train": float(risk_labels[train_idx].mean()),
                "risk_positive_validation": float(risk_labels[val_idx].mean()),
                "risk_positive_test": float(risk_labels[test_idx].mean()),
                "target_only_columns": ";".join(TARGET_ONLY_RISK_COLUMNS),
                "input_columns": ";".join(sum(GROUPS.values(), [])),
            }
        )

        clean_runs = {}
        for method, frequency in frequencies.items():
            print(f"{method}: clean", flush=True)
            clean = run_federated(
                arrays, labels, df, train_idx, val_idx, test_idx, seed, protocol, device,
                stress=False, gsa_frequency=frequency,
            )
            print(f"{method}: moderate stale stress", flush=True)
            stressed = run_federated(
                arrays, labels, df, train_idx, val_idx, test_idx, seed, protocol, device,
                stress=True, gsa_frequency=frequency,
            )
            clean_runs[method] = clean
            stress_slice = slice(protocol.stress_start - 1, protocol.stress_end)
            recovery_slice = slice(protocol.stress_end, protocol.rounds)
            damage = float(np.sum(stressed["val_nll"][stress_slice] - clean["val_nll"][stress_slice]))
            recovery = float(
                np.sum(stressed["val_nll"][recovery_slice] - clean["val_nll"][recovery_slice])
            )
            all_lowfreq.append(
                {
                    "seed": seed,
                    "method": method,
                    "gsa_frequency": frequency or 0,
                    "gsa_calls": stressed["gsa_calls"],
                    "stress_damage_nll": damage,
                    "recovery_residual_nll": recovery,
                    "stress_dspec_pre": float(stressed["dspec_pre"][stress_slice].mean()),
                    "stress_dspec_post": float(stressed["dspec_post"][stress_slice].mean()),
                    "test_macro_f1": stressed["macro_f1"],
                    "test_nll": stressed["nll"],
                    "worst_client_f1": stressed["worst_client_f1"],
                    "runtime_sec": stressed["runtime_sec"],
                    "mean_actual_alpha": float(stressed["alpha_used"].mean()),
                }
            )
            probe = safety_probe(
                clean["model"], arrays, risk_labels, train_idx, val_idx, test_idx, device
            )
            all_safety.append({"seed": seed, "representation": method, **probe})

        print("Local-only common-test audit", flush=True)
        personalized, global_local, cross = run_local_audit(
            arrays, labels, df, train_idx, test_idx, seed, protocol, device
        )
        fedavg_clean = clean_runs["SA_only"]
        all_local.extend(
            [
                {"seed": seed, "evaluation": "Local_personalized_routing", **personalized},
                {
                    "seed": seed,
                    "evaluation": "FedAvg_single_global_model",
                    "macro_f1": fedavg_clean["macro_f1"],
                    "nll": fedavg_clean["nll"],
                    "worst_client_f1": fedavg_clean["worst_client_f1"],
                },
                {
                    "seed": seed,
                    "evaluation": "Local_models_global_test_mean",
                    "macro_f1": float(np.mean([row["macro_f1"] for row in global_local])),
                    "nll": float(np.mean([row["nll"] for row in global_local])),
                    "worst_client_f1": np.nan,
                },
            ]
        )
        all_cross.extend(cross)
        all_global_local.extend({"seed": seed, **row} for row in global_local)

    lowfreq = pd.DataFrame(all_lowfreq)
    safety = pd.DataFrame(all_safety)
    local = pd.DataFrame(all_local)
    cross = pd.DataFrame(all_cross)
    global_local = pd.DataFrame(all_global_local)
    audit = pd.DataFrame(audits)

    lowfreq.to_csv(out / "low_frequency_gsa_by_seed.csv", index=False, encoding="utf-8-sig")
    safety.to_csv(out / "safety_proxy_by_seed.csv", index=False, encoding="utf-8-sig")
    local.to_csv(out / "local_only_audit_by_seed.csv", index=False, encoding="utf-8-sig")
    cross.to_csv(out / "local_cross_evaluation.csv", index=False, encoding="utf-8-sig")
    global_local.to_csv(out / "local_global_test_by_model.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "data_and_leakage_audit.csv", index=False, encoding="utf-8-sig")

    lowfreq_summary = summarize(
        lowfreq,
        ["method", "gsa_frequency"],
        [
            "gsa_calls",
            "stress_damage_nll",
            "recovery_residual_nll",
            "stress_dspec_post",
            "test_macro_f1",
            "test_nll",
            "worst_client_f1",
            "runtime_sec",
        ],
    )
    safety_summary = summarize(
        safety,
        ["representation"],
        ["auc", "f1", "brier", "ece_10bin", "test_positive_rate"],
    )
    local_summary = summarize(
        local,
        ["evaluation"],
        ["macro_f1", "nll", "worst_client_f1"],
    )
    lowfreq_summary.to_csv(out / "low_frequency_gsa_summary.csv", index=False, encoding="utf-8-sig")
    safety_summary.to_csv(out / "safety_proxy_summary.csv", index=False, encoding="utf-8-sig")
    local_summary.to_csv(out / "local_only_audit_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "status": "supplementary_analysis",
        "date": "2026-08-06",
        "data_path": str(DATA_PATH),
        "output_path": str(out),
        "device": str(device),
        "seeds": args.seeds,
        "protocol": asdict(protocol),
        "input_groups": GROUPS,
        "safety_target_only_columns": TARGET_ONLY_RISK_COLUMNS,
        "safety_label": (
            "top-quartile train-defined operational-exposure score from max speed, "
            "speed variability, low pressure, and wind speed; all defining columns excluded from inputs"
        ),
        "interpretation_boundary": (
            "The safety label is an operational-exposure proxy, not an observed accident outcome."
        ),
    }
    (out / "protocol.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# RESS supplementary experiments",
        "",
        "These analyses are stored separately from the primary experiments.",
        "",
        "## Low-frequency GSA",
        "",
        "```csv",
        lowfreq_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Safety-oriented operational-exposure proxy",
        "",
        "The label-defining variables are excluded from all encoder inputs. The label is a proxy and is not an accident outcome.",
        "",
        "```csv",
        safety_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Local-only evaluation audit",
        "",
        "```csv",
        local_summary.to_csv(index=False).strip(),
        "```",
    ]
    (out / "README_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print("\nDONE", flush=True)
    print(lowfreq_summary.to_string(index=False), flush=True)
    print(safety_summary.to_string(index=False), flush=True)
    print(local_summary.to_string(index=False), flush=True)
    print(f"Output: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[67, 68, 69])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
