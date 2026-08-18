"""Leakage-safe four-modal preparation with five ship-type federated nodes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODALITY_COLUMNS = {
    "ais": ["traj_points", "avg_speed", "max_speed", "speed_std", "avg_lat", "avg_lon", "avg_cog"],
    "weather": ["surface_pressure", "temperature_2m", "relative_humidity_2m", "u_wind_10m", "v_wind_10m", "weather_time_diff_hours"],
    "registry_numeric": ["vessel_type_ais"],
    "registry_categorical": ["ReportingPeriod", "ShipType", "PortOfRegistry", "TechnicalEfficiency", "MRVSheet"],
    "emissions": [
        "total_fuel_consumption_tonnes", "fuel_on_laden_tonnes", "total_co2_emissions_tonnes",
        "co2_on_laden_tonnes", "total_ch4_emissions_tonnes", "total_n2o_emissions_tonnes",
        "total_co2eq_emissions_tonnes", "time_at_sea_hours", "fuel_per_distance_kg_per_nm",
        "co2_per_distance_kg_per_nm", "co2_per_hour_tonnes", "average_cargo_density",
    ],
}
MODALITIES = ["ais", "weather", "registry", "emissions"]
NODES = ["container", "tanker", "bulk", "general_cargo", "passenger"]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass
class PreparedData:
    entity_ids: np.ndarray
    node_labels: np.ndarray
    features: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    splits: SplitIndices
    node_indices: dict[str, dict[str, np.ndarray]]
    input_dims: dict[str, int]
    preprocessing: dict[str, Any]
    split_hash: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False) if pd.api.types.is_bool_dtype(series) else series.astype("string").str.lower().isin({"true", "1", "yes"})


def load_complete_entities(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = ["MMSI", "IMO", "federated_node", "has_ais", "has_weather", "has_static", "has_emissions"]
    required += MODALITY_COLUMNS["ais"] + MODALITY_COLUMNS["weather"] + MODALITY_COLUMNS["registry_numeric"]
    required += MODALITY_COLUMNS["registry_categorical"] + MODALITY_COLUMNS["emissions"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Run prepare_dataset.py.")
    if frame["MMSI"].duplicated().any():
        raise ValueError("MMSI must be unique.")
    for flag in ("has_ais", "has_weather", "has_static", "has_emissions"):
        frame = frame[_truthy(frame[flag])]
    unknown_nodes = sorted(set(frame["federated_node"]) - set(NODES))
    if unknown_nodes:
        raise ValueError(f"Unknown federated nodes: {unknown_nodes}")
    if set(frame["federated_node"]) != set(NODES):
        raise ValueError("All five ship-type nodes must be present.")
    if len(frame) < 500:
        raise ValueError(f"Only {len(frame)} complete four-modal entities; formal experiment requires at least 500.")
    return frame.reset_index(drop=True)


def make_stratified_split(frame: pd.DataFrame, seed: int, test_fraction: float, validation_fraction: float) -> SplitIndices:
    indices = np.arange(len(frame))
    train_validation, test = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=frame["federated_node"]
    )
    relative_validation = validation_fraction / (1.0 - test_fraction)
    train, validation = train_test_split(
        train_validation, test_size=relative_validation, random_state=seed + 10_000,
        stratify=frame.iloc[train_validation]["federated_node"],
    )
    return SplitIndices(np.sort(train), np.sort(validation), np.sort(test))


def _numeric_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def _pipeline_metadata(pipeline: Pipeline) -> dict[str, list[float]]:
    return {
        "imputer_statistics": np.asarray(pipeline.named_steps["impute"].statistics_).tolist(),
        "scale_mean": np.asarray(pipeline.named_steps["scale"].mean_).tolist(),
        "scale": np.asarray(pipeline.named_steps["scale"].scale_).tolist(),
    }


def prepare_data(
    frame: pd.DataFrame, seed: int, test_fraction: float, validation_fraction: float,
    splits_override: SplitIndices | None = None,
) -> PreparedData:
    splits = splits_override or make_stratified_split(frame, seed, test_fraction, validation_fraction)
    train = frame.iloc[splits.train]
    features: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for modality in ("ais", "weather", "emissions"):
        configured_columns = MODALITY_COLUMNS[modality]
        columns = [column for column in configured_columns if train[column].notna().any()]
        if not columns:
            raise ValueError(f"Modality {modality} has no non-empty training features.")
        pipeline = _numeric_pipeline()
        features[modality] = pipeline.fit(train[columns]).transform(frame[columns]).astype(np.float32)
        metadata[modality] = {
            "columns": columns,
            "excluded_all_missing_columns": sorted(set(configured_columns) - set(columns)),
            **_pipeline_metadata(pipeline),
        }
    registry_numeric_pipeline = _numeric_pipeline()
    registry_numeric = registry_numeric_pipeline.fit(train[MODALITY_COLUMNS["registry_numeric"]]).transform(
        frame[MODALITY_COLUMNS["registry_numeric"]]
    )
    registry_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    registry_encoder.fit(train[MODALITY_COLUMNS["registry_categorical"]].fillna("<MISSING>").astype(str))
    registry_categorical = registry_encoder.transform(
        frame[MODALITY_COLUMNS["registry_categorical"]].fillna("<MISSING>").astype(str)
    )
    features["registry"] = np.concatenate([registry_numeric.astype(np.float32), registry_categorical], axis=1)
    metadata["registry"] = {
        "numeric_columns": MODALITY_COLUMNS["registry_numeric"],
        "categorical_columns": MODALITY_COLUMNS["registry_categorical"],
        "numeric": _pipeline_metadata(registry_numeric_pipeline),
        "categories": [values.tolist() for values in registry_encoder.categories_],
    }
    # Targets are kept outside modality inputs by task runners when target leakage would occur.
    targets = {
        "co2": pd.to_numeric(frame["total_co2_emissions_tonnes"], errors="coerce").to_numpy(np.float32),
        "position": frame[["sample_lat", "sample_lon"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32),
    }
    node_labels = frame["federated_node"].astype(str).to_numpy()
    node_indices: dict[str, dict[str, np.ndarray]] = {}
    for node in NODES:
        node_indices[node] = {
            split_name: indices[node_labels[indices] == node]
            for split_name, indices in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
        }
    split_payload = {
        name: frame.iloc[indices]["MMSI"].astype(str).tolist()
        for name, indices in (("train", splits.train), ("validation", splits.validation), ("test", splits.test))
    }
    split_hash = hashlib.sha256(json.dumps(split_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return PreparedData(
        entity_ids=frame["MMSI"].astype(str).to_numpy(), node_labels=node_labels, features=features,
        targets=targets, splits=splits, node_indices=node_indices,
        input_dims={name: values.shape[1] for name, values in features.items()},
        preprocessing=metadata, split_hash=split_hash,
    )
