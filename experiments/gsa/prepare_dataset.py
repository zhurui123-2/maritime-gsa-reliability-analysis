"""Build the four-modal table and quantify ship-type-node Non-IIDness.

This script never fabricates a cargo/EDI modality.  The fourth modality consists
only of measured EU MRV fuel and greenhouse-gas reporting fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMISSION_COLUMNS = [
    "total_fuel_consumption_tonnes", "fuel_on_laden_tonnes",
    "total_co2_emissions_tonnes", "co2_on_laden_tonnes",
    "total_ch4_emissions_tonnes", "total_n2o_emissions_tonnes",
    "total_co2eq_emissions_tonnes", "time_at_sea_hours",
    "fuel_per_distance_kg_per_nm", "co2_per_distance_kg_per_nm",
    "co2_per_hour_tonnes", "average_cargo_density",
]


def node_for_ship_type(value: object) -> str:
    text = str(value).strip().lower()
    if "passenger" in text or "ro-pax" in text:
        return "passenger"
    if "container" in text:
        return "container"
    if "bulk" in text:
        return "bulk"
    if any(token in text for token in ("tanker", "gas carrier", "lng carrier")):
        return "tanker"
    return "general_cargo"


def numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce")


def non_iid_report(frame: pd.DataFrame) -> dict:
    target = "total_co2_emissions_tonnes"
    valid_target = frame[target].dropna()
    if valid_target.nunique() >= 5:
        edges = np.unique(valid_target.quantile(np.linspace(0, 1, 6)).to_numpy())
        frame = frame.copy()
        frame["_emission_bin"] = pd.cut(frame[target], bins=edges, include_lowest=True, labels=False)
        categories = list(range(max(0, len(edges) - 1)))
    else:
        frame = frame.copy()
        frame["_emission_bin"] = np.nan
        categories = []
    global_distribution = (
        frame["_emission_bin"].value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
        if categories else np.array([])
    )
    nodes = {}
    for node, group in frame.groupby("federated_node", sort=True):
        distribution = (
            group["_emission_bin"].value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
            if categories else np.array([])
        )
        nodes[node] = {
            "samples": int(len(group)),
            "fraction": float(len(group) / len(frame)),
            "co2_mean": float(group[target].mean()) if group[target].notna().any() else None,
            "co2_std": float(group[target].std()) if group[target].notna().any() else None,
            "co2_missing_rate": float(group[target].isna().mean()),
            "emission_quintile_distribution": distribution.tolist(),
            "js_divergence_from_global": float(jensenshannon(distribution, global_distribution) ** 2)
            if categories and distribution.sum() > 0 and global_distribution.sum() > 0 else None,
        }
    counts = frame["federated_node"].value_counts().to_numpy(float)
    return {
        "node_definition": "ship type as a real-world data-ownership proxy",
        "mapping": {
            "container": "ship type contains container",
            "tanker": "oil/chemical tanker, gas carrier, or LNG carrier",
            "bulk": "bulk carrier",
            "general_cargo": "general/vehicle/refrigerated/ro-ro/other cargo",
            "passenger": "passenger/cruise/ro-pax",
        },
        "nodes": nodes,
        "sample_count_coefficient_of_variation": float(counts.std() / counts.mean()),
        "non_iid_metric": "Jensen-Shannon divergence between node and global CO2-emission quintile distributions",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "aligned_samples.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "multimodal_dataset.csv")
    parser.add_argument("--min-traj-points", type=int, default=10)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, low_memory=False)
    missing = [column for column in EMISSION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Aligned data does not contain real MRV emission fields. Re-run align_shipping_data.py first. "
            f"Missing: {missing}"
        )
    for column in EMISSION_COLUMNS:
        frame[column] = numeric_clean(frame[column])
    before_filter = len(frame)
    frame = frame[pd.to_numeric(frame["traj_points"], errors="coerce") >= args.min_traj_points].copy()
    frame["federated_node"] = frame["ShipType"].map(node_for_ship_type)
    frame["has_emissions"] = frame["total_co2_emissions_tonnes"].notna()
    if set(frame["federated_node"].unique()) != {"container", "tanker", "bulk", "general_cargo", "passenger"}:
        raise ValueError("At least one of the five pre-registered ship-type nodes has no samples.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    weather_error = pd.to_numeric(frame["weather_time_diff_hours"], errors="coerce")
    report = {
        "input_rows": before_filter,
        "rows": len(frame),
        "unique_mmsi": int(frame["MMSI"].nunique()),
        "mmsi_uniqueness_rate": float(frame["MMSI"].nunique() / len(frame)),
        "trajectory_threshold": args.min_traj_points,
        "trajectory_points": {
            "min": int(frame["traj_points"].min()), "median": float(frame["traj_points"].median()),
            "p95": float(frame["traj_points"].quantile(0.95)), "max": int(frame["traj_points"].max()),
        },
        "modality_complete": {
            "ais": int(frame["has_ais"].astype(bool).sum()),
            "weather": int(frame["has_weather"].astype(bool).sum()),
            "registry": int(frame["has_static"].astype(bool).sum()),
            "emissions": int(frame["has_emissions"].sum()),
        },
        "weather_interpolation_error_hours": {
            "mean": float(weather_error.mean()), "median": float(weather_error.median()),
            "p95": float(weather_error.quantile(0.95)), "max": float(weather_error.max()),
        },
        "non_iid": non_iid_report(frame),
        "integrity_note": "Fourth modality is measured EU MRV reporting data; no synthetic bill-of-lading data is used.",
    }
    args.output.with_suffix(".quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
