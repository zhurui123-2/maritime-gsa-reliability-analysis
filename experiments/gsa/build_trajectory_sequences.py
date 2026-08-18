"""Build AIS+HRRR sequences for genuine future-position prediction."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from align_shipping_data import clean_mmsi, list_weather_times, nearest_weather_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_FEATURES = [
    "LAT", "LON", "SOG", "cog_sin", "cog_cos", "surface_pressure",
    "temperature_2m", "relative_humidity_2m", "u_wind_10m", "v_wind_10m",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "multimodal_dataset.csv")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "trajectory_sequences_h10.npz")
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-points-per-ship", type=int, default=2000)
    parser.add_argument("--max-windows-per-ship", type=int, default=32)
    parser.add_argument("--max-weather-error-hours", type=float, default=1.0)
    parser.add_argument("--max-gap-minutes", type=float, default=60.0)
    args = parser.parse_args()
    entities = pd.read_csv(args.data, usecols=["MMSI", "federated_node"])
    target = set(entities["MMSI"].astype("int64"))
    node_lookup = dict(zip(entities["MMSI"].astype("int64"), entities["federated_node"].astype(str)))
    points: dict[int, list[pd.DataFrame]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    for path in sorted(args.data_dir.glob("AIS_*.csv")):
        for chunk in pd.read_csv(path, usecols=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"], chunksize=500_000):
            chunk["MMSI"] = clean_mmsi(chunk["MMSI"])
            chunk = chunk[chunk["MMSI"].isin(target)].dropna(subset=["MMSI"])
            for mmsi, group in chunk.groupby("MMSI", sort=False):
                remaining = args.max_points_per_ship - counts[int(mmsi)]
                if remaining > 0:
                    selected = group.head(remaining).copy()
                    points[int(mmsi)].append(selected)
                    counts[int(mmsi)] += len(selected)
    rows = []
    for mmsi, chunks in points.items():
        ship = pd.concat(chunks, ignore_index=True)
        ship["time"] = pd.to_datetime(ship["BaseDateTime"], errors="coerce", utc=True)
        ship = ship.dropna(subset=["time", "LAT", "LON"]).sort_values("time").drop_duplicates("time")
        ship["MMSI_real"] = mmsi
        rows.append(ship)
    observations = pd.concat(rows, ignore_index=True)
    weather_times = list_weather_times(args.data_dir).sort_values("weather_time").reset_index(drop=True)
    weather_ns = weather_times["weather_time"].astype("int64").to_numpy()
    observation_ns = observations["time"].astype("int64").to_numpy()
    location = np.searchsorted(weather_ns, observation_ns)
    left = np.clip(location - 1, 0, len(weather_ns) - 1)
    right = np.clip(location, 0, len(weather_ns) - 1)
    chosen = np.where(np.abs(weather_ns[right] - observation_ns) < np.abs(weather_ns[left] - observation_ns), right, left)
    observations["weather_path"] = weather_times.iloc[chosen]["path"].to_numpy()
    observations["weather_error_hours"] = np.abs(weather_ns[chosen] - observation_ns) / 3.6e12
    observations = observations[observations["weather_error_hours"] <= args.max_weather_error_hours].copy()
    weather_parts = []
    observations["point_id"] = np.arange(len(observations))
    for path, group in observations.groupby("weather_path", sort=False):
        request = pd.DataFrame({
            "MMSI": group["point_id"].to_numpy(),
            "sample_lat": pd.to_numeric(group["LAT"], errors="coerce"),
            "sample_lon": pd.to_numeric(group["LON"], errors="coerce"),
        })
        values = nearest_weather_values(Path(path), request).rename(columns={"MMSI": "point_id"})
        weather_parts.append(values)
    observations = observations.merge(pd.concat(weather_parts, ignore_index=True), on="point_id", how="inner")
    cog = np.deg2rad(pd.to_numeric(observations["COG"], errors="coerce"))
    observations["cog_sin"], observations["cog_cos"] = np.sin(cog), np.cos(cog)
    observations[SEQUENCE_FEATURES] = observations[SEQUENCE_FEATURES].apply(pd.to_numeric, errors="coerce")
    observations = observations.dropna(subset=SEQUENCE_FEATURES)
    x, y, mmsi_values, nodes = [], [], [], []
    for mmsi, ship in observations.groupby("MMSI_real", sort=False):
        ship = ship.sort_values("time")
        values = ship[SEQUENCE_FEATURES].to_numpy(np.float32)
        times = ship["time"].astype("int64").to_numpy()
        stop = len(values) - args.history - args.horizon + 1
        starts = np.arange(0, max(0, stop), args.stride)
        starts = np.asarray([
            start for start in starts
            if (
                np.diff(times[start:start + args.history + args.horizon]).max(initial=0)
                <= args.max_gap_minutes * 60.0 * 1e9
            )
        ], dtype=int)
        if len(starts) > args.max_windows_per_ship:
            starts = starts[np.linspace(0, len(starts) - 1, args.max_windows_per_ship, dtype=int)]
        for start in starts:
            x.append(values[start:start + args.history])
            y.append(values[start + args.history:start + args.history + args.horizon, :2])
            mmsi_values.append(mmsi)
            nodes.append(node_lookup[mmsi])
    if not x:
        raise ValueError("No valid trajectory windows were generated.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, x=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.float32),
        mmsi=np.asarray(mmsi_values), node=np.asarray(nodes), feature_names=np.asarray(SEQUENCE_FEATURES),
        history=args.history, horizon=args.horizon,
        max_windows_per_ship=args.max_windows_per_ship,
        max_gap_minutes=args.max_gap_minutes,
    )
    summary = pd.DataFrame({"MMSI": mmsi_values, "node": nodes}).groupby("node").agg(
        windows=("MMSI", "size"), unique_ships=("MMSI", "nunique")
    )
    print(summary.to_string(), flush=True)
    print(
        f"windows={len(x)} ships={len(set(mmsi_values))} "
        f"target_shape={np.asarray(y).shape} max_gap_minutes={args.max_gap_minutes}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
