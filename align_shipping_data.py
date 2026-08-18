"""AIS + HRRR weather + EU MRV ship registry alignment.

It expects raw files under the repository-local ``input`` folder and reads the
very large AIS CSV files in chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Conda installs the ecCodes DLLs under <environment>/Library/bin on Windows.
# VS Code's ordinary "Run Python File" button may select the correct Python
# interpreter without adding that DLL directory to PATH. Keep the handle alive
# for the whole process so cfgrib/eccodes can always load their native library.
_ECCODES_DLL_HANDLE = None
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _eccodes_dll_dir = Path(sys.prefix) / "Library" / "bin"
    if _eccodes_dll_dir.is_dir():
        os.environ["PATH"] = str(_eccodes_dll_dir) + os.pathsep + os.environ.get("PATH", "")
        _ECCODES_DLL_HANDLE = os.add_dll_directory(str(_eccodes_dll_dir))


AIS_COLUMNS = [
    "MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "VesselType", "IMO"
]
MRV_COLUMNS = {
    "IMO Number": "IMO",
    "Name": "ShipName",
    "Ship type": "ShipType",
    "Reporting Period": "ReportingPeriod",
    "Technical efficiency": "TechnicalEfficiency",
    "Port of Registry": "PortOfRegistry",
    "Total fuel consumption [m tonnes]": "total_fuel_consumption_tonnes",
    "Fuel consumptions assigned to On laden [m tonnes]": "fuel_on_laden_tonnes",
    "Total CO₂ emissions [m tonnes]": "total_co2_emissions_tonnes",
    "CO₂ emissions assigned to On laden [m tonnes]": "co2_on_laden_tonnes",
    "Total CH₄ emissions [m tonnes]": "total_ch4_emissions_tonnes",
    "Total N₂O emissions [m tonnes]": "total_n2o_emissions_tonnes",
    "Total CO₂eq emissions [m tonnes]": "total_co2eq_emissions_tonnes",
    "Time spent at sea [hours]": "time_at_sea_hours",
    "Fuel consumption per distance [kg / n mile]": "fuel_per_distance_kg_per_nm",
    "CO₂ emissions per distance [kg CO₂ / n mile]": "co2_per_distance_kg_per_nm",
    "CO₂ emissions per time spent at sea [m tonnes CO₂ / hour]": "co2_per_hour_tonnes",
    "Average density of the cargo transported [m tonnes / m³]": "average_cargo_density",
}
WEATHER_NAMES = {
    "sp": "surface_pressure",
    "pres": "surface_pressure",
    "t2m": "temperature_2m",
    "2t": "temperature_2m",
    "r2": "relative_humidity_2m",
    "2r": "relative_humidity_2m",
    "u10": "u_wind_10m",
    "10u": "u_wind_10m",
    "v10": "v_wind_10m",
    "10v": "v_wind_10m",
}


def clean_imo(values: pd.Series) -> pd.Series:
    """Convert values such as 'IMO7938024' or 7938024.0 to nullable strings."""
    extracted = values.astype("string").str.extract(r"(?i)(?:IMO\s*)?(\d{7})", expand=False)
    return extracted.astype("string")


def clean_mmsi(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.between(100_000_000, 999_999_999)
    return numeric.where(valid).astype("Int64")


def read_mrv(path: Path) -> pd.DataFrame:
    """Read both MRV sheets; the real column names are on Excel row 3."""
    book = pd.ExcelFile(path)
    frames = []
    for sheet in book.sheet_names:
        header = pd.read_excel(path, sheet_name=sheet, header=2, nrows=0).columns
        wanted = [c for c in MRV_COLUMNS if c in header]
        if "IMO Number" not in wanted:
            print(f"  Warning: skip MRV sheet without IMO Number: {sheet}")
            continue
        frame = pd.read_excel(path, sheet_name=sheet, header=2, usecols=wanted)
        frame = frame.rename(columns=MRV_COLUMNS)
        frame["IMO"] = clean_imo(frame["IMO"])
        frame["MRVSheet"] = sheet
        frames.append(frame.dropna(subset=["IMO"]))
    if not frames:
        raise ValueError("No readable MRV sheet containing 'IMO Number' was found.")
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(subset="IMO", keep="first")


@dataclass
class ShipAgg:
    count: int = 0
    speed_count: int = 0
    speed_sum: float = 0.0
    speed_sq_sum: float = 0.0
    max_speed: float = float("-inf")
    lat_count: int = 0
    lat_sum: float = 0.0
    lon_sum: float = 0.0
    cog_count: int = 0
    cog_sin_sum: float = 0.0
    cog_cos_sum: float = 0.0

    def add_group(self, group: pd.DataFrame) -> None:
        self.count += len(group)
        speeds = pd.to_numeric(group["SOG"], errors="coerce").dropna().to_numpy(float)
        if speeds.size:
            self.speed_count += speeds.size
            self.speed_sum += float(speeds.sum())
            self.speed_sq_sum += float(np.square(speeds).sum())
            self.max_speed = max(self.max_speed, float(speeds.max()))
        lat = pd.to_numeric(group["LAT"], errors="coerce")
        lon = pd.to_numeric(group["LON"], errors="coerce")
        valid_pos = lat.between(-90, 90) & lon.between(-180, 180)
        if valid_pos.any():
            self.lat_count += int(valid_pos.sum())
            self.lat_sum += float(lat[valid_pos].sum())
            self.lon_sum += float(lon[valid_pos].sum())
        cog = pd.to_numeric(group["COG"], errors="coerce")
        cog = cog[cog.between(0, 360)]
        if len(cog):
            radians = np.deg2rad(cog.to_numpy(float))
            self.cog_count += len(cog)
            self.cog_sin_sum += float(np.sin(radians).sum())
            self.cog_cos_sum += float(np.cos(radians).sum())

    def finish(self) -> dict:
        avg_speed = self.speed_sum / self.speed_count if self.speed_count else np.nan
        variance = (
            max(0.0, self.speed_sq_sum / self.speed_count - avg_speed**2)
            if self.speed_count else np.nan
        )
        avg_cog = (
            np.degrees(np.arctan2(self.cog_sin_sum, self.cog_cos_sum)) % 360
            if self.cog_count else np.nan
        )
        return {
            "traj_points": self.count,
            "avg_speed": avg_speed,
            "max_speed": self.max_speed if self.speed_count else np.nan,
            "speed_std": np.sqrt(variance),
            "avg_lat": self.lat_sum / self.lat_count if self.lat_count else np.nan,
            "avg_lon": self.lon_sum / self.lat_count if self.lat_count else np.nan,
            "avg_cog": avg_cog,
        }


def first_pass(ais_files: list[Path], chunksize: int) -> tuple[dict, Counter, dict]:
    """Aggregate AIS features and count candidate MMSI-IMO pairs without full loading."""
    aggregates: dict[int, ShipAgg] = defaultdict(ShipAgg)
    imo_counts: Counter = Counter()
    vessel_types: dict[int, object] = {}
    total = 0
    for file_index, path in enumerate(ais_files, 1):
        print(f"  [{file_index}/{len(ais_files)}] {path.name}")
        for chunk in pd.read_csv(path, usecols=AIS_COLUMNS, chunksize=chunksize, low_memory=False):
            chunk["MMSI"] = clean_mmsi(chunk["MMSI"])
            chunk["IMO_clean"] = clean_imo(chunk["IMO"])
            chunk = chunk.dropna(subset=["MMSI"])
            chunk["MMSI"] = chunk["MMSI"].astype("int64")
            total += len(chunk)

            pairs = chunk.dropna(subset=["IMO_clean"])[["MMSI", "IMO_clean"]]
            # Update with grouped counts, rather than expanding millions of pairs in memory.
            imo_counts.update(pairs.value_counts().to_dict())

            types = chunk.dropna(subset=["VesselType"]).drop_duplicates("MMSI", keep="first")
            for row in types.itertuples(index=False):
                vessel_types.setdefault(int(row.MMSI), row.VesselType)

            for mmsi, group in chunk.groupby("MMSI", sort=False):
                aggregates[int(mmsi)].add_group(group)
    print(f"  Read {total:,} valid AIS rows, covering {len(aggregates):,} MMSIs")
    return aggregates, imo_counts, vessel_types


def choose_imo(imo_counts: Counter) -> pd.DataFrame:
    best: dict[int, tuple[str, int]] = {}
    for (mmsi, imo), count in imo_counts.items():
        if mmsi not in best or count > best[mmsi][1]:
            best[mmsi] = (imo, count)
    return pd.DataFrame(
        [(mmsi, imo, count) for mmsi, (imo, count) in best.items()],
        columns=["MMSI", "IMO", "imo_observations"],
    )


def list_weather_times(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("hrrr_3km_*.grib2")):
        match = re.search(r"(\d{10})$", path.stem)
        if not match:
            continue
        rows.append({"path": path, "weather_time": pd.to_datetime(match.group(1), format="%Y%m%d%H", utc=True)})
    return pd.DataFrame(rows)


def second_pass_representatives(
    ais_files: list[Path], target_mmsi: set[int], weather_times: pd.DataFrame, chunksize: int
) -> pd.DataFrame:
    """Keep, per target ship, the AIS point nearest to any available HRRR hour."""
    if weather_times.empty:
        return pd.DataFrame(columns=["MMSI", "ais_time", "weather_time", "weather_path", "weather_time_diff_hours", "sample_lat", "sample_lon"])
    weather_ns = weather_times["weather_time"].astype("int64").to_numpy()
    best: dict[int, tuple[int, dict]] = {}
    usecols = ["MMSI", "BaseDateTime", "LAT", "LON"]
    for file_index, path in enumerate(ais_files, 1):
        print(f"  weather-point pass [{file_index}/{len(ais_files)}] {path.name}")
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk["MMSI"] = clean_mmsi(chunk["MMSI"])
            chunk = chunk[chunk["MMSI"].isin(target_mmsi)].copy()
            if chunk.empty:
                continue
            chunk["ais_time"] = pd.to_datetime(chunk["BaseDateTime"], errors="coerce", utc=True)
            chunk["LAT"] = pd.to_numeric(chunk["LAT"], errors="coerce")
            chunk["LON"] = pd.to_numeric(chunk["LON"], errors="coerce")
            chunk = chunk.dropna(subset=["ais_time", "LAT", "LON"])
            chunk = chunk[chunk["LAT"].between(-90, 90) & chunk["LON"].between(-180, 180)]
            if chunk.empty:
                continue
            times_ns = chunk["ais_time"].astype("int64").to_numpy()
            pos = np.searchsorted(weather_ns, times_ns)
            left = np.clip(pos - 1, 0, len(weather_ns) - 1)
            right = np.clip(pos, 0, len(weather_ns) - 1)
            choose_right = np.abs(weather_ns[right] - times_ns) < np.abs(weather_ns[left] - times_ns)
            nearest_idx = np.where(choose_right, right, left)
            diff = np.abs(weather_ns[nearest_idx] - times_ns)
            chunk["weather_idx"] = nearest_idx
            chunk["diff_ns"] = diff
            for mmsi, group in chunk.groupby("MMSI", sort=False):
                row = group.loc[group["diff_ns"].idxmin()]
                record = {
                    "MMSI": int(mmsi), "ais_time": row["ais_time"],
                    "weather_time": weather_times.iloc[int(row["weather_idx"])] ["weather_time"],
                    "weather_path": weather_times.iloc[int(row["weather_idx"])] ["path"],
                    "weather_time_diff_hours": float(row["diff_ns"]) / 3.6e12,
                    "sample_lat": float(row["LAT"]), "sample_lon": float(row["LON"]),
                }
                if int(mmsi) not in best or int(row["diff_ns"]) < best[int(mmsi)][0]:
                    best[int(mmsi)] = (int(row["diff_ns"]), record)
    return pd.DataFrame([item[1] for item in best.values()])


def open_grib(path: Path):
    try:
        import cfgrib
    except ImportError as exc:
        raise RuntimeError("Weather reading needs cfgrib and eccodes. See README_VSCode.md.") from exc
    datasets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    return datasets


def find_weather_variable(datasets, aliases: set[str]):
    for dataset in datasets:
        for name in dataset.data_vars:
            attrs = dataset[name].attrs
            candidates = {name, str(attrs.get("GRIB_shortName", ""))}
            if candidates & aliases:
                return dataset[name]
    return None


def nearest_weather_values(path: Path, ships: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    datasets = open_grib(path)
    reference = next((d for d in datasets if "latitude" in d.coords and "longitude" in d.coords), None)
    if reference is None:
        raise ValueError(f"No latitude/longitude grid in {path.name}")
    lat_grid = np.asarray(reference["latitude"].values)
    lon_grid = np.asarray(reference["longitude"].values)
    lon_grid = ((lon_grid + 180) % 360) - 180
    # Scaled lon makes the nearest search less distorted at mid/high latitudes.
    mean_lat = float(np.nanmean(ships["sample_lat"]))
    scale = max(0.2, float(np.cos(np.deg2rad(mean_lat))))
    tree = cKDTree(np.column_stack([lat_grid.ravel(), lon_grid.ravel() * scale]))
    _, flat_idx = tree.query(np.column_stack([ships["sample_lat"], ships["sample_lon"] * scale]))

    out = ships[["MMSI"]].copy()
    for output_name in ["surface_pressure", "temperature_2m", "relative_humidity_2m", "u_wind_10m", "v_wind_10m"]:
        aliases = {key for key, value in WEATHER_NAMES.items() if value == output_name}
        variable = find_weather_variable(datasets, aliases)
        if variable is None:
            out[output_name] = np.nan
            continue
        values = np.asarray(variable.squeeze(drop=True).values).ravel()
        out[output_name] = values[flat_idx] if values.size == lat_grid.size else np.nan
    for dataset in datasets:
        dataset.close()
    return out


def extract_weather(representatives: pd.DataFrame, max_time_diff_hours: float) -> pd.DataFrame:
    if representatives.empty:
        return pd.DataFrame(columns=["MMSI"])
    rows = []
    for path, ships in representatives.groupby("weather_path", sort=False):
        print(f"  extracting {Path(path).name} for {len(ships)} ships")
        values = nearest_weather_values(Path(path), ships)
        rows.append(values)
    weather = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["MMSI"])
    metadata = representatives.drop(columns="weather_path").copy()
    result = metadata.merge(weather, on="MMSI", how="left")
    result["weather_within_tolerance"] = result["weather_time_diff_hours"] <= max_time_diff_hours
    value_cols = list(set(WEATHER_NAMES.values()))
    result.loc[~result["weather_within_tolerance"], value_cols] = np.nan
    return result


def parse_args() -> argparse.Namespace:
    default_data = Path(__file__).resolve().parent / "input"
    parser = argparse.ArgumentParser(description="Align AIS, HRRR and EU MRV by MMSI/IMO and time-space proximity.")
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("aligned_samples.csv"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--max-ships",
        type=int,
        default=0,
        help="Maximum matched ships to output; 0 means all (default, for formal experiments)",
    )
    parser.add_argument("--min-traj-points", type=int, default=10)
    parser.add_argument("--max-weather-time-diff-hours", type=float, default=1.0)
    parser.add_argument("--skip-weather", action="store_true", help="Align only AIS + MRV; useful for dependency/path testing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    ais_files = sorted(data_dir.glob("AIS_*.csv"))
    mrv_files = sorted(data_dir.glob("*EU MRV Publication of information.xlsx"))
    if not ais_files:
        raise FileNotFoundError(f"No AIS_*.csv files under {data_dir}")
    if not mrv_files:
        raise FileNotFoundError(f"No EU MRV workbook under {data_dir}")

    print("1/5 Read MRV registry")
    mrv = pd.concat([read_mrv(p) for p in mrv_files], ignore_index=True).drop_duplicates("IMO")
    print(f"  {len(mrv):,} unique IMO records")

    print("2/5 Stream and aggregate AIS")
    aggregates, imo_counts, vessel_types = first_pass(ais_files, args.chunksize)
    ais_imo = choose_imo(imo_counts)
    matched = ais_imo.merge(mrv, on="IMO", how="inner")
    features = pd.DataFrame([
        {"MMSI": mmsi, "vessel_type_ais": vessel_types.get(mmsi), **agg.finish()}
        for mmsi, agg in aggregates.items()
    ])
    matched = matched.merge(features, on="MMSI", how="inner")
    matched_before_traj_filter = len(matched)
    matched = matched[matched["traj_points"] >= args.min_traj_points]
    matched = matched.sort_values(["traj_points", "MMSI"], ascending=[False, True])
    if args.max_ships > 0:
        matched = matched.head(args.max_ships)
    print(
        f"  {matched_before_traj_filter:,} ships match AIS→IMO→MRV; "
        f"{len(matched):,} pass traj_points >= {args.min_traj_points}"
    )

    if args.skip_weather:
        result = matched.copy()
    else:
        print("3/5 Select AIS observations nearest available HRRR times")
        weather_times = list_weather_times(data_dir)
        representatives = second_pass_representatives(
            ais_files, set(matched["MMSI"].astype(int)), weather_times, args.chunksize
        )
        print("4/5 Extract nearest HRRR grid cells")
        weather = extract_weather(representatives, args.max_weather_time_diff_hours)
        result = matched.merge(weather, on="MMSI", how="left")

    print("5/5 Write output and quality report")
    result["has_ais"] = result["traj_points"].notna()
    result["has_static"] = result["ShipName"].notna()
    result["has_weather"] = result.get("temperature_2m", pd.Series(False, index=result.index)).notna()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    report = {
        "output": str(args.output.resolve()),
        "rows": len(result),
        "has_ais": int(result["has_ais"].sum()),
        "has_static": int(result["has_static"].sum()),
        "has_weather": int(result["has_weather"].sum()),
        "matched_before_trajectory_filter": matched_before_traj_filter,
        "min_trajectory_points": args.min_traj_points,
        "weather_time_error_hours": {
            "mean": float(pd.to_numeric(result.get("weather_time_diff_hours"), errors="coerce").mean()) if "weather_time_diff_hours" in result else None,
            "p95": float(pd.to_numeric(result.get("weather_time_diff_hours"), errors="coerce").quantile(0.95)) if "weather_time_diff_hours" in result else None,
            "max": float(pd.to_numeric(result.get("weather_time_diff_hours"), errors="coerce").max()) if "weather_time_diff_hours" in result else None,
        },
        "weather_note": "Weather values outside the configured time tolerance are deliberately blank.",
    }
    report_path = args.output.with_suffix(".quality_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
