"""Run the pre-registered A-E plus nonlinear GSA experiment (training entry only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import MODALITIES, NODES, load_complete_entities, prepare_data, sha256_file
    from experiments.gsa.federated import ABLATIONS, FederatedTrainer
    from experiments.gsa.metrics import paired_ablation_tests, summarize_runs
else:
    from .data import MODALITIES, NODES, load_complete_entities, prepare_data, sha256_file
    from .federated import ABLATIONS, FederatedTrainer
    from .metrics import paired_ablation_tests, summarize_runs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--ablations", default=None, help="Override config, e.g. A,D; primary run leaves this unset")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def choose_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if config["modalities"] != MODALITIES or config["nodes"] != NODES:
        raise ValueError("Modality/node order differs from the configured protocol.")
    ablations = config["ablations"] if args.ablations is None else [x.strip().upper() for x in args.ablations.split(",")]
    unknown = sorted(set(ablations) - set(ABLATIONS))
    if unknown:
        raise ValueError(f"Unknown ablations: {unknown}")
    input_path = project_path(config["input"]).resolve()
    output_dir = project_path(config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite explicitly.")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_complete_entities(input_path)
    device = choose_device(args.device)
    protocol = {
        "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment_name"], "journal_target": "Information Fusion",
        "input": str(input_path), "input_sha256": sha256_file(input_path), "rows": len(frame),
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "method": "local multi-epoch training -> sample-weighted FedAvg parameter deltas; different-dimensional modality deltas -> learned A_i -> shared spectrum -> back-projected correction",
        "federated_nodes": "five ship-type ownership proxies",
        "scope_limits": [
            "offline federated simulation", "MRV emissions is the fourth modality, not bill-of-lading text",
            "privacy is not claimed until MIA results are produced",
        ],
        "environment": {
            "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
            "device": str(device), "git_revision": git_revision(),
        },
    }
    write_json(output_dir / "protocol.json", protocol)
    all_rows = []
    for seed in map(int, config["seeds"]):
        prepared = prepare_data(frame, seed, float(config["test_fraction"]), float(config["validation_fraction"]))
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        write_json(seed_dir / "split.json", {
            "seed": seed, "split_hash": prepared.split_hash,
            "train_mmsi": prepared.entity_ids[prepared.splits.train].tolist(),
            "validation_mmsi": prepared.entity_ids[prepared.splits.validation].tolist(),
            "test_mmsi": prepared.entity_ids[prepared.splits.test].tolist(),
            "node_counts": {
                node: {split: int(len(values)) for split, values in splits.items()}
                for node, splits in prepared.node_indices.items()
            },
            "preprocessing": prepared.preprocessing,
        })
        for ablation in ablations:
            print(f"seed={seed} ablation={ablation}", flush=True)
            artifact = FederatedTrainer(prepared, config, ablation, seed, device).train()
            run_dir = seed_dir / ablation
            run_dir.mkdir(exist_ok=True)
            pd.DataFrame(artifact.history).to_csv(run_dir / "convergence.csv", index=False)
            pd.DataFrame(artifact.gradient_diagnostics).to_csv(run_dir / "gradient_spectrum_diagnostics.csv", index=False)
            pd.DataFrame(artifact.test_rows).to_csv(run_dir / "test_retrieval.csv", index=False)
            torch.save(artifact.checkpoint, run_dir / "best_checkpoint.pt")
            all_rows.extend(artifact.test_rows)
    raw = pd.DataFrame(all_rows)
    raw.to_csv(output_dir / "retrieval_runs.csv", index=False, encoding="utf-8-sig")
    summarize_runs(raw).to_csv(output_dir / "retrieval_summary.csv", index=False, encoding="utf-8-sig")
    paired_ablation_tests(raw).to_csv(output_dir / "paired_tests.csv", index=False, encoding="utf-8-sig")
    protocol.update({"status": "COMPLETED", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(output_dir / "protocol.json", protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
