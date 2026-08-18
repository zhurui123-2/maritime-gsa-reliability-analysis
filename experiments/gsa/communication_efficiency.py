"""Audit parameter storage, actual simulated communication, and convergence rounds."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .data import load_complete_entities, prepare_data
from .federated import FederatedTrainer

ROOT = Path(__file__).resolve().parents[2]


def count(module) -> int:
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def main() -> int:
    config = json.loads((Path(__file__).with_name("configs") / "experiment.json").read_text(encoding="utf-8"))
    frame = load_complete_entities(ROOT / config["input"])
    prepared = prepare_data(frame, 42, config["test_fraction"], config["validation_fraction"])
    output = ROOT / config["output_dir"] / "communication_efficiency"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in ("A", "B", "C", "D", "E", "D_NONLINEAR"):
        trainer = FederatedTrainer(prepared, config, method, 42, torch.device("cpu"))
        model_parameters = count(trainer.model) + count(trainer.classifier)
        server_extra = count(trainer.gsa)
        # Current implementation broadcasts the full model and receives full
        # model deltas from each of five clients. GSA is applied server-side.
        per_round_bytes = len(config["nodes"]) * 2 * model_parameters * 4
        convergence = ROOT / config["output_dir"] / "seed_42" / method / "convergence.csv"
        curve = pd.read_csv(convergence) if convergence.exists() else pd.DataFrame()
        best_round = int(curve.loc[curve["validation_mean_mrr"].idxmax(), "round"]) if len(curve) else None
        executed_rounds = int(curve["round"].max()) if len(curve) else None
        rows.append({
            "method": method,
            "trainable_model_parameters": model_parameters,
            "server_only_extra_parameters": server_extra,
            "total_stored_parameters": model_parameters + server_extra,
            "actual_per_round_communication_mb": per_round_bytes / 1024**2,
            "best_validation_round_seed42": best_round,
            "executed_rounds_seed42": executed_rounds,
            "actual_protocol": "full model download + full delta upload; GSA projection is server-side",
        })
    pd.DataFrame(rows).to_csv(output / "communication_efficiency_audit.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
