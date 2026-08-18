"""Leakage-safe CO2 estimation by linear probing AIS+weather+registry embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import load_complete_entities, prepare_data
    from experiments.gsa.models import FourModalModel
else:
    from .data import load_complete_entities, prepare_data
    from .models import FourModalModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def representations(model: FourModalModel, data, indices: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor_index = torch.as_tensor(indices, dtype=torch.long, device=device)
    outputs = model({name: torch.as_tensor(values, dtype=torch.float32, device=device)[tensor_index] for name, values in data.features.items()})
    # Emissions tower is deliberately excluded: it contains the prediction target.
    return torch.cat([outputs["ais"], outputs["weather"], outputs["registry"]], dim=1).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ablations", default="A,B,C,D,E,D_NONLINEAR")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    frame = load_complete_entities(PROJECT_ROOT / config["input"])
    root = PROJECT_ROOT / config["output_dir"]
    output = root / "downstream_emissions"
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "emissions_runs.partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {(int(row["seed"]), str(row["ablation"])) for row in rows}
    for seed in map(int, config["seeds"]):
        data = prepare_data(frame, seed, float(config["test_fraction"]), float(config["validation_fraction"]))
        for ablation in [x.strip().upper() for x in args.ablations.split(",")]:
            if (seed, ablation) in completed:
                print(f"seed={seed} ablation={ablation} resumed", flush=True)
                continue
            print(f"seed={seed} ablation={ablation} probing", flush=True)
            checkpoint_path = root / f"seed_{seed}" / ablation / "best_checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
            model = FourModalModel(
                data.input_dims, int(config["hidden_dim"]), int(config["embedding_dim"]),
                use_bn=ablation == "C",
            ).to(args.device)
            model.load_state_dict(checkpoint["states"]["model"])
            train_x = representations(model, data, data.splits.train, torch.device(args.device))
            validation_x = representations(model, data, data.splits.validation, torch.device(args.device))
            test_x = representations(model, data, data.splits.test, torch.device(args.device))
            target = np.log1p(data.targets["co2"])
            best_alpha, best_mae = None, float("inf")
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                probe = Ridge(alpha=alpha).fit(train_x, target[data.splits.train])
                mae = mean_absolute_error(target[data.splits.validation], probe.predict(validation_x))
                if mae < best_mae:
                    best_alpha, best_mae = alpha, mae
            probe = Ridge(alpha=best_alpha).fit(
                np.concatenate([train_x, validation_x]), target[np.concatenate([data.splits.train, data.splits.validation])]
            )
            prediction = np.expm1(probe.predict(test_x)).clip(min=0)
            truth = data.targets["co2"][data.splits.test]
            rows.append({
                "seed": seed, "ablation": ablation, "alpha": best_alpha, "n_test": len(truth),
                "mae_tonnes": float(mean_absolute_error(truth, prediction)),
                "rmse_tonnes": float(mean_squared_error(truth, prediction) ** 0.5),
                "r2": float(r2_score(truth, prediction)),
            })
            completed.add((seed, ablation))
            pd.DataFrame(rows).to_csv(partial_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "emissions_runs.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(json.dumps({
        "task": "AIS + weather + registry -> total MRV CO2 emissions",
        "leakage_control": "The emissions tower and every MRV emissions input are excluded from probe features.",
        "split": "MMSI-disjoint and identical to the representation experiment",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
