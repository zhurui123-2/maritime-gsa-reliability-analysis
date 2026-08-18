"""Ship-type classification criterion and sensitive relationship inference attack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import load_complete_entities, prepare_data
    from experiments.gsa.models import FourModalModel
else:
    from .data import load_complete_entities, prepare_data
    from .models import FourModalModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def embedding(model: FourModalModel, data, indices: np.ndarray, device: torch.device) -> np.ndarray:
    index = torch.as_tensor(indices, dtype=torch.long, device=device)
    outputs = model({name: torch.as_tensor(values, dtype=torch.float32, device=device)[index] for name, values in data.features.items()})
    return torch.cat(list(outputs.values()), dim=1).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ablations", default="A,D,D_NONLINEAR")
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    frame = load_complete_entities(PROJECT_ROOT / config["input"])
    root = PROJECT_ROOT / config["output_dir"]
    output = root / "classification_and_link_attack"
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "runs.partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    counts = pd.DataFrame(rows).groupby(["seed", "ablation"])["experiment"].nunique() if rows else pd.Series(dtype=int)
    completed = {(int(seed), str(ablation)) for (seed, ablation), count in counts.items() if count == 2}
    for seed in map(int, config["seeds"]):
        data = prepare_data(frame, seed, float(config["test_fraction"]), float(config["validation_fraction"]))
        node_encoder = LabelEncoder().fit(data.node_labels)
        node_target = node_encoder.transform(data.node_labels)
        # Registry-port relation is the private link attribute attacked from non-registry modalities.
        port = frame["PortOfRegistry"].fillna("<MISSING>").astype(str)
        common_ports = set(port.iloc[data.splits.train].value_counts().head(10).index)
        port_target = port.map(lambda value: value if value in common_ports else "<OTHER>")
        port_encoder = LabelEncoder().fit(port_target.iloc[data.splits.train])
        for ablation in [x.strip().upper() for x in args.ablations.split(",")]:
            if (seed, ablation) in completed:
                print(f"seed={seed} ablation={ablation} resumed", flush=True)
                continue
            print(f"seed={seed} ablation={ablation} evaluating", flush=True)
            checkpoint = torch.load(root / f"seed_{seed}" / ablation / "best_checkpoint.pt", map_location=args.device, weights_only=False)
            model = FourModalModel(data.input_dims, int(config["hidden_dim"]), int(config["embedding_dim"]), use_bn=ablation == "C").to(args.device)
            model.load_state_dict(checkpoint["states"]["model"])
            train_x = embedding(model, data, data.splits.train, torch.device(args.device))
            test_x = embedding(model, data, data.splits.test, torch.device(args.device))
            classifier = LogisticRegression(max_iter=5000, class_weight="balanced").fit(train_x, node_target[data.splits.train])
            prediction = classifier.predict(test_x)
            rows.append({
                "seed": seed, "ablation": ablation, "experiment": "ship_type_classification",
                "accuracy": float(accuracy_score(node_target[data.splits.test], prediction)),
                "macro_f1": float(f1_score(node_target[data.splits.test], prediction, average="macro")),
            })
            # Attack only with AIS+weather+emissions embeddings, excluding registry where the port link originates.
            with torch.no_grad():
                train_index = torch.as_tensor(data.splits.train, device=args.device)
                test_index = torch.as_tensor(data.splits.test, device=args.device)
                tensors = {name: torch.as_tensor(values, dtype=torch.float32, device=args.device) for name, values in data.features.items()}
                train_outputs = model({name: value[train_index] for name, value in tensors.items()})
                test_outputs = model({name: value[test_index] for name, value in tensors.items()})
                attack_train_x = torch.cat([train_outputs[name] for name in ("ais", "weather", "emissions")], 1).cpu().numpy()
                attack_test_x = torch.cat([test_outputs[name] for name in ("ais", "weather", "emissions")], 1).cpu().numpy()
            train_port = port_encoder.transform(port_target.iloc[data.splits.train])
            test_port_text = port_target.iloc[data.splits.test]
            known = test_port_text.isin(port_encoder.classes_).to_numpy()
            attack = LogisticRegression(max_iter=5000, class_weight="balanced").fit(attack_train_x, train_port)
            attack_probability = attack.predict_proba(attack_test_x[known])
            attack_truth = port_encoder.transform(test_port_text[known])
            rows.append({
                "seed": seed, "ablation": ablation, "experiment": "registry_port_link_inference",
                "accuracy": float(accuracy_score(attack_truth, attack_probability.argmax(1))),
                "macro_f1": float(f1_score(attack_truth, attack_probability.argmax(1), average="macro")),
                "macro_auc_ovr": float(roc_auc_score(attack_truth, attack_probability, multi_class="ovr", average="macro"))
                if len(np.unique(attack_truth)) > 1 else np.nan,
            })
            completed.add((seed, ablation))
            pd.DataFrame(rows).to_csv(partial_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "runs.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
