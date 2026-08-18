"""Gradient-inversion attack, reported separately from membership inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.gsa.data import MODALITIES, load_complete_entities, prepare_data
    from experiments.gsa.federated import FederatedTrainer
else:
    from .data import MODALITIES, load_complete_entities, prepare_data
    from .federated import FederatedTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def differentiable_flat_gradient(loss: torch.Tensor, parameters: list[torch.nn.Parameter], create_graph: bool) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, create_graph=create_graph, allow_unused=True)
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def transmitted_gradient(
    trainer: FederatedTrainer, modality: str, features: torch.Tensor, label: int, create_graph: bool,
) -> torch.Tensor:
    tower = trainer.model.towers[modality]
    parameters = list(tower.parameters())
    embedding = tower(features)
    target = torch.as_tensor([label], dtype=torch.long, device=features.device)
    loss = F.cross_entropy(trainer.classifier(embedding), target)
    gradient = differentiable_flat_gradient(loss, parameters, create_graph=create_graph)
    if trainer.gsa is not None:
        gradient = trainer.gsa.projectors[modality].project(gradient)
    return gradient


def invert_one(
    trainer: FederatedTrainer, modality: str, position: int, steps: int, learning_rate: float,
) -> dict[str, float | int | str]:
    trainer.model.eval()
    true_feature = trainer.features[modality][position:position + 1].detach()
    label = trainer.train_label[position]
    observed = transmitted_gradient(trainer, modality, true_feature, label, create_graph=False).detach()
    dummy = torch.randn_like(true_feature, requires_grad=True)
    optimizer = torch.optim.Adam([dummy], lr=learning_rate)
    best_loss, best_feature = float("inf"), None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        candidate = transmitted_gradient(trainer, modality, dummy, label, create_graph=True)
        matching = 1.0 - F.cosine_similarity(candidate, observed, dim=0)
        matching = matching + F.mse_loss(candidate / candidate.norm().clamp_min(1e-12), observed / observed.norm().clamp_min(1e-12))
        matching.backward()
        optimizer.step()
        if float(matching.detach()) < best_loss:
            best_loss = float(matching.detach())
            best_feature = dummy.detach().clone()
    reconstruction_mse = F.mse_loss(best_feature, true_feature).item() if best_feature is not None else float("nan")
    return {
        "modality": modality, "position": position, "mmsi": trainer.data.entity_ids[position],
        "transmitted_dimension": int(observed.numel()), "matching_loss": best_loss,
        "standardized_reconstruction_mse": reconstruction_mse,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "experiment.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--ablations", default="A,D,D_NONLINEAR")
    parser.add_argument("--samples-per-modality", type=int, default=20)
    parser.add_argument("--attack-steps", type=int, default=1000)
    parser.add_argument("--attack-lr", type=float, default=0.05)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    frame = load_complete_entities(PROJECT_ROOT / config["input"])
    seed = int(config["seeds"][0])
    data = prepare_data(frame, seed, float(config["test_fraction"]), float(config["validation_fraction"]))
    rng = np.random.default_rng(seed + 700_000)
    output = PROJECT_ROOT / config["output_dir"] / "gradient_inversion"
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "gradient_inversion_runs.partial.csv"
    rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {
        (str(row["ablation"]), str(row["modality"]), int(row["position"])) for row in rows
    }
    # Use identical entities for every ablation so reconstruction errors are
    # paired rather than confounded by different attack samples.
    selected = rng.choice(data.splits.train, size=min(args.samples_per_modality, len(data.splits.train)), replace=False)
    for ablation in [x.strip().upper() for x in args.ablations.split(",")]:
        device = torch.device(args.device)
        trainer = FederatedTrainer(data, config, ablation, seed, device)
        checkpoint_path = PROJECT_ROOT / config["output_dir"] / f"seed_{seed}" / ablation / "best_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["states"]["model"])
        trainer.classifier.load_state_dict(checkpoint["states"]["classifier"])
        if trainer.gsa is not None and checkpoint["states"]["gsa"] is not None:
            trainer.gsa.load_state_dict(checkpoint["states"]["gsa"])
        print(f"ablation={ablation} checkpoint={checkpoint_path} loaded", flush=True)
        for modality in MODALITIES:
            for position in selected:
                key = (ablation, modality, int(position))
                if key in completed:
                    print(f"ablation={ablation} modality={modality} position={position} resumed", flush=True)
                    continue
                print(f"ablation={ablation} modality={modality} position={position} attacking", flush=True)
                rows.append({
                    "ablation": ablation,
                    **invert_one(trainer, modality, int(position), args.attack_steps, args.attack_lr),
                })
                completed.add(key)
                pd.DataFrame(rows).to_csv(partial_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(output / "gradient_inversion_runs.csv", index=False, encoding="utf-8-sig")
    (output / "protocol.json").write_text(json.dumps({
        "attack": "optimization-based gradient matching",
        "distinction": "This is a gradient inversion/reconstruction attack, not MIA.",
        "metric": "standardized input reconstruction MSE; larger is more private",
        "ablations": [x.strip().upper() for x in args.ablations.split(",")],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
