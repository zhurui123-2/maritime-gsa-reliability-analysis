"""True FedAvg with ship-type clients and cross-modal GSA delta coordination."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import MODALITIES, NODES, PreparedData
from .gsa import GradientSpectrumAlignment
from .metrics import evaluate_all_pairs
from .models import (
    EntityClassifier, FourModalModel, apply_flat_delta, flatten_parameters,
)


ABLATIONS: dict[str, dict[str, bool | str]] = {
    "A": {"name": "bare_fedavg", "gsa": "none", "contrastive": False, "batch_norm": False},
    "B": {"name": "contrastive_only", "gsa": "none", "contrastive": True, "batch_norm": False},
    "C": {"name": "batchnorm_substitute", "gsa": "none", "contrastive": True, "batch_norm": True},
    "D": {"name": "linear_gsa_full", "gsa": "linear", "contrastive": True, "batch_norm": False},
    "E": {"name": "linear_gsa_only", "gsa": "linear", "contrastive": False, "batch_norm": False},
    "D_NONLINEAR": {"name": "nonlinear_gsa_full", "gsa": "nonlinear", "contrastive": True, "batch_norm": False},
    "FEDAVG": {"name": "fedavg_multimodal", "gsa": "none", "contrastive": True, "batch_norm": False, "algorithm": "fedavg"},
    "FEDPROX": {"name": "fedprox", "gsa": "none", "contrastive": True, "batch_norm": False, "algorithm": "fedprox"},
    "FEDBN": {"name": "fedbn", "gsa": "none", "contrastive": True, "batch_norm": True, "algorithm": "fedbn"},
    "GSA_FEDBN": {"name": "gsa_fedbn", "gsa": "linear", "contrastive": True, "batch_norm": True, "algorithm": "fedbn_gsa"},
    "GSA_STRUCTURED_FEDBN": {"name": "gsa_structured_fedbn", "gsa": "structured", "contrastive": True, "batch_norm": True, "algorithm": "fedbn_gsa"},
    "GSA_HASH_FEDBN": {"name": "gsa_hash_fedbn", "gsa": "hash", "contrastive": True, "batch_norm": True, "algorithm": "fedbn_gsa"},
    "FEDAVGM": {"name": "fedavgm", "gsa": "none", "contrastive": True, "batch_norm": False, "algorithm": "fedavgm"},
    "FEDPROTO": {"name": "fedproto_adapted", "gsa": "none", "contrastive": True, "batch_norm": False, "algorithm": "fedproto"},
    "FEDCMR": {"name": "fedcmr_adapted", "gsa": "none", "contrastive": True, "batch_norm": False, "algorithm": "fedcmr"},
    "D_SKETCH": {"name": "gsa_structured_client_upload", "gsa": "structured", "contrastive": True, "batch_norm": False, "algorithm": "gsa_sketch"},
}


@dataclass
class TrainingArtifacts:
    test_rows: list[dict[str, Any]]
    history: list[dict[str, Any]]
    gradient_diagnostics: list[dict[str, Any]]
    checkpoint: dict[str, Any]


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Fused/memory-efficient attention backward is nondeterministic on CUDA.
        # Force the math implementation so the formal fixed-seed runs satisfy
        # the protocol instead of merely emitting a warning and continuing.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)


def _info_nce(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = left @ right.T / temperature
    labels = torch.arange(len(left), device=left.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def multimodal_contrastive(embeddings: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
    losses = []
    for index, left in enumerate(MODALITIES):
        for right in MODALITIES[index + 1:]:
            losses.append(_info_nce(embeddings[left], embeddings[right], temperature))
    return torch.stack(losses).mean()


def _cpu_state(module: nn.Module | None) -> dict[str, torch.Tensor] | None:
    return None if module is None else {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class FederatedTrainer:
    def __init__(self, data: PreparedData, config: dict[str, Any], ablation: str, seed: int, device: torch.device) -> None:
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        self.data, self.config, self.ablation, self.seed, self.device = data, config, ablation, seed, device
        self.spec = ABLATIONS[ablation]
        self.algorithm = str(self.spec.get("algorithm", "standard"))
        seed_everything(seed, bool(config["deterministic"]))
        self.model = FourModalModel(
            data.input_dims, int(config["hidden_dim"]), int(config["embedding_dim"]),
            use_bn=bool(self.spec["batch_norm"]),
            use_common_subspace=self.algorithm == "fedcmr",
        ).to(device)
        self.classifier = EntityClassifier(int(config["embedding_dim"]), len(data.splits.train)).to(device)
        gsa_kind = str(self.spec["gsa"])
        self.gsa = None if gsa_kind == "none" else GradientSpectrumAlignment(
            self.model.gradient_dimensions(), int(config["spectrum_dim"]),
            nonlinear=gsa_kind == "nonlinear",
            structured_dim=(
                int(config.get("structured_sketch_dim", 128))
                if gsa_kind in {"structured", "hash"} else None
            ),
            structured_mode="hash" if gsa_kind == "hash" else "pool",
        ).to(device)
        self.gsa_optimizer = None if self.gsa is None else torch.optim.AdamW(
            self.gsa.parameters(), lr=float(config["gsa_learning_rate"]), weight_decay=float(config["gsa_weight_decay"])
        )
        self.features = {name: torch.as_tensor(values, dtype=torch.float32, device=device) for name, values in data.features.items()}
        self.train_label = {int(position): label for label, position in enumerate(data.splits.train)}
        self.server_velocity: dict[str, torch.Tensor] | None = None
        self.classifier_velocity: torch.Tensor | None = None
        self.global_prototypes: dict[str, torch.Tensor] | None = None
        self.client_bn_states = {
            node: {
                name: _cpu_state(self.model.towers[name].normalizer)
                for name in MODALITIES
            }
            for node in NODES
        } if self.algorithm in {"fedbn", "fedbn_gsa"} else None

    def _local_loss(
        self, model: FourModalModel, classifier: EntityClassifier, positions: np.ndarray,
    ) -> torch.Tensor:
        index = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        labels = torch.as_tensor([self.train_label[int(position)] for position in positions], dtype=torch.long, device=self.device)
        embeddings = model({name: self.features[name][index] for name in MODALITIES})
        classification = torch.stack([F.cross_entropy(classifier(embeddings[name]), labels) for name in MODALITIES]).mean()
        contrastive = (
            multimodal_contrastive(embeddings, float(self.config["temperature"]))
            if self.spec["contrastive"] else torch.zeros((), device=self.device)
        )
        loss = float(self.config["classification_weight"]) * classification + float(self.config["contrastive_weight"]) * contrastive
        if self.algorithm == "fedproto" and self.global_prototypes is not None:
            loss = loss + float(self.config.get("fedproto_weight", 0.1)) * torch.stack([
                F.mse_loss(embeddings[name].mean(0), self.global_prototypes[name])
                for name in MODALITIES
            ]).mean()
        return loss

    def _client_update(
        self, node: str, round_index: int,
    ) -> tuple[
        dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None,
        dict[str, torch.Tensor] | None, float, int,
    ]:
        """Run local SGD, then return server-minus-client parameter deltas."""
        local_model = copy.deepcopy(self.model)
        if self.algorithm in {"fedbn", "fedbn_gsa"} and self.client_bn_states is not None:
            for modality in MODALITIES:
                local_model.towers[modality].normalizer.load_state_dict(
                    self.client_bn_states[node][modality]
                )
        local_classifier = copy.deepcopy(self.classifier)
        optimizer = torch.optim.AdamW(
            list(local_model.parameters()) + list(local_classifier.parameters()),
            lr=float(self.config["learning_rate"]), weight_decay=float(self.config["weight_decay"]),
        )
        positions = self.data.node_indices[node]["train"]
        rng = np.random.default_rng(self.seed + round_index * 1009 + NODES.index(node))
        losses = []
        local_model.train()
        for _ in range(int(self.config["local_epochs"])):
            shuffled = rng.permutation(positions)
            for start in range(0, len(shuffled), int(self.config["batch_size_per_node"])):
                batch = shuffled[start:start + int(self.config["batch_size_per_node"])]
                optimizer.zero_grad(set_to_none=True)
                loss = self._local_loss(local_model, local_classifier, batch)
                if self.algorithm == "fedprox":
                    mu = float(self.config.get("fedprox_mu", 0.01))
                    proximal = sum(
                        (local - global_.detach()).square().sum()
                        for local, global_ in zip(
                            list(local_model.parameters()) + list(local_classifier.parameters()),
                            list(self.model.parameters()) + list(self.classifier.parameters()),
                        )
                    )
                    loss = loss + 0.5 * mu * proximal
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(local_model.parameters()) + list(local_classifier.parameters()),
                    float(self.config["gradient_clip_norm"]),
                )
                optimizer.step()
                losses.append(float(loss.detach()))
        deltas = {
            modality: flatten_parameters(self.model.modality_parameters(modality))
            - flatten_parameters(local_model.modality_parameters(modality))
            for modality in MODALITIES
        }
        if self.algorithm in {"fedbn", "fedbn_gsa"} and self.client_bn_states is not None:
            for modality in MODALITIES:
                self.client_bn_states[node][modality] = _cpu_state(
                    local_model.towers[modality].normalizer
                )
                offset = 0
                for name, parameter in self.model.towers[modality].named_parameters():
                    count = parameter.numel()
                    if name.startswith("normalizer."):
                        deltas[modality][offset:offset + count] = 0
                    offset += count
        classifier_delta = flatten_parameters(list(self.classifier.parameters())) - flatten_parameters(list(local_classifier.parameters()))
        shared_delta = None
        if self.model.shared_parameters():
            shared_delta = (
                flatten_parameters(self.model.shared_parameters())
                - flatten_parameters(local_model.shared_parameters())
            )
        local_prototypes = None
        if self.algorithm == "fedproto":
            local_model.eval()
            with torch.no_grad():
                chunks: dict[str, list[torch.Tensor]] = {name: [] for name in MODALITIES}
                batch_size = int(self.config["evaluation_batch_size"])
                for start in range(0, len(positions), batch_size):
                    index = torch.as_tensor(
                        positions[start:start + batch_size], dtype=torch.long,
                        device=self.device,
                    )
                    embedded = local_model({
                        name: self.features[name][index] for name in MODALITIES
                    })
                    for name in MODALITIES:
                        chunks[name].append(embedded[name])
                local_prototypes = {
                    name: torch.cat(values).mean(0).detach()
                    for name, values in chunks.items()
                }
        return (
            deltas, classifier_delta, shared_delta, local_prototypes,
            float(np.mean(losses)), len(positions),
        )

    @staticmethod
    def _weighted_average(items: list[tuple[float, torch.Tensor]]) -> torch.Tensor:
        total = sum(weight for weight, _ in items)
        return sum(weight * value for weight, value in items) / max(total, 1.0)

    def _active_nodes(self, round_index: int) -> list[str]:
        dropout = float(self.config.get("client_dropout_rate", 0.0))
        rng = np.random.default_rng(self.seed + 900_000 + round_index)
        active = [node for node in NODES if rng.random() >= dropout]
        return active or [NODES[int(rng.integers(len(NODES)))]]

    def _federated_round(self, round_index: int) -> tuple[float, dict[str, float]]:
        modality_items: dict[str, list[tuple[float, torch.Tensor]]] = {name: [] for name in MODALITIES}
        sketch_items: dict[str, list[tuple[float, torch.Tensor]]] = {name: [] for name in MODALITIES}
        norm_items: dict[str, list[tuple[float, torch.Tensor]]] = {name: [] for name in MODALITIES}
        classifier_items: list[tuple[float, torch.Tensor]] = []
        shared_items: list[tuple[float, torch.Tensor]] = []
        prototype_items: dict[str, list[tuple[float, torch.Tensor]]] = {
            name: [] for name in MODALITIES
        }
        losses = []
        active_nodes = self._active_nodes(round_index)
        for node in active_nodes:
            (
                deltas, classifier_delta, shared_delta, local_prototypes,
                loss, count,
            ) = self._client_update(node, round_index)
            for modality in MODALITIES:
                if self.algorithm == "gsa_sketch" and self.gsa is not None:
                    sketch_items[modality].append(
                        (float(count), self.gsa.projectors[modality].project(deltas[modality]))
                    )
                    norm_items[modality].append((float(count), deltas[modality].norm().detach()))
                else:
                    modality_items[modality].append((float(count), deltas[modality]))
            classifier_items.append((float(count), classifier_delta))
            if shared_delta is not None:
                shared_items.append((float(count), shared_delta))
            if local_prototypes is not None:
                for modality in MODALITIES:
                    prototype_items[modality].append(
                        (float(count), local_prototypes[modality])
                    )
            losses.append(loss)
        if self.algorithm == "gsa_sketch" and self.gsa is not None:
            projected = {name: self._weighted_average(values) for name, values in sketch_items.items()}
            norms = {name: self._weighted_average(values) for name, values in norm_items.items()}
            corrected, projected_payload = self.gsa.align_projected(
                projected, norms, float(self.config["gsa_correction_strength"])
            )
            alignment_loss, gsa_diagnostics = self.gsa.alignment_loss(projected_payload)
            self.gsa_optimizer.zero_grad(set_to_none=True)
            alignment_loss.backward()
            self.gsa_optimizer.step()
            aggregated = {name: value.detach() for name, value in corrected.items()}
        else:
            aggregated = {name: self._weighted_average(values) for name, values in modality_items.items()}
            projected_payload, gsa_diagnostics = None, {}
        packet_loss = float(self.config.get("gradient_packet_loss", 0.0))
        noise_std = float(self.config.get("gradient_noise_std", 0.0))
        for name, value in aggregated.items():
            if packet_loss > 0:
                keep = (torch.rand_like(value) >= packet_loss).to(value.dtype)
                value = value * keep / max(1.0 - packet_loss, 1e-6)
            if noise_std > 0:
                value = value + torch.randn_like(value) * noise_std * value.norm() / max(value.numel() ** 0.5, 1.0)
            aggregated[name] = value
        diagnostics: dict[str, float] = {f"raw_delta_norm_{name}": float(value.norm()) for name, value in aggregated.items()}
        if self.gsa is not None and self.gsa_optimizer is not None and self.algorithm != "gsa_sketch":
            warmup = int(self.config.get("gsa_warmup_rounds", 0))
            ramp = int(self.config.get("gsa_ramp_rounds", 0))
            if round_index <= warmup:
                schedule = 0.0
            elif ramp > 0:
                schedule = min(1.0, (round_index - warmup) / ramp)
            else:
                schedule = 1.0
            correction_strength = float(self.config["gsa_correction_strength"]) * schedule
            corrected, projected = self.gsa.align(aggregated, correction_strength)
            alignment_loss, gsa_diagnostics = self.gsa.alignment_loss(projected)
            self.gsa_optimizer.zero_grad(set_to_none=True)
            (float(self.config["gsa_objective_weight"]) * alignment_loss).backward()
            nn.utils.clip_grad_norm_(self.gsa.parameters(), float(self.config["gradient_clip_norm"]))
            self.gsa_optimizer.step()
            aggregated = {name: value.detach() for name, value in corrected.items()}
            diagnostics.update(gsa_diagnostics)
            diagnostics["projected_pairwise_cosine"] = self.gsa.pairwise_cosine(projected)
            diagnostics["gsa_effective_correction_strength"] = correction_strength
        elif self.algorithm == "gsa_sketch" and projected_payload is not None:
            diagnostics.update(gsa_diagnostics)
            diagnostics["projected_pairwise_cosine"] = self.gsa.pairwise_cosine(projected_payload)
        if self.algorithm == "fedavgm":
            momentum = float(self.config.get("fedavgm_momentum", 0.9))
            if self.server_velocity is None:
                self.server_velocity = {name: torch.zeros_like(value) for name, value in aggregated.items()}
            for name in aggregated:
                self.server_velocity[name] = momentum * self.server_velocity[name] + aggregated[name]
                aggregated[name] = self.server_velocity[name]
            classifier_delta = self._weighted_average(classifier_items)
            if self.classifier_velocity is None:
                self.classifier_velocity = torch.zeros_like(classifier_delta)
            self.classifier_velocity = momentum * self.classifier_velocity + classifier_delta
        else:
            classifier_delta = self._weighted_average(classifier_items)
        server_scale = float(self.config.get("server_mixing", 1.0))
        for modality in MODALITIES:
            apply_flat_delta(self.model.modality_parameters(modality), aggregated[modality], server_scale)
        apply_flat_delta(
            list(self.classifier.parameters()),
            self.classifier_velocity if self.algorithm == "fedavgm" else classifier_delta,
            server_scale,
        )
        if shared_items:
            apply_flat_delta(
                self.model.shared_parameters(),
                self._weighted_average(shared_items),
                server_scale,
            )
        if self.algorithm == "fedproto":
            self.global_prototypes = {
                name: self._weighted_average(prototype_items[name]).detach()
                for name in MODALITIES
            }
        diagnostics["active_nodes"] = float(len(active_nodes))
        return float(np.mean(losses)), diagnostics

    @torch.no_grad()
    def embeddings(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        if self.algorithm in {"fedbn", "fedbn_gsa"} and self.client_bn_states is not None:
            ordered = {name: np.zeros((len(positions), int(self.config["embedding_dim"])), dtype=np.float32) for name in MODALITIES}
            for node in NODES:
                mask = np.where(self.data.node_labels[positions] == node)[0]
                if not len(mask):
                    continue
                for modality in MODALITIES:
                    self.model.towers[modality].normalizer.load_state_dict(
                        self.client_bn_states[node][modality]
                    )
                node_result = self._embeddings_standard(positions[mask])
                for name in MODALITIES:
                    ordered[name][mask] = node_result[name]
            return ordered
        return self._embeddings_standard(positions)

    @torch.no_grad()
    def _embeddings_standard(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        self.model.eval()
        result: dict[str, list[torch.Tensor]] = {name: [] for name in MODALITIES}
        batch_size = int(self.config["evaluation_batch_size"])
        for start in range(0, len(positions), batch_size):
            index = torch.as_tensor(positions[start:start + batch_size], dtype=torch.long, device=self.device)
            batch = self.model({name: self.features[name][index] for name in MODALITIES})
            for name in MODALITIES:
                result[name].append(batch[name].cpu())
        self.model.train()
        return {name: torch.cat(chunks).numpy() for name, chunks in result.items()}

    def _validation_mrr(self) -> float:
        rows = evaluate_all_pairs(self.embeddings(self.data.splits.validation))
        return float(np.mean([float(row["mrr"]) for row in rows]))

    def membership_features(self, positions: np.ndarray) -> np.ndarray:
        rows = []
        self.model.eval()
        all_parameters = list(self.model.parameters())
        for position in positions:
            index = torch.as_tensor([int(position)], dtype=torch.long, device=self.device)
            embeddings = self.model({name: self.features[name][index] for name in MODALITIES})
            scores = torch.stack([
                F.cosine_similarity(embeddings[left], embeddings[right], dim=-1).squeeze(0)
                for left_index, left in enumerate(MODALITIES) for right in MODALITIES[left_index + 1:]
            ])
            alignment_loss = 1.0 - scores.mean()
            gradients = torch.autograd.grad(alignment_loss, all_parameters, allow_unused=True)
            gradient_norm = torch.sqrt(sum(g.detach().square().sum() for g in gradients if g is not None)).item()
            probabilities = torch.softmax(scores.detach(), dim=0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
            embedding_norm = torch.stack([value.detach().norm(dim=-1) for value in embeddings.values()]).mean().item()
            rows.append([float(alignment_loss.detach()), float(torch.sigmoid(scores.detach()).mean()), entropy, embedding_norm, gradient_norm])
        self.model.train()
        return np.asarray(rows, dtype=np.float32)

    def train(self) -> TrainingArtifacts:
        history, diagnostics = [], []
        best_metric, best_round, best_state, patience = -math.inf, -1, None, 0
        for round_index in range(1, int(self.config["rounds"]) + 1):
            loss, diagnostic = self._federated_round(round_index)
            validation_mrr = self._validation_mrr()
            history.append({"round": round_index, "train_loss": loss, "validation_mean_mrr": validation_mrr})
            diagnostics.append({"round": round_index, **diagnostic})
            if validation_mrr > best_metric + float(self.config["early_stopping_min_delta"]):
                best_metric, best_round, patience = validation_mrr, round_index, 0
                best_state = {"model": _cpu_state(self.model), "classifier": _cpu_state(self.classifier), "gsa": _cpu_state(self.gsa)}
            else:
                patience += 1
                if patience >= int(self.config["early_stopping_patience"]):
                    break
        if best_state is None:
            raise RuntimeError("No valid checkpoint was produced.")
        self.model.load_state_dict(best_state["model"])
        self.classifier.load_state_dict(best_state["classifier"])
        if self.gsa is not None and best_state["gsa"] is not None:
            self.gsa.load_state_dict(best_state["gsa"])
        test_rows = []
        if not bool(self.config.get("skip_test_evaluation", False)):
            test_rows = evaluate_all_pairs(self.embeddings(self.data.splits.test))
            for row in test_rows:
                row.update({"ablation": self.ablation, "seed": self.seed, "best_round": best_round, "best_validation_mean_mrr": best_metric})
        return TrainingArtifacts(
            test_rows=test_rows, history=history, gradient_diagnostics=diagnostics,
            checkpoint={
                "ablation": self.ablation, "seed": self.seed, "best_round": best_round,
                "best_validation_mean_mrr": best_metric, "gradient_dimensions": self.model.gradient_dimensions(),
                "states": best_state,
            },
        )
