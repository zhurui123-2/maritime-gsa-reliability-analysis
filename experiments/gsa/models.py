"""Four heterogeneous encoders used by every ship-type federated node."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AISFeatureTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(1, hidden_dim)
        self.feature_tokens = nn.Parameter(torch.empty(1, input_dim, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.feature_tokens, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self.value(x.unsqueeze(-1)) + self.feature_tokens).mean(1))


class WeatherCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        width = hidden_dim // 2
        self.network = nn.Sequential(
            nn.Conv1d(1, width, 3, padding=1), nn.GELU(),
            nn.Conv1d(width, hidden_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.unsqueeze(1))


class RegistryMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EmissionsResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.input(x)
        return self.norm(base + self.block(base))


class ModalityTower(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, embedding_dim: int, use_bn: bool) -> None:
        super().__init__()
        self.encoder = encoder
        # No running statistics: otherwise heterogeneous client buffers require a
        # separate aggregation rule and confound the BN-vs-GSA ablation.
        self.normalizer = nn.BatchNorm1d(hidden_dim, track_running_stats=False) if use_bn else nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.normalizer(self.encoder(x))), dim=-1)


class FourModalModel(nn.Module):
    def __init__(
        self, input_dims: dict[str, int], hidden_dim: int, embedding_dim: int,
        use_bn: bool, use_common_subspace: bool = False,
    ) -> None:
        super().__init__()
        encoders = {
            "ais": AISFeatureTransformer(input_dims["ais"], hidden_dim),
            "weather": WeatherCNN(input_dims["weather"], hidden_dim),
            "registry": RegistryMLP(input_dims["registry"], hidden_dim),
            "emissions": EmissionsResidualMLP(input_dims["emissions"], hidden_dim),
        }
        self.towers = nn.ModuleDict({
            name: ModalityTower(encoder, hidden_dim, embedding_dim, use_bn)
            for name, encoder in encoders.items()
        })
        self.common_subspace = (
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            if use_common_subspace else nn.Identity()
        )
        if isinstance(self.common_subspace, nn.Linear):
            nn.init.eye_(self.common_subspace.weight)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: F.normalize(self.common_subspace(self.towers[name](features[name])), dim=-1)
            for name in self.towers
        }

    def modality_parameters(self, modality: str) -> list[nn.Parameter]:
        return list(self.towers[modality].parameters())

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.common_subspace.parameters())

    def gradient_dimensions(self) -> dict[str, int]:
        return {name: sum(parameter.numel() for parameter in tower.parameters()) for name, tower in self.towers.items()}


class EntityClassifier(nn.Module):
    def __init__(self, embedding_dim: int, n_train_entities: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_train_entities, embedding_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings @ F.normalize(self.weight, dim=-1).T


def flatten_gradients(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if parameter.grad is None else parameter.grad.reshape(-1)
        for parameter in parameters
    ])


def flatten_parameters(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


@torch.no_grad()
def assign_flat_gradient(parameters: list[nn.Parameter], gradient: torch.Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = gradient[offset:offset + count].view_as(parameter).clone()
        offset += count
    if offset != gradient.numel():
        raise ValueError("Gradient dimension mismatch.")


@torch.no_grad()
def apply_flat_delta(parameters: list[nn.Parameter], delta: torch.Tensor, scale: float = 1.0) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.sub_(scale * delta[offset:offset + count].view_as(parameter))
        offset += count
    if offset != delta.numel():
        raise ValueError("Parameter-delta dimension mismatch.")
