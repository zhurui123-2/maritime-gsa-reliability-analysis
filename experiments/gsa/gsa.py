"""Gradient Spectrum Alignment for heterogeneous, unequal-dimensional gradients."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def power_spectrum(vector: torch.Tensor) -> torch.Tensor:
    """Unit-normalized one-sided gradient power spectrum."""
    spectrum = torch.fft.rfft(vector, norm="ortho").abs().square()
    return F.normalize(spectrum, dim=0)


class GradientProjector(nn.Module):
    """A_i: R^(d_i) -> R^k, with tied-transpose decoding back to R^(d_i)."""

    def __init__(
        self, input_dim: int, spectrum_dim: int, nonlinear: bool = False,
        structured_dim: int | None = None, structured_mode: str = "pool",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.spectrum_dim = spectrum_dim
        self.nonlinear = nonlinear
        self.structured_dim = structured_dim
        self.structured_mode = structured_mode
        if structured_dim is not None and structured_mode == "hash":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(input_dim * 1_000_003 + structured_dim * 10_007) % (2**31)
            )
            index = torch.randint(
                0, structured_dim, (input_dim,), generator=generator
            )
            sign = (
                torch.randint(0, 2, (input_dim,), generator=generator)
                .mul(2).sub(1).float()
            )
            counts = torch.bincount(index, minlength=structured_dim).clamp_min(1)
            weight = sign / counts[index].float().sqrt()
            self.register_buffer("hash_index", index, persistent=True)
            self.register_buffer("hash_weight", weight, persistent=True)
        encoder_dim = structured_dim or input_dim
        if nonlinear:
            hidden = max(spectrum_dim * 2, 64)
            self.encoder = nn.Sequential(
                nn.Linear(encoder_dim, hidden, bias=False), nn.GELU(),
                nn.Linear(hidden, spectrum_dim, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.Linear(spectrum_dim, hidden, bias=False), nn.GELU(),
                nn.Linear(hidden, encoder_dim, bias=False),
            )
        else:
            self.encoder = nn.Linear(encoder_dim, spectrum_dim, bias=False)
            self.decoder = None
            nn.init.orthogonal_(self.encoder.weight)

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.structured_dim is not None:
            if self.structured_mode == "hash":
                sketch = gradient.new_zeros(self.structured_dim)
                sketch.scatter_add_(
                    0, self.hash_index, gradient * self.hash_weight
                )
                gradient = sketch
            else:
                gradient = F.adaptive_avg_pool1d(
                    gradient.reshape(1, 1, -1), self.structured_dim
                ).reshape(-1)
        return self.encoder(gradient)

    def back_project(self, shared_residual: torch.Tensor) -> torch.Tensor:
        if self.decoder is not None:
            result = self.decoder(shared_residual)
        else:
            result = F.linear(shared_residual, self.encoder.weight.T)
        if self.structured_dim is not None:
            if self.structured_mode == "hash":
                result = result[self.hash_index] * self.hash_weight
            else:
                result = F.interpolate(
                    result.reshape(1, 1, -1), size=self.input_dim, mode="linear",
                    align_corners=False,
                ).reshape(-1)
        return result


class GradientSpectrumAlignment(nn.Module):
    def __init__(
        self, gradient_dims: dict[str, int], spectrum_dim: int,
        nonlinear: bool = False, structured_dim: int | None = None,
        structured_mode: str = "pool",
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: GradientProjector(
                size, spectrum_dim, nonlinear=nonlinear,
                structured_dim=structured_dim, structured_mode=structured_mode,
            )
            for name, size in gradient_dims.items()
        })

    def align(
        self, gradients: dict[str, torch.Tensor], correction_strength: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        projected = {name: self.projectors[name].project(gradient) for name, gradient in gradients.items()}
        consensus = torch.stack([F.normalize(value, dim=0) for value in projected.values()]).mean(0)
        consensus = F.normalize(consensus, dim=0)
        corrected = {
            name: gradient + correction_strength * self.projectors[name].back_project(consensus - F.normalize(projected[name], dim=0))
            for name, gradient in gradients.items()
        }
        return corrected, {**projected, "consensus": consensus}

    def align_projected(
        self, projected: dict[str, torch.Tensor], norms: dict[str, torch.Tensor],
        correction_strength: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        consensus = F.normalize(
            torch.stack([F.normalize(value, dim=0) for value in projected.values()]).mean(0),
            dim=0,
        )
        corrected_projected = {
            name: value + correction_strength * (consensus - F.normalize(value, dim=0))
            for name, value in projected.items()
        }
        corrected = {}
        for name, value in corrected_projected.items():
            reconstructed = self.projectors[name].back_project(value)
            corrected[name] = F.normalize(reconstructed, dim=0) * norms[name]
        return corrected, {**projected, "consensus": consensus}

    def alignment_loss(self, projected: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        consensus = projected["consensus"].detach()
        direction_losses, spectrum_losses = [], []
        consensus_spectrum = power_spectrum(consensus).detach()
        for name, value in projected.items():
            if name == "consensus":
                continue
            direction_losses.append(1.0 - F.cosine_similarity(value, consensus, dim=0))
            spectrum_losses.append(F.mse_loss(power_spectrum(value), consensus_spectrum))
        direction = torch.stack(direction_losses).mean()
        spectrum = torch.stack(spectrum_losses).mean()
        return direction + spectrum, {
            "gsa_direction_loss": float(direction.detach()),
            "gsa_spectrum_loss": float(spectrum.detach()),
        }

    @staticmethod
    def pairwise_cosine(projected: dict[str, torch.Tensor]) -> float:
        names = [name for name in projected if name != "consensus"]
        values = []
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                values.append(float(F.cosine_similarity(projected[left], projected[right], dim=0).detach()))
        return float(sum(values) / len(values))
