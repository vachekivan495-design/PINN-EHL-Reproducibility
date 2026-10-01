from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .elasticity import ELASTIC_COEFFICIENT, influence_quadrant


class GaussianFourierFeatures(nn.Module):
    def __init__(self, input_dim: int, features: int, scale: float, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        matrix = torch.randn(input_dim, features, generator=generator) * float(scale)
        self.register_buffer("matrix", matrix)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * math.pi * values @ self.matrix
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class ConditionEncoder(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.layers(condition)


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gamma1 = nn.Linear(hidden_dim, hidden_dim)
        self.beta1 = nn.Linear(hidden_dim, hidden_dim)
        self.gamma2 = nn.Linear(hidden_dim, hidden_dim)
        self.beta2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = self.norm1(hidden)
        value = value * (1.0 + 0.1 * torch.tanh(self.gamma1(condition)))
        value = value + 0.1 * self.beta1(condition)
        value = F.silu(self.fc1(value))
        value = self.norm2(value)
        value = value * (1.0 + 0.1 * torch.tanh(self.gamma2(condition)))
        value = value + 0.1 * self.beta2(condition)
        return hidden + self.fc2(F.silu(value))


class PressureOffsetNet(nn.Module):
    """Shared backbone for physics-regularized and data-only experiments."""

    def __init__(
        self,
        condition_dim: int = 4,
        hidden_dim: int = 128,
        depth: int = 6,
        fourier_features: int = 64,
        fourier_scale: float = 6.0,
        x_bounds: tuple[float, float] = (-2.0, 2.0),
        y_bounds: tuple[float, float] = (-2.0, 2.0),
        pressure_scale: float = 1.0,
        film_scale: float = 1.0,
        h0_scale: float = 1.0,
        h0_initial_value: float = -0.8,
        hard_pressure_boundary: bool = True,
        learn_h0: bool = True,
        seed: int = 20260907,
        film_mode: str = "both",
        h0_mode: str = "condition",
        transverse_symmetry: bool = True,
        nonnegative_pressure: bool = True,
        spatial_chunk_size: int = 0,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.x_min, self.x_max = map(float, x_bounds)
        self.y_min, self.y_max = map(float, y_bounds)
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("Coordinate bounds must be strictly increasing")
        y_center = 0.5 * (self.y_min + self.y_max)
        if not math.isclose(y_center, 0.0, abs_tol=1e-12):
            raise ValueError("V0.1 requires a transverse domain symmetric about y=0")
        self.register_buffer("pressure_scale", torch.tensor(float(pressure_scale)))
        self.register_buffer("film_scale", torch.tensor(float(film_scale)))
        self.register_buffer("h0_scale", torch.tensor(float(h0_scale)))
        self.h0_initial_value = float(h0_initial_value)
        if not math.isfinite(self.h0_initial_value):
            raise ValueError("h0_initial_value must be finite")
        if float(h0_scale) == 0.0:
            raise ValueError("h0_scale must be non-zero")
        self.hard_pressure_boundary = bool(hard_pressure_boundary)
        self.learn_h0 = bool(learn_h0)
        if film_mode not in {"both", "elastic_closure", "direct"}:
            raise ValueError("Invalid film_mode")
        if h0_mode not in {"condition", "shared", "fixed_zero"}:
            raise ValueError("Invalid h0_mode")
        self.film_mode = film_mode
        self.h0_mode = h0_mode if learn_h0 else "fixed_zero"
        self.transverse_symmetry = bool(transverse_symmetry)
        self.nonnegative_pressure = bool(nonnegative_pressure)
        self.spatial_chunk_size = int(spatial_chunk_size)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.spatial_chunk_size < 0:
            raise ValueError("spatial_chunk_size must be nonnegative")

        self.fourier = GaussianFourierFeatures(2, fourier_features, fourier_scale, seed)
        spatial_dim = 2 + 2 * fourier_features
        self.spatial = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition = ConditionEncoder(condition_dim, hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([FiLMBlock(hidden_dim) for _ in range(depth)])
        self.pressure_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.direct_film_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.reset_parameters()
        # Keep identical seeded backbone initialization across controlled baselines.
        # Dormant heads are excluded from the optimizer and effective parameter count.
        self.shared_h0 = nn.Parameter(torch.zeros(1), requires_grad=(film_mode != "direct" and self.h0_mode == "shared"))
        self.offset_head.requires_grad_(film_mode != "direct" and self.h0_mode == "condition")
        self.direct_film_head.requires_grad_(film_mode != "elastic_closure")

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "effective_trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # h0 is an unconstrained rigid approach, not a positive film output.
        # Start every condition at the established nondimensional EHL scale;
        # condition dependence is then learned without an external baseline.
        last_offset = self.offset_head[-1]
        if isinstance(last_offset, nn.Linear):
            nn.init.zeros_(last_offset.weight)
            nn.init.constant_(
                last_offset.bias,
                self.h0_initial_value / float(self.h0_scale.detach().cpu()),
            )

    def coordinate_features(
        self, coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = coordinates[..., 0:1]
        y = coordinates[..., 1:2]
        x01 = (x - self.x_min) / (self.x_max - self.x_min)
        y_half = 0.5 * (self.y_max - self.y_min)
        y_even01 = (y / y_half).square()
        y_mapped = 2.0 * y_even01 - 1.0 if self.transverse_symmetry else y / y_half
        mapped = torch.cat([2.0 * x01 - 1.0, y_mapped], dim=-1)
        x_envelope = 4.0 * x01 * (1.0 - x01)
        y_envelope = 1.0 - y_even01
        envelope = torch.clamp(x_envelope * y_envelope, min=0.0)
        if not self.hard_pressure_boundary:
            envelope = torch.ones_like(envelope)
        return mapped, envelope

    def _spatial_fields(
        self,
        coordinates: torch.Tensor,
        encoded_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mapped, envelope = self.coordinate_features(coordinates)
        spatial = torch.cat([mapped, self.fourier(mapped)], dim=-1)
        hidden = self.spatial(spatial)
        condition = encoded_condition
        while condition.ndim < hidden.ndim:
            condition = condition.unsqueeze(-2)
        hidden = hidden + self.condition_projection(condition)
        for block in self.blocks:
            hidden = block(hidden, condition)
        logits = self.pressure_head(hidden)
        pressure = (F.softplus(logits) if self.nonnegative_pressure else logits) * envelope * self.pressure_scale
        direct_film = (
            F.softplus(self.direct_film_head(hidden)) * self.film_scale
            if self.film_mode != "elastic_closure" else torch.zeros_like(pressure)
        )
        return pressure, direct_film

    def forward(self, coordinates: torch.Tensor, normalized_condition: torch.Tensor):
        encoded = self.condition(normalized_condition)
        if self.film_mode == "direct":
            h0 = None  # Not identifiable from this model's training objective.
        elif self.h0_mode == "shared":
            h0 = self.shared_h0.expand(*encoded.shape[:-1], 1) * self.h0_scale
        elif self.h0_mode == "fixed_zero":
            h0 = encoded.new_zeros((*encoded.shape[:-1], 1))
        else:
            h0 = self.offset_head(encoded) * self.h0_scale
        size = self.spatial_chunk_size or coordinates.shape[-2]
        fields = []
        for chunk in coordinates.split(size, dim=-2):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                fields.append(checkpoint(self._spatial_fields, chunk, encoded, use_reentrant=False))
            else:
                fields.append(self._spatial_fields(chunk, encoded))
        pressure = torch.cat([f[0] for f in fields], dim=-2)
        direct_film = torch.cat([f[1] for f in fields], dim=-2)
        return pressure, h0, direct_film


class TorchElasticConvolver(nn.Module):
    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        pressure_to_deformation_ratio: float = 1.0,
    ):
        super().__init__()
        self.nx = int(nx)
        self.ny = int(ny)
        self.output_shape = (2 * self.nx - 1, 2 * self.ny - 1)
        quadrant = influence_quadrant(self.nx, self.ny)
        full = torch.zeros(self.output_shape, dtype=torch.float64)
        q = torch.from_numpy(quadrant)
        full[: self.nx, : self.ny] = q
        full[: self.nx, self.ny :] = torch.fliplr(q[:, 1:])
        full[self.nx :, self.ny :] = torch.flipud(torch.fliplr(q[1:, 1:]))
        full[self.nx :, : self.ny] = torch.flipud(q[1:, :])
        self.register_buffer("kernel_fft", torch.fft.rfft2(full))
        coefficient = (
            float(dx)
            * ELASTIC_COEFFICIENT
            * float(pressure_to_deformation_ratio)
        )
        self.register_buffer("coefficient", torch.tensor(coefficient, dtype=torch.float64))

    def forward(self, pressure: torch.Tensor) -> torch.Tensor:
        squeeze = pressure.ndim == 2
        if squeeze:
            pressure = pressure.unsqueeze(0)
        if pressure.ndim != 3 or tuple(pressure.shape[-2:]) != (self.nx, self.ny):
            raise ValueError(
                f"Expected pressure [..., {self.nx}, {self.ny}], got {tuple(pressure.shape)}"
            )
        padded = F.pad(pressure, (0, self.ny - 1, 0, self.nx - 1))
        deformation = torch.fft.irfft2(
            torch.fft.rfft2(padded) * self.kernel_fft.to(
                torch.complex128 if pressure.dtype == torch.float64 else torch.complex64
            ),
            s=self.output_shape,
        )[..., : self.nx, : self.ny]
        deformation = deformation * self.coefficient.to(pressure.dtype)
        return deformation.squeeze(0) if squeeze else deformation


def close_film(
    pressure: torch.Tensor,
    h0: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    convolver: TorchElasticConvolver,
    film_floor: float | torch.Tensor,
    beta: float = 50.0,
    return_raw: bool = False,
) -> torch.Tensor:
    """FDM-identical floor. beta is retained only for old call-site compatibility.

    Clipping has the correct piecewise derivative; use a separate raw-film
    violation penalty during training instead of changing the closure equation.
    """
    squeeze = pressure.ndim == 2
    pressure_batch = pressure.unsqueeze(0) if squeeze else pressure
    h0_batch = h0.reshape(-1, 1, 1)
    geometry = 0.5 * (x_grid.square() + y_grid.square())
    raw = h0_batch + geometry.unsqueeze(0) + convolver(pressure_batch)
    floor = torch.as_tensor(film_floor, dtype=raw.dtype, device=raw.device)
    if floor.ndim == 1:
        floor = floor.reshape(-1, 1, 1)
    film = torch.maximum(raw, floor)
    if squeeze:
        film, raw = film.squeeze(0), raw.squeeze(0)
    return (film, raw) if return_raw else film
