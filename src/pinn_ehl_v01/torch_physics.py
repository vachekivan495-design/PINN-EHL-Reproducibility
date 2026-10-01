from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def central_difference_x(values: torch.Tensor, dx: float) -> torch.Tensor:
    result = torch.zeros_like(values)
    result[..., 1:-1, :] = (values[..., 2:, :] - values[..., :-2, :]) / (2.0 * dx)
    result[..., 0, :] = (values[..., 1, :] - values[..., 0, :]) / dx
    result[..., -1, :] = (values[..., -1, :] - values[..., -2, :]) / dx
    return result


def central_difference_y(values: torch.Tensor, dy: float) -> torch.Tensor:
    result = torch.zeros_like(values)
    result[..., :, 1:-1] = (values[..., :, 2:] - values[..., :, :-2]) / (2.0 * dy)
    result[..., :, 0] = (values[..., :, 1] - values[..., :, 0]) / dy
    result[..., :, -1] = (values[..., :, -1] - values[..., :, -2]) / dy
    return result


def constitutive_ratios(
    pressure: torch.Tensor,
    eta0_pa_s: torch.Tensor,
    alpha_pa_inv: torch.Tensor,
    maximum_pressure_pa: torch.Tensor,
    reference_pressure_pa: float | torch.Tensor = 1.96e8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eta0 = eta0_pa_s.reshape(-1, 1, 1)
    alpha = alpha_pa_inv.reshape(-1, 1, 1)
    p_hertz = maximum_pressure_pa.reshape(-1, 1, 1)
    reference_pressure_pa = torch.as_tensor(reference_pressure_pa, dtype=pressure.dtype, device=pressure.device).reshape(-1, 1, 1)
    z = alpha * reference_pressure_pa / (torch.log(eta0) + 9.67)
    pressure_pa = p_hertz * torch.clamp_min(pressure, 0.0)
    base = torch.clamp_min(1.0 + pressure_pa / reference_pressure_pa, 1e-12)
    exponent = (torch.log(eta0) + 9.67) * (torch.pow(base, z) - 1.0)
    viscosity = torch.exp(torch.clamp(exponent, -60.0, 60.0))
    pressure_gpa = pressure_pa * 1e-9
    density = (0.59 + 1.34 * pressure_gpa) / (0.59 + pressure_gpa)
    return viscosity, density, z.reshape(-1)


def conservative_flux_residual(
    pressure: torch.Tensor,
    mobility: torch.Tensor,
    transported_film: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    if pressure.shape != mobility.shape or pressure.shape != transported_film.shape:
        raise ValueError("pressure, mobility, and transported_film must have equal shapes")
    residual = torch.zeros_like(pressure)
    center = pressure[..., 1:-1, 1:-1]
    mobility_center = mobility[..., 1:-1, 1:-1]
    east = 0.5 * (mobility_center + mobility[..., 2:, 1:-1])
    west = 0.5 * (mobility_center + mobility[..., :-2, 1:-1])
    north = 0.5 * (mobility_center + mobility[..., 1:-1, 2:])
    south = 0.5 * (mobility_center + mobility[..., 1:-1, :-2])
    diffusion_x = (
        east * (pressure[..., 2:, 1:-1] - center)
        - west * (center - pressure[..., :-2, 1:-1])
    ) / (dx * dx)
    diffusion_y = (
        north * (pressure[..., 1:-1, 2:] - center)
        - south * (center - pressure[..., 1:-1, :-2])
    ) / (dy * dy)
    transport_x = (
        transported_film[..., 1:-1, 1:-1]
        - transported_film[..., :-2, 1:-1]
    ) / dx
    residual[..., 1:-1, 1:-1] = diffusion_x + diffusion_y - transport_x
    return residual


def reynolds_residual(
    pressure: torch.Tensor,
    film: torch.Tensor,
    dx: float,
    dy: float,
    eta0_pa_s: torch.Tensor,
    alpha_pa_inv: torch.Tensor,
    entrainment_speed_m_s: torch.Tensor,
    maximum_pressure_pa: torch.Tensor,
    contact_radius_m: torch.Tensor,
    equivalent_radius_m: torch.Tensor,
    reference_pressure_pa: float | torch.Tensor = 1.96e8,
) -> torch.Tensor:
    viscosity, density, _ = constitutive_ratios(
        pressure,
        eta0_pa_s,
        alpha_pa_inv,
        maximum_pressure_pa,
        reference_pressure_pa,
    )
    enda = (
        12.0
        * entrainment_speed_m_s
        * eta0_pa_s
        * equivalent_radius_m.square()
        / (maximum_pressure_pa * contact_radius_m.pow(3))
    ).reshape(-1, 1, 1)
    epsilon = density * film.pow(3) / torch.clamp_min(viscosity * enda, 1e-30)
    return conservative_flux_residual(
        pressure,
        epsilon,
        density * film,
        dx,
        dy,
    )


def reynolds_obstacle_loss(
    pressure: torch.Tensor, residual: torch.Tensor,
    pressure_scale: float = 1.0, residual_scale: float = 1.0,
) -> torch.Tensor:
    """Fischer-Burmeister for p>=0, -R>=0, p*(-R)=0 with frozen scales.

    Scales are dimensionless protocol constants, never functions of predictions
    or held-out labels. Exact zero has a finite (zero) subgradient via norm.
    """
    if not all(math.isfinite(s) and s > 0 for s in (pressure_scale, residual_scale)):
        raise ValueError("Frozen obstacle scales must be finite and positive")
    p = pressure / pressure_scale
    dual = -residual / residual_scale
    fb = torch.linalg.vector_norm(torch.stack((p, dual)), dim=0) - p - dual
    interior = fb[..., 1:-1, 1:-1]
    return torch.mean(interior.square())


def trapezoid_load(pressure: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return torch.trapezoid(torch.trapezoid(pressure, dx=dy, dim=-1), dx=dx, dim=-1)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(prediction - target, dim=(-2, -1))
    denominator = torch.linalg.vector_norm(target, dim=(-2, -1)).clamp_min(1e-12)
    return numerator / denominator


def supervised_field_loss(
    pressure: torch.Tensor,
    film: torch.Tensor,
    target_pressure: torch.Tensor,
    target_film: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if not torch.any(mask):
        raise ValueError("Supervision mask is empty")
    p_loss = F.mse_loss(pressure[mask], target_pressure[mask])
    h_loss = F.mse_loss(film[mask], target_film[mask])
    return p_loss, h_loss
