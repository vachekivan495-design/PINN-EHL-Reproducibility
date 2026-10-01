from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .data import (
    CaseArtifact,
    ConditionScaler,
    load_case_artifact,
    load_case_metadata,
    make_supervision_mask,
)
from .model import PressureOffsetNet, TorchElasticConvolver, close_film
from .physics import DIMENSIONLESS_LOAD_TARGET
from .splits import SplitAssignment
from .contracts import (CHECKPOINT_SCHEMA, project_path, atomic_json, atomic_write,
                        data_contract, require_same_contract, runtime_environment)
from .metrics import field_metrics
from .torch_physics import (
    relative_l2,
    reynolds_obstacle_loss,
    reynolds_residual,
    supervised_field_loss,
    trapezoid_load,
)


@dataclass(frozen=True)
class LossWeights:
    pressure_data: float = 1.0
    film_data: float = 1.0
    reynolds_obstacle: float = 1e-3
    load_balance: float = 1e-2
    boundary_pressure: float = 1e-2
    film_floor_violation: float = 1e-2

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError(f"Loss weights must be finite and non-negative: {values}")


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    case_dirs: tuple[str, ...]
    split_manifest: str
    output_root: str = "outputs/unlocked_runs"
    mode: str = "physics_regularized"
    film_mode: str = "elastic_closure"
    hard_pressure_boundary: bool = True
    learn_h0: bool = True
    h0_mode: str = "condition"
    h0_initial_value: float = -0.8
    transverse_symmetry: bool = True
    nonnegative_pressure: bool = True
    supervised_fraction: float = 0.05
    supervision_mode: str = "uniform"
    experiment_scope: str = "multi_condition"
    single_case_id: str | None = None
    seed: int = 20260907
    epochs: int = 1000
    steps_per_epoch: int = 0
    learning_rate: float = 2e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    hidden_dim: int = 128
    depth: int = 6
    fourier_features: int = 64
    fourier_scale: float = 6.0
    film_floor_beta: float = 50.0
    reynolds_pressure_scale: float = 1.0
    reynolds_residual_scale: float = 1.0
    spatial_chunk_size: int = 8192
    activation_checkpointing: bool = True
    deterministic: bool = False
    run_purpose: str = "pilot"
    reference_validation_report: str | None = None
    residual_calibration_report: str | None = None
    residual_calibration_sha256: str | None = None
    device: str = "auto"
    validation_interval: int = 10
    gradient_diagnostics_interval: int = 10
    checkpoint_interval: int = 50
    resume_checkpoint: str | None = None
    loss_weights: LossWeights = LossWeights()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TrainingConfig":
        payload = dict(raw)
        payload["case_dirs"] = tuple(str(path) for path in payload.get("case_dirs", ()))
        payload["loss_weights"] = LossWeights(**payload.get("loss_weights", {}))
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.run_name.strip() or not self.case_dirs:
            raise ValueError("run_name and case_dirs are required")
        if self.mode not in {"physics_regularized", "data_only"}:
            raise ValueError("mode must be 'physics_regularized' or 'data_only'")
        if self.film_mode not in {"elastic_closure", "direct"}:
            raise ValueError("film_mode must be 'elastic_closure' or 'direct'")
        if not 0.0 < self.supervised_fraction <= 1.0:
            raise ValueError("supervised_fraction must lie in (0, 1]")
        if self.supervision_mode not in {"uniform", "sensor_lines", "sensor_lines_plus_random"}:
            raise ValueError("Unsupported supervision_mode")
        if self.experiment_scope not in {"multi_condition", "single_condition_spatial"}:
            raise ValueError("experiment_scope must be multi_condition or single_condition_spatial")
        if self.experiment_scope == "single_condition_spatial":
            if not self.single_case_id:
                raise ValueError("single_condition_spatial requires single_case_id")
            if self.supervised_fraction >= 1.0:
                raise ValueError("single_condition_spatial requires an unobserved spatial holdout")
        elif self.single_case_id is not None:
            raise ValueError("single_case_id is only valid for single_condition_spatial")
        if self.epochs < 1 or self.validation_interval < 1 or self.checkpoint_interval < 1:
            raise ValueError("Epoch and interval counts must be positive")
        if self.gradient_diagnostics_interval < 0:
            raise ValueError("gradient_diagnostics_interval cannot be negative")
        if self.steps_per_epoch < 0:
            raise ValueError("steps_per_epoch cannot be negative")
        if self.learning_rate <= 0.0 or self.grad_clip <= 0.0:
            raise ValueError("learning_rate and grad_clip must be positive")
        self.loss_weights.validate()
        if self.h0_mode not in {"condition", "shared", "fixed_zero"}:
            raise ValueError("Invalid h0_mode")
        if not math.isfinite(self.h0_initial_value):
            raise ValueError("h0_initial_value must be finite")
        if self.run_purpose not in {"pilot", "production"}:
            raise ValueError("run_purpose must be pilot or production")
        if self.spatial_chunk_size < 0 or self.hidden_dim < 2 or self.depth < 1:
            raise ValueError("Invalid network size")
        if not all(math.isfinite(v) and v > 0 for v in (self.reynolds_pressure_scale, self.reynolds_residual_scale, self.learning_rate, self.grad_clip)):
            raise ValueError("Loss scales and optimization constants must be finite and positive")
        if self.mode == "physics_regularized" and not self.nonnegative_pressure:
            raise ValueError("Physics model requires nonnegative pressure")


def model_configuration(config: TrainingConfig, x_bounds, y_bounds) -> dict:
    return {
        "condition_dim": 4, "hidden_dim": config.hidden_dim, "depth": config.depth,
        "fourier_features": config.fourier_features, "fourier_scale": config.fourier_scale,
        "x_bounds": tuple(x_bounds), "y_bounds": tuple(y_bounds),
        "pressure_scale": 1.0, "film_scale": 1.0, "h0_scale": 1.0,
        "h0_initial_value": config.h0_initial_value,
        "hard_pressure_boundary": config.hard_pressure_boundary, "learn_h0": config.learn_h0,
        "seed": config.seed, "film_mode": config.film_mode, "h0_mode": config.h0_mode,
        "transverse_symmetry": config.transverse_symmetry,
        "nonnegative_pressure": config.nonnegative_pressure,
        "spatial_chunk_size": config.spatial_chunk_size,
        "activation_checkpointing": config.activation_checkpointing,
    }


def baseline_description(config: TrainingConfig) -> str:
    if config.mode == "physics_regularized":
        return "physics_regularized_" + config.film_mode
    if config.film_mode == "elastic_closure":
        return "supervised_elastic_closure_no_pde_or_load_loss"
    if config.hard_pressure_boundary or config.transverse_symmetry:
        return "geometric_prior_data_only"
    return "direct_data_only_with_positive_outputs" if config.nonnegative_pressure else "direct_data_only_unconstrained_pressure"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _same_grid(cases: Sequence[CaseArtifact]) -> tuple[int, int, float, float]:
    if not cases:
        raise ValueError("At least one case is required")
    reference = cases[0]
    nx, ny = reference.pressure.shape
    dx = float(reference.x[1] - reference.x[0])
    dy = float(reference.y[1] - reference.y[0])
    for case in cases[1:]:
        if case.pressure.shape != (nx, ny):
            raise ValueError("All neural-training cases must use the same grid shape")
        if not np.allclose(case.x, reference.x) or not np.allclose(case.y, reference.y):
            raise ValueError("All neural-training cases must use identical coordinates")
    return nx, ny, dx, dy


def _case_tensors(case: CaseArtifact, device: torch.device) -> dict[str, torch.Tensor]:
    manifest_case = case.manifest["case"]
    scaling = case.manifest["scaling"]
    return {
        "pressure": torch.as_tensor(case.pressure, dtype=torch.float32, device=device).unsqueeze(0),
        "film": torch.as_tensor(case.film, dtype=torch.float32, device=device).unsqueeze(0),
        "eta0": torch.tensor([manifest_case["eta0_pa_s"]], dtype=torch.float32, device=device),
        "alpha": torch.tensor([manifest_case["alpha_pa_inv"]], dtype=torch.float32, device=device),
        "speed": torch.tensor([manifest_case["entrainment_speed_m_s"]], dtype=torch.float32, device=device),
        "p_hertz": torch.tensor([scaling["maximum_pressure_pa"]], dtype=torch.float32, device=device),
        "contact_radius": torch.tensor([scaling["contact_radius_m"]], dtype=torch.float32, device=device),
        "radius": torch.tensor([scaling["equivalent_radius_m"]], dtype=torch.float32, device=device),
        "reference_pressure": torch.tensor([manifest_case["roelands_reference_pressure_pa"]], dtype=torch.float32, device=device),
        "film_floor": torch.tensor(
            float(manifest_case["solver"]["film_floor_m"]) / float(scaling["film_scale_m"]),
            dtype=torch.float32,
            device=device,
        ),
    }


class CaseRuntime:
    def __init__(
        self,
        case: CaseArtifact,
        scaler: ConditionScaler,
        device: torch.device,
        supervision_fraction: float,
        supervision_mode: str,
        seed: int,
    ):
        self.case = case
        self.constants = _case_tensors(case, device)
        X, Y = np.meshgrid(case.x, case.y, indexing="ij")
        coordinates = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
        self.coordinates = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
        normalized = scaler.transform(case.condition).astype(np.float32)
        self.condition = torch.as_tensor(normalized, dtype=torch.float32, device=device).reshape(1, -1)
        mask = make_supervision_mask(
            case,
            fraction=supervision_fraction,
            seed=seed,
            mode=supervision_mode,
        )
        self.mask = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)


def predict_case(
    model: PressureOffsetNet,
    convolver: TorchElasticConvolver,
    runtime: CaseRuntime,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    film_floor_beta: float,
    film_mode: str,
    return_raw: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_pressure, h0, direct_film = model(runtime.coordinates, runtime.condition)
    pressure = raw_pressure.reshape(1, convolver.nx, convolver.ny)
    if film_mode == "elastic_closure":
        film, raw_film = close_film(
            pressure,
            h0,
            x_grid,
            y_grid,
            convolver,
            film_floor=runtime.constants["film_floor"],
            beta=film_floor_beta,
            return_raw=True,
        )
    elif film_mode == "direct":
        film = direct_film.reshape(1, convolver.nx, convolver.ny)
        film = film + runtime.constants["film_floor"]
        raw_film = film
    else:
        raise ValueError(f"Unsupported film_mode={film_mode!r}")
    result = (pressure, film, h0.reshape(-1) if h0 is not None else None)
    return (*result, raw_film) if return_raw else result


def loss_for_case(
    model: PressureOffsetNet,
    convolver: TorchElasticConvolver,
    runtime: CaseRuntime,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    dx: float,
    dy: float,
    config: TrainingConfig,
    return_weighted_components: bool = False,
) -> tuple[torch.Tensor, dict[str, float]] | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    pressure, film, _, raw_film = predict_case(
        model,
        convolver,
        runtime,
        x_grid,
        y_grid,
        config.film_floor_beta,
        config.film_mode,
        return_raw=True,
    )
    target_pressure = runtime.constants["pressure"]
    target_film = runtime.constants["film"]
    pressure_data, film_data = supervised_field_loss(
        pressure,
        film,
        target_pressure,
        target_film,
        runtime.mask,
    )
    weights = config.loss_weights
    weighted_components = {
        "pressure_data": weights.pressure_data * pressure_data,
        "film_data": weights.film_data * film_data,
    }
    edge_values = torch.cat(
        [
            pressure[..., 0, :].reshape(-1),
            pressure[..., -1, :].reshape(-1),
            pressure[..., :, 0].reshape(-1),
            pressure[..., :, -1].reshape(-1),
        ]
    )
    boundary_loss = torch.mean(edge_values.square())
    if config.mode == "physics_regularized":
        residual = reynolds_residual(
            pressure,
            film,
            dx,
            dy,
            runtime.constants["eta0"],
            runtime.constants["alpha"],
            runtime.constants["speed"],
            runtime.constants["p_hertz"],
            runtime.constants["contact_radius"],
            runtime.constants["radius"],
            runtime.constants["reference_pressure"],
        )
        obstacle = reynolds_obstacle_loss(pressure, residual, config.reynolds_pressure_scale, config.reynolds_residual_scale)
        floor_violation = torch.relu(runtime.constants["film_floor"] - raw_film).square().mean()
        load = trapezoid_load(pressure, dx, dy)
        load_loss = torch.mean(
            ((load - DIMENSIONLESS_LOAD_TARGET) / DIMENSIONLESS_LOAD_TARGET) ** 2
        )
        weighted_components.update(
            {
                "reynolds_obstacle": weights.reynolds_obstacle * obstacle,
                "load_balance": weights.load_balance * load_loss,
                "boundary_pressure": weights.boundary_pressure * boundary_loss,
                "film_floor_violation": weights.film_floor_violation * floor_violation,
            }
        )
    else:
        obstacle = torch.zeros((), dtype=pressure.dtype, device=pressure.device)
        load_loss = torch.zeros((), dtype=pressure.dtype, device=pressure.device)
        floor_violation = torch.zeros((), dtype=pressure.dtype, device=pressure.device)
        zero = torch.zeros((), dtype=pressure.dtype, device=pressure.device)
        weighted_components.update(
            {
                "reynolds_obstacle": zero,
                "load_balance": zero,
                "boundary_pressure": zero,
                "film_floor_violation": zero,
            }
        )
    total = sum(weighted_components.values())
    diagnostics = {
        "total": float(total.detach().cpu()),
        "pressure_data": float(pressure_data.detach().cpu()),
        "film_data": float(film_data.detach().cpu()),
        "reynolds_obstacle": float(obstacle.detach().cpu()),
        "load_balance": float(load_loss.detach().cpu()),
        "boundary_pressure": float(boundary_loss.detach().cpu()),
        "film_floor_violation": float(floor_violation.detach().cpu()),
    }
    diagnostics.update(
        {
            f"weighted_{name}": float(value.detach().cpu())
            for name, value in weighted_components.items()
        }
    )
    if return_weighted_components:
        return total, diagnostics, weighted_components
    return total, diagnostics


def weighted_component_gradient_norms(
    model: nn.Module,
    weighted_components: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Measure each term's actual optimizer-scale gradient without changing .grad."""
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    result: dict[str, float] = {}
    for name, component in weighted_components.items():
        if not component.requires_grad:
            result[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            component,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = sum(
            gradient.detach().double().square().sum()
            for gradient in gradients
            if gradient is not None
        )
        result[name] = float(torch.sqrt(squared_norm).cpu()) if not isinstance(squared_norm, int) else 0.0
    return result


@torch.no_grad()
def evaluate_cases(
    model: PressureOffsetNet,
    convolver: TorchElasticConvolver,
    runtimes: Sequence[CaseRuntime],
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    dx: float,
    dy: float,
    film_floor_beta: float,
    film_mode: str,
    reynolds_pressure_scale: float = 1.0,
    reynolds_residual_scale: float = 1.0,
    evaluation_masks: Mapping[str, torch.Tensor] | None = None,
    selection_score_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    for runtime in runtimes:
        pressure, film, h0 = predict_case(
            model, convolver, runtime, x_grid, y_grid, film_floor_beta, film_mode
        )
        evaluation_mask = None if evaluation_masks is None else evaluation_masks.get(runtime.case.case_id)
        if evaluation_mask is None:
            p_rel = float(relative_l2(pressure, runtime.constants["pressure"])[0].cpu())
            h_rel = float(relative_l2(film, runtime.constants["film"])[0].cpu())
            evaluation_region = "full_field"
            evaluated_locations = pressure.numel()
        else:
            if evaluation_mask.shape != pressure.shape or not bool(evaluation_mask.any()):
                raise ValueError(f"Invalid evaluation mask for {runtime.case.case_id}")
            pressure_error = pressure[evaluation_mask] - runtime.constants["pressure"][evaluation_mask]
            film_error = film[evaluation_mask] - runtime.constants["film"][evaluation_mask]
            p_rel = float((torch.linalg.vector_norm(pressure_error) / torch.clamp(torch.linalg.vector_norm(runtime.constants["pressure"][evaluation_mask]), min=1e-12)).cpu())
            h_rel = float((torch.linalg.vector_norm(film_error) / torch.clamp(torch.linalg.vector_norm(runtime.constants["film"][evaluation_mask]), min=1e-12)).cpu())
            evaluation_region = "unobserved_spatial_holdout"
            evaluated_locations = int(evaluation_mask.sum().cpu())
        load = float(trapezoid_load(pressure, dx, dy)[0].cpu())
        pmax_true = float(torch.max(runtime.constants["pressure"]).cpu())
        pmax_pred = float(torch.max(pressure).cpu())
        hmin_true = float(torch.min(runtime.constants["film"]).cpu())
        hmin_pred = float(torch.min(film).cpu())
        residual = reynolds_residual(
            pressure,
            film,
            dx,
            dy,
            runtime.constants["eta0"],
            runtime.constants["alpha"],
            runtime.constants["speed"],
            runtime.constants["p_hertz"],
            runtime.constants["contact_radius"],
            runtime.constants["radius"],
            runtime.constants["reference_pressure"],
        )
        obstacle = float(reynolds_obstacle_loss(pressure, residual, reynolds_pressure_scale, reynolds_residual_scale).cpu())
        residual_rms = float(
            torch.sqrt(torch.mean(residual[..., 1:-1, 1:-1].square())).cpu()
        )
        pressure_np = pressure[0].cpu().numpy()
        film_np = film[0].cpu().numpy()
        p_true_np = runtime.case.pressure
        h_true_np = runtime.case.film
        p_pred_index = np.unravel_index(int(np.argmax(pressure_np)), pressure_np.shape)
        p_true_index = np.unravel_index(int(np.argmax(p_true_np)), p_true_np.shape)
        h_pred_index = np.unravel_index(int(np.argmin(film_np)), film_np.shape)
        h_true_index = np.unravel_index(int(np.argmin(h_true_np)), h_true_np.shape)

        def location_distance(first: tuple[int, int], second: tuple[int, int]) -> float:
            return float(
                np.hypot(
                    runtime.case.x[first[0]] - runtime.case.x[second[0]],
                    runtime.case.y[first[1]] - runtime.case.y[second[1]],
                )
            )

        boundary_max = float(
            max(
                np.max(np.abs(pressure_np[0, :])),
                np.max(np.abs(pressure_np[-1, :])),
                np.max(np.abs(pressure_np[:, 0])),
                np.max(np.abs(pressure_np[:, -1])),
            )
        )
        rows.append(
            {
                "case_id": runtime.case.case_id,
                "evaluation_region": evaluation_region,
                "evaluated_locations": evaluated_locations,
                "pressure_relative_l2": p_rel,
                "film_relative_l2": h_rel,
                "load_relative_error": abs(load - DIMENSIONLESS_LOAD_TARGET) / DIMENSIONLESS_LOAD_TARGET,
                "peak_pressure_relative_error": abs(pmax_pred - pmax_true) / max(abs(pmax_true), 1e-12),
                "peak_pressure_location_error": location_distance(p_pred_index, p_true_index),
                "minimum_film_relative_error": abs(hmin_pred - hmin_true) / max(abs(hmin_true), 1e-12),
                "minimum_film_location_error": location_distance(h_pred_index, h_true_index),
                "maximum_boundary_pressure": boundary_max,
                "maximum_symmetry_error": float(np.max(np.abs(pressure_np - pressure_np[:, ::-1]))),
                "reynolds_residual_rms": residual_rms,
                "reynolds_obstacle_loss": obstacle,
                "predicted_h0": float(h0[0].cpu()) if h0 is not None else None,
            }
        )
        scaling = runtime.case.manifest["scaling"]
        rows[-1].update(field_metrics(
            pressure_np, film_np, p_true_np, h_true_np, runtime.case.x, runtime.case.y,
            residual[0].cpu().numpy(), h0=float(h0[0].cpu()) if h0 is not None else None,
            h0_true=runtime.case.h0, pressure_pa=scaling["maximum_pressure_pa"],
            film_m=scaling["film_scale_m"], radius_m=scaling["contact_radius_m"],
            symmetric=model.transverse_symmetry,
        ))
        if h0 is not None:
            _, raw = close_film(pressure, h0, x_grid, y_grid, convolver,
                                runtime.constants["film_floor"], return_raw=True)
            rows[-1]["raw_film_floor_violation_max"] = float(torch.relu(runtime.constants["film_floor"] - raw).max().cpu())
    score_fields = tuple(selection_score_fields or (
        "pressure_relative_l2",
        "film_relative_l2",
        "load_relative_error",
        "minimum_film_relative_error",
    ))
    means = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in score_fields
    }
    diagnostic_fields = (
        "peak_pressure_relative_error",
        "peak_pressure_location_error",
        "minimum_film_location_error",
        "maximum_boundary_pressure",
        "maximum_symmetry_error",
        "reynolds_residual_rms",
        "reynolds_obstacle_loss",
    )
    means.update(
        {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in diagnostic_fields
        }
    )
    means["selection_score"] = float(sum(means[name] for name in score_fields))
    for name in rows[0]:
        values = [row[name] for row in rows if isinstance(row.get(name), (int, float))]
        if values:
            means.setdefault(name, float(np.mean(values)))
    spread = {}
    for name in score_fields:
        values = np.asarray([row[name] for row in rows], dtype=float)
        rng = np.random.default_rng(20260909)
        bootstrap = rng.choice(values, size=(1000, len(values)), replace=True).mean(axis=1)
        spread[name] = {"median": float(np.median(values)), "max": float(values.max()),
                        "case_bootstrap_mean_ci95": np.quantile(bootstrap, [.025, .975]).tolist()}
    return {"means": means, "cases": rows, "condition_statistics": spread,
            "obstacle_scales": {"pressure": reynolds_pressure_scale, "residual": reynolds_residual_scale},
            "confidence_interval_note": "Cases, not grid nodes, are resampled; a single-case interval is not generalization evidence."}


def _assert_resume_compatible(
    saved: Mapping[str, Any], current: TrainingConfig
) -> None:
    current_payload = asdict(current)
    operational_fields = {"epochs", "checkpoint_interval", "resume_checkpoint", "device"}
    mismatches = {
        name: (saved.get(name), value)
        for name, value in current_payload.items()
        if name not in operational_fields and saved.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Resume changes immutable training-contract fields: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    scaler: ConditionScaler,
    config: TrainingConfig,
    model_config: Mapping[str, Any],
    validation: Mapping[str, Any],
    numpy_rng_state: Mapping[str, Any],
    training_elapsed_seconds: float,
    contract: Mapping[str, Any] | None = None,
    history: list | None = None,
) -> None:
    payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "condition_scaler": scaler.to_dict(),
            "training_config": asdict(config),
            "model_config": dict(model_config),
            "validation": dict(validation),
            "numpy_rng_state": dict(numpy_rng_state),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_elapsed_seconds": float(training_elapsed_seconds),
            "data_contract": contract,
            "history": list(history or []),
            "python_rng_state": random.getstate(),
            "environment": runtime_environment(),
        }
    atomic_write(path, lambda temporary: torch.save(payload, temporary), retain_previous=True)


def _cpu_cuda_rng_states(states: Sequence[Any]) -> list[torch.Tensor]:
    """Return CUDA RNG states in the CPU byte-tensor form required by PyTorch."""
    return [
        state.detach().to(device="cpu", dtype=torch.uint8)
        if isinstance(state, torch.Tensor)
        else torch.as_tensor(state, dtype=torch.uint8, device="cpu")
        for state in states
    ]


def _validate_residual_calibration(
    config: TrainingConfig,
    contract: Mapping[str, Any],
    model_config: Mapping[str, Any],
    scaler: ConditionScaler,
) -> dict[str, Any] | None:
    if not config.residual_calibration_report:
        if config.run_purpose == "production" and config.mode == "physics_regularized":
            raise ValueError("Production training requires a bound residual_calibration_report")
        if config.mode == "physics_regularized" and not math.isclose(config.reynolds_residual_scale, 1.0):
            raise ValueError("A non-default Reynolds residual scale requires a bound residual_calibration_report")
        return None
    path = project_path(config.residual_calibration_report)
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if config.residual_calibration_sha256 and actual_sha256 != config.residual_calibration_sha256:
        raise ValueError("Residual calibration report checksum mismatch")
    if config.run_purpose == "production" and not config.residual_calibration_sha256:
        raise ValueError("Production training requires residual_calibration_sha256")
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("labels_used_for_scale") is not False:
        raise ValueError("Residual calibration must be label-free")
    frozen_scale = float(calibration.get("frozen_residual_scale", math.nan))
    if not math.isclose(frozen_scale, config.reynolds_residual_scale, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Configured Reynolds residual scale differs from the bound calibration")
    require_same_contract(calibration.get("contract"), contract)
    calibration_scope = calibration.get("experiment_scope", "multi_condition")
    if calibration_scope != config.experiment_scope or calibration.get("single_case_id") != config.single_case_id:
        raise ValueError("Residual calibration experiment scope does not match training")
    saved_scaler = calibration.get("condition_scaler", {})
    if not np.allclose(saved_scaler.get("mean"), scaler.mean) or not np.allclose(saved_scaler.get("scale"), scaler.scale):
        raise ValueError("Residual calibration condition scaler does not match training")
    architecture_fields = (
        "hidden_dim", "depth", "fourier_features", "fourier_scale", "x_bounds", "y_bounds",
        "h0_initial_value",
    )
    saved_model = calibration.get("model_config", {})
    mismatches = {}
    for name in architecture_fields:
        saved_value, current_value = saved_model.get(name), model_config.get(name)
        matches = bool(np.allclose(saved_value, current_value)) if name in {"x_bounds", "y_bounds"} else saved_value == current_value
        if not matches:
            mismatches[name] = (saved_value, current_value)
    if mismatches:
        raise ValueError("Residual calibration model contract mismatch: " + json.dumps(mismatches, sort_keys=True))
    return {"path": str(path), "sha256": actual_sha256, "frozen_residual_scale": frozen_scale}


def train(config: TrainingConfig) -> Path:
    config.validate()
    config = replace(config, case_dirs=tuple(str(project_path(p)) for p in config.case_dirs),
                     split_manifest=str(project_path(config.split_manifest)), output_root=str(project_path(config.output_root)),
                     reference_validation_report=str(project_path(config.reference_validation_report)) if config.reference_validation_report else None,
                     residual_calibration_report=str(project_path(config.residual_calibration_report)) if config.residual_calibration_report else None,
                     resume_checkpoint=str(project_path(config.resume_checkpoint)) if config.resume_checkpoint else None)
    if config.run_purpose == "production":
        if not config.reference_validation_report:
            raise ValueError("Complete production training requires a reference_validation_report; run a short pilot after the core fixes")
        gate = json.loads(project_path(config.reference_validation_report).read_text(encoding="utf-8"))
        if gate.get("status") != "passed" or not all(gate.get(name, {}).get("passed") for name in ("grid", "domain", "benchmark")):
            raise ValueError("Reference validation must include passed grid, domain and independent benchmark checks")
    seed_everything(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    device = select_device(config.device)
    split = SplitAssignment.load(config.split_manifest)
    metadata = [load_case_metadata(path, require_strict=True, verify_field_checksum=True) for path in config.case_dirs]
    metadata_groups = split.bind(metadata)
    material = metadata_groups["train"][0].manifest["case"]["material"]
    p_ref = metadata_groups["train"][0].manifest["case"]["roelands_reference_pressure_pa"]
    if any(m.manifest["case"]["material"] != material or m.manifest["case"]["roelands_reference_pressure_pa"] != p_ref for m in metadata):
        raise ValueError("Four-input surrogate requires fixed material/geometry and fixed Roelands reference pressure across cases")
    context = metadata_groups["train"][0].manifest["case"]
    if any(m.manifest["case"]["grid"] != context["grid"] or m.manifest["case"]["solver"]["film_floor_m"] != context["solver"]["film_floor_m"] for m in metadata):
        raise ValueError("All cases must share the inference grid and dimensional film floor")
    contract = data_contract(metadata, config.split_manifest)
    if config.run_purpose == "production" and gate.get("reference_field_sha256") != {m.case_id: m.manifest["field_sha256"] for m in metadata}:
        raise ValueError("Reference validation report is not bound to these reference fields")
    if config.experiment_scope == "single_condition_spatial":
        selected = [item for item in metadata_groups["train"] if item.case_id == config.single_case_id]
        if len(selected) != 1:
            raise ValueError("single_case_id must identify exactly one training-split condition")
        training_metadata = selected
        validation_metadata = []
    else:
        training_metadata = metadata_groups["train"]
        validation_metadata = metadata_groups["validation"]
    training_cases = [
        load_case_artifact(item.run_dir, require_strict=True)
        for item in training_metadata
    ]
    validation_cases = [
        load_case_artifact(item.run_dir, require_strict=True)
        for item in validation_metadata
    ]
    scaler = ConditionScaler.fit(training_cases)
    development_cases = training_cases + validation_cases
    nx, ny, dx, dy = _same_grid(development_cases)
    reference = training_cases[0]
    x_bounds = (float(reference.x[0]), float(reference.x[-1]))
    y_bounds = (float(reference.y[0]), float(reference.y[-1]))
    model_config = model_configuration(config, x_bounds, y_bounds)
    residual_calibration = _validate_residual_calibration(config, contract, model_config, scaler)
    model = PressureOffsetNet(**model_config).to(device)
    convolver = TorchElasticConvolver(nx, ny, dx).to(device)
    X, Y = np.meshgrid(reference.x, reference.y, indexing="ij")
    x_grid = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_grid = torch.as_tensor(Y, dtype=torch.float32, device=device)

    train_runtimes = [
        CaseRuntime(
            case,
            scaler,
            device,
            config.supervised_fraction,
            config.supervision_mode,
            config.seed + index * 1009,
        )
        for index, case in enumerate(training_cases)
    ]
    if config.experiment_scope == "single_condition_spatial":
        validation_runtimes = train_runtimes
        evaluation_masks = {
            runtime.case.case_id: ~runtime.mask
            for runtime in train_runtimes
        }
        selection_score_fields = ("pressure_relative_l2", "film_relative_l2")
        checkpoint_selection = "single_condition_unobserved_spatial_holdout"
    else:
        validation_runtimes = [
            CaseRuntime(case, scaler, device, 1.0, "uniform", config.seed)
            for case in validation_cases
        ]
        evaluation_masks = None
        selection_score_fields = None
        checkpoint_selection = "condition_validation_split"
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_dir = Path(config.output_root) / config.run_name
    if config.resume_checkpoint:
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "split_manifest": str(Path(config.split_manifest).resolve()),
        "split_manifest_sha256": hashlib.sha256(
            Path(config.split_manifest).read_bytes()
        ).hexdigest(),
        "case_manifests": {
            case.case_id: {
                "path": str((case.run_dir / "manifest.json").resolve()),
                "sha256": hashlib.sha256(
                    (case.run_dir / "manifest.json").read_bytes()
                ).hexdigest(),
            }
            for case in metadata
        },
        "checkpoint_selection_split": checkpoint_selection,
        "test_field_arrays_loaded_during_training": False,
    }
    provenance["data_contract"] = contract
    provenance["residual_calibration"] = residual_calibration

    steps = config.steps_per_epoch or len(train_runtimes)
    rng = np.random.default_rng(config.seed)
    best_score = math.inf
    prior_elapsed_seconds = 0.0
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if config.resume_checkpoint:
        resume = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        if resume.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("Old checkpoints used a different film/residual protocol; start a new run")
        require_same_contract(resume.get("data_contract"), contract)
        _assert_resume_compatible(resume.get("training_config", {}), config)
        if resume.get("model_config") != model_config:
            raise ValueError("Resume checkpoint model configuration does not match")
        saved_scaler = resume.get("condition_scaler", {})
        if not np.allclose(saved_scaler.get("mean"), scaler.mean) or not np.allclose(
            saved_scaler.get("scale"), scaler.scale
        ):
            raise ValueError("Resume checkpoint condition scaler does not match")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        validation_payload = resume.get("validation", {})
        best_score = float(
            validation_payload.get(
                "best_selection_score",
                validation_payload.get("means", {}).get("selection_score", math.inf),
            )
        )
        history = resume.get("history", [])
        if "python_rng_state" in resume:
            random.setstate(resume["python_rng_state"])
        if "numpy_rng_state" in resume:
            rng.bit_generator.state = resume["numpy_rng_state"]
        if "torch_rng_state" in resume:
            torch.set_rng_state(resume["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                _cpu_cuda_rng_states(resume["cuda_rng_state_all"])
            )
        prior_elapsed_seconds = float(resume.get("training_elapsed_seconds", 0.0))
    if start_epoch > config.epochs + 1:
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch - 1}, "
            f"which is not below requested epochs={config.epochs}"
        )
    atomic_json(run_dir / "resolved_config.json", asdict(config))
    atomic_json(run_dir / "condition_scaler.json", scaler.to_dict())
    atomic_json(run_dir / "provenance.json", provenance)
    label_budget = {
        "training_supervision_fraction": config.supervised_fraction,
        "observed_training_locations": sum(int(r.mask.sum().cpu()) for r in train_runtimes),
        "full_training_locations": sum(r.case.pressure.size for r in train_runtimes),
        "validation_locations_for_selection": (
            sum(int(mask.sum().cpu()) for mask in evaluation_masks.values())
            if evaluation_masks is not None
            else sum(r.case.pressure.size for r in validation_runtimes)
        ),
        "selection_protocol": checkpoint_selection,
        "observed_fields_per_location": ["P", "H"],
        "sampling": "nested_permutation_v2_" + config.supervision_mode,
        "note": "Multi-condition runs use full labels only in the condition-validation split. Single-condition runs select checkpoints only on the complement of the training supervision mask.",
    }
    atomic_json(run_dir / "label_budget.json", label_budget)
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        aggregate: dict[str, float] = {}
        gradient_diagnostics: dict[str, Any] = {}
        diagnostic_epoch = config.gradient_diagnostics_interval > 0 and (
            epoch == 1 or epoch % config.gradient_diagnostics_interval == 0 or epoch == config.epochs
        )
        for step_index in range(steps):
            runtime = train_runtimes[int(rng.integers(0, len(train_runtimes)))]
            optimizer.zero_grad(set_to_none=True)
            if diagnostic_epoch and step_index == 0:
                total, diagnostics, weighted_components = loss_for_case(
                    model, convolver, runtime, x_grid, y_grid, dx, dy, config,
                    return_weighted_components=True,
                )
                gradient_diagnostics = {
                    f"weighted_gradient_norm_{name}": value
                    for name, value in weighted_component_gradient_norms(model, weighted_components).items()
                }
                gradient_diagnostics["gradient_diagnostic_case_id"] = runtime.case.case_id
            else:
                total, diagnostics = loss_for_case(
                    model, convolver, runtime, x_grid, y_grid, dx, dy, config
                )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, case {runtime.case.case_id}; optimizer not advanced")
            total.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            diagnostics["gradient_norm_before_clip"] = float(norm.detach().cpu())
            diagnostics["gradient_clip_fraction"] = float(norm > config.grad_clip)
            optimizer.step()
            for name, value in diagnostics.items():
                aggregate[name] = aggregate.get(name, 0.0) + value / steps
        row = {"epoch": float(epoch), **aggregate, **gradient_diagnostics}

        if epoch == 1 or epoch % config.validation_interval == 0 or epoch == config.epochs:
            validation = evaluate_cases(
                model,
                convolver,
                validation_runtimes,
                x_grid,
                y_grid,
                dx,
                dy,
                config.film_floor_beta,
                config.film_mode,
                config.reynolds_pressure_scale,
                config.reynolds_residual_scale,
                evaluation_masks,
                selection_score_fields,
            )
            score = float(validation["means"]["selection_score"])
            if not math.isfinite(score):
                raise FloatingPointError(f"Non-finite validation score at epoch {epoch}")
            row["validation_score"] = score
            row["training_elapsed_seconds"] = prior_elapsed_seconds + (
                time.perf_counter() - started
            )
            if score < best_score:
                best_score = score
                _save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    scaler,
                    config,
                    model_config,
                    validation,
                    rng.bit_generator.state,
                    row["training_elapsed_seconds"],
                    contract, history + [row],
                )
        row["training_elapsed_seconds"] = prior_elapsed_seconds + (
            time.perf_counter() - started
        )
        history.append(row)
        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            _save_checkpoint(
                run_dir / "last.pt",
                model,
                optimizer,
                epoch,
                scaler,
                config,
                model_config,
                {"best_selection_score": best_score},
                rng.bit_generator.state,
                row["training_elapsed_seconds"],
                contract, history,
            )
            atomic_json(run_dir / "training_history.json", history)

    segment_elapsed_seconds = time.perf_counter() - started
    cumulative_elapsed_seconds = prior_elapsed_seconds + segment_elapsed_seconds
    atomic_json(run_dir / "training_summary.json",
            {
                "best_validation_score": best_score,
                "elapsed_seconds_this_invocation": segment_elapsed_seconds,
                "training_elapsed_seconds": cumulative_elapsed_seconds,
                "device": str(device),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                **model.parameter_counts(),
                "baseline": baseline_description(config),
                "run_purpose": config.run_purpose,
                "completed_epochs": config.epochs,
                "optimizer_updates": config.epochs * steps,
                "environment": runtime_environment(),
                "checkpoint_selected_by": checkpoint_selection,
                "test_field_arrays_loaded": False,
            })
    return run_dir
