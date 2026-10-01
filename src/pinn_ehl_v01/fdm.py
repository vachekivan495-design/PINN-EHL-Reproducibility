from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
import time
from typing import Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .config import CaseConfig
from .elasticity import ELASTIC_COEFFICIENT, ElasticConvolver
from .grid import Grid
from .physics import (
    DIMENSIONLESS_LOAD_TARGET,
    HertzScaling,
    dowson_higginson_density_ratio,
    hertz_pressure,
    integrate_load,
    moes_parameters,
    reynolds_speed_parameter,
    roelands_viscosity_ratio,
)


@dataclass
class PressureSolveResult:
    pressure: np.ndarray
    film: np.ndarray
    h0: float
    iterations: int
    pressure_change: float
    stable_checks: int
    converged: bool
    history: list[dict[str, float]]


@dataclass
class FDMSolution:
    case: CaseConfig
    grid: Grid
    scaling: HertzScaling
    pressure: np.ndarray
    film: np.ndarray
    h0: float
    load_integral: float
    load_relative_error: float
    pressure_change: float
    pressure_iterations: int
    load_iterations: int
    pressure_converged: bool
    load_converged: bool
    strict_converged: bool
    elapsed_seconds: float
    pressure_history: list[dict[str, float]]
    load_history: list[dict[str, float]]
    continuation_levels: list[dict[str, float | bool]]

    def summary(self) -> dict[str, Any]:
        m_value, l_value = moes_parameters(self.case, self.scaling)
        return {
            "schema_version": "fdm_solution_v0.1",
            "case_id": self.case.case_id,
            "case_fingerprint": self.case.fingerprint,
            "roelands_parameterization": "alpha_primary_z_derived",
            "roelands_z": self.case.roelands_z,
            "cavitation_model": "classical_nonnegative_reynolds_boundary",
            "mass_conserving_cavitation": False,
            "boundary_condition": "p=0 on all four far-field edges",
            "symmetry_treatment": "full domain with explicit even projection",
            "solver_algorithm": (
                "multilevel_continuation_projected_jacobi_"
                "bracketed_secant_v0.2"
            ),
            "continuation_levels": self.continuation_levels,
            "grid_shape": list(self.pressure.shape),
            "h0": self.h0,
            "load_target": DIMENSIONLESS_LOAD_TARGET,
            "load_integral": self.load_integral,
            "load_relative_error": self.load_relative_error,
            "pressure_change": self.pressure_change,
            "pressure_iterations": self.pressure_iterations,
            "load_iterations": self.load_iterations,
            "pressure_converged": self.pressure_converged,
            "load_converged": self.load_converged,
            "strict_converged": self.strict_converged,
            "peak_pressure": float(np.max(self.pressure)),
            "minimum_film_thickness": float(np.min(self.film)),
            "maximum_boundary_pressure": maximum_boundary_pressure(self.pressure),
            "maximum_symmetry_error": maximum_symmetry_error(self.pressure),
            "elapsed_seconds": self.elapsed_seconds,
            "moes_M": m_value,
            "moes_L": l_value,
            "scaling": self.scaling.to_dict(),
            "case": self.case.to_dict(),
        }


def maximum_boundary_pressure(pressure: np.ndarray) -> float:
    return float(
        max(
            np.max(np.abs(pressure[0, :])),
            np.max(np.abs(pressure[-1, :])),
            np.max(np.abs(pressure[:, 0])),
            np.max(np.abs(pressure[:, -1])),
        )
    )


def maximum_symmetry_error(field: np.ndarray) -> float:
    return float(np.max(np.abs(field - field[:, ::-1])))


def _relative_change(current: float, previous: float) -> float:
    return abs(current - previous) / max(abs(previous), 1e-12)


def _film_from_pressure(
    pressure: np.ndarray,
    h0: float,
    geometry: np.ndarray,
    convolver: ElasticConvolver,
    film_floor: float,
) -> np.ndarray:
    deformation = convolver(pressure)
    return np.maximum(h0 + geometry + deformation, film_floor)


def solve_pressure_fixed_h0(
    case: CaseConfig,
    grid: Grid,
    scaling: HertzScaling,
    convolver: ElasticConvolver,
    h0: float,
    initial_pressure: np.ndarray | None = None,
) -> PressureSolveResult:
    solver = case.solver
    if initial_pressure is None:
        pressure = hertz_pressure(grid.X, grid.Y)
    else:
        if initial_pressure.shape != grid.shape:
            raise ValueError("initial_pressure shape does not match the configured grid")
        pressure = np.maximum(np.asarray(initial_pressure, dtype=np.float64), 0.0).copy()
    pressure[[0, -1], :] = 0.0
    pressure[:, [0, -1]] = 0.0
    pressure = 0.5 * (pressure + pressure[:, ::-1])

    geometry = 0.5 * (
        grid.X**2 + scaling.transverse_radius_ratio * grid.Y**2
    )
    film_floor = case.solver.film_floor_m / scaling.film_scale_m
    speed_parameter = reynolds_speed_parameter(case, scaling)
    ak00 = float(convolver.kernel[0, 0])
    ak10 = float(convolver.kernel[1, 0])
    coupling = (
        scaling.pressure_to_deformation_ratio * ELASTIC_COEFFICIENT
    )

    history: list[dict[str, float]] = []
    previous_check: tuple[float, float, float] | None = None
    stable_checks = 0
    pressure_change = math.inf
    film = _film_from_pressure(pressure, h0, geometry, convolver, film_floor)
    # The load balance is the outer solve's acceptance criterion.  Its
    # per-check drift must therefore be tighter than the final load tolerance;
    # otherwise a slowly relaxing pressure field can be accepted too early and
    # make repeated evaluations at the same h0 return different loads.
    stability_tolerance = min(solver.field_tolerance, 0.25 * solver.load_tolerance)
    effective_min_pressure_iterations = max(
        solver.min_pressure_iterations,
        500 if grid.shape[0] >= 513 else solver.min_pressure_iterations,
    )

    for iteration in range(1, solver.max_pressure_iterations + 1):
        viscosity = roelands_viscosity_ratio(pressure, case, scaling)
        density = dowson_higginson_density_ratio(pressure, scaling)
        epsilon = density * film**3 / np.maximum(viscosity * speed_parameter, 1e-30)

        center = pressure[1:-1, 1:-1]
        left = pressure[:-2, 1:-1]
        right = pressure[2:, 1:-1]
        down = pressure[1:-1, :-2]
        up = pressure[1:-1, 2:]

        eps_center = epsilon[1:-1, 1:-1]
        d1 = 0.5 * (epsilon[:-2, 1:-1] + eps_center)
        d2 = 0.5 * (epsilon[2:, 1:-1] + eps_center)
        d4 = 0.5 * (epsilon[1:-1, :-2] + eps_center)
        d5 = 0.5 * (epsilon[1:-1, 2:] + eps_center)

        density_center = density[1:-1, 1:-1]
        density_left = density[:-2, 1:-1]
        d8 = coupling * density_center * ak00
        d9 = coupling * density_left * ak10
        denominator = d1 + d2 + d4 + d5 + (d8 - d9) * grid.dx
        denominator = np.maximum(denominator, 1e-30)
        neighbor_sum = d1 * left + d2 * right + d4 * down + d5 * up
        wedge = (
            density_center * film[1:-1, 1:-1]
            - d8 * center
            - density_left * film[:-2, 1:-1]
            + d9 * center
        ) * grid.dx
        candidate = np.maximum((neighbor_sum - wedge) / denominator, 0.0)

        updated = pressure.copy()
        omega = solver.pressure_relaxation
        updated[1:-1, 1:-1] = (1.0 - omega) * center + omega * candidate
        updated[[0, -1], :] = 0.0
        updated[:, [0, -1]] = 0.0
        updated = 0.5 * (updated + updated[:, ::-1])

        pressure_change = float(
            np.sum(np.abs(updated - pressure)) / max(np.sum(np.abs(updated)), 1e-30)
        )
        pressure = updated
        film = _film_from_pressure(pressure, h0, geometry, convolver, film_floor)

        should_check = (
            iteration == 1
            or iteration % solver.history_interval == 0
            or iteration == solver.max_pressure_iterations
        )
        if should_check:
            load_value = integrate_load(pressure, grid.x, grid.y)
            pmax = float(np.max(pressure))
            hmin = float(np.min(film))
            current_check = (pmax, hmin, load_value)
            if previous_check is not None:
                local_change = max(
                    _relative_change(pmax, previous_check[0]),
                    _relative_change(hmin, previous_check[1]),
                    _relative_change(load_value, previous_check[2]),
                )
                stable = local_change <= stability_tolerance
                stable_checks = stable_checks + 1 if stable else 0
            else:
                stable_checks = 0
            previous_check = current_check
            history.append(
                {
                    "iteration": float(iteration),
                    "pressure_change": pressure_change,
                    "load_integral": load_value,
                    "peak_pressure": pmax,
                    "minimum_film_thickness": hmin,
                    "stable_checks": float(stable_checks),
                }
            )

        converged = (
            iteration >= effective_min_pressure_iterations
            and pressure_change <= solver.pressure_tolerance
            and stable_checks >= solver.stable_checks_required
        )
        if converged:
            return PressureSolveResult(
                pressure=pressure,
                film=film,
                h0=float(h0),
                iterations=iteration,
                pressure_change=pressure_change,
                stable_checks=stable_checks,
                converged=True,
                history=history,
            )

    return PressureSolveResult(
        pressure=pressure,
        film=film,
        h0=float(h0),
        iterations=solver.max_pressure_iterations,
        pressure_change=pressure_change,
        stable_checks=stable_checks,
        converged=False,
        history=history,
    )


def _initial_h0(
    grid: Grid,
    scaling: HertzScaling,
    convolver: ElasticConvolver,
) -> float:
    pressure = hertz_pressure(grid.X, grid.Y)
    geometry = 0.5 * (
        grid.X**2 + scaling.transverse_radius_ratio * grid.Y**2
    )
    base = geometry + convolver(pressure)
    return float(0.1 - np.min(base))


def _solve_case_single_grid(
    case: CaseConfig,
    progress_callback: Callable[[dict[str, float | str | bool]], None] | None = None,
    initial_pressure: np.ndarray | None = None,
    initial_h0: float | None = None,
) -> FDMSolution:
    case.validate()
    start = time.perf_counter()
    grid = Grid.from_config(case.grid)
    scaling = HertzScaling.circular_point_contact(case)
    convolver = ElasticConvolver.build(
        case.grid.nx,
        case.grid.ny,
        grid.dx,
        pressure_to_deformation_ratio=scaling.pressure_to_deformation_ratio,
    )

    load_history: list[dict[str, float]] = []
    pressure_histories: list[dict[str, float]] = []
    center = (
        float(initial_h0)
        if initial_h0 is not None
        else _initial_h0(grid, scaling, convolver)
    )
    step = 0.01 if initial_h0 is not None else 0.08
    def evaluate(
        h0: float,
        stage: str,
        warm_pressure: np.ndarray | None,
    ) -> tuple[PressureSolveResult, float]:
        result = solve_pressure_fixed_h0(
            case,
            grid,
            scaling,
            convolver,
            h0,
            initial_pressure=warm_pressure,
        )
        if not result.converged:
            raise RuntimeError(
                f"Pressure solve did not converge during {stage} at h0={h0:.8g}; "
                "the load residual is not admissible for bracketing"
            )
        load_value = integrate_load(result.pressure, grid.x, grid.y)
        # A warm-started nonlinear solve can satisfy the local pressure
        # criterion while its integrated load is still drifting.  Polish the
        # same h0 until two consecutive pressure solves agree in load.
        load_stable = False
        max_polish_passes = int(os.environ.get("PINN_EHL_LOAD_POLISH_PASSES", "12"))
        if max_polish_passes < 1:
            raise ValueError("PINN_EHL_LOAD_POLISH_PASSES must be positive")
        load_stability_factor = 0.1 if stage == "bracketed_secant" else 0.5
        load_stability_limit = load_stability_factor * case.solver.load_tolerance
        polish_passes = 0
        load_drift = math.inf
        for polish_passes in range(1, max_polish_passes + 1):
            polished = solve_pressure_fixed_h0(
                case,
                grid,
                scaling,
                convolver,
                h0,
                initial_pressure=result.pressure,
            )
            if not polished.converged:
                raise RuntimeError(
                    f"Pressure polishing did not converge during {stage} at h0={h0:.8g}"
                )
            polished_load = integrate_load(polished.pressure, grid.x, grid.y)
            load_drift = abs(polished_load - load_value) / DIMENSIONLESS_LOAD_TARGET
            result = polished
            load_value = polished_load
            if load_drift <= load_stability_limit:
                load_stable = True
                break
        if not load_stable:
            raise RuntimeError(
                f"Pressure load did not stabilize during {stage} at h0={h0:.8g}; "
                f"last relative drift={load_drift:.3e}, "
                f"required<={load_stability_limit:.3e}"
            )
        residual = load_value - DIMENSIONLESS_LOAD_TARGET
        progress = {
                "evaluation": float(len(load_history) + 1),
                "stage": stage,
                "h0": float(h0),
                "load_integral": load_value,
                "load_residual": residual,
                "load_relative_error": abs(residual) / DIMENSIONLESS_LOAD_TARGET,
                "load_stability_relative_drift": load_drift,
                "load_stability_limit": load_stability_limit,
                "load_polish_passes": float(polish_passes),
                "pressure_iterations": float(result.iterations),
                "pressure_change": result.pressure_change,
                "pressure_converged": float(result.converged),
            }
        load_history.append(progress)
        if progress_callback is not None:
            progress_callback(progress)
        pressure_histories.extend(
            {"h0": float(h0), **row} for row in result.history
        )
        return result, residual

    center_result, center_residual = evaluate(center, "center", initial_pressure)
    best_result = center_result
    best_residual = center_residual

    def retain_best(result: PressureSolveResult, residual: float) -> None:
        nonlocal best_result, best_residual
        if abs(residual) < abs(best_residual):
            best_result = result
            best_residual = residual

    if center_residual >= 0.0:
        low, low_result, low_residual = center, center_result, center_residual
        high = center + step
        high_result, high_residual = evaluate(high, "bracket", low_result.pressure)
        retain_best(high_result, high_residual)
        for _ in range(12):
            if high_residual <= 0.0:
                break
            step *= 1.7
            high += step
            high_result, high_residual = evaluate(
                high, "bracket", high_result.pressure
            )
            retain_best(high_result, high_residual)
    else:
        high, high_result, high_residual = center, center_result, center_residual
        low = center - step
        low_result, low_residual = evaluate(low, "bracket", high_result.pressure)
        retain_best(low_result, low_residual)
        for _ in range(12):
            if low_residual >= 0.0:
                break
            step *= 1.7
            low -= step
            low_result, low_residual = evaluate(low, "bracket", low_result.pressure)
            retain_best(low_result, low_residual)

    if not (low_residual >= 0.0 and high_residual <= 0.0):
        raise RuntimeError(
            "Could not bracket the load-balanced film offset; inspect load_history"
        )

    final_result = best_result
    final_residual = best_residual
    load_iterations = 0
    for load_iterations in range(1, case.solver.max_load_iterations + 1):
        span = high - low
        h0_resolution = 16.0 * np.spacing(max(abs(low), abs(high), 1.0))
        if span <= h0_resolution:
            break

        residual_span = low_residual - high_residual
        fraction = (
            low_residual / residual_span
            if residual_span > 0.0 and math.isfinite(residual_span)
            else 0.5
        )
        if not 0.05 <= fraction <= 0.95:
            fraction = 0.5
        candidate_h0 = low + fraction * span
        if not low < candidate_h0 < high:
            candidate_h0 = 0.5 * (low + high)

        high_weight = (candidate_h0 - low) / span
        warm_pressure = (
            (1.0 - high_weight) * low_result.pressure
            + high_weight * high_result.pressure
        )
        candidate_result, candidate_residual = evaluate(
            candidate_h0, "bracketed_secant", warm_pressure
        )
        retain_best(candidate_result, candidate_residual)
        final_result = best_result
        final_residual = best_residual
        load_error = abs(final_residual) / DIMENSIONLESS_LOAD_TARGET
        if final_result.converged and load_error <= case.solver.load_tolerance:
            break
        if candidate_residual >= 0.0:
            low, low_result, low_residual = (
                candidate_h0,
                candidate_result,
                candidate_residual,
            )
        else:
            high, high_result, high_residual = (
                candidate_h0,
                candidate_result,
                candidate_residual,
            )

    load_value = integrate_load(final_result.pressure, grid.x, grid.y)
    load_relative_error = abs(load_value - DIMENSIONLESS_LOAD_TARGET) / DIMENSIONLESS_LOAD_TARGET
    pressure_ok = bool(final_result.converged)
    load_ok = bool(load_relative_error <= case.solver.load_tolerance)
    strict = bool(
        pressure_ok
        and load_ok
        and np.all(np.isfinite(final_result.pressure))
        and np.all(np.isfinite(final_result.film))
        and maximum_boundary_pressure(final_result.pressure) == 0.0
        and maximum_symmetry_error(final_result.pressure) <= 1e-12
    )
    return FDMSolution(
        case=case,
        grid=grid,
        scaling=scaling,
        pressure=final_result.pressure,
        film=final_result.film,
        h0=final_result.h0,
        load_integral=load_value,
        load_relative_error=load_relative_error,
        pressure_change=final_result.pressure_change,
        pressure_iterations=final_result.iterations,
        load_iterations=load_iterations,
        pressure_converged=pressure_ok,
        load_converged=load_ok,
        strict_converged=strict,
        elapsed_seconds=time.perf_counter() - start,
        pressure_history=pressure_histories,
        load_history=load_history,
        continuation_levels=[],
    )


def continuation_grid_sizes(target_nodes: int) -> list[int]:
    if target_nodes < 17 or target_nodes % 2 == 0:
        raise ValueError("target_nodes must be odd and at least 17")
    levels = [int(target_nodes)]
    while True:
        next_nodes = (levels[-1] + 1) // 2
        if next_nodes < 65:
            break
        levels.append(next_nodes)
    return list(reversed(levels))


def _interpolate_pressure(source: FDMSolution, target_grid: Grid) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (source.grid.x, source.grid.y),
        source.pressure,
        method="linear",
        bounds_error=True,
    )
    points = np.stack([target_grid.X.reshape(-1), target_grid.Y.reshape(-1)], axis=1)
    pressure = interpolator(points).reshape(target_grid.shape)
    pressure = np.maximum(pressure, 0.0)
    pressure[[0, -1], :] = 0.0
    pressure[:, [0, -1]] = 0.0
    return 0.5 * (pressure + pressure[:, ::-1])


def solve_case(
    case: CaseConfig,
    progress_callback: Callable[[dict[str, float | str | bool]], None] | None = None,
) -> FDMSolution:
    case.validate()
    if case.grid.nx != case.grid.ny:
        raise NotImplementedError("V0.1 multilevel continuation requires a square grid")
    started = time.perf_counter()
    levels = continuation_grid_sizes(case.grid.nx)
    previous: FDMSolution | None = None
    initial_pressure: np.ndarray | None = None
    initial_h0: float | None = None
    pressure_history: list[dict[str, float]] = []
    load_history: list[dict[str, float]] = []
    level_summaries: list[dict[str, float | bool]] = []

    for nodes in levels:
        level_case = replace(case, grid=replace(case.grid, nx=nodes, ny=nodes))
        if previous is not None:
            target_grid = Grid.from_config(level_case.grid)
            initial_pressure = _interpolate_pressure(previous, target_grid)
            initial_h0 = previous.h0

        def level_progress(row: dict[str, float | str | bool]) -> None:
            if progress_callback is not None:
                progress_callback({"grid_n": float(nodes), **row})

        result = _solve_case_single_grid(
            level_case,
            progress_callback=level_progress,
            initial_pressure=initial_pressure,
            initial_h0=initial_h0,
        )
        if not result.strict_converged:
            raise RuntimeError(f"Continuation level {nodes} did not pass its strict gates")
        pressure_history.extend(
            {"grid_n": float(nodes), **row} for row in result.pressure_history
        )
        load_history.extend(
            {"grid_n": float(nodes), **row} for row in result.load_history
        )
        level_summaries.append(
            {
                "grid_n": float(nodes),
                "elapsed_seconds": result.elapsed_seconds,
                "pressure_iterations_final": float(result.pressure_iterations),
                "load_iterations": float(result.load_iterations),
                "load_relative_error": result.load_relative_error,
                "pressure_change": result.pressure_change,
                "strict_converged": result.strict_converged,
            }
        )
        previous = result

    if previous is None:
        raise RuntimeError("No continuation level was executed")
    return replace(
        previous,
        case=case,
        elapsed_seconds=time.perf_counter() - started,
        pressure_history=pressure_history,
        load_history=load_history,
        continuation_levels=level_summaries,
    )
