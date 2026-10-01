"""Evaluation-only metrics; none of these reference masks enters training."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def extrema_distance(pred, true, x, y, minimum=False, symmetric=False) -> float:
    extreme = np.min if minimum else np.max
    def points(a):
        value = extreme(a)
        tolerance = max(abs(float(value)) * 1e-6, 1e-10)
        i, j = np.where(np.abs(a - value) <= tolerance)
        return np.column_stack((x[i], np.abs(y[j]) if symmetric else y[j]))
    # Both directions prevent an extensive false plateau from scoring zero.
    a, b = points(pred), points(true)
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def field_metrics(p, h, p_true, h_true, x, y, residual, *, h0=None, h0_true=None,
                  pressure_pa=1.0, film_m=1.0, radius_m=1.0, symmetric=False):
    delta = p - p_true
    result = {
        "pressure_rmse": float(np.sqrt(np.mean(delta**2))),
        "pressure_max_absolute_error": float(np.max(np.abs(delta))),
        "film_rmse": float(np.sqrt(np.mean((h - h_true)**2))),
        "peak_pressure_location_error": extrema_distance(p, p_true, x, y, symmetric=symmetric),
        "minimum_film_location_error": extrema_distance(h, h_true, x, y, minimum=True, symmetric=symmetric),
        "maximum_film_symmetry_error": float(np.max(np.abs(h - h[:, ::-1]))),
        "minimum_predicted_pressure": float(p.min()),
        "peak_pressure_pa": float(p.max() * pressure_pa),
        "minimum_film_m": float(h.min() * film_m),
        "predicted_h0": h0,
        "reference_h0": float(h0_true) if h0 is not None else None,
        "h0_absolute_error": abs(h0 - h0_true) if h0 is not None else None,
        "h0_absolute_error_m": abs(h0 - h0_true) * film_m if h0 is not None else None,
    }
    for fraction, name in [(0.01, "top_1pct"), (0.005, "top_0_5pct")]:
        k = max(1, int(np.ceil(p_true.size * fraction)))
        indices = np.argsort(-p_true.reshape(-1), kind="stable")[:k]
        truth = p_true.reshape(-1)[indices]
        error = delta.reshape(-1)[indices]
        result[name + "_pressure_relative_l2"] = float(np.linalg.norm(error) / max(np.linalg.norm(truth), 1e-12))
        result[name + "_pressure_rmse"] = float(np.sqrt(np.mean(error**2)))
    for region_name, mask in [("reference_active", p_true[1:-1, 1:-1] > 1e-6),
                              ("predicted_active", p[1:-1, 1:-1] > 1e-6)]:
        r = residual[1:-1, 1:-1]
        result[region_name + "_residual_rms"] = float(np.sqrt(np.mean(r[mask]**2))) if mask.any() else None
        result[region_name + "_cavitation_positive_residual_max"] = float(np.maximum(r[~mask], 0).max()) if (~mask).any() else None
    result["minimum_film_location_error_m"] = result["minimum_film_location_error"] * radius_m
    result["peak_pressure_location_error_m"] = result["peak_pressure_location_error"] * radius_m
    outlet = (x >= 0.5) & (x <= 1.5)
    if outlet.any():
        denominator = max(float(p_true[outlet].max()), 1e-12)
        result["outlet_window_peak_relative_error"] = abs(float(p[outlet].max()) - float(p_true[outlet].max())) / denominator
    return result
