from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import CaseConfig


DIMENSIONLESS_LOAD_TARGET = 2.0 * math.pi / 3.0


@dataclass(frozen=True)
class HertzScaling:
    contact_radius_m: float
    maximum_pressure_pa: float
    film_scale_m: float
    reduced_modulus_pa: float
    equivalent_radius_m: float
    transverse_radius_ratio: float
    pressure_to_deformation_ratio: float

    @classmethod
    def circular_point_contact(cls, case: CaseConfig) -> "HertzScaling":
        material = case.material
        rx = material.rx_m
        ry = material.ry_m
        if not math.isclose(rx, ry, rel_tol=1e-8, abs_tol=1e-12):
            raise NotImplementedError(
                "V0.1 currently supports circular point contact only; "
                "elliptical Hertz scaling must be implemented before using Rx != Ry"
            )
        e_star = material.reduced_modulus_pa
        radius = rx
        contact_radius = (3.0 * case.load_n * radius / (4.0 * e_star)) ** (1.0 / 3.0)
        maximum_pressure = 3.0 * case.load_n / (2.0 * math.pi * contact_radius**2)
        return cls(
            contact_radius_m=contact_radius,
            maximum_pressure_pa=maximum_pressure,
            film_scale_m=contact_radius**2 / radius,
            reduced_modulus_pa=e_star,
            equivalent_radius_m=radius,
            transverse_radius_ratio=1.0,
            pressure_to_deformation_ratio=1.0,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "contact_radius_m": self.contact_radius_m,
            "maximum_pressure_pa": self.maximum_pressure_pa,
            "film_scale_m": self.film_scale_m,
            "reduced_modulus_pa": self.reduced_modulus_pa,
            "equivalent_radius_m": self.equivalent_radius_m,
            "transverse_radius_ratio": self.transverse_radius_ratio,
            "pressure_to_deformation_ratio": self.pressure_to_deformation_ratio,
        }


def hertz_pressure(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    radial = 1.0 - x**2 - y**2
    return np.sqrt(np.maximum(radial, 0.0))


def roelands_viscosity_ratio(
    pressure_dimensionless: np.ndarray,
    case: CaseConfig,
    scaling: HertzScaling,
) -> np.ndarray:
    pressure_pa = scaling.maximum_pressure_pa * np.maximum(pressure_dimensionless, 0.0)
    base = np.maximum(1.0 + pressure_pa / case.roelands_reference_pressure_pa, 1e-12)
    exponent = (math.log(case.eta0_pa_s) + 9.67) * (np.power(base, case.roelands_z) - 1.0)
    return np.exp(np.clip(exponent, -60.0, 60.0))


def dowson_higginson_density_ratio(
    pressure_dimensionless: np.ndarray,
    scaling: HertzScaling,
) -> np.ndarray:
    pressure_gpa = 1e-9 * scaling.maximum_pressure_pa * np.maximum(pressure_dimensionless, 0.0)
    return (0.59 + 1.34 * pressure_gpa) / (0.59 + pressure_gpa)


def reynolds_speed_parameter(case: CaseConfig, scaling: HertzScaling) -> float:
    value = (
        12.0
        * case.entrainment_speed_m_s
        * case.eta0_pa_s
        * scaling.equivalent_radius_m**2
        / (scaling.maximum_pressure_pa * scaling.contact_radius_m**3)
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid Reynolds speed parameter: {value!r}")
    return value


def moes_parameters(case: CaseConfig, scaling: HertzScaling) -> tuple[float, float]:
    e_prime = 2.0 * scaling.reduced_modulus_pa
    radius = scaling.equivalent_radius_m
    u = case.eta0_pa_s * case.entrainment_speed_m_s / (e_prime * radius)
    w = case.load_n / (e_prime * radius**2)
    g = case.alpha_pa_inv * e_prime
    return float(w * u ** (-0.75)), float(g * u**0.25)


def integrate_load(pressure: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(pressure, y, axis=1), x, axis=0))

