from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


@dataclass(frozen=True)
class MaterialConfig:
    e1_pa: float = 210e9
    e2_pa: float = 81e9
    nu1: float = 0.3
    nu2: float = 0.208
    rx1_m: float = 0.01268
    rx2_m: float = math.inf
    ry1_m: float = 0.01268
    ry2_m: float = math.inf

    def validate(self) -> None:
        _positive("e1_pa", self.e1_pa)
        _positive("e2_pa", self.e2_pa)
        for name, value in (("nu1", self.nu1), ("nu2", self.nu2)):
            if not math.isfinite(value) or not (-1.0 < value < 0.5):
                raise ValueError(f"{name} must lie in (-1, 0.5), got {value!r}")
        for name, value in (
            ("rx1_m", self.rx1_m),
            ("rx2_m", self.rx2_m),
            ("ry1_m", self.ry1_m),
            ("ry2_m", self.ry2_m),
        ):
            if not (math.isinf(value) or (math.isfinite(value) and value > 0.0)):
                raise ValueError(f"{name} must be positive or infinity, got {value!r}")

    @staticmethod
    def _combined_radius(r1: float, r2: float) -> float:
        inverse = (0.0 if math.isinf(r1) else 1.0 / r1) + (
            0.0 if math.isinf(r2) else 1.0 / r2
        )
        if inverse <= 0.0:
            raise ValueError("At least one finite radius is required in each direction")
        return 1.0 / inverse

    @property
    def reduced_modulus_pa(self) -> float:
        return 1.0 / (
            (1.0 - self.nu1**2) / self.e1_pa
            + (1.0 - self.nu2**2) / self.e2_pa
        )

    @property
    def rx_m(self) -> float:
        return self._combined_radius(self.rx1_m, self.rx2_m)

    @property
    def ry_m(self) -> float:
        return self._combined_radius(self.ry1_m, self.ry2_m)

    def to_dict(self) -> dict[str, float | None]:
        payload = asdict(self)
        return {
            name: (None if isinstance(value, float) and math.isinf(value) else value)
            for name, value in payload.items()
        }


@dataclass(frozen=True)
class GridConfig:
    nx: int = 65
    ny: int = 65
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -2.0
    y_max: float = 2.0

    def validate(self) -> None:
        if self.nx < 17 or self.ny < 17:
            raise ValueError("nx and ny must both be at least 17")
        if self.nx % 2 == 0 or self.ny % 2 == 0:
            raise ValueError("nx and ny must be odd so the symmetry line is represented")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("Grid bounds must be strictly increasing")
        dx = (self.x_max - self.x_min) / (self.nx - 1)
        dy = (self.y_max - self.y_min) / (self.ny - 1)
        if not math.isclose(dx, dy, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("V0.1 FDM requires dx == dy")


@dataclass(frozen=True)
class SolverConfig:
    pressure_tolerance: float = 1e-6
    load_tolerance: float = 1e-3
    field_tolerance: float = 5e-5
    max_pressure_iterations: int = 3000
    min_pressure_iterations: int = 50
    max_load_iterations: int = 24
    pressure_relaxation: float = 0.45
    film_floor_m: float = 1e-9
    history_interval: int = 10
    stable_checks_required: int = 3

    def validate(self) -> None:
        for name, value in (
            ("pressure_tolerance", self.pressure_tolerance),
            ("load_tolerance", self.load_tolerance),
            ("field_tolerance", self.field_tolerance),
            ("film_floor_m", self.film_floor_m),
        ):
            _positive(name, value)
        if not 0.0 < self.pressure_relaxation <= 1.0:
            raise ValueError("pressure_relaxation must lie in (0, 1]")
        if self.max_pressure_iterations < self.min_pressure_iterations:
            raise ValueError("max_pressure_iterations must be >= min_pressure_iterations")
        if self.max_load_iterations < 1 or self.history_interval < 1:
            raise ValueError("Iteration counts must be positive")
        if self.stable_checks_required < 1:
            raise ValueError("stable_checks_required must be positive")


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    load_n: float
    entrainment_speed_m_s: float
    eta0_pa_s: float
    alpha_pa_inv: float
    roelands_reference_pressure_pa: float = 1.96e8
    density_kg_m3: float = 818.2
    material: MaterialConfig = field(default_factory=MaterialConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def validate(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty")
        for name, value in (
            ("load_n", self.load_n),
            ("entrainment_speed_m_s", self.entrainment_speed_m_s),
            ("eta0_pa_s", self.eta0_pa_s),
            ("alpha_pa_inv", self.alpha_pa_inv),
            ("roelands_reference_pressure_pa", self.roelands_reference_pressure_pa),
            ("density_kg_m3", self.density_kg_m3),
        ):
            _positive(name, value)
        self.material.validate()
        self.grid.validate()
        self.solver.validate()
        if not 0.0 < self.roelands_z < 2.0:
            raise ValueError(f"Derived Roelands z is outside (0, 2): {self.roelands_z}")

    @property
    def roelands_z(self) -> float:
        denominator = math.log(self.eta0_pa_s) + 9.67
        if denominator <= 0.0:
            raise ValueError("eta0_pa_s is incompatible with the Roelands relation")
        return self.alpha_pa_inv * self.roelands_reference_pressure_pa / denominator

    @property
    def canonical_condition(self) -> tuple[float, float, float, float]:
        return (
            round(self.load_n, 12),
            round(self.entrainment_speed_m_s, 12),
            round(self.eta0_pa_s, 12),
            round(self.alpha_pa_inv, 18),
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "load_n": self.load_n,
            "entrainment_speed_m_s": self.entrainment_speed_m_s,
            "eta0_pa_s": self.eta0_pa_s,
            "alpha_pa_inv": self.alpha_pa_inv,
            "roelands_reference_pressure_pa": self.roelands_reference_pressure_pa,
            "density_kg_m3": self.density_kg_m3,
            "material": self.material.to_dict(),
            "grid": asdict(self.grid),
            "solver": asdict(self.solver),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CaseConfig":
        payload = dict(raw)
        material = dict(payload.get("material", {}))
        for name in ("rx1_m", "rx2_m", "ry1_m", "ry2_m"):
            if name in material and material[name] is None:
                material[name] = math.inf
        payload["material"] = MaterialConfig(**material)
        payload["grid"] = GridConfig(**payload.get("grid", {}))
        payload["solver"] = SolverConfig(**payload.get("solver", {}))
        case = cls(**payload)
        case.validate()
        return case
