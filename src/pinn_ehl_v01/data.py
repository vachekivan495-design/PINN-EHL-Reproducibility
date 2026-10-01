from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .io import sha256_file


CONDITION_FIELDS = (
    "load_n",
    "entrainment_speed_m_s",
    "eta0_pa_s",
    "alpha_pa_inv",
)


@dataclass(frozen=True)
class CaseArtifact:
    case_id: str
    run_dir: Path
    manifest: Mapping[str, object]
    x: np.ndarray
    y: np.ndarray
    pressure: np.ndarray
    film: np.ndarray
    h0: float
    condition: np.ndarray

    @property
    def canonical_condition(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.condition)


@dataclass(frozen=True)
class CaseMetadata:
    case_id: str
    run_dir: Path
    manifest: Mapping[str, object]
    condition: np.ndarray

    @property
    def canonical_condition(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.condition)


def _physical_condition(manifest: Mapping[str, object]) -> np.ndarray:
    case = manifest.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("Manifest is missing its case configuration")
    missing = [name for name in CONDITION_FIELDS if name not in case]
    if missing:
        raise ValueError(f"Manifest is missing condition fields: {missing}")
    values = np.asarray([float(case[name]) for name in CONDITION_FIELDS], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"Invalid physical condition: {values.tolist()}")
    return values


def load_case_metadata(
    run_dir: str | Path,
    require_strict: bool = True,
    verify_field_checksum: bool = False,
) -> CaseMetadata:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_strict and not bool(manifest.get("strict_converged", False)):
        raise ValueError(f"Case {manifest.get('case_id')} is not a strict FDM solution")

    field_path = root / str(manifest.get("field_file", "fields.npz"))
    if not field_path.is_file():
        raise FileNotFoundError(field_path)
    if verify_field_checksum:
        expected_hash = str(manifest.get("field_sha256", ""))
        actual_hash = sha256_file(field_path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"Field checksum mismatch for {field_path}")
    return CaseMetadata(
        case_id=str(manifest["case_id"]),
        run_dir=root,
        manifest=manifest,
        condition=_physical_condition(manifest),
    )


def load_case_artifact(run_dir: str | Path, require_strict: bool = True) -> CaseArtifact:
    metadata = load_case_metadata(
        run_dir,
        require_strict=require_strict,
        verify_field_checksum=True,
    )
    root = metadata.run_dir
    manifest = metadata.manifest
    field_path = root / str(manifest.get("field_file", "fields.npz"))

    with np.load(field_path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        pressure = np.asarray(payload["P"], dtype=np.float64)
        film = np.asarray(payload["H"], dtype=np.float64)
        h0 = float(payload["h0"])
    if pressure.shape != (x.size, y.size) or film.shape != pressure.shape:
        raise ValueError(f"Malformed field arrays in {field_path}")
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(film)):
        raise ValueError(f"Non-finite field values in {field_path}")
    return CaseArtifact(
        case_id=str(manifest["case_id"]),
        run_dir=root,
        manifest=manifest,
        x=x,
        y=y,
        pressure=pressure,
        film=film,
        h0=h0,
        condition=metadata.condition,
    )


@dataclass(frozen=True)
class ConditionScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_cases: Sequence[CaseArtifact]) -> "ConditionScaler":
        if not training_cases:
            raise ValueError("ConditionScaler requires at least one training case")
        values = np.stack([case.condition for case in training_cases], axis=0)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-14, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def canonical_condition_key(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != len(CONDITION_FIELDS):
        raise ValueError(f"Expected {len(CONDITION_FIELDS)} condition values")
    return (
        round(float(values[0]), 12),
        round(float(values[1]), 12),
        round(float(values[2]), 12),
        round(float(values[3]), 18),
    )


def assert_unique_conditions(cases: Iterable[CaseArtifact | CaseMetadata]) -> None:
    owner: dict[tuple[float, ...], str] = {}
    for case in cases:
        key = canonical_condition_key(case.condition)
        if key in owner:
            raise ValueError(
                f"Duplicate physical condition: {owner[key]!r} and {case.case_id!r}"
            )
        owner[key] = case.case_id


def make_supervision_mask(
    case: CaseArtifact,
    fraction: float,
    seed: int,
    mode: str = "uniform",
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if mode not in {"uniform", "sensor_lines", "sensor_lines_plus_random"}:
        raise ValueError("Unsupported supervision mode")
    total = case.pressure.size
    count = max(1, min(total, int(round(total * fraction))))
    rng = np.random.default_rng(int(seed))
    if mode == "uniform":
        selected = rng.permutation(total)[:count]
    else:
        nx, ny = case.pressure.shape
        center_y = ny // 2
        center_x = nx // 2
        candidates = np.unique(
            np.concatenate(
                [
                    np.ravel_multi_index((np.arange(nx), np.full(nx, center_y)), (nx, ny)),
                    np.ravel_multi_index((np.full(ny, center_x), np.arange(ny)), (nx, ny)),
                ]
            )
        )
        if candidates.size >= count:
            selected = rng.permutation(candidates)[:count]
        else:
            if mode == "sensor_lines":
                raise ValueError(f"sensor_lines contains only {candidates.size}/{total} points; reduce fraction or explicitly use sensor_lines_plus_random")
            remaining = np.setdiff1d(np.arange(total), candidates, assume_unique=False)
            extra = rng.permutation(remaining)[:count - candidates.size]
            selected = np.concatenate([candidates, extra])
    mask = np.zeros(total, dtype=bool)
    mask[selected] = True
    return mask.reshape(case.pressure.shape)
