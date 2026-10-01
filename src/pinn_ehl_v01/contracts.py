"""Content-based training identity and crash-safe small artifact writes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import torch

from .io import sha256_file

CHECKPOINT_SCHEMA = "pinn_checkpoint_v0.3"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def atomic_json(path: Path, payload) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"))


def atomic_write(path: Path, write, retain_previous: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    try:
        write(temp)
        with temp.open("r+b") as stream:
            os.fsync(stream.fileno())
        if retain_previous and path.is_file():
            previous = path.with_name(path.stem + ".previous" + path.suffix)
            atomic_write(previous, lambda p: shutil.copyfile(path, p))
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def data_contract(metadata, split_path) -> dict:
    """Hashes raw files without unpacking held-out fields or using their values."""
    case = metadata[0].manifest["case"]
    fixed_context_fields = (
        "density_kg_m3",
        "material",
        "roelands_reference_pressure_pa",
        "grid",
        "solver",
    )
    return {
        "protocol": CHECKPOINT_SCHEMA,
        "physical_context": {name: case[name] for name in fixed_context_fields},
        "split_sha256": sha256_file(split_path),
        "cases": {
            item.case_id: {
                "manifest_sha256": sha256_file(item.run_dir / "manifest.json"),
                "field_sha256": sha256_file(item.run_dir / item.manifest.get("field_file", "fields.npz")),
            }
            for item in metadata
        },
        "source_sha256": {
            p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def require_same_contract(saved, current) -> None:
    if saved != current:
        raise ValueError("Saved data/source contract mismatch: split, reference fields, manifests or source changed; regenerate the artifact")


def runtime_environment() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
    }
