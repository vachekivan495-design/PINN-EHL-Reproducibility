from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import CaseConfig
from .fdm import FDMSolution


def load_case_config(path: str | Path) -> CaseConfig:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    return CaseConfig.from_mapping(raw)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_solution(solution: FDMSolution, output_root: str | Path) -> Path:
    root = Path(output_root)
    run_dir = root / f"{solution.case.case_id}_{solution.case.fingerprint[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    npz_path = run_dir / "fields.npz"
    np.savez_compressed(
        npz_path,
        x=solution.grid.x,
        y=solution.grid.y,
        X=solution.grid.X,
        Y=solution.grid.Y,
        P=solution.pressure,
        H=solution.film,
        h0=np.float64(solution.h0),
    )
    _write_csv(run_dir / "pressure_history.csv", solution.pressure_history)
    _write_csv(run_dir / "load_history.csv", solution.load_history)
    summary = solution.summary()
    summary["field_file"] = npz_path.name
    summary["field_sha256"] = sha256_file(npz_path)
    _write_json(run_dir / "manifest.json", summary)
    return run_dir

