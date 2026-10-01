from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_ehl_v01.inference import Predictor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - reference) / np.linalg.norm(reference))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the held-out ti_006 prediction")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "primary_1pct_seed1_inference.pt",
    )
    parser.add_argument("--case", type=Path, default=ROOT / "data" / "sample_ti_006")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sample_ti_006_metrics.json")
    args = parser.parse_args()

    manifest = json.loads((args.case / "manifest.json").read_text(encoding="utf-8"))
    fields_path = args.case / manifest["field_file"]
    if sha256(fields_path) != manifest["field_sha256"]:
        raise RuntimeError("Sample field checksum does not match manifest")
    fields = np.load(fields_path)
    case = manifest["case"]
    conditions = [[
        case["load_n"],
        case["entrainment_speed_m_s"],
        case["eta0_pa_s"],
        case["alpha_pa_inv"],
    ]]

    predictor = Predictor(args.checkpoint, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = predictor(conditions)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    pressure = prediction["P"][0]
    film = prediction["H"][0]
    p_true = fields["P"]
    h_true = fields["H"]
    metrics = {
        "schema_version": "sample_reproduction_v1",
        "case_id": manifest["case_id"],
        "device": args.device,
        "elapsed_seconds_single_run": elapsed,
        "timing_note": "Do not compare this smoke timing across different hardware or protocols.",
        "pressure_relative_l2": relative_l2(pressure, p_true),
        "film_relative_l2": relative_l2(film, h_true),
        "peak_pressure_relative_error": abs(float(pressure.max()) - float(p_true.max())) / float(p_true.max()),
        "minimum_film_relative_error": abs(float(film.min()) - float(h_true.min())) / float(h_true.min()),
        "h0_absolute_error": abs(float(prediction["h0"][0, 0]) - float(fields["h0"])),
        "maximum_boundary_pressure": float(max(
            pressure[0, :].max(), pressure[-1, :].max(), pressure[:, 0].max(), pressure[:, -1].max()
        )),
        "maximum_transverse_symmetry_error": float(np.max(np.abs(pressure - pressure[:, ::-1]))),
        "checkpoint_release_provenance": predictor.checkpoint.get("release_provenance", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

