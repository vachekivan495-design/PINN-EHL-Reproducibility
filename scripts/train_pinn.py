from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinn_ehl_v01.training import TrainingConfig, train


def resolve_repository_paths(raw: dict) -> dict:
    resolved = dict(raw)
    resolved["case_dirs"] = [str((ROOT / path).resolve()) for path in raw["case_dirs"]]
    for key in (
        "output_root",
        "split_manifest",
        "reference_validation_report",
        "residual_calibration_report",
    ):
        if resolved.get(key):
            resolved[key] = str((ROOT / resolved[key]).resolve())
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the formal PINN-EHL surrogate")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = train(TrainingConfig.from_mapping(resolve_repository_paths(raw)))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()

