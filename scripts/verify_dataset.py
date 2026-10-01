from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all 60 formal D2 reference cases")
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    index = json.loads((ROOT / "data" / "full_dataset_index.json").read_text(encoding="utf-8"))
    errors = []
    for item in index["cases"]:
        directory = args.data_root / item["directory"]
        manifest_path = directory / "manifest.json"
        fields_path = directory / "fields.npz"
        if not manifest_path.is_file() or not fields_path.is_file():
            errors.append(f"{item['case_id']}: missing manifest.json or fields.npz")
            continue
        if sha256(fields_path) != item["field_sha256"]:
            errors.append(f"{item['case_id']}: fields.npz checksum mismatch")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(index['cases'])} cases under {args.data_root}")


if __name__ == "__main__":
    main()

