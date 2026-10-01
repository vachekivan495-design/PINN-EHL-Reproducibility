import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "training.py": "6d0acfee90ae36c694b58f8a46aa5826bf9a58757eb839abf8a34999f3dcebec",
    "model.py": "3899db2bf6c08038a50faee7d36317374a2286fe1da103d4d4f7902f971f15e5",
    "data.py": "e5d1a2e57e2a7147fb540ffc4f215a248bd56d4b4cfe6290944d27d0ee42796b",
    "torch_physics.py": "c024a001096913393c47c38c66462d981b93abb232e823f59b360cc82e4fd816",
    "inference.py": "deb44c3e1e877fc2b9d171588d67fd766eb97ff1f29105cda26244246a5a9a00",
    "metrics.py": "57ad0661b2c0f83cd807c3b2db2d59052b4d7c8c788b6846618836777ba7dc6a",
}


def test_formal_source_hashes() -> None:
    source = ROOT / "src" / "pinn_ehl_v01"
    for filename, expected in EXPECTED.items():
        actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        assert actual == expected, filename


def test_frozen_split_hash() -> None:
    split = ROOT / "configs" / "split_manifest.json"
    actual = hashlib.sha256(split.read_bytes()).hexdigest()
    assert actual == "1a483e23e1fe9f7ee235af8c2086622cab664a68a8833b7db99497bfb9e358e4"

