# Source provenance and version selection

The production run stored SHA-256 hashes for every source module. The local
source selected for this release was matched against those recorded hashes.

Key matches:

| Module | SHA-256 |
|---|---|
| `training.py` | `6d0acfee90ae36c694b58f8a46aa5826bf9a58757eb839abf8a34999f3dcebec` |
| `model.py` | `3899db2bf6c08038a50faee7d36317374a2286fe1da103d4d4f7902f971f15e5` |
| `data.py` | `e5d1a2e57e2a7147fb540ffc4f215a248bd56d4b4cfe6290944d27d0ee42796b` |
| `torch_physics.py` | `c024a001096913393c47c38c66462d981b93abb232e823f59b360cc82e4fd816` |
| `inference.py` | `deb44c3e1e877fc2b9d171588d67fd766eb97ff1f29105cda26244246a5a9a00` |
| `metrics.py` | `57ad0661b2c0f83cd807c3b2db2d59052b4d7c8c788b6846618836777ba7dc6a` |

The selected source is
`pinn_ehl_formal_v04_clean/src/pinn_ehl_v01`. Its `training.py` hash matches the
formal 1% production run exactly. Files under
`pinn_ehl_legacy_v03_20260917` have different hashes and were excluded.

The formal split manifest has SHA-256
`1a483e23e1fe9f7ee235af8c2086622cab664a68a8833b7db99497bfb9e358e4`.
The released inference checkpoint records the original checkpoint hash and the
same training-code and split hashes in `release_provenance`.

No source from the superseded M27 or Speed/Load/Viscosity experiments is part of
this repository.

