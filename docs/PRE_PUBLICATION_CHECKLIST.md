# Pre-publication checklist

- [x] Formal source selected by production-run SHA-256 hashes.
- [x] Legacy M27 and earlier Speed/Load/Viscosity code excluded.
- [x] Checkpoint stripped of optimizer, RNG states, and machine-local paths.
- [x] Result JSON files stripped of local checkpoint paths.
- [x] One held-out case and its checksum included.
- [x] Frozen condition split and formal training configuration included.
- [x] Code and data licenses included.
- [x] Full 60-case data archive prepared and integrity-tested.
- [x] Upload the 60-case data archive as a versioned GitHub Release asset.
- [x] Insert the versioned dataset URL in `docs/DATA_AVAILABILITY.md`.
- [x] `CITATION.cff` follows the current manuscript author list and order.
- [x] Confirm the repository owner and final public repository name.

