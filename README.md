# PINN-EHL surrogate reproducibility package

This repository contains the compact reproducibility package for the manuscript
**“Physics-informed neural prediction of coupled pressure and film-thickness
fields in point-contact elastohydrodynamic lubrication.”**

The released model maps four operating parameters—load, entrainment speed,
ambient viscosity, and pressure–viscosity coefficient—to coupled pressure and
film-thickness fields on a fixed `769 × 769` grid. Film thickness is recovered
through the elastic closure rather than predicted as an unconstrained second
field.

## What is included

- the exact formal-training implementation identified by hashes stored with the
  production run;
- the frozen 36/8/8/8 condition split;
- the 1% sparse-supervision configuration;
- a validation-selected inference checkpoint from seed 1;
- one held-out interpolation case (`ti_006`) for an end-to-end smoke
  reproduction;
- machine-readable formal evaluation results for 0.1%, 1%, and 100%
  supervision;
- a 60-case data index with SHA-256 checksums.

Legacy M27, Speed/Load/Viscosity, 5% pilot, and superseded solver experiments
are not included.

## Quick reproduction

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/reproduce_sample.py
```

The command writes `outputs/sample_ti_006_metrics.json` and checks the pressure,
film thickness, hard pressure boundary, transverse symmetry, and checkpoint
provenance. Runtime from this smoke run should not be compared with the
manuscript timing unless the hardware and timing protocol are the same.

Run the integrity and unit checks with:

```bash
python scripts/verify_release.py
python -m pytest
```

## Full training reproduction

The complete D2 dataset contains 60 converged `769 × 769` reference cases and is
about 300 MB. It is kept outside Git so that the repository remains small. Place
the extracted dataset at:

```text
data/full/reference_d2_769/<case-directory>/manifest.json
data/full/reference_d2_769/<case-directory>/fields.npz
```

Verify it before training:

```bash
python scripts/verify_dataset.py data/full/reference_d2_769
python scripts/train_pinn.py --config configs/primary_1pct_seed1.json
```

Before making this repository public, add the permanent Zenodo or institutional
archive URL for the full dataset to `docs/DATA_AVAILABILITY.md`.

## Experimental contract

- grid: `769 × 769`;
- domain: `x ∈ [-4, 2]`, `y ∈ [-3, 3]`;
- split: 36 training, 8 validation, 8 interpolation test, 8 one-parameter
  extrapolation test cases;
- main architecture: 64 Gaussian Fourier features, six FiLM residual blocks,
  width 128;
- training: AdamW, learning rate `2e-4`, weight decay `1e-6`, 500 epochs and 32
  updates per epoch;
- loss weights: pressure data `1`, film data `1`, Reynolds obstacle `1e-3`,
  load balance `1e-2`, and film-floor penalty `1e-2`;
- checkpoint selection uses the condition-level validation split only.

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md) for the exact audit trail.

## License

Code is released under the MIT License. The included sample data and result JSON
files are released under CC BY 4.0; see `DATA_LICENSE`.

