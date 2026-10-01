# Data availability

The repository includes one held-out interpolation case so that reviewers can
run an end-to-end prediction immediately. The complete data archive contains 60
converged FAS reference solutions on the fixed `769 × 769` D2 grid. Each case is
indexed in `data/full_dataset_index.json` with its field checksum and convergence
diagnostics.

**Versioned full-data archive:**
[GitHub Release v1.0.0](https://github.com/vachekivan495-design/PINN-EHL-Reproducibility/releases/tag/v1.0.0)

**Direct download:**
[`PINN_EHL_D2_769_full_dataset.zip`](https://github.com/vachekivan495-design/PINN-EHL-Reproducibility/releases/download/v1.0.0/PINN_EHL_D2_769_full_dataset.zip)

After downloading the archive, run:

```bash
python scripts/verify_dataset.py data/full/reference_d2_769
```

The archive should preserve the case-directory names listed in
`data/full_dataset_index.json`.

Published archive:

- filename: `PINN_EHL_D2_769_full_dataset.zip`;
- size: 300.64 MB;
- members: 123 files for 60 cases;
- SHA-256: `68fb843e115b16726f13c600fdbd64a7d93ad2e0474c3b8038510945276fb53e`.

