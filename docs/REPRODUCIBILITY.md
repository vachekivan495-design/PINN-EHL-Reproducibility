# Reproducibility scope

## Reproduced directly from this repository

`scripts/reproduce_sample.py` loads the validation-selected 1% checkpoint,
predicts pressure and film thickness for held-out case `ti_006`, and compares
the prediction with the frozen reference field. This exercises the model,
condition scaling, hard boundary construction, elastic convolution, dimensional
case scaling, and field metrics on all 591,361 grid nodes.

## Reproduced with the full data archive

After the 60-case data archive is placed under `data/full/reference_d2_769`, the
included configuration reproduces the seed-1, 1% sparse-supervision training
contract. The formal results also used seed 2 and seed 3 for the 1% setting and
single runs for 0.1% and 100% supervision. Their evaluation JSON files are
provided under `results/formal_evaluations`.

The full dataset is excluded from Git because it is about 300 MB. It is
published as the versioned GitHub Release asset
`PINN_EHL_D2_769_full_dataset.zip` at
<https://github.com/vachekivan495-design/PINN-EHL-Reproducibility/releases/tag/v1.0.0>.
Its SHA-256 is
`68fb843e115b16726f13c600fdbd64a7d93ad2e0474c3b8038510945276fb53e`.

## Timing interpretation

The manuscript reports approximately 2.00 s per case for PINN inference under
the server timing protocol. The mean recorded reference-generation time was
2417.3388 s per held-out case. Those values describe different computational
tasks and hardware protocols; the repository does not turn them into a strict
same-platform speedup claim. The smoke script records its own device and wall
time for transparency.

## Determinism

The formal CUDA runs did not request deterministic algorithms. The frozen
checkpoint gives deterministic inference on a fixed software stack, subject to
the normal numerical variation of hardware and PyTorch kernels. Retraining may
show small run-to-run variation, which is why the 1% setting was repeated over
three seeds.

