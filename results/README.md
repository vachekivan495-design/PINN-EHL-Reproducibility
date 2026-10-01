# Formal result records

The JSON files in `formal_evaluations` are the frozen evaluations used for the
manuscript's sparse-supervision comparison:

- 0.1% supervision: one seed;
- 1% supervision: three seeds;
- 100% supervision: one seed;
- two disjoint test sets: interpolation and one-parameter extrapolation.

Every evaluation records its checkpoint SHA-256 and states that model selection
used validation data only. Machine-local checkpoint paths were replaced during
release preparation; numerical content was not changed.

`pinn_1pct_seed_aggregate.json` contains the three-seed aggregate. The three
training summaries preserve elapsed training time, parameter count, software
versions, epoch count, and optimizer-update count.

