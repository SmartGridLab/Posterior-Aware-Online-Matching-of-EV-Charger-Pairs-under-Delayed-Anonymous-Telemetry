# Canonical Manuscript Accuracy Data

This folder contains the numerical evidence used by the submitted manuscript.

- `canonical_run_20260626_170431_105970/` is the Section VI full reproduction
  run (four main matchers, 324 operating points × 20 repeats) used for the
  manuscript tables and the Figure 3–4 data.
- `ablation_run_20260711_164832_170489/` is the Section VI-C ablation
  reproduction run (base-load grid, 20 repeats) executed on the canonical
  base-load scenario seeds; its full-design rows equal the canonical run
  exactly. Produced by `python scripts/reproduce_all.py --mode ablation`.
- `figure6_ablation_simulated.csv` is the Figure 6 data table tabulated from the
  ablation run by `python scripts/build_figure6_data.py`.
- `figure5_posterior_trace.csv` is the measured posterior trace behind Figure 5,
  regenerated bit-identically by `python scripts/extract_figure5_posterior_trace.py`.
