# Accuracy evidence behind the paper

This folder contains the numerical evidence behind every table and data figure of the paper. Every
run folder was produced by the current simulator (run dates 2026-09-07 to 2026-09-10);
`python scripts/reproduce_all.py --mode verify-existing` checks that no older run id
appears in a folder name or in a `run_id` column. Each run folder holds `summary.csv`,
`raw_runs.csv`, `report.json`, `config.json`, `resolved_runtime_config.json`,
`physical_model_manifest.json` and `git_revision.txt`; full event/cost traces are not committed
(they are written under `.generated/` when a run uses `--trace-level full`).

Table and figure numbers below are those of the current manuscript and its Supplementary Material;
the figure data files keep the names under which they were first produced (see
`docs/PAPER_CODE_MAP.md`).

## Section VI grid and base load

- `canonical_run_20260907_111056_515561/` — the Section VI full reproduction run (four matchers,
  324 operating points × 20 repeats, base seed 2187631072): TABLE 6 accuracy and latency,
  Figures 3–4, the watermark and candidate-margin sweeps, and the Wilcoxon/Holm comparisons.
  Produced by `python scripts/reproduce_all.py --mode full`.
- `impairment_default_run_20260910_091432_644397/` — the representative cell (EV = 500,
  Δt_EV = 30 s, d_max = 10 s, Δ = 120 s) run alone: it reproduces the canonical accuracies exactly
  and supplies the TABLE 6 runtime column and the TABLE 7 control point.
- `trace_representative_run_20260909_112529_182312/` — summary files of the full-trace replay of
  the representative cell (A1 + A3, seeds 2187632432–2451); the ≈ 1.5 GB of traces stay under
  `.generated/`. Produced by
  `python scripts/reproduce_all.py --mode trace-representative --algorithms single_only,bayesian_windowed --repeats 20`.
  From these traces: `table9_failure_taxonomy.csv` and `failure_cases_representative.csv` (A3,
  one row per misidentified session) and `table9_failure_taxonomy_a1.csv` (A1) — TABLE 8, by
  `python scripts/analyze_failure_taxonomy.py`; `latency_decomposition.csv` — the per-repeat
  decomposition of the 416 s p90 decision latency, by `python scripts/analyze_latency_decomposition.py`
  (`latency_decomposition_representative.csv` is a byte-identical copy);
  `codebook_realised_statistics.csv` and `codebook_candidate_separation.csv` — the realised
  statistics of the command codebook the runs used (random-unique terminal branch of the generator:
  minimum pairwise ℓ1 4–13 A per repeat, mean 8 A; mean pairwise ℓ1 42 A) and the conditional error
  rate against the ℓ1 distance to the nearest other gated candidate, by
  `python scripts/analyze_codebook_separation.py` (the pool statistics were recomputed from this run
  on 2026-09-11 and equal the committed file).
- `figure5_posterior_trace.csv` — the measured posterior trace behind Supplementary Fig. S1, regenerated
  bit-identically by `python scripts/extract_figure5_posterior_trace.py`.
- `saturation_SA_run_20260907_111056_513694/`, `saturation_SB_run_20260907_111056_525003/` — the
  saturation points S-A (2000 requested sessions on 50 slots) and S-B (500 on 15 slots); four
  matchers, 20 repeats. Produced by `python scripts/reproduce_all.py --mode saturation --point SA`
  (and `--point SB`) `--repeats 20`; tabulated with the control point into `table7_saturation.csv`
  (TABLE 7) by `python scripts/build_table7_saturation.py`.
- `static_limit_ideal_sensing_run_20260908_170955_253099/`, `static_limit_paper_sensing_run_20260909_020513_080213/`
  — the static-limit sanity check (300 simultaneous sessions on 300 slots, complete waveforms,
  ingestion disabled, legacy slot-fixed codebook; A1/A2/A3, base seed 2187631072, 20 repeats;
  identical scenario seeds and hashes within a cell). Produced by
  `python scripts/reproduce_all.py --mode static-limit --cell <ideal|paper> --repeats 20`; tabulated
  into `table_s2_static_limit.csv` (Supplementary Table S2) by `python scripts/build_table_s2_static_limit.py`.
  `static_limit_a2_ideal_by_repeat.csv` joins the per-repeat accuracies of the ideal-sensing cell with
  the legacy codebook's near-duplicate pair counts (`python scripts/build_static_limit_codebook_table.py`);
  the prior static cost identifies every pair in all 20 repeats of that cell.
  For these runs (`ingestion_enabled: false`), `resolved_runtime_config.json` → `ingestion_effective` lists the
  nominal ingestion values; the simulator applies zero delay and loss, as `report.json` →
  `online_operational_conditions.ingestion_model.ev` records (checked by `--mode verify-existing`).

## Robustness sweeps (Section VI-B)

- `impairment_<cell>_run_<id>/` — the impairment sweeps at the representative cell (four matchers,
  20 paired repeats, canonical EV = 500 seeds 2187632432–2451): `default`, `loss_0.05/0.15/0.30`,
  `delay_5_2.5_15`, `delay_10_5_30`, `delay_20_10_60`, `delay_20_10_60_dmax60`, `noise_x2/x4/x8`
  (11 cells). Produced by
  `python scripts/reproduce_all.py --mode impairment-sweep --axis <axis> [--cell-index i] --repeats 20`;
  tabulated into `figure7_impairment_sweeps.csv` (Figure 5) by `python scripts/build_figure7_data.py`.
  `impairment_realised_loss.csv` gives the realised share of lost measurements per loss cell
  (0.52 / 5.11 / 15.05 / 30.07 % at 0.005 / 0.05 / 0.15 / 0.30), replayed from the recorded ingestion
  seeds by `python scripts/analyze_realised_loss.py`. The two `impairment_noise_x{4,8}_costterms_run_<id>/`
  folders hold the cost-term variants of Supplementary Table S3.
- `heterogeneous_response_run_20260908_170955_253072/` — the base-load grid (EV 100/300/500, four
  matchers, 20 repeats, canonical base-load seeds) with per-session response dynamics drawn in
  Layer 1 (first-step lag ~ U(2, 30) s, ramp ~ U(1, 10) A/s; `response_*` columns in `raw_runs.csv`
  log the realised draws); tabulated against the canonical run into
  `figure8_heterogeneous_response.csv` (Figure 6) by `python scripts/build_figure8_data.py`.
- `codebook_band_{B20,B16,B12}_run_<id>/` — the command set-point band sweep at the representative
  cell (A1/A2/A3, canonical seeds; `codebook_*` columns log the realised separation), tabulated with
  the canonical 6–30 A cell into `figure9_codebook_separation.csv` (Figure 7) by
  `python scripts/build_figure9_data.py`.
- `hyperparam_<knob>_<value>_run_<id>/` (12 folders) — one-knob-at-a-time sensitivity runs of the
  Posterior matcher (γ_prev, η_mix, β_curr; A3 only; canonical EV = 500 evaluation repeats).
  Produced by `python scripts/reproduce_all.py --mode hyperparam-sweep --cell-index i --repeats 20`;
  tabulated into `table8_hyperparam_sensitivity.csv` (Supplementary Table S4) by `python scripts/build_table8_hyperparam.py`.
- `sampling_diag_<cell>_run_<id>/` (9 folders: `noisefree_5/15/30`, `blockmean_5/15`,
  `termpart_5/15/30`, `offsetfree_5`) — the dense-sampling diagnostic on the canonical Δt_EV cell
  seeds: sample-noise-free control, 30 s block-mean front end, Eq. (7) term partition (Greedy
  DTW-only `single_only_step_off` and step-only `single_only_dtw_off`) and the alignment control
  (EV meter offset fixed at 5 s). Produced by
  `python scripts/reproduce_all.py --mode sampling-diagnostic --cell-index i --repeats 20`; tabulated
  into `table_s1_sampling_diagnostic.csv` (Supplementary Table S1) by `python scripts/build_table_s1_sampling_diagnostic.py`.

## Ablation (Section VI-C)

- `ablation_run_20260909_112655_937821/` — the Section VI-C ablation run (base-load grid,
  20 repeats) on the canonical base-load scenario seeds; its full-design rows equal the canonical run
  exactly. Produced by `python scripts/reproduce_all.py --mode ablation`.
- `ablation_two_prefix_run_20260908_170955_253026/` — the stateless two-prefix variant
  (`two_prefix_greedy`, `two_prefix_hungarian`) on the same seeds (identical `scenario_hash` per
  repeat); TABLE 9 via `python scripts/build_table10_two_prefix.py`. Produced by
  `python scripts/reproduce_all.py --mode ablation --algorithms two_prefix_greedy,two_prefix_hungarian`.
- `figure6_ablation_simulated.csv` — the Figure 8 data table, tabulated from the two ablation runs
  by `python scripts/build_figure6_data.py`.
- `table_s3_cost_term_partition.csv` — Supplementary Table S3 (Greedy full / DTW-only / step-only and Global
  full / no-correlation / no-DTW at sensing multipliers 1, 4, 8), by
  `python scripts/build_table_s3_cost_term_partition.py` from the ablation run and the two
  cost-term impairment runs.
