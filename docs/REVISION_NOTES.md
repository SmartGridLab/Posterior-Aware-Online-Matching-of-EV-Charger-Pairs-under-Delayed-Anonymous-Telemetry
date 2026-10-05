# Revision notes

This file records what changed in the simulator, the reproduction harness and the committed evidence
since the first public release of this repository (commit `c8fb959f923339b06d8cc2510de13fa4a1b48d93`,
July 2026). The current version of the repository accompanies the paper "Watermark-Based Online
EV–Charger Matching from Delayed Slot-Unlabeled Telemetry".

New experiments are new `--mode` entries of `scripts/reproduce_all.py`; runs are written to
`.generated/algorithm_compare/<run_id>/` and committed under
`case_study/scalability_analysis/accuracy/<kind>_run_<run_id>/` when they become manuscript
evidence. Figure and table numbers below are those of the current version of the paper
(`docs/PAPER_CODE_MAP.md` maps them to the figure files).

## Corrections to the simulator

Two defects in the generation of the EV-side telemetry were corrected, and every result of the current
version of the paper was regenerated with the corrected simulator (Section V-A, Layer 3):

1. **EV-side sampling.** The sampler produced the last 5–25 s of each observation window from its own
   rounded output instead of from the slot current, re-applying gain, bias and the 1 A rounding at every
   step. Every EV-side sample now reads the physical slot current once, and gain, bias, noise and the
   1 A ceiling are applied once per sample (`pair_identification/matching_core.py`,
   `build_dataset_with_random_arrivals`; `tests/test_sensor_window_tail.py`).
2. **Ingestion.** Delay and loss were drawn once per 5 s grid copy of a measurement; they are now drawn
   once per EV-side measurement, and every copy inherits the draw
   (`pair_identification/runtime_event_loop.py::_simulate_ingest_times`;
   `tests/test_per_measurement_ingestion.py`).

The scenario seeds, arrivals, command patterns, candidate gating, decision timing and the timing-only
baseline are identical before and after the corrections; no matcher, hyperparameter or evaluation
protocol was changed. In addition, the command-codebook generator breaks ties in its candidate
ordering with a stable sort, so a scenario is identical on every platform (the default sort reordered
equal distances differently on x86-64 builds of NumPy that dispatch sorting to SIMD kernels; the
committed codebooks are the stable-sort result).

## Changes to the reproduction harness

Every new keyword argument defaults to the configuration of the paper, so the harness changes leave the
canonical results bit-for-bit reproducible (checked by replaying canonical cells against the committed
canonical run; `tests/test_replay_canonical_cell.py`).

| Change | Where |
|---|---|
| `evse_count` kwarg (was hard-wired to 50) | `case_study.section_vi_reproduction.run_algorithm_compare_report` → `ExperimentConfig`, `config.json`, `raw_runs.csv`, `report.json` |
| `simultaneous_arrivals` kwarg (threads `arrival_schedule_sec=[0]*n_evs`) | same → `pair_identification.matching_core.build_dataset_with_random_arrivals` |
| Sensor-model kwargs `ev_sensor_extra_delay_max_s`, `ev_sensor_gain_std`, `ev_sensor_bias_std_a`, `ev_sensor_noise_std_a`, `ev_start_est_jitter_s` (`None` = module default) | same → `_override_sensor_model` context manager, `resolved_runtime_config.json["sensor_model"]`, `physical_model_manifest.json` |
| Mode registry in the reproduction driver | `scripts/reproduce_all.py` |
| A3 hyperparameter kwargs `a3_posterior_prev_power`, `a3_time_prior_mix`, `a3_current_like_beta` (`None` = TABLE 2) | same → `_override_a3_hyperparams` |
| Response-model kwargs `first_step_lag_s`, `step_lag_s`, `ramp_rate_a_per_s` of `generate_charger_current` (defaults = fitted model bit-for-bit); per-session draws `RESPONSE_LAG_RANGE_S` / `RESPONSE_RAMP_RANGE_A_PER_S` in `build_dataset_with_random_arrivals` (dedicated RNG stream; scenario realisation unchanged), stored in `ev_meta` and the slot sessions | harness kwargs `response_lag_range_s`, `response_ramp_range_a_per_s` → `_override_response_model`, `physical_model_manifest.json` |
| Codebook band `command_setpoint_range` (None = 6–30 A) → `COMMAND_SETPOINT_RANGE` / `generate_command_patterns(setpoint_range=)` (only the `allowed` set-points change; default bit-identical); `codebook_statistics` → `dataset_diag["codebook"]` → `raw_runs.csv` `codebook_*` columns (unique patterns, pairwise ℓ1 min/mean, internal gap min, consecutive-step min, terminal_branch, ladder_rungs, band) in every run; log line when the ladder is exhausted | same → `_override_codebook_setpoint_range` |
| EV-grid front end `ev_grid_aggregation` {hold, block_mean} (`EV_GRID_AGGREGATION`, `EV_GRID_BLOCK_S` in `matching_core`; `RuntimeConfig`; block-wise ingestion availability in the event loop) | same → `_override_ev_grid_aggregation` |

## Evidence added

| Addition | Driver mode and scripts | Committed evidence | Manuscript |
|---|---|---|---|
| Impairment sweeps (measurement loss, ingestion delay, sensing distortion) at the representative cell, all four matchers | `--mode impairment-sweep --axis {default,loss,delay,noise} [--cell-index i]`; `scripts/build_figure7_data.py`; `scripts/analyze_realised_loss.py` | `impairment_<cell>_run_<id>/` (11 cells), `figure7_impairment_sweeps.csv`, `impairment_realised_loss.csv`, `paper/figure7_impairment_sweeps.pdf` | Section VI-B, Figure 5; Section V-A (realised loss shares) |
| Per-session response heterogeneity (first-step lag ~ U(2, 30) s, ramp ~ U(1, 10) A/s) | `--mode heterogeneous-response`; `scripts/build_figure8_data.py` | `heterogeneous_response_run_20260908_170955_253072/`, `figure8_heterogeneous_response.csv`, `paper/figure8_heterogeneous_response.pdf` | Section VI-B, Figure 6 |
| Command set-point band sweep (codebook separation) and realised codebook statistics | `--mode codebook-band-sweep --cell-index {1,2,3}`; `scripts/build_figure9_data.py`; `scripts/analyze_codebook_separation.py` | `codebook_band_{B20,B16,B12}_run_<id>/`, `figure9_codebook_separation.csv`, `codebook_realised_statistics.csv`, `codebook_candidate_separation.csv`, `paper/figure9_codebook_separation.pdf` | Section V-A, Section VI-B, Figure 7 |
| Saturation operating points S-A (2000 requested sessions on 50 slots) and S-B (500 on 15 slots) | `--mode saturation --point {SA,SB}`; `scripts/build_table7_saturation.py` | `saturation_SA_run_20260907_111056_513694/`, `saturation_SB_run_20260907_111056_525003/`, control `impairment_default_run_20260910_091432_644397/`, `table7_saturation.csv` | Section VI-A, TABLE 7 |
| Hyperparameter sensitivity of the Posterior matcher (one knob at a time; values not re-selected) | `--mode hyperparam-sweep [--cell-index i]`; `scripts/build_table8_hyperparam.py` | `hyperparam_<knob>_<value>_run_<id>/` (12), `table8_hyperparam_sensitivity.csv` | Section VI-B, Supplementary Table S4 |
| Residual-error taxonomy from full decision traces and the p90 latency decomposition | `--mode trace-representative --algorithms single_only,bayesian_windowed`; `scripts/analyze_failure_taxonomy.py`; `scripts/analyze_latency_decomposition.py` | `trace_representative_run_20260909_112529_182312/`, `table9_failure_taxonomy.csv`, `table9_failure_taxonomy_a1.csv`, `failure_cases_representative.csv`, `latency_decomposition.csv` | Section VI-A, Section VI-C, TABLE 8 |
| Stateless two-prefix variant of the Posterior matcher (greedy and Hungarian) | `--mode ablation --algorithms two_prefix_greedy,two_prefix_hungarian`; `scripts/build_table10_two_prefix.py`; `scripts/build_figure6_data.py` | `ablation_two_prefix_run_20260908_170955_253026/`, `table10_two_prefix_ablation.csv`, `figure6_ablation_simulated.csv` | Section VI-C, TABLE 9, Figure 8 |
| Dense-sampling diagnostic (noise-free control, block-mean front end, Eq. (7) term partition, alignment control) | `--mode sampling-diagnostic [--cell-index i]`; `scripts/build_table_s1_sampling_diagnostic.py` | `sampling_diag_<cell>_run_<id>/` (9), `table_s1_sampling_diagnostic.csv` | Section VI-B, Supplementary Table S1 |
| Static-limit sanity check (300 simultaneous sessions on 300 slots, complete waveforms, ingestion off) | `--mode static-limit --cell {ideal,paper}`; `scripts/build_table_s2_static_limit.py`; `scripts/build_static_limit_codebook_table.py` | `static_limit_{ideal,paper}_sensing_run_<id>/`, `table_s2_static_limit.csv`, `static_limit_a2_ideal_by_repeat.csv` | Section VI-A, Supplementary Table S2 |
| Cost-term partition under sensing distortion | `--mode ablation`; `--mode impairment-sweep --axis noise --cell-index 1\|2 --algorithms <cost-term variants>`; `scripts/build_table_s3_cost_term_partition.py` | `ablation_run_20260909_112655_937821/`, `impairment_noise_x{4,8}_costterms_run_<id>/`, `table_s3_cost_term_partition.csv` | Section VI-C, Supplementary Table S3 |

All run folders are under `case_study/scalability_analysis/accuracy/`; `python scripts/reproduce_all.py --mode verify-existing`
checks the values the manuscript prints and that every committed run id dates from 2026-09-07 or later.

## Supplementary Material and claim checks

The detailed diagnostics are in the Supplementary Material (`paper/supplementary_material.pdf`, source
`paper/supplementary_material.tex`): Supplementary Tables S1–S4, Supplementary Fig. S1 and Supplementary
Section S6. Data files, scripts and claim-check labels keep the names under which they were first produced
(`docs/PAPER_CODE_MAP.md`, identifier map).

- Hyperparameters: the TABLE 2 values were selected on the calibration slice (EV = 300, repeats 1–5) before the
  simulator corrections above and were not re-selected afterwards.
- New claim checks in `verify-existing`: the Wilcoxon–Holm comparisons of Section V-B, recomputed from the canonical
  `raw_runs.csv` (`wilcoxon_holm_*`), and the near-tie statistics of Section VI-C and Supplementary Section S5, from
  `failure_cases_representative.csv` (`near_tie_*`).

### Simulator details stated in the manuscript and the Supplementary Material

- Start estimate (`pair_identification/matching_core.py`, `build_dataset_with_random_arrivals`): the first 1 s EV
  reading at or above 1 A plus a start-estimate jitter drawn from U{−20, …, 20} s. Readings are rounded up to whole
  amperes, so a sensor that reads a small positive value at 0 A already reports 1 A during the 0 A first step. The
  estimate is kept within [t_arr − Δt_EV, t_arr + max(Δt_EV, W/2)]; with the Section V-A defaults this bound is not
  reached at Δt_EV = 30 s, and it can bind at 5 and 15 s sampling and in the strongest sensing cells.
- Global matcher DTW (`case_study/section_vi_reproduction.py`, `_nomura_calculate_dtw_distance` and
  `_online_assign_nomura_original_hungarian`): unbanded, absolute-difference local cost on the raw currents,
  normalized by the alignment-path length and min–max scaled across the finite candidate pairs of the batch.
- Step signature at the 60 % stage (`pair_identification/matching_core.py`, `step_medians`): steps that lie beyond
  the prefix take the median of the whole prefix, on both planes.
- Pattern assignment (`_pick_pattern_maximin`, `_proxy_dist`): each arriving session receives the free pattern with
  the largest minimum distance to the active patterns, the distance being the Euclidean distance of the z-normalized
  six-step patterns plus 0.05, 0.02 and 0.01 times the level, step-change and mean-level differences.
- Sensing multiplier of the impairment sweep (`scripts/reproduce_all.py`, `_representative_point_kwargs`): scales
  the noise, gain and bias standard deviations; the gain stays clipped to [0.8, 1.2].
- Pattern re-issue (`charging_current_generation/command_charger_current.py`): each re-issued pattern starts from
  0 A without the 4 A/s fall ramp; the EV 1 s series ends at t_arr + W, so a decision-grid tail past t_arr + W holds
  the reading at t_arr + W.
- Static limit with the Section V-A sensing layer: the spread of the start estimates puts the 300 decisions on
  several watermark ticks, so each tick's one-to-one assignment covers only the sessions decided at it; the
  ideal-sensing cell decides all 300 sessions at one tick.
