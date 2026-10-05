#!/usr/bin/env python3
"""Verify or reproduce the IEEE Access Section VI result grid.

Default mode is intentionally lightweight and verifies that the published
aggregate CSV supports the headline paper claims. Use ``--mode full`` for the
complete 6480-run sweep used by the manuscript, and ``--mode ablation`` to
re-simulate the Section VI-C ablation variants (manuscript Figure 8) on the base-load grid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from scipy import stats
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_study.section_vi_reproduction import (  # noqa: E402
    PAPER_ABLATION_ALGORITHM_IDS,
    run_algorithm_compare_report,
)
from pair_identification import matching_core as _algo  # noqa: E402
from manuscript_figure_generation import (  # noqa: E402
    FIGURE7_CSV,
    PAPER_FIGURE_FILES,
    generate_figure3_base_load_comparison,
    generate_figure4_sampling_sensitivity,
    generate_figure5_posterior_diagnostic,
    generate_figure6_ablation_summary,
    generate_figure7_impairment_sweeps,
)

PAPER_RUN_ID = "20260907_111056_515561"
PAPER_ABLATION_RUN_ID = "20260909_112655_937821"
PAPER_ACCURACY_DIR = (
    REPO_ROOT
    / "case_study"
    / "scalability_analysis"
    / "accuracy"
)
PAPER_RESULTS_DIR = PAPER_ACCURACY_DIR / f"canonical_run_{PAPER_RUN_ID}"
PAPER_ABLATION_RESULTS_DIR = PAPER_ACCURACY_DIR / f"ablation_run_{PAPER_ABLATION_RUN_ID}"
GENERATED_RUNS_DIR = REPO_ROOT / ".generated" / "algorithm_compare"
PAPER_BASE_SEED = 2187631072
# Scenario counter of the first (Δt_EV = 30 s, d_max = 10 s, Δ = 120 s) base-load cell
# inside the canonical 324-point grid. Seeding a single-cell run with
# PAPER_BASE_SEED + 1320 replays exactly the same EV ∈ {100, 300, 500} scenario
# realizations (seeds 2187632392..2187632451) as the canonical full sweep, without
# re-running the whole grid.
PAPER_ABLATION_BASE_SEED = PAPER_BASE_SEED + 1320
# Scenario counter of the representative (EV = 500, Δt_EV = 30 s, d_max = 10 s, Δ = 120 s)
# cell: the 40 base-load scenarios at EV ∈ {100, 300} precede it, so a single-cell run with
# ev_counts=[500] seeded here replays the canonical EV = 500 realizations
# (seeds 2187632432..2187632451 for repeats 1..20).
PAPER_REPRESENTATIVE_BASE_SEED = PAPER_ABLATION_BASE_SEED + 40
# TABLE 3 cloud-ingestion defaults shared by every simulation mode.
PAPER_INGESTION_DEFAULTS: dict[str, Any] = {
    "ingestion_enabled": True,
    "ev_ingest_delay_mean_s": 2.0,
    "ev_ingest_delay_jitter_s": 1.0,
    "ev_ingest_delay_max_s": 10.0,
    "ev_ingest_loss_prob": 0.005,
}
PAPER_ALGORITHMS = [
    "time_only_baseline",
    "single_only",
    "nomura_original_interval_hungarian",
    "bayesian_windowed",
]


_COST_TERM_RUNS = {
    1.0: ["ablation_run_20260909_112655_937821"],
    4.0: ["impairment_noise_x4_costterms_run_20260909_112756_275599", "impairment_noise_x4_run_20260908_194258_023727"],
    8.0: ["impairment_noise_x8_costterms_run_20260909_112756_282217", "impairment_noise_x8_run_20260908_214145_238517"],
}


def _cost_term_counts(multiplier: float, algorithm_id: str) -> tuple[float | None, float | None]:
    """Correct and total session counts of one cost-term cell, for the printed-digit rule."""
    for name in _COST_TERM_RUNS.get(float(multiplier), []):
        path = PAPER_ACCURACY_DIR / name / "raw_runs.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        sel = raw[(raw["algorithm_id"] == algorithm_id) & (raw["ev_count"] == 500)]
        if len(sel):
            return float(sel["correct"].sum()), float(sel["total_evs"].sum())
    return None, None


def printed_accuracy(correct: float | None, total: float | None, value: float, places: int = 3) -> float:
    """The digits the manuscript prints: the exact mean rounded half-up.

    Accuracy is a ratio of counts, so the printed value is Sigma correct / N rounded half-up. Taking it
    from the float mean instead can round the wrong way when the exact value sits on the midpoint --
    8685/10000 is stored as 0.8684999999999998 and would print 0.868 rather than 0.869.
    """
    if correct is not None and total not in (None, 0):
        exact = Decimal(int(round(float(correct)))) / Decimal(int(round(float(total))))
    else:
        exact = Decimal(repr(round(float(value), 4)))
    return float(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _summary_path(run_id: str = PAPER_RUN_ID) -> Path:
    if run_id == PAPER_RUN_ID:
        return PAPER_RESULTS_DIR / "summary.csv"
    return GENERATED_RUNS_DIR / run_id / "summary.csv"


def _raw_path(run_id: str = PAPER_RUN_ID) -> Path:
    if run_id == PAPER_RUN_ID:
        return PAPER_RESULTS_DIR / "raw_runs.csv"
    return GENERATED_RUNS_DIR / run_id / "raw_runs.csv"


def _load_summary(run_id: str = PAPER_RUN_ID) -> pd.DataFrame:
    path = _summary_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    return pd.read_csv(path)


def _row(summary: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=summary.index)
    for key, value in filters.items():
        if key not in summary.columns:
            raise KeyError(f"summary.csv is missing required column: {key}")
        mask &= summary[key] == value
    rows = summary[mask]
    if len(rows) != 1:
        raise AssertionError(f"Expected one row for {filters}, found {len(rows)}")
    return rows.iloc[0]


def verify_existing() -> int:
    summary = _load_summary()
    raw_path = _raw_path()
    raw_rows = None
    if raw_path.exists():
        raw_rows = int(sum(1 for _ in raw_path.open("r", encoding="utf-8")) - 1)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: float, expected: float, tol: float) -> None:
        ok = math.isfinite(float(value)) and abs(float(value) - float(expected)) <= float(tol)
        checks.append(
            {
                "name": name,
                "value": float(value),
                "expected": float(expected),
                "tolerance": float(tol),
                "ok": bool(ok),
            }
        )

    default = {
        "ev_count": 500,
        "ev_sample_s": 30,
        "matcher_delay_max_s": 10.0,
        "candidate_margin_s": 120,
    }
    expected_default = {
        "time_only_baseline": 0.7121,
        "single_only": 0.9732,
        "nomura_original_interval_hungarian": 0.9993,
        "bayesian_windowed": 0.9838,
    }
    for algorithm_id, expected in expected_default.items():
        row = _row(summary, algorithm_id=algorithm_id, **default)
        add_check(f"default_500_accuracy_{algorithm_id}", row["accuracy_mean"], expected, 1e-6)
        add_check(f"default_500_p90_latency_{algorithm_id}", row["p90_latency_mean"], 416.035, 1e-6)

    for ev_count, expected in [(100, 0.9975), (300, 0.990667), (500, 0.9838)]:
        row = _row(
            summary,
            algorithm_id="bayesian_windowed",
            ev_count=ev_count,
            ev_sample_s=30,
            matcher_delay_max_s=10.0,
            candidate_margin_s=120,
        )
        add_check(f"a3_base_load_accuracy_ev_{ev_count}", row["accuracy_mean"], expected, 5e-6)

    # EV sampling period decides which cost wins: at 5 s the staged posterior leads the prior static
    # cost at EV 500 in every (d_max, margin) cell, and at 15 s and 30 s the prior cost leads.
    raw_grid = pd.read_csv(_raw_path())
    def _paired_a3_minus_a2(ev_count: int, ev_sample_s: int):
        held = {100: 1, 300: 6, 500: 1}[ev_count]
        sub = raw_grid[(raw_grid["ev_count"] == ev_count) & (raw_grid["ev_sample_s"] == ev_sample_s)
                       & (raw_grid["repeat_idx"] >= held)]
        out = []
        for (_dmax, _mg), grp in sub.groupby(["matcher_delay_max_s", "candidate_margin_s"]):
            a3 = grp[grp["algorithm_id"] == "bayesian_windowed"].sort_values("repeat_idx")["accuracy"].to_numpy(dtype=float)
            a2 = grp[grp["algorithm_id"] == "nomura_original_interval_hungarian"].sort_values("repeat_idx")["accuracy"].to_numpy(dtype=float)
            d = a3 - a2
            half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
            out.append((float(d.mean()), float(d.mean() - half), float(d.mean() + half)))
        return out

    cells_500_5 = _paired_a3_minus_a2(500, 5)
    add_check("sampling_period_5s_a3_leads_a2_ev500", float(sum(1 for _m, lo, _hi in cells_500_5 if lo > 0)), 9, 0)
    add_check("sampling_period_5s_a3_lead_min_mean", float(min(m for m, _lo, _hi in cells_500_5)), 0.0085, 1e-4)
    add_check("sampling_period_5s_a3_lead_max_mean", float(max(m for m, _lo, _hi in cells_500_5)), 0.0206, 1e-4)
    add_check("sampling_period_5s_ev300_no_significant_cell",
              float(sum(1 for _m, lo, hi in _paired_a3_minus_a2(300, 5) if lo > 0 or hi < 0)), 0, 0)
    coarse = [(ev, dt) for ev in (100, 300, 500) for dt in (15, 30)]
    add_check("sampling_period_15s_30s_a2_leads_mean_all_cells",
              float(sum(1 for ev, dt in coarse for m, _lo, _hi in _paired_a3_minus_a2(ev, dt) if m < 0)), 54, 0)
    add_check("sampling_period_15s_30s_a2_leads_ci_ev300_ev500",
              float(sum(1 for ev in (300, 500) for dt in (15, 30) for _m, _lo, hi in _paired_a3_minus_a2(ev, dt) if hi < 0)), 36, 0)
    add_check("sampling_period_15s_30s_a2_leads_ci_ev100",
              float(sum(1 for dt in (15, 30) for _m, _lo, hi in _paired_a3_minus_a2(100, dt) if hi < 0)), 13, 0)

    for ev_sample_s, expected in [(5, 0.9495), (15, 0.9683), (30, 0.9838)]:
        row = _row(
            summary,
            algorithm_id="bayesian_windowed",
            ev_count=500,
            ev_sample_s=ev_sample_s,
            matcher_delay_max_s=10.0,
            candidate_margin_s=120,
        )
        add_check(f"a3_sampling_accuracy_{ev_sample_s}s", row["accuracy_mean"], expected, 1e-6)

    # Section VI-C ablation claims (manuscript Figure 8; committed ablation run on the
    # canonical base-load scenario seeds).
    ablation_path = PAPER_ABLATION_RESULTS_DIR / "summary.csv"
    ablation_ok = True
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation_ok = tuple(ablation.shape) == (36, 128)

        def _acc(df: pd.DataFrame, algorithm_id: str, ev_count: int) -> float:
            row = _row(
                df,
                algorithm_id=algorithm_id,
                ev_count=ev_count,
                ev_sample_s=30,
                matcher_delay_max_s=10.0,
                candidate_margin_s=120,
            )
            return float(row["accuracy_mean"])

        # Full-design rows must replay the canonical means exactly.
        for algorithm_id in PAPER_ALGORITHMS[1:]:
            for ev_count in (100, 300, 500):
                add_check(
                    f"ablation_full_matches_canonical_{algorithm_id}_ev_{ev_count}",
                    _acc(ablation, algorithm_id, ev_count),
                    _acc(summary, algorithm_id, ev_count),
                    1e-9,
                )
        # Removing the carry-over (gamma_prev = 0) also removes the 60 % prefix from the decision;
        # A3 then returns to the greedy level at the representative point.
        add_check("ablation_carry_over_off_500", _acc(ablation, "bayesian_windowed_memory_off", 500), 0.9736, 1e-6)
        add_check("ablation_a3_all_off_500", _acc(ablation, "bayesian_windowed_all_off", 500), 0.9732, 1e-6)
    else:
        ablation = None

    # Evidence checks. Each block checks a committed artefact only when it exists.
    revision_ok = True
    # Failure taxonomy of the representative point (A3, 20 repeats)
    taxonomy_path = PAPER_ACCURACY_DIR / "table9_failure_taxonomy.csv"
    if taxonomy_path.exists():
        taxonomy = pd.read_csv(taxonomy_path)
        all_row = taxonomy[(taxonomy["breakdown"] == "all") & (taxonomy["bin"] == "all sessions")].iloc[0]
        add_check("taxonomy_sessions", all_row["sessions"], 10000, 0)
        add_check("taxonomy_error_rate_matches_a3_accuracy", 1.0 - float(all_row["error_rate"]), 0.9838, 1e-6)
        cause_rows = taxonomy[taxonomy["breakdown"] == "cause"]
        add_check("taxonomy_cause_shares_sum_to_one", float(cause_rows["share_of_errors"].sum()), 1.0, 1e-9)
        add_check("taxonomy_cause_errors_sum", float(cause_rows["errors"].sum()), float(all_row["errors"]), 0)
    # Sec. VII-E: duplicate slot claims across decision batches (same traced run)
    duplicates_path = PAPER_ACCURACY_DIR / "duplicate_slot_claims.csv"
    if duplicates_path.exists():
        duplicates = pd.read_csv(duplicates_path)
        posterior = duplicates[duplicates["algorithm_id"] == "bayesian_windowed"].iloc[0]
        prior = duplicates[duplicates["algorithm_id"] == "nomura_original_interval_hungarian"].iloc[0]
        add_check("duplicate_claim_pairs_posterior", float(posterior["duplicate_pairs"]), 198, 0)
        add_check("duplicate_claim_pairs_prior_static", float(prior["duplicate_pairs"]), 8, 0)
        add_check("duplicate_claim_misidentified_involved_posterior", float(posterior["misidentified_sessions_involved"]), 159, 0)
        add_check("duplicate_claim_same_batch_pairs", float(duplicates["same_batch_pairs"].sum()), 0, 0)
        add_check("duplicate_claim_every_pair_has_an_error",
                  float(bool((duplicates["duplicate_pairs"] == duplicates["pairs_one_misidentified"] + duplicates["pairs_both_misidentified"]).all())), 1.0, 0)
    # Latency decomposition reproduces the canonical p90 per repeat
    latency_path = PAPER_ACCURACY_DIR / "latency_decomposition_representative.csv"
    if latency_path.exists():
        latency = pd.read_csv(latency_path)
        add_check("latency_repeats", float(len(latency)), 20, 0)
        add_check("latency_p90_matches_raw_all_repeats", float(latency["p90_match"].astype(bool).all()), 1.0, 0)
        add_check("latency_p90_mean", float(latency["p90_latency_raw_s"].mean()), 416.035, 1e-6)
        add_check("latency_window_is_w", float(latency["window_s_mean"].mean()), 360.0, 1e-9)
        add_check("latency_watermark_is_dmax", float(latency["watermark_s_mean"].mean()), 10.0, 1e-9)
        # Sec. VII-D: the modulated time T is bounded by the longest arrival-to-finalization time
        add_check("latency_sessions_total", float(latency["sessions"].sum()), 10000, 0)
        add_check("latency_max_all_repeats", float(latency["decision_latency_s_max"].max()), 464.0, 0)

    # Two-prefix baselines on the canonical base-load seeds (table10_ / figure6_ data files; TABLE 9 and Figure 8 of the manuscript)
    table10_path = PAPER_ACCURACY_DIR / "table10_two_prefix_ablation.csv"
    if table10_path.exists():
        table10 = pd.read_csv(table10_path)
        add_check("table10_table10_rows", float(len(table10)), 6, 0)
        for variant in ("two_prefix_greedy", "two_prefix_hungarian"):
            row = table10[(table10["variant"] == variant) & (table10["ev_count"] == 500)].iloc[0]
            add_check(f"table10_{variant}_ev500_runs", float(row["runs"]), 20, 0)
            add_check(f"table10_{variant}_ev500_accuracy", float(row["accuracy_mean"]), 0.9798, 1e-6)
            add_check(f"table10_{variant}_ev500_a3_reference", float(row["a3_mean"]), 0.9838, 1e-6)
            # decomposition of the A3 - A1 gain along Greedy -> two-prefix -> Posterior (Section VI-C)
            add_check(f"table10_{variant}_ev500_minus_greedy", float(row["diff_vs_a1_mean"]), 0.0066, 1e-6)
            add_check(f"table10_{variant}_ev500_posterior_minus_two_prefix", float(row["diff_vs_a3_mean"]), 0.0040, 1e-6)
        fig6_path = PAPER_ACCURACY_DIR / "figure6_ablation_simulated.csv"
        fig6 = pd.read_csv(fig6_path)
        add_check("figure6_series_count", float(len(fig6[["matcher_family", "variant"]].drop_duplicates())), 14, 0)

    # Static-limit sanity check (300 simultaneous sessions, 300 slots, ingestion off)
    table_s2_path = PAPER_ACCURACY_DIR / "table_s2_static_limit.csv"
    if table_s2_path.exists():
        table_s2 = pd.read_csv(table_s2_path)
        add_check("static_limit_table_s2_rows", float(len(table_s2)), 6, 0)
        add_check("static_limit_runs_all_20", float(table_s2["runs"].min()), 20, 0)
        add_check("static_limit_no_blocking", float(table_s2["blocked_evs_mean"].max()), 0.0, 0)
        add_check("static_limit_no_early_session_end", float(table_s2["session_end_before_watermark_ratio_mean"].max()), 0.0, 0)
        add_check("static_limit_all_slots_are_candidates", float(table_s2["avg_candidates_per_ev_mean"].min()), 300.0, 1e-9)
        # TABLE S2 intervals are Student-t over the per-repeat accuracies (Section V-B rule for the Supplementary tables),
        # and the static-limit runs applied zero ingestion delay and loss (report.json records the effective model).
        for cell_dir in sorted(PAPER_ACCURACY_DIR.glob("static_limit_*_sensing_run_*")):
            cell = "ideal_sensing" if "_ideal_" in cell_dir.name else "paper_sensing"
            raw_s2 = pd.read_csv(cell_dir / "raw_runs.csv")
            worst = 0.0
            for algorithm_id, grp in raw_s2.groupby("algorithm_id"):
                x = grp["accuracy"].to_numpy(dtype=float)
                half = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
                printed = float(table_s2[(table_s2["cell"] == cell) & (table_s2["algorithm_id"] == algorithm_id)]["accuracy_ci95"].iloc[0])
                worst = max(worst, abs(printed - half))
            add_check(f"static_limit_{cell}_ci95_student_t", worst, 0.0, 1e-6)
            report = json.loads((cell_dir / "report.json").read_text(encoding="utf-8"))
            ev_ingestion = report["online_operational_conditions"]["ingestion_model"]["ev"]
            applied = max(abs(float(ev_ingestion[k])) for k in ("delay_mean_s", "delay_max_s", "loss_prob", "watermark_delay_max_s"))
            add_check(f"static_limit_{cell}_ingestion_applied_zero", applied, 0.0, 0)
        for cell, algorithm_id, expected in (
            ("ideal_sensing", "bayesian_windowed", 1.0),
            ("ideal_sensing", "nomura_original_interval_hungarian", 1),
            ("ideal_sensing", "single_only", 0.677167),
            ("paper_sensing", "nomura_original_interval_hungarian", 0.952333),
            ("paper_sensing", "bayesian_windowed", 0.625667),
            ("paper_sensing", "single_only", 0.5805),
        ):
            row = table_s2[(table_s2["cell"] == cell) & (table_s2["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"static_limit_{cell}_{algorithm_id}_accuracy", float(row["accuracy_mean"]), expected, 1e-6)

    # Impairment sweeps at the representative point (manuscript Figure 5; data file figure7_impairment_sweeps.csv)
    fig7_path = PAPER_ACCURACY_DIR / "figure7_impairment_sweeps.csv"
    if fig7_path.exists():
        fig7 = pd.read_csv(fig7_path)
        add_check("impairment_figure7_rows", float(len(fig7)), 52, 0)  # 11 cells (default counted on 3 axes) x 4 matchers
        add_check("impairment_figure7_cells", float(fig7["cell"].nunique()), 11, 0)
        add_check("impairment_all_cells_20_repeats", float(fig7["runs"].min()), 20, 0)
        for algorithm_id, expected in expected_default.items():
            row = fig7[(fig7["cell"] == "default") & (fig7["axis"] == "loss") & (fig7["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"impairment_default_cell_{algorithm_id}", float(row["accuracy_mean"]), expected, 1e-6)
        for cell, algorithm_id, expected in (
            ("loss_0.30", "bayesian_windowed", 0.8136),
            ("loss_0.30", "single_only", 0.8072),
            ("delay_20_10_60", "bayesian_windowed", 0.985),
            ("noise_x8", "bayesian_windowed", 0.8708),
            ("noise_x8", "single_only", 0.8685),
        ):
            row = fig7[(fig7["cell"] == cell) & (fig7["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"impairment_{cell}_{algorithm_id}", float(row["accuracy_mean"]), expected, 1e-6)
        matched = fig7[fig7["matched_dmax"].astype(bool)]
        add_check("impairment_matched_dmax_p90_latency", float(matched["p90_latency_mean"].iloc[0]), 465.98, 1e-2)
        # ordering stated for the impairment sweeps: A2 > A3 >= A1 > A0 in every cell
        ok_rank = True
        for cell, grp in fig7.drop_duplicates(["cell", "algorithm_id"]).groupby("cell"):
            acc = grp.set_index("algorithm_id")["accuracy_mean"]
            ok_rank &= bool(acc["nomura_original_interval_hungarian"] > acc["bayesian_windowed"] >= acc["single_only"] > acc["time_only_baseline"])
        add_check("impairment_ranking_a2_a3_a1_a0_holds_in_all_cells", float(ok_rank), 1.0, 0)
    # realised share of lost EV measurements on the measurement-loss axis (replayed ingestion draws)
    loss_path = PAPER_ACCURACY_DIR / "impairment_realised_loss.csv"
    if loss_path.exists():
        rl = pd.read_csv(loss_path).set_index("cell")
        add_check("impairment_realised_loss_cells", float(len(rl)), 4, 0)
        add_check("impairment_realised_loss_measurements_per_cell", float(rl["measurements"].min()), 130000, 0)
        for cell, expected in (("default", 0.0052), ("loss_0.05", 0.0511), ("loss_0.15", 0.1505), ("loss_0.30", 0.3007)):
            add_check(f"impairment_realised_loss_share_{cell}", float(rl.loc[cell, "realised_share"]), expected, 5e-5)

    # Saturation operating points (TABLE 7)
    table7_path = PAPER_ACCURACY_DIR / "table7_saturation.csv"
    if table7_path.exists():
        table7 = pd.read_csv(table7_path)
        add_check("saturation_table7_rows", float(len(table7)), 12, 0)
        add_check("saturation_all_points_20_repeats", float(table7["runs"].min()), 20, 0)
        for algorithm_id, expected in expected_default.items():
            row = table7[(table7["point"] == "control") & (table7["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"saturation_control_{algorithm_id}", float(row["accuracy_mean"]), expected, 1e-6)
        for point, algorithm_id, key, expected, tol in (
            ("SA", "bayesian_windowed", "accuracy_mean", 0.9545, 1e-4),
            ("SA", "bayesian_windowed", "accuracy_requested_mean", 0.44970, 1e-4),
            ("SA", "bayesian_windowed", "blocked_ratio_mean", 0.5289, 1e-4),
            ("SB", "bayesian_windowed", "accuracy_mean", 0.9837, 1e-4),
            ("SB", "bayesian_windowed", "accuracy_requested_mean", 0.5281, 1e-4),
            ("SB", "bayesian_windowed", "blocked_ratio_mean", 0.4631, 1e-4),
        ):
            row = table7[(table7["point"] == point) & (table7["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"saturation_{point}_{algorithm_id}_{key}", float(row[key]), expected, tol)
        sat = table7[table7["point"].isin(["SA", "SB"])]
        add_check("saturation_saturation_points_block", float(sat["blocked_ratio_mean"].min() > 0.0), 1.0, 0)
        add_check("saturation_requested_below_admitted", float((sat["accuracy_requested_mean"] < sat["accuracy_mean"]).all()), 1.0, 0)
        add_check("saturation_a2_top_rank_on_admitted_sessions", float(all(
            grp.set_index("algorithm_id")["accuracy_mean"].idxmax() == "nomura_original_interval_hungarian" for _, grp in table7.groupby("point")
        )), 1.0, 0)
        add_check("saturation_saturation_ordering_a2_a3_a1_a0", float(all(
            (lambda r: r["nomura_original_interval_hungarian"] > r["bayesian_windowed"] >= r["single_only"] > r["time_only_baseline"])(
                grp.set_index("algorithm_id")["accuracy_mean"]) for _, grp in sat.groupby("point")
        )), 1.0, 0)

    # A3 hyperparameter sensitivity (Supplementary Table S4; data file table8_hyperparam_sensitivity.csv)
    table8_path = PAPER_ACCURACY_DIR / "table8_hyperparam_sensitivity.csv"
    if table8_path.exists():
        table8 = pd.read_csv(table8_path)
        add_check("hyperparam_table8_rows", float(len(table8)), 15, 0)
        add_check("hyperparam_all_cells_20_repeats", float(table8["runs"].min()), 20, 0)
        nondefault = table8[~table8["is_default"].astype(bool)]
        add_check("hyperparam_nondefault_cells", float(len(nondefault)), 12, 0)
        add_check("hyperparam_min_accuracy_over_grid", float(nondefault["accuracy_mean"].min()), 0.9815, 1e-4)
        add_check("hyperparam_max_accuracy_over_grid", float(nondefault["accuracy_mean"].max()), 0.9838, 1e-4)
        add_check("hyperparam_every_cell_above_a1_reference", float((nondefault["accuracy_mean"] > 0.9732).all()), 1.0, 0)
        for knob, value, expected in (("gamma_prev", 0.3, 0.9815), ("gamma_prev", 0.9, 0.983), ("eta_mix", 0.7, 0.9838), ("beta_curr", 2.0, 0.9837)):
            row = table8[(table8["knob"] == knob) & (table8["value"] == value)].iloc[0]
            add_check(f"hyperparam_{knob}_{value:g}", float(row["accuracy_mean"]), expected, 1e-4)

    # Dense-sampling diagnostic (TABLE S1)
    table_s1_path = PAPER_ACCURACY_DIR / "table_s1_sampling_diagnostic.csv"
    if table_s1_path.exists():
        table_s1 = pd.read_csv(table_s1_path)
        add_check("sampling_diag_table_s1_rows", float(len(table_s1)), 26, 0)
        add_check("sampling_diag_all_cells_20_repeats", float(table_s1["runs"].min()), 20, 0)
        for variant, dt, algorithm_id, expected in (
            ("canonical", 5, "bayesian_windowed", 0.9495), ("canonical", 15, "bayesian_windowed", 0.9683), ("canonical", 30, "bayesian_windowed", 0.9838),
            ("noise_free", 5, "bayesian_windowed", 0.9491), ("noise_free", 15, "bayesian_windowed", 0.9658), ("noise_free", 30, "bayesian_windowed", 0.9832),
            ("block_mean", 5, "bayesian_windowed", 0.9439), ("block_mean", 15, "bayesian_windowed", 0.9602),
            ("noise_free", 5, "single_only", 0.9466), ("block_mean", 5, "single_only", 0.92990),
        ):
            row = table_s1[(table_s1["variant"] == variant) & (table_s1["ev_sample_s"] == dt) & (table_s1["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"sampling_diag_{variant}_{dt}s_{algorithm_id}", float(row["accuracy_mean"]), expected, 1e-4)
        nf = table_s1[(table_s1["variant"] == "noise_free") & (table_s1["algorithm_id"] == "bayesian_windowed")].sort_values("ev_sample_s")
        add_check("sampling_diag_noise_free_trend_persists", float(list(nf["accuracy_mean"]) == sorted(nf["accuracy_mean"])), 1.0, 0)
        # Eq. (7) term partition: the 5 -> 30 s rise lives in the banded-DTW term; the step-median term moves the other way
        for dt, algorithm_id, expected in (
            (5, "single_only_step_off", 0.9028), (15, "single_only_step_off", 0.9773), (30, "single_only_step_off", 0.9963),
            (5, "single_only_dtw_off", 0.9122), (15, "single_only_dtw_off", 0.9424), (30, "single_only_dtw_off", 0.9213),
        ):
            row = table_s1[(table_s1["variant"] == "term_partition") & (table_s1["ev_sample_s"] == dt) & (table_s1["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"sampling_diag_term_partition_{dt}s_{algorithm_id}", float(row["accuracy_mean"]), expected, 1e-4)
            add_check(f"sampling_diag_term_partition_{dt}s_{algorithm_id}_runs", float(row["runs"]), 20, 0)
        tp = table_s1[table_s1["variant"] == "term_partition"]
        dtw = tp[tp["algorithm_id"] == "single_only_step_off"].set_index("ev_sample_s")["accuracy_mean"]
        stp = tp[tp["algorithm_id"] == "single_only_dtw_off"].set_index("ev_sample_s")["accuracy_mean"]
        add_check("sampling_diag_term_partition_dtw_only_rises_5_to_30s", float(bool(dtw[5] < dtw[15] < dtw[30])), 1.0, 0)
        add_check("sampling_diag_term_partition_step_only_does_not_rise_monotonically", float(bool(not (stp[5] < stp[15] < stp[30]))), 1.0, 0)
        add_check("sampling_diag_term_partition_dtw_only_5s_vs_30s_deficit", float(dtw[30] - dtw[5]), 0.0935, 1e-4)

    # Per-session response heterogeneity on the base-load grid — manuscript Figure 6 (data file figure8_heterogeneous_response.csv)
    fig8_path = PAPER_ACCURACY_DIR / "figure8_heterogeneous_response.csv"
    if fig8_path.exists():
        fig8 = pd.read_csv(fig8_path)
        add_check("heterogeneity_figure8_rows", float(len(fig8)), 24, 0)
        het = fig8[fig8["model"] == "heterogeneous"]
        # TABLE 5 protocol: EV 300 rows use the held-out repeats 6-20 (15 paired runs); EV 100 / 500 all 20
        add_check("heterogeneity_heterogeneous_ev300_held_out_runs", float(het[het["ev_count"] == 300]["runs"].min()), 15, 0)
        add_check("heterogeneity_heterogeneous_ev100_ev500_runs", float(het[het["ev_count"] != 300]["runs"].min()), 20, 0)
        for aid, ev, expected in (("bayesian_windowed", 100, 0.9950), ("bayesian_windowed", 300, 0.9793), ("bayesian_windowed", 500, 0.965),
                                  ("single_only", 500, 0.9493), ("nomura_original_interval_hungarian", 500, 0.9991), ("time_only_baseline", 500, 0.7133)):
            row = het[(het["algorithm_id"] == aid) & (het["ev_count"] == ev)].iloc[0]
            add_check(f"heterogeneity_heterogeneous_{aid}_ev{ev}_accuracy", float(row["accuracy_mean"]), expected, 1e-4)
        a3_500 = het[(het["algorithm_id"] == "bayesian_windowed") & (het["ev_count"] == 500)].iloc[0]
        add_check("heterogeneity_a3_ev500_delta_vs_nominal", float(a3_500["delta_vs_nominal_mean"]), -0.0188, 1e-4)
        add_check("heterogeneity_a3_ev500_delta_ci_excludes_zero", float(bool(abs(float(a3_500["delta_vs_nominal_mean"])) > float(a3_500["delta_vs_nominal_ci95"]))), 1.0, 0)
        a2_500 = het[(het["algorithm_id"] == "nomura_original_interval_hungarian") & (het["ev_count"] == 500)].iloc[0]
        add_check("heterogeneity_a2_ev500_unchanged_under_heterogeneity", float(abs(float(a2_500["delta_vs_nominal_mean"]))), 0.0, 0.005)
        r500 = het[het["ev_count"] == 500].set_index("algorithm_id")["accuracy_mean"]
        add_check("heterogeneity_ordering_a2_a3_a1_a0_under_heterogeneity", float(all(
            (lambda r: r["nomura_original_interval_hungarian"] > r["bayesian_windowed"] > r["single_only"] > r["time_only_baseline"])(
                het[het["ev_count"] == ev].set_index("algorithm_id")["accuracy_mean"]) for ev in (100, 300, 500)
        )), 1.0, 0)

    # Cost-term partition (TABLE S3 of the Supplementary Material): which term of each cost fails under sensing distortion
    table_s3_path = PAPER_ACCURACY_DIR / "table_s3_cost_term_partition.csv"
    if table_s3_path.exists():
        s3 = pd.read_csv(table_s3_path)
        add_check("table_s3_rows", float(len(s3)), 18, 0)
        add_check("table_s3_all_cells_20_repeats", float(s3["runs"].min()), 20, 0)
        expected_s3 = {
            (1.0, "single_only"): 0.9732, (1.0, "single_only_step_off"): 0.9963, (1.0, "single_only_dtw_off"): 0.9214,
            (1.0, "nomura_original_interval_hungarian"): 0.9993, (1.0, "nomura_original_interval_hungarian_corr_off"): 0.9716,
            (1.0, "nomura_original_interval_hungarian_dtw_off"): 0.9703,
            (4.0, "single_only"): 0.9408, (4.0, "single_only_step_off"): 0.9953, (4.0, "single_only_dtw_off"): 0.8601,
            (4.0, "nomura_original_interval_hungarian"): 0.9752, (4.0, "nomura_original_interval_hungarian_corr_off"): 0.7472,
            (4.0, "nomura_original_interval_hungarian_dtw_off"): 0.9685,
            (8.0, "single_only"): 0.8685, (8.0, "single_only_step_off"): 0.9922, (8.0, "single_only_dtw_off"): 0.7433,
            (8.0, "nomura_original_interval_hungarian"): 0.9206, (8.0, "nomura_original_interval_hungarian_corr_off"): 0.5353,
            (8.0, "nomura_original_interval_hungarian_dtw_off"): 0.9633,
        }
        for (multiplier, algorithm_id), expected in expected_s3.items():
            row = s3[(s3["sensing_multiplier"] == multiplier) & (s3["algorithm_id"] == algorithm_id)].iloc[0]
            add_check(f"table_s3_x{multiplier:g}_{algorithm_id}", float(row["accuracy_mean"]), expected, 1e-4)
        # the diagnosis the table exists for: dropping the ampere-scale step term helps under distortion,
        # dropping correlation from the prior cost hurts, and both effects grow with the multiplier
        dtw_only = s3[s3["algorithm_id"] == "single_only_step_off"].set_index("sensing_multiplier")["accuracy_mean"]
        step_only = s3[s3["algorithm_id"] == "single_only_dtw_off"].set_index("sensing_multiplier")["accuracy_mean"]
        corr_off = s3[s3["algorithm_id"] == "nomura_original_interval_hungarian_corr_off"].set_index("sensing_multiplier")["accuracy_mean"]
        dtw_off = s3[s3["algorithm_id"] == "nomura_original_interval_hungarian_dtw_off"].set_index("sensing_multiplier")["accuracy_mean"]
        add_check("table_s3_dtw_only_arm_is_flat", float(dtw_only[1.0] - dtw_only[8.0]), 0.0034, 5e-3)
        add_check("table_s3_step_only_arm_collapses", float(step_only[1.0] - step_only[8.0]), 0.1781, 5e-3)
        add_check("table_s3_corr_off_arm_collapses", float(corr_off[1.0] - corr_off[8.0]), 0.4365, 5e-3)
        add_check("table_s3_corr_only_arm_is_flat", float(dtw_off[1.0] - dtw_off[8.0]), 0.0066, 5e-3)

    # TABLE S1 (v) alignment control and the Section VI-B statements it supports
    table_s1_path = PAPER_ACCURACY_DIR / "table_s1_sampling_diagnostic.csv"
    if table_s1_path.exists():
        s1 = pd.read_csv(table_s1_path)
        align = s1[s1["variant"] == "alignment_control"].set_index("algorithm_id")
        add_check("sampling_diag_alignment_control_rows", float(len(align)), 4, 0)
        add_check("sampling_diag_alignment_control_is_5s_only", float((s1[s1["variant"] == "alignment_control"]["ev_sample_s"] == 5).all()), 1.0, 0)
        for algorithm_id, expected in (("single_only", 0.0641), ("bayesian_windowed", 0.0501),
                                       ("nomura_original_interval_hungarian", 0.0615), ("single_only_step_off", 0.0968)):
            row = align.loc[algorithm_id]
            add_check(f"sampling_diag_alignment_gain_{algorithm_id}", float(row["delta_vs_canonical_mean"]), expected, 1e-4)
            # every arm's gain is significant: the paired CI stays above zero
            add_check(f"sampling_diag_alignment_gain_ci_above_zero_{algorithm_id}",
                      float(bool(float(row["delta_vs_canonical_mean"]) - float(row["delta_vs_canonical_ci95"]) > 0.0)), 1.0, 0)
        # with the offset fixed the 5 s advantage of the staged posterior over the prior static cost is gone
        align_raw = pd.read_csv(PAPER_ACCURACY_DIR / "sampling_diag_offsetfree_5_run_20260910_180742_322660" / "raw_runs.csv")
        a3 = align_raw[align_raw["algorithm_id"] == "bayesian_windowed"].sort_values("repeat_idx")["accuracy"].to_numpy(dtype=float)
        a2 = align_raw[align_raw["algorithm_id"] == "nomura_original_interval_hungarian"].sort_values("repeat_idx")["accuracy"].to_numpy(dtype=float)
        d = a3 - a2
        half = float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
        add_check("sampling_diag_alignment_a3_minus_a2_mean", float(d.mean()), -0.0002, 1e-4)
        add_check("sampling_diag_alignment_a3_minus_a2_ci_includes_zero", float(bool(d.mean() - half < 0.0 < d.mean() + half)), 1.0, 0)

    # printed digits follow the half-up rule on the exact mean
    if table_s3_path.exists():
        printed_expected = {
            (1.0, "single_only"): 0.973, (1.0, "single_only_step_off"): 0.996, (1.0, "single_only_dtw_off"): 0.921,
            (1.0, "nomura_original_interval_hungarian"): 0.999, (1.0, "nomura_original_interval_hungarian_corr_off"): 0.972,
            (1.0, "nomura_original_interval_hungarian_dtw_off"): 0.970,
            (4.0, "single_only"): 0.941, (4.0, "single_only_step_off"): 0.995, (4.0, "single_only_dtw_off"): 0.860,
            (4.0, "nomura_original_interval_hungarian"): 0.975, (4.0, "nomura_original_interval_hungarian_corr_off"): 0.747,
            (4.0, "nomura_original_interval_hungarian_dtw_off"): 0.969,
            (8.0, "single_only"): 0.869, (8.0, "single_only_step_off"): 0.992, (8.0, "single_only_dtw_off"): 0.743,
            (8.0, "nomura_original_interval_hungarian"): 0.921, (8.0, "nomura_original_interval_hungarian_corr_off"): 0.535,
            (8.0, "nomura_original_interval_hungarian_dtw_off"): 0.963,
        }
        s3p = pd.read_csv(table_s3_path)
        for (multiplier, algorithm_id), expected in printed_expected.items():
            row = s3p[(s3p["sensing_multiplier"] == multiplier) & (s3p["algorithm_id"] == algorithm_id)].iloc[0]
            correct, total = _cost_term_counts(multiplier, algorithm_id)
            add_check(f"table_s3_printed_x{multiplier:g}_{algorithm_id}",
                      printed_accuracy(correct, total, float(row["accuracy_mean"])), expected, 0)

    # Zero-run codebook facts: the manuscript's codebook is the generator's random-unique branch
    cb_stats_path = PAPER_ACCURACY_DIR / "codebook_realised_statistics.csv"
    cb_sep_path = PAPER_ACCURACY_DIR / "codebook_candidate_separation.csv"
    if cb_stats_path.exists() and cb_sep_path.exists():
        cb = pd.read_csv(cb_stats_path); cb_all = cb[cb["repeat_idx"].astype(str) == "all"].iloc[0]; cb_rep = cb[cb["repeat_idx"].astype(str) != "all"]
        add_check("codebook_repeats", float(len(cb_rep)), 20, 0)
        add_check("codebook_pairwise_l1_min_a", float(cb_all["pairwise_l1_min_a"]), 4.0, 0)
        add_check("codebook_pairwise_l1_min_max_over_repeats_a", float(cb_rep["pairwise_l1_min_a"].max()), 13.0, 0)
        add_check("codebook_pairwise_l1_mean_a", float(cb_all["pairwise_l1_mean_a"]), 41.874, 1e-3)
        add_check("codebook_internal_gap_min_a", float(cb_all["internal_gap_min_a"]), 1.0, 0)
        add_check("codebook_consec_step_min_a", float(cb_all["consec_step_min_a"]), 1.0, 0)
        add_check("codebook_patterns_violating_internal_4a_share", float(cb_all["patterns_violating_internal_4a"]) / float(cb_all["patterns_used"]), 907.0 / 934.0, 1e-6)
        add_check("codebook_patterns_violating_consec_8a_share", float(cb_all["patterns_violating_consec_8a"]) / float(cb_all["patterns_used"]), 878.0 / 934.0, 1e-6)
        sep = pd.read_csv(cb_sep_path)
        for aid, label, expected in (("bayesian_windowed", "<= 10", 1.0 / 13.0), ("bayesian_windowed", "> 40", 0.0), ("bayesian_windowed", "all", 0.0162),
                                     ("single_only", "<= 10", 3.0 / 13.0), ("single_only", "all", 0.0268)):
            row = sep[(sep["algorithm_id"] == aid) & (sep["separation_bin"] == label)].iloc[0]
            add_check(f"codebook_candidate_separation_{aid}_{label.replace(' ', '').replace('<=', 'le').replace('>', 'gt')}", float(row["conditional_error_rate"]), expected, 1e-6)
        le10 = sep[(sep["algorithm_id"] == "bayesian_windowed") & (sep["separation_bin"] == "<= 10")].iloc[0]
        add_check("codebook_candidate_separation_le10_sessions", float(le10["sessions"]), 13, 0)
        a3 = sep[(sep["algorithm_id"] == "bayesian_windowed") & (sep["separation_bin"].isin(["<= 10", "11-20", "21-30", "31-40", "> 40"]))]
        a3 = a3.set_index("separation_bin").loc[["<= 10", "11-20", "21-30", "31-40", "> 40"], "conditional_error_rate"].to_numpy(dtype=float)
        add_check("codebook_candidate_separation_a3_error_rate_monotone", float(bool((np.diff(a3) <= 0).all())), 1.0, 0)

    # Set-point band sweep (figure9_ data file, Figure 7 of the manuscript): accuracy vs realised codebook separation; ranking preserved at every band
    fig9_path = PAPER_ACCURACY_DIR / "figure9_codebook_separation.csv"
    if fig9_path.exists():
        fig9 = pd.read_csv(fig9_path)
        add_check("codebook_figure9_rows", float(len(fig9)), 12, 0)
        add_check("codebook_band_cells_20_runs", float(fig9["runs"].min()), 20, 0)
        sep = fig9.drop_duplicates("cell").set_index("cell")["realised_l1_mean_a"]
        add_check("codebook_realised_separation_monotone_in_band", float(bool(sep["B30"] > sep["B20"] > sep["B16"] > sep["B12"])), 1.0, 0)
        for cell, aid, expected in (("B20", "bayesian_windowed", 0.9738), ("B20", "single_only", 0.9675), ("B20", "nomura_original_interval_hungarian", 0.9962),
                                    ("B16", "bayesian_windowed", 0.9484), ("B16", "single_only", 0.9364), ("B16", "nomura_original_interval_hungarian", 0.9809),
                                    ("B12", "bayesian_windowed", 0.8952), ("B12", "single_only", 0.8811), ("B12", "nomura_original_interval_hungarian", 0.9231)):
            row = fig9[(fig9["cell"] == cell) & (fig9["algorithm_id"] == aid)].iloc[0]
            add_check(f"codebook_{cell}_{aid}_accuracy", float(row["accuracy_mean"]), expected, 1e-4)
        for cell in ("B30", "B20", "B16", "B12"):
            r = fig9[fig9["cell"] == cell].set_index("algorithm_id")["accuracy_mean"]
            add_check(f"codebook_{cell}_ordering_a2_a3_a1", float(bool(r["nomura_original_interval_hungarian"] > r["bayesian_windowed"] > r["single_only"])), 1.0, 0)
        a3 = fig9[fig9["algorithm_id"] == "bayesian_windowed"].set_index("cell")["accuracy_mean"]
        add_check("codebook_a3_accuracy_monotone_in_separation", float(bool(a3["B30"] > a3["B20"] > a3["B16"] > a3["B12"])), 1.0, 0)
        for aid in ("nomura_original_interval_hungarian", "single_only"):
            m = fig9[fig9["algorithm_id"] == aid].set_index("cell")["accuracy_mean"]
            add_check(f"codebook_{aid}_accuracy_monotone_in_separation", float(bool(m["B30"] > m["B20"] > m["B16"] > m["B12"])), 1.0, 0)
        add_check("codebook_a3_above_0p90_down_to_B16", float(bool(a3["B16"] > 0.90 > a3["B12"])), 1.0, 0)
        for cell, expected in (("B20", 24.91), ("B16", 14.74), ("B12", 8.06)):
            add_check(f"codebook_{cell}_realised_l1_mean_a", float(sep[cell]), expected, 5e-3)

    # Static-limit anchor diagnostics: per-repeat A2-ideal accuracy vs legacy-codebook near-duplicates
    s2_rep_path = PAPER_ACCURACY_DIR / "static_limit_a2_ideal_by_repeat.csv"
    if s2_rep_path.exists():
        s2r_all = pd.read_csv(s2_rep_path)
        # the run carries the cell's three matchers; these checks are about the prior static cost
        s2r = s2r_all[s2r_all["algorithm_id"] == "nomura_original_interval_hungarian"]
        add_check("static_limit_ideal_cell_rows", float(len(s2r_all)), 60, 0)
        add_check("static_limit_a2_ideal_by_repeat_rows", float(len(s2r)), 20, 0)
        add_check("static_limit_legacy_codebook_distinct_patterns_min", float(s2r["codebook_distinct_patterns"].min()), 300, 0)
        add_check("static_limit_legacy_codebook_pairwise_l1_min_max_a", float(s2r["pairwise_l1_min_a"].max()), 4, 0)
        add_check("static_limit_legacy_codebook_pairs_le_4a_max", float(s2r["pairs_l1_le_4a"].max()), 7, 0)
        add_check("static_limit_legacy_codebook_pairs_le_8a_min", float(s2r["pairs_l1_le_8a"].min()), 38, 0)
        add_check("static_limit_legacy_codebook_pairs_le_8a_max", float(s2r["pairs_l1_le_8a"].max()), 68, 0)
        add_check("static_limit_a2_ideal_repeats_at_or_above_0p99", float((s2r["accuracy"] >= 0.99 - 1e-9).sum()), 20, 0)
    add_check("static_limit_a2_ideal_errors_total", float(s2r["errors"].sum()) if s2_rep_path.exists() else float("nan"), 0, 0)

    # Section V-B: two-sided Wilcoxon signed-rank tests with a Holm adjustment over the twelve base-load
    # comparisons (the prior static cost against each other matcher and the Posterior matcher against its
    # greedy base, at EV = 100, 300 and 500; EV = 300 on its held-out repeats 6-20). Zero differences are
    # dropped and the normal approximation is used without continuity correction.
    from scipy.stats import wilcoxon as _wilcoxon
    base_grid = raw_grid[(raw_grid["ev_sample_s"] == 30) & (raw_grid["matcher_delay_max_s"] == 10.0) & (raw_grid["candidate_margin_s"] == 120)]
    comparisons = (("nomura_original_interval_hungarian", "time_only_baseline"), ("nomura_original_interval_hungarian", "single_only"),
                   ("nomura_original_interval_hungarian", "bayesian_windowed"), ("bayesian_windowed", "single_only"))
    tests_vb: list[dict[str, Any]] = []
    for ev_count in (100, 300, 500):
        cell = base_grid[base_grid["ev_count"] == ev_count]
        if ev_count == 300:
            cell = cell[cell["repeat_idx"] >= 6]
        acc = cell.pivot_table(index="repeat_idx", columns="algorithm_id", values="accuracy")
        for first, second in comparisons:
            diff = (acc[first] - acc[second]).to_numpy(dtype=float)
            p_value = float(_wilcoxon(acc[first], acc[second], zero_method="wilcox", correction=False,
                                      alternative="two-sided", method="approx").pvalue)
            tests_vb.append({"key": f"{first}_vs_{second}_ev_{ev_count}", "p": p_value, "ties": int(np.sum(np.abs(diff) < 1e-12))})
    running = 0.0
    for rank, idx in enumerate(np.argsort([t["p"] for t in tests_vb], kind="stable")):
        running = max(running, (len(tests_vb) - rank) * tests_vb[idx]["p"])
        tests_vb[idx]["holm"] = min(1.0, running)
    holm = {t["key"]: t for t in tests_vb}
    add_check("wilcoxon_holm_comparisons", float(len(tests_vb)), 12, 0)
    gp = holm["nomura_original_interval_hungarian_vs_bayesian_windowed_ev_100"]
    add_check("wilcoxon_holm_global_vs_posterior_ev_100", gp["holm"], 0.076, 5e-4)
    add_check("wilcoxon_holm_global_vs_posterior_ev_100_tied_repeats", float(gp["ties"]), 15, 0)
    for ev_count in (100, 300):
        add_check(f"wilcoxon_holm_posterior_vs_greedy_ev_{ev_count}", holm[f"bayesian_windowed_vs_single_only_ev_{ev_count}"]["holm"], 0.51, 5e-3)
    exempt = {"nomura_original_interval_hungarian_vs_bayesian_windowed_ev_100", "bayesian_windowed_vs_single_only_ev_100", "bayesian_windowed_vs_single_only_ev_300"}
    others = [t["holm"] for t in tests_vb if t["key"] not in exempt]
    add_check("wilcoxon_holm_other_nine_below_0p02", float(bool(len(others) == 9 and max(others) < 0.02)), 1.0, 0)

    # Section VI-C and Supplementary Section S5: posterior of the chosen minus the true slot over the Posterior
    # matcher's errors at the representative point (near-tie: gap below 0.05)
    fc_path = PAPER_ACCURACY_DIR / "failure_cases_representative.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path)
        near = (fc["posterior_chosen_slot"] - fc["posterior_true_slot"]) < 0.05
        add_check("near_tie_errors", float(len(fc)), 162, 0)
        add_check("near_tie_count", float(near.sum()), 70, 0)
        add_check("near_tie_share_printed_percent", float(round(100.0 * float(near.mean()))), 43, 0)
        add_check("near_tie_posterior_chosen_mean", float(fc["posterior_chosen_slot"].mean()), 0.340, 5e-4)
        add_check("near_tie_posterior_true_mean", float(fc["posterior_true_slot"].mean()), 0.247, 5e-4)
        add_check("near_tie_posterior_chosen_mean_within_near_ties", float(fc.loc[near, "posterior_chosen_slot"].mean()), 0.275, 5e-4)
        add_check("near_tie_posterior_true_mean_within_near_ties", float(fc.loc[near, "posterior_true_slot"].mean()), 0.252, 5e-4)
    add_check("near_tie_source_present", float(fc_path.exists()), 1.0, 0)

    # Evidence lineage: every committed run folder, and every run id quoted in a top-level evidence
    # CSV, comes from the corrected simulator (run dates 2026-09-07 or later).
    import re as _re
    lineage_min_date = 20260907
    folder_dates = [int(m.group(1)) for d in PAPER_ACCURACY_DIR.iterdir() if d.is_dir()
                    for m in [_re.search(r"_run_(\d{8})_\d{6}", d.name)] if m]
    csv_dates: list[int] = []
    for csv_path in sorted(PAPER_ACCURACY_DIR.glob("*.csv")):
        frame = pd.read_csv(csv_path)
        if "run_id" in frame.columns:
            csv_dates += [int(x) for x in _re.findall(r"(20\d{6})_\d{6}", " ".join(frame["run_id"].astype(str)))]
    add_check("evidence_lineage_run_folders_on_or_after_20260907", float(bool(folder_dates) and min(folder_dates) >= lineage_min_date), 1.0, 0)
    add_check("evidence_lineage_csv_run_ids_on_or_after_20260907", float(bool(csv_dates) and min(csv_dates) >= lineage_min_date), 1.0, 0)

    causal_ok = (
        "candidate_query_lead_max_s_mean" in summary.columns
        and "candidate_future_query_violation_count_mean" in summary.columns
        and float(summary["candidate_query_lead_max_s_mean"].max()) <= 0.0
        and float(summary["candidate_future_query_violation_count_mean"].max()) == 0.0
    )
    shape_ok = tuple(summary.shape) == (324, 128)
    raw_ok = raw_rows in (None, 6480)
    ok = bool(shape_ok and raw_ok and causal_ok and ablation_ok and revision_ok and all(c["ok"] for c in checks))
    print(
        json.dumps(
            {
                "ok": ok,
                "run_id": PAPER_RUN_ID,
                "ablation_run_id": PAPER_ABLATION_RUN_ID if ablation is not None else None,
                "summary_shape": list(summary.shape),
                "raw_rows": raw_rows,
                "causal_candidate_checks_ok": bool(causal_ok),
                "ablation_grid_ok": bool(ablation_ok),
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if ok else 1


def regenerate_figures() -> int:
    output_dir = REPO_ROOT / "paper"
    generators = {
        3: generate_figure3_base_load_comparison,
        4: generate_figure4_sampling_sensitivity,
        5: generate_figure5_posterior_diagnostic,
        6: generate_figure6_ablation_summary,
    }
    if FIGURE7_CSV.exists():  # figure file figure7 (manuscript Figure 5); generated only once its committed data table exists
        generators[7] = generate_figure7_impairment_sweeps
    written: dict[str, str] = {}
    for figure_number, generator in generators.items():
        generator(output_dir, save_png=False)
        written[f"Figure {figure_number}"] = str(output_dir / PAPER_FIGURE_FILES[figure_number])
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": PAPER_RUN_ID,
                "figures": written,
            },
            indent=2,
        )
    )
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    base_seed = PAPER_BASE_SEED if args.base_seed is None else int(args.base_seed)
    result = run_algorithm_compare_report(
        ev_counts=[int(args.smoke_ev_count)],
        repeats=int(args.smoke_repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=["time_only_baseline", "bayesian_windowed"],
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        **PAPER_INGESTION_DEFAULTS,
        trace_level="summary",
    )
    print(json.dumps({"ok": True, "mode": "smoke", "run_id": result.get("run_id")}, indent=2))
    return 0


def _parse_algorithm_filter(raw: str | None, allowed: list[str]) -> list[str]:
    """Comma-separated algorithm ids restricted to ``allowed`` (order preserved, duplicates dropped)."""
    if raw is None or str(raw).strip() == "":
        return list(allowed)
    out: list[str] = []
    for token in str(raw).split(","):
        aid = token.strip()
        if aid == "":
            continue
        if aid not in allowed:
            raise SystemExit(f"--algorithms: unknown id {aid!r}; allowed: {', '.join(allowed)}")
        if aid not in out:
            out.append(aid)
    if len(out) == 0:
        raise SystemExit("--algorithms: no algorithm id given")
    return out


def run_ablation(args: argparse.Namespace) -> int:
    base_seed = PAPER_ABLATION_BASE_SEED if args.base_seed is None else int(args.base_seed)
    algorithm_ids = _parse_algorithm_filter(getattr(args, "algorithms", None), list(PAPER_ABLATION_ALGORITHM_IDS))
    result = run_algorithm_compare_report(
        ev_counts=[100, 300, 500],
        repeats=int(args.repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        pattern_unique_count=50,
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=list(algorithm_ids),
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        **PAPER_INGESTION_DEFAULTS,
        trace_level=str(args.trace_level),
    )
    print(json.dumps({"ok": True, "mode": "ablation", "run_id": result.get("run_id")}, indent=2))
    return 0


STATIC_LIMIT_CELLS: dict[str, dict[str, Any]] = {
    # Static limit: sensing-distortion layer switched off (the prior static study had none)
    "ideal_sensing": {
        "ev_sensor_extra_delay_max_s": 0,
        "ev_sensor_gain_std": 0.0,
        "ev_sensor_bias_std_a": 0.0,
        "ev_sensor_noise_std_a": 0.0,
        "ev_start_est_jitter_s": 0,
    },
    # TABLE 3 / Sec. V-A sensing defaults (module values; nothing overridden)
    "paper_sensing": {},
}
STATIC_LIMIT_ALGORITHMS = ["nomura_original_interval_hungarian", "single_only", "bayesian_windowed"]


def run_static_limit(args: argparse.Namespace) -> int:
    """Static-limit sanity check.

    300 sessions arrive simultaneously on 300 slots, every waveform is complete (W = 360 s
    inside a 10–60 min session), cloud ingestion is disabled (zero delay / jitter / loss,
    watermark bound 0 s) and the codebook is the legacy interval generator with fixed
    slot→pattern binding — the setting of the prior static study. One run per cell
    (``--cell ideal|paper|both``); the two cells differ only in the sensing layer. A 300 × 300
    batch decision costs minutes per matcher and repeat, so ``--algorithms`` allows one process
    per matcher (scenario and ingestion seeds do not depend on the matcher set, so the runs
    stay paired).
    """
    base_seed = PAPER_BASE_SEED if args.base_seed is None else int(args.base_seed)
    wanted = str(getattr(args, "cell", "both") or "both").strip().lower()
    cells = list(STATIC_LIMIT_CELLS) if wanted == "both" else [f"{wanted}_sensing"]
    for cell in cells:
        if cell not in STATIC_LIMIT_CELLS:
            raise SystemExit(f"--cell must be ideal, paper or both (got {wanted!r})")
    algorithm_ids = _parse_algorithm_filter(getattr(args, "algorithms", None), list(STATIC_LIMIT_ALGORITHMS))
    run_ids: dict[str, str] = {}
    for cell in cells:
        result = run_algorithm_compare_report(
            ev_counts=[300],
            repeats=int(args.repeats),
            error_mode="ci95",
            base_seed=int(base_seed),
            pattern_mode="legacy",
            pattern_assignment_mode="legacy_slot_fixed",
            pattern_unique_count=50,
            command_step_count=6,
            tau_values=[60],
            algorithm_ids=list(algorithm_ids),
            charger_sample_values=[5],
            ev_sample_values=[30],
            matcher_delay_max_values_s=[0.0],
            candidate_margin_values_s=[120],
            **{**PAPER_INGESTION_DEFAULTS, "ingestion_enabled": False},
            trace_level=str(args.trace_level),
            evse_count=300,
            simultaneous_arrivals=True,
            **STATIC_LIMIT_CELLS[cell],
        )
        run_ids[cell] = str(result.get("run_id"))
    print(json.dumps({"ok": True, "mode": "static-limit", "run_ids": run_ids}, indent=2))
    return 0


def run_trace_representative(args: argparse.Namespace) -> int:
    """Replay the representative EV = 500 cell with full traces (failure taxonomy and latency decomposition).

    Seeds ``PAPER_REPRESENTATIVE_BASE_SEED + k`` (k = 0..repeats-1) reproduce the canonical
    EV = 500, Δt_EV = 30 s, d_max = 10 s, Δ = 120 s realizations exactly, so the per-EV
    event/cost traces and the per-session metadata export describe the manuscript's own
    residual errors. Default matcher: A3 (``--algorithms`` may add ``single_only``).
    """
    base_seed = PAPER_REPRESENTATIVE_BASE_SEED if args.base_seed is None else int(args.base_seed)
    algorithm_ids = _parse_algorithm_filter(getattr(args, "algorithms", None), list(PAPER_ALGORITHMS))
    if getattr(args, "algorithms", None) is None:
        algorithm_ids = ["bayesian_windowed"]
    result = run_algorithm_compare_report(
        ev_counts=[500],
        repeats=int(args.repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        pattern_unique_count=50,
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=list(algorithm_ids),
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        **PAPER_INGESTION_DEFAULTS,
        trace_level=str(args.trace_level),
    )
    print(json.dumps({"ok": True, "mode": "trace-representative", "run_id": result.get("run_id")}, indent=2))
    return 0


# --------------------------------------------------------------------------- Impairment sweeps
# One-factor-at-a-time impairment sweeps at the representative point. Every cell is one
# single-cell compare run (the harness varies ingestion / sensing parameters per call, not
# per grid axis), seeded with PAPER_REPRESENTATIVE_BASE_SEED so repeat k replays the canonical
# EV = 500 scenario k. The TABLE 3 default cell is shared by the three axes (and is the saturation
# run's control point), so it is run once via ``--axis default``.
IMPAIRMENT_DEFAULT_CELL: dict[str, Any] = {
    "label": "default",
    "ev_ingest_loss_prob": 0.005,
    "ev_ingest_delay_mean_s": 2.0,
    "ev_ingest_delay_jitter_s": 1.0,
    "ev_ingest_delay_max_s": 10.0,
    "matcher_delay_max_s": 10.0,
    "sensing_multiplier": 1.0,
}
IMPAIRMENT_CELLS: dict[str, list[dict[str, Any]]] = {
    "default": [dict(IMPAIRMENT_DEFAULT_CELL)],
    "loss": [
        {**IMPAIRMENT_DEFAULT_CELL, "label": "loss_0.05", "ev_ingest_loss_prob": 0.05},
        {**IMPAIRMENT_DEFAULT_CELL, "label": "loss_0.15", "ev_ingest_loss_prob": 0.15},
        {**IMPAIRMENT_DEFAULT_CELL, "label": "loss_0.30", "ev_ingest_loss_prob": 0.30},
    ],
    "delay": [
        {**IMPAIRMENT_DEFAULT_CELL, "label": "delay_5_2.5_15", "ev_ingest_delay_mean_s": 5.0, "ev_ingest_delay_jitter_s": 2.5, "ev_ingest_delay_max_s": 15.0},
        {**IMPAIRMENT_DEFAULT_CELL, "label": "delay_10_5_30", "ev_ingest_delay_mean_s": 10.0, "ev_ingest_delay_jitter_s": 5.0, "ev_ingest_delay_max_s": 30.0},
        {**IMPAIRMENT_DEFAULT_CELL, "label": "delay_20_10_60", "ev_ingest_delay_mean_s": 20.0, "ev_ingest_delay_jitter_s": 10.0, "ev_ingest_delay_max_s": 60.0},
        # matched watermark: the deployed d_max follows the true ingestion bound (remedy + its latency cost)
        {**IMPAIRMENT_DEFAULT_CELL, "label": "delay_20_10_60_dmax60", "ev_ingest_delay_mean_s": 20.0, "ev_ingest_delay_jitter_s": 10.0, "ev_ingest_delay_max_s": 60.0, "matcher_delay_max_s": 60.0},
    ],
    "noise": [
        {**IMPAIRMENT_DEFAULT_CELL, "label": "noise_x2", "sensing_multiplier": 2.0},
        {**IMPAIRMENT_DEFAULT_CELL, "label": "noise_x4", "sensing_multiplier": 4.0},
        {**IMPAIRMENT_DEFAULT_CELL, "label": "noise_x8", "sensing_multiplier": 8.0},
    ],
}


def _representative_point_kwargs(cell: dict[str, Any], *, repeats: int, base_seed: int, trace_level: str, evse_count: int = 50, ev_count: int = 500) -> dict[str, Any]:
    """Keyword arguments of one representative-point compare run for an impairment cell."""
    m = float(cell.get("sensing_multiplier", 1.0))
    return dict(
        ev_counts=[int(ev_count)],
        repeats=int(repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        pattern_unique_count=min(50, int(evse_count)),
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=list(PAPER_ALGORITHMS),
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[float(cell["matcher_delay_max_s"])],
        candidate_margin_values_s=[120],
        ingestion_enabled=True,
        ev_ingest_delay_mean_s=float(cell["ev_ingest_delay_mean_s"]),
        ev_ingest_delay_jitter_s=float(cell["ev_ingest_delay_jitter_s"]),
        ev_ingest_delay_max_s=float(cell["ev_ingest_delay_max_s"]),
        ev_ingest_loss_prob=float(cell["ev_ingest_loss_prob"]),
        trace_level=str(trace_level),
        evse_count=int(evse_count),
        # sensing multiplier scales the three distortion std parameters jointly (module defaults x m)
        ev_sensor_noise_std_a=float(_algo.EV_SENSOR_NOISE_STD_A) * m,
        ev_sensor_gain_std=float(_algo.EV_SENSOR_GAIN_STD) * m,
        ev_sensor_bias_std_a=float(_algo.EV_SENSOR_BIAS_STD_A) * m,
    )


# The cost-term ablation variants may also be swept, so that the component responsible for a
# matcher's behaviour under an impairment can be named rather than inferred (the sampling
# diagnostic allows the same ids on its own axis).
IMPAIRMENT_ALGORITHMS = list(PAPER_ALGORITHMS) + [
    "single_only_step_off",
    "single_only_dtw_off",
    "nomura_original_interval_hungarian_corr_off",
    "nomura_original_interval_hungarian_dtw_off",
]


def run_impairment_sweep(args: argparse.Namespace) -> int:
    """Impairment sweeps (measurement loss / ingestion delay / sensing distortion), one run per cell."""
    axis = str(getattr(args, "axis", None) or "").strip().lower()
    if axis not in IMPAIRMENT_CELLS:
        raise SystemExit(f"--axis must be one of {', '.join(IMPAIRMENT_CELLS)} (got {axis!r})")
    cells = list(IMPAIRMENT_CELLS[axis])
    cell_index = getattr(args, "cell_index", None)
    if cell_index is not None:
        if not (0 <= int(cell_index) < len(cells)):
            raise SystemExit(f"--cell-index must be in 0..{len(cells) - 1} for axis {axis}")
        cells = [cells[int(cell_index)]]
    base_seed = PAPER_REPRESENTATIVE_BASE_SEED if args.base_seed is None else int(args.base_seed)
    # default stays the four reported matchers; the cost-term variants are opt-in via --algorithms
    _requested = getattr(args, "algorithms", None)
    algorithm_ids = (
        list(PAPER_ALGORITHMS)
        if _requested is None or str(_requested).strip() == ""
        else _parse_algorithm_filter(_requested, list(IMPAIRMENT_ALGORITHMS))
    )
    out: list[dict[str, Any]] = []
    for cell in cells:
        kwargs = _representative_point_kwargs(cell, repeats=int(args.repeats), base_seed=int(base_seed), trace_level=str(args.trace_level))
        kwargs["algorithm_ids"] = list(algorithm_ids)
        kwargs["algorithm_ids"] = list(algorithm_ids)
        result = run_algorithm_compare_report(**kwargs)
        out.append({"axis": axis, "cell": str(cell["label"]), "run_id": str(result.get("run_id"))})
    print(json.dumps({"ok": True, "mode": "impairment-sweep", "axis": axis, "cells": out}, indent=2))
    return 0


# --------------------------------------------------------------------------- Saturation operating points
SATURATION_POINTS: dict[str, dict[str, int]] = {
    "SA": {"ev_count": 2000, "evse_count": 50},   # heavy port contention: ~180 arrivals/h vs ~85/h capacity
    "SB": {"ev_count": 500, "evse_count": 15},    # same demand as the representative point, 15 slots
    "control": {"ev_count": 500, "evse_count": 50},  # = representative point (canonical cell; also the impairment sweeps' default cell)
}


def run_saturation(args: argparse.Namespace) -> int:
    """Saturation operating points where blocking occurs (admitted vs requested accuracy)."""
    point = str(getattr(args, "point", None) or "").strip()
    if point not in SATURATION_POINTS:
        raise SystemExit(f"--point must be one of {', '.join(SATURATION_POINTS)} (got {point!r})")
    spec = SATURATION_POINTS[point]
    base_seed = PAPER_REPRESENTATIVE_BASE_SEED if args.base_seed is None else int(args.base_seed)
    result = run_algorithm_compare_report(
        **_representative_point_kwargs(
            IMPAIRMENT_DEFAULT_CELL,
            repeats=int(args.repeats),
            base_seed=int(base_seed),
            trace_level=str(args.trace_level),
            evse_count=int(spec["evse_count"]),
            ev_count=int(spec["ev_count"]),
        )
    )
    print(json.dumps({"ok": True, "mode": "saturation", "point": point, **spec, "run_id": result.get("run_id")}, indent=2))
    return 0


# --------------------------------------------------------------------------- A3 hyperparameter sensitivity
# One-knob-at-a-time sensitivity of the Posterior matcher's TABLE 2 hyperparameters at the
# representative point, evaluation slice only (EV = 500 repeats 1-20); the TABLE 2 cell itself
# is the canonical representative cell of TABLE 6 and is not re-run.
HYPERPARAM_CELLS: list[dict[str, Any]] = (
    [{"label": f"gamma_prev_{v:g}", "knob": "gamma_prev", "value": v, "a3_posterior_prev_power": v} for v in (0.30, 0.45, 0.75, 0.90)]
    + [{"label": f"eta_mix_{v:g}", "knob": "eta_mix", "value": v, "a3_time_prior_mix": v} for v in (0.15, 0.25, 0.50, 0.70)]
    + [{"label": f"beta_curr_{v:g}", "knob": "beta_curr", "value": v, "a3_current_like_beta": v} for v in (0.50, 0.75, 1.50, 2.00)]
)
HYPERPARAM_DEFAULTS = {"gamma_prev": 0.60, "eta_mix": 0.35, "beta_curr": 1.00}


def run_hyperparam_sweep(args: argparse.Namespace) -> int:
    """A3 hyperparameter sensitivity (gamma_prev, eta_mix, beta_curr), one cell per run."""
    cells = list(HYPERPARAM_CELLS)
    cell_index = getattr(args, "cell_index", None)
    if cell_index is not None:
        if not (0 <= int(cell_index) < len(cells)):
            raise SystemExit(f"--cell-index must be in 0..{len(cells) - 1}")
        cells = [cells[int(cell_index)]]
    base_seed = PAPER_REPRESENTATIVE_BASE_SEED if args.base_seed is None else int(args.base_seed)
    out: list[dict[str, Any]] = []
    for cell in cells:
        kwargs = _representative_point_kwargs(IMPAIRMENT_DEFAULT_CELL, repeats=int(args.repeats), base_seed=int(base_seed), trace_level=str(args.trace_level))
        kwargs["algorithm_ids"] = ["bayesian_windowed"]
        for key in ("a3_posterior_prev_power", "a3_time_prior_mix", "a3_current_like_beta"):
            if key in cell:
                kwargs[key] = float(cell[key])
        result = run_algorithm_compare_report(**kwargs)
        out.append({"cell": str(cell["label"]), "knob": str(cell["knob"]), "value": float(cell["value"]), "run_id": str(result.get("run_id"))})
    print(json.dumps({"ok": True, "mode": "hyperparam-sweep", "cells": out}, indent=2))
    return 0


# --------------------------------------------------------------------------- Dense-sampling diagnostic
# Dense-sampling diagnostic at EV = 500 (d_max 10 s, Δ 120 s): each cell replays the canonical
# Δt_EV cell's scenarios (base seed = canonical scenario counter of that cell + 40 for EV = 500)
# with either the sensing sample noise switched off (control) or the block-mean EV-grid front
# end. Matchers A1 and A3. The canonical ("hold", TABLE 3 sensing) rows are not re-run.
SAMPLING_CELL_BASE_SEED = {5: PAPER_BASE_SEED + 240 + 40, 15: PAPER_BASE_SEED + 780 + 40, 30: PAPER_REPRESENTATIVE_BASE_SEED}
SAMPLING_DIAG_CELLS: list[dict[str, Any]] = [
    {"label": "noisefree_5", "ev_sample_s": 5, "variant": "noise_free"},
    {"label": "noisefree_15", "ev_sample_s": 15, "variant": "noise_free"},
    {"label": "noisefree_30", "ev_sample_s": 30, "variant": "noise_free"},
    {"label": "blockmean_5", "ev_sample_s": 5, "variant": "block_mean"},
    {"label": "blockmean_15", "ev_sample_s": 15, "variant": "block_mean"},
    # Eq. (7) term partition — the Greedy ablation variants
    # (DTW-only = single_only_step_off, step-only = single_only_dtw_off) on the unchanged
    # manuscript front end at Δt_EV ∈ {5, 15, 30} s (canonical Δt_EV seeds), 20 paired repeats.
    {"label": "termpart_5", "ev_sample_s": 5, "variant": "manuscript"},
    {"label": "termpart_15", "ev_sample_s": 15, "variant": "manuscript"},
    {"label": "termpart_30", "ev_sample_s": 30, "variant": "manuscript"},
    # Alignment control: the EV sensor offset is fixed at CHARGER_SAMPLE instead of being drawn over
    # 0-20 s, so every session's samples fall inside the banded-DTW window. Everything else matches
    # termpart_5, on the same seeds, so the two pair repeat by repeat.
    {"label": "offsetfree_5", "ev_sample_s": 5, "variant": "manuscript", "ev_sensor_extra_delay_max_s": 0},
]
SAMPLING_DIAG_ALGORITHMS = [
    "single_only",
    "bayesian_windowed",
    "nomura_original_interval_hungarian",
    "single_only_step_off",
    "single_only_dtw_off",
]


def run_sampling_diagnostic(args: argparse.Namespace) -> int:
    """Dense-sampling diagnostic: noise-free control, block-mean front end and the Eq. (7) term partition at Δt_EV ∈ {5, 15, 30} s."""
    cells = list(SAMPLING_DIAG_CELLS)
    cell_index = getattr(args, "cell_index", None)
    if cell_index is not None:
        if not (0 <= int(cell_index) < len(cells)):
            raise SystemExit(f"--cell-index must be in 0..{len(cells) - 1}")
        cells = [cells[int(cell_index)]]
    algorithm_ids = _parse_algorithm_filter(getattr(args, "algorithms", None), list(SAMPLING_DIAG_ALGORITHMS))
    out: list[dict[str, Any]] = []
    for cell in cells:
        dt = int(cell["ev_sample_s"])
        base_seed = SAMPLING_CELL_BASE_SEED[dt] if args.base_seed is None else int(args.base_seed)
        kwargs = _representative_point_kwargs(IMPAIRMENT_DEFAULT_CELL, repeats=int(args.repeats), base_seed=int(base_seed), trace_level=str(args.trace_level))
        kwargs["algorithm_ids"] = list(algorithm_ids)
        kwargs["ev_sample_values"] = [dt]
        if cell["variant"] == "noise_free":
            kwargs["ev_sensor_noise_std_a"] = 0.0
        elif cell["variant"] == "block_mean":
            kwargs["ev_grid_aggregation"] = "block_mean"
        elif cell["variant"] != "manuscript":
            raise SystemExit(f"unknown sampling-diagnostic variant {cell['variant']!r}")
        if "ev_sensor_extra_delay_max_s" in cell:
            kwargs["ev_sensor_extra_delay_max_s"] = int(cell["ev_sensor_extra_delay_max_s"])
        result = run_algorithm_compare_report(**kwargs)
        out.append({"cell": str(cell["label"]), "ev_sample_s": dt, "variant": str(cell["variant"]), "base_seed": int(base_seed), "run_id": str(result.get("run_id"))})
    print(json.dumps({"ok": True, "mode": "sampling-diagnostic", "cells": out}, indent=2))
    return 0


# --------------------------------------------------------------------------- Response heterogeneity
# Per-session response heterogeneity on the base-load grid: first-step lag ~ U(2, 30) s and ramp
# rate ~ U(1, 10) A/s drawn per admitted session, replayed on the canonical base-load seeds so
# every cell pairs with the canonical/ablation runs.
HETEROGENEITY_LAG_RANGE_S = (2.0, 30.0)
HETEROGENEITY_RAMP_RANGE_A_PER_S = (1.0, 10.0)


def run_heterogeneous_response(args: argparse.Namespace) -> int:
    """Response heterogeneity: base-load grid (EV 100/300/500) with per-session lag and ramp draws, four matchers."""
    base_seed = PAPER_ABLATION_BASE_SEED if args.base_seed is None else int(args.base_seed)
    algorithm_ids = _parse_algorithm_filter(getattr(args, "algorithms", None), list(PAPER_ALGORITHMS))
    result = run_algorithm_compare_report(
        ev_counts=[100, 300, 500],
        repeats=int(args.repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        pattern_unique_count=50,
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=list(algorithm_ids),
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        **PAPER_INGESTION_DEFAULTS,
        trace_level=str(args.trace_level),
        response_lag_range_s=list(HETEROGENEITY_LAG_RANGE_S),
        response_ramp_range_a_per_s=list(HETEROGENEITY_RAMP_RANGE_A_PER_S),
    )
    print(json.dumps({"ok": True, "mode": "heterogeneous-response", "run_id": result.get("run_id"), "lag_range_s": list(HETEROGENEITY_LAG_RANGE_S), "ramp_range_a_per_s": list(HETEROGENEITY_RAMP_RANGE_A_PER_S)}, indent=2))
    return 0


def run_full(args: argparse.Namespace) -> int:
    base_seed = PAPER_BASE_SEED if args.base_seed is None else int(args.base_seed)
    result = run_algorithm_compare_report(
        ev_counts=[100, 300, 500],
        repeats=int(args.repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        pattern_unique_count=50,
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=list(PAPER_ALGORITHMS),
        charger_sample_values=[5],
        ev_sample_values=[5, 15, 30],
        matcher_delay_max_values_s=[5.0, 10.0, 15.0],
        candidate_margin_values_s=[60, 120, 180],
        **PAPER_INGESTION_DEFAULTS,
        trace_level=str(args.trace_level),
    )
    print(json.dumps({"ok": True, "mode": "full", "run_id": result.get("run_id")}, indent=2))
    return 0


# Mode registry: name -> handler(args). New evidence modes register here instead of
# extending the if-chain in main().
# --------------------------------------------------------------------------- Codebook band sweep
# Command set-point band sweep at the representative point: the codebook generator is unchanged
# (its relaxation ladder is never satisfied for 50 patterns, so the operative branch is the
# random-unique draw); narrowing the band {lo..hi} A forces similar commands. B30 is the canonical
# band (rows taken from the canonical summary; runnable here only for the pairing check).
CODEBOOK_BAND_CELLS: dict[str, tuple[int, int]] = {"B30": (6, 30), "B20": (6, 20), "B16": (8, 16), "B12": (8, 12)}
CODEBOOK_BAND_ALGORITHMS = ["single_only", "nomura_original_interval_hungarian", "bayesian_windowed"]


def run_codebook_band_sweep(args: argparse.Namespace) -> int:
    """Set-point band sweep (EV = 500, canonical representative seeds, A1/A2/A3), one cell per --cell-index."""
    labels = list(CODEBOOK_BAND_CELLS)
    cell_index = getattr(args, "cell_index", None)
    if cell_index is not None:
        if not (0 <= int(cell_index) < len(labels)):
            raise SystemExit(f"--cell-index must be in 0..{len(labels) - 1}")
        labels = [labels[int(cell_index)]]
    else:
        labels = [l for l in labels if l != "B30"]
    algorithm_ids = _parse_algorithm_filter(getattr(args, "algorithms", None), list(CODEBOOK_BAND_ALGORITHMS))
    base_seed = PAPER_REPRESENTATIVE_BASE_SEED if args.base_seed is None else int(args.base_seed)
    out: list[dict[str, Any]] = []
    for label in labels:
        band = CODEBOOK_BAND_CELLS[label]
        kwargs = _representative_point_kwargs(IMPAIRMENT_DEFAULT_CELL, repeats=int(args.repeats), base_seed=int(base_seed), trace_level=str(args.trace_level))
        kwargs["algorithm_ids"] = list(algorithm_ids)
        kwargs["command_setpoint_range"] = [int(band[0]), int(band[1])]
        result = run_algorithm_compare_report(**kwargs)
        out.append({"cell": label, "setpoint_range_a": list(band), "base_seed": int(base_seed), "run_id": str(result.get("run_id"))})
    print(json.dumps({"ok": True, "mode": "codebook-band-sweep", "cells": out}, indent=2))
    return 0


MODES: dict[str, Callable[[argparse.Namespace], int]] = {
    "verify-existing": lambda args: verify_existing(),
    "figures": lambda args: regenerate_figures(),
    "smoke": run_smoke,
    "ablation": run_ablation,
    "full": run_full,
    # sweep and diagnostic modes
    "static-limit": run_static_limit,
    "codebook-band-sweep": run_codebook_band_sweep,
    "trace-representative": run_trace_representative,
    "impairment-sweep": run_impairment_sweep,
    "saturation": run_saturation,
    "hyperparam-sweep": run_hyperparam_sweep,
    "sampling-diagnostic": run_sampling_diagnostic,
    "heterogeneous-response": run_heterogeneous_response,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="verify-existing",
        help=(
            "Action to run. Full mode reruns the complete manuscript sweep; "
            "ablation mode reruns the Section VI-C ablation variants (manuscript Figure 8)."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help=(
            f"Scenario base seed. Defaults to {PAPER_BASE_SEED} "
            f"({PAPER_ABLATION_BASE_SEED} in ablation mode, which replays the "
            "canonical base-load scenario realizations)."
        ),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--smoke-ev-count", type=int, default=20)
    parser.add_argument("--smoke-repeats", type=int, default=1)
    parser.add_argument("--trace-level", choices=["summary", "full"], default="summary")
    parser.add_argument(
        "--cell",
        choices=["ideal", "paper", "both"],
        default="both",
        help="static-limit mode only: which sensing cell(s) to run (default both).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help=(
            "Ablation / static-limit / trace-representative modes: comma-separated subset of the "
            "mode's algorithm ids to run (default: all). Example: two_prefix_greedy,two_prefix_hungarian"
        ),
    )
    parser.add_argument("--axis", choices=list(IMPAIRMENT_CELLS), default=None, help="impairment-sweep mode: which axis to run (default = the shared TABLE 3 cell).")
    parser.add_argument("--cell-index", type=int, default=None, help="impairment-sweep / hyperparam-sweep modes: run only the i-th cell (one process per cell).")
    parser.add_argument("--point", choices=list(SATURATION_POINTS), default=None, help="saturation mode: SA (EV 2000 / 50 slots), SB (EV 500 / 15 slots) or control (EV 500 / 50 slots).")
    args = parser.parse_args()

    handler = MODES.get(str(args.mode))
    if handler is None:
        raise AssertionError(f"Unhandled mode: {args.mode}")
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
