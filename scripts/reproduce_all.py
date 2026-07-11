#!/usr/bin/env python3
"""Verify or reproduce the IEEE Access Section VI result grid.

Default mode is intentionally lightweight and verifies that the published
aggregate CSV supports the headline paper claims. Use ``--mode full`` for the
complete 6480-run sweep used by the manuscript, and ``--mode ablation`` to
re-simulate the Section VI-C / Figure 6 ablation variants on the base-load grid.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_study.section_vi_reproduction import (  # noqa: E402
    PAPER_ABLATION_ALGORITHM_IDS,
    run_algorithm_compare_report,
)
from manuscript_figure_generation import (  # noqa: E402
    PAPER_FIGURE_FILES,
    generate_figure3_base_load_comparison,
    generate_figure4_sampling_sensitivity,
    generate_figure5_posterior_diagnostic,
    generate_figure6_ablation_summary,
)

PAPER_RUN_ID = "20260626_170431_105970"
PAPER_ABLATION_RUN_ID = "20260711_164832_170489"
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
PAPER_ALGORITHMS = [
    "time_only_baseline",
    "single_only",
    "nomura_original_interval_hungarian",
    "bayesian_windowed",
]


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
        "single_only": 0.9374,
        "nomura_original_interval_hungarian": 0.8952,
        "bayesian_windowed": 0.9695,
    }
    for algorithm_id, expected in expected_default.items():
        row = _row(summary, algorithm_id=algorithm_id, **default)
        add_check(f"default_500_accuracy_{algorithm_id}", row["accuracy_mean"], expected, 1e-6)
        add_check(f"default_500_p90_latency_{algorithm_id}", row["p90_latency_mean"], 416.035, 1e-6)

    for ev_count, expected in [(100, 0.9930), (300, 0.9810), (500, 0.9695)]:
        row = _row(
            summary,
            algorithm_id="bayesian_windowed",
            ev_count=ev_count,
            ev_sample_s=30,
            matcher_delay_max_s=10.0,
            candidate_margin_s=120,
        )
        add_check(f"a3_base_load_accuracy_ev_{ev_count}", row["accuracy_mean"], expected, 5e-6)

    for ev_sample_s, expected in [(5, 0.9533), (15, 0.9579), (30, 0.9695)]:
        row = _row(
            summary,
            algorithm_id="bayesian_windowed",
            ev_count=500,
            ev_sample_s=ev_sample_s,
            matcher_delay_max_s=10.0,
            candidate_margin_s=120,
        )
        add_check(f"a3_sampling_accuracy_{ev_sample_s}s", row["accuracy_mean"], expected, 1e-6)

    # Section VI-C / Figure 6 ablation claims (committed ablation run on the
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
        # Carry-over is the decisive component: removing it collapses A3 to the
        # greedy-baseline level at the representative point.
        add_check("ablation_carry_over_off_500", _acc(ablation, "bayesian_windowed_memory_off", 500), 0.9376, 1e-6)
        add_check("ablation_a3_all_off_500", _acc(ablation, "bayesian_windowed_all_off", 500), 0.9374, 1e-6)
    else:
        ablation = None

    causal_ok = (
        "candidate_query_lead_max_s_mean" in summary.columns
        and "candidate_future_query_violation_count_mean" in summary.columns
        and float(summary["candidate_query_lead_max_s_mean"].max()) <= 0.0
        and float(summary["candidate_future_query_violation_count_mean"].max()) == 0.0
    )
    shape_ok = tuple(summary.shape) == (324, 128)
    raw_ok = raw_rows in (None, 6480)
    ok = bool(shape_ok and raw_ok and causal_ok and ablation_ok and all(c["ok"] for c in checks))
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
        ingestion_enabled=True,
        ev_ingest_delay_mean_s=2.0,
        ev_ingest_delay_jitter_s=1.0,
        ev_ingest_delay_max_s=10.0,
        ev_ingest_loss_prob=0.005,
        trace_level="summary",
    )
    print(json.dumps({"ok": True, "mode": "smoke", "run_id": result.get("run_id")}, indent=2))
    return 0


def run_ablation(args: argparse.Namespace) -> int:
    base_seed = PAPER_ABLATION_BASE_SEED if args.base_seed is None else int(args.base_seed)
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
        algorithm_ids=list(PAPER_ABLATION_ALGORITHM_IDS),
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        ingestion_enabled=True,
        ev_ingest_delay_mean_s=2.0,
        ev_ingest_delay_jitter_s=1.0,
        ev_ingest_delay_max_s=10.0,
        ev_ingest_loss_prob=0.005,
        trace_level=str(args.trace_level),
    )
    print(json.dumps({"ok": True, "mode": "ablation", "run_id": result.get("run_id")}, indent=2))
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
        ingestion_enabled=True,
        ev_ingest_delay_mean_s=2.0,
        ev_ingest_delay_jitter_s=1.0,
        ev_ingest_delay_max_s=10.0,
        ev_ingest_loss_prob=0.005,
        trace_level=str(args.trace_level),
    )
    print(json.dumps({"ok": True, "mode": "full", "run_id": result.get("run_id")}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["verify-existing", "figures", "smoke", "ablation", "full"],
        default="verify-existing",
        help=(
            "Action to run. Full mode reruns the complete manuscript sweep; "
            "ablation mode reruns the Section VI-C / Figure 6 ablation variants."
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
    args = parser.parse_args()

    if args.mode == "verify-existing":
        return verify_existing()
    if args.mode == "figures":
        return regenerate_figures()
    if args.mode == "smoke":
        return run_smoke(args)
    if args.mode == "ablation":
        return run_ablation(args)
    if args.mode == "full":
        return run_full(args)
    raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
