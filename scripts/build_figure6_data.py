#!/usr/bin/env python3
"""Build the figure6 data table (manuscript Figure 8) from the committed ablation reproduction runs.

Aggregates the per-run results (``raw_runs.csv``) of the Section VI-C ablation
run (produced by ``python scripts/reproduce_all.py --mode ablation`` on the
canonical base-load scenario seeds) and of the two-prefix baseline run
(``--mode ablation --algorithms two_prefix_greedy,two_prefix_hungarian``, same
seeds; two-prefix baseline) into the plotting table consumed by
``generate_figure6_ablation_summary()``:

    case_study/scalability_analysis/accuracy/figure6_ablation_simulated.csv

Aggregation follows the calibration/evaluation protocol of TABLE 5 in the
manuscript: EV in {100, 500} average repeats 1-20, while EV = 300 averages the
held-out evaluation repeats 6-20 (repeats 1-5 are the hyperparameter
calibration slice and are excluded from every reported number).

Three integrity gates run before anything is written:

* every (variant, EV) cell must aggregate exactly the protocol repeat count
  (20 at EV in {100, 500}; 15 held-out repeats at EV = 300),
* the three full-design rows (A1/A2/A3) must equal the canonical run's
  ``raw_runs.csv`` aggregated under the same protocol exactly, which proves
  that the ablation run replayed the identical scenario realizations, and
* every (EV, repeat) scenario of the two-prefix run carries the same
  ``scenario_hash`` as the committed ablation run, so the two-prefix rows are
  paired with every existing series of the figure.

Coinciding curves in the Posterior panel (the time-prior-removed variant stays
within the repeat confidence interval of the full design, and all-removed ~=
without-carry-over within 0.1 pt) are separated by a small horizontal marker
offset (``dodge`` column, in EV-session units): lines are drawn at the true x
positions, only the markers are displaced, and the offset is declared in the
Figure 8 caption.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
ABLATION_RUN_ID = "20260909_112655_937821"
# Two-prefix baselines run on the same base-load scenario seeds.
TWO_PREFIX_RUN_ID = "20260908_170955_253026"
ABLATION_RAW = ACCURACY_DIR / f"ablation_run_{ABLATION_RUN_ID}" / "raw_runs.csv"
TWO_PREFIX_RAW = ACCURACY_DIR / f"ablation_two_prefix_run_{TWO_PREFIX_RUN_ID}" / "raw_runs.csv"
CANONICAL_RAW = ACCURACY_DIR / "canonical_run_20260907_111056_515561" / "raw_runs.csv"
OUT_CSV = ACCURACY_DIR / "figure6_ablation_simulated.csv"

EV_COUNTS = (100, 300, 500)
HELD_OUT_MIN_REPEAT = {100: 1, 300: 6, 500: 1}  # TABLE 5: EV=300 -> repeats 6-20
PROTOCOL_RUNS = {100: 20, 300: 15, 500: 20}

# (algorithm_id, family, legend label, marker, linestyle, dodge, markerfill)
# CSV row order == draw order. In the Posterior panel the band anchors
# (full; without carry-over) keep their lines, every variant keeps its own
# linestyle (coinciding lines overlap), and markers are dodged horizontally.
SERIES: list[tuple[str, str, str, str, str, float, str]] = [
    ("single_only", "Greedy", "Greedy (full)", "o", "-", 0.0, "filled"),
    ("single_only_step_off", "Greedy", "without step-signature", "s", "-.", 0.0, "open"),
    ("single_only_dtw_off", "Greedy", "without DTW", "^", "--", 0.0, "open"),
    ("single_only_all_off", "Greedy", "all terms removed", "D", ":", 0.0, "open"),
    # the two two-prefix baselines coincide exactly (same per-repeat accuracies), so their
    # markers are dodged like the coinciding Posterior-panel variants (declared in the caption)
    ("two_prefix_greedy", "Greedy", "two-prefix greedy", "v", "--", -9.0, "open"),
    ("two_prefix_hungarian", "Greedy", "two-prefix Hungarian", "P", "-.", 9.0, "open"),
    ("nomura_original_interval_hungarian", "Global", "Global (full)", "o", "-", 0.0, "filled"),
    ("nomura_original_interval_hungarian_corr_off", "Global", "without correlation", "s", "-.", 0.0, "open"),
    ("nomura_original_interval_hungarian_dtw_off", "Global", "without DTW", "^", "--", 0.0, "open"),
    ("nomura_original_interval_hungarian_all_off", "Global", "all terms removed", "D", ":", 0.0, "open"),
    ("bayesian_windowed", "Posterior", "Posterior (full)", "o", "-", -9.0, "filled"),
    ("bayesian_windowed_timeprior_off", "Posterior", "without time prior", "D", ":", 9.0, "open"),
    ("bayesian_windowed_memory_off", "Posterior", "without carry-over", "s", "-.", -9.0, "open"),
    ("bayesian_windowed_all_off", "Posterior", "all components removed", "v", ":", 9.0, "open"),
]

FAMILY_COLORS = {"Greedy": "#ff7f0e", "Global": "#2ca02c", "Posterior": "#1f77b4"}
FULL_ANCHORS = ("single_only", "nomura_original_interval_hungarian", "bayesian_windowed")

BASE_LOAD = dict(ev_sample_s=30, matcher_delay_max_s=10.0, candidate_margin_s=120)


def _protocol_cell(raw: pd.DataFrame, algorithm_id: str, ev_count: int) -> tuple[float, int]:
    m = (raw["algorithm_id"] == algorithm_id) & (raw["ev_count"] == ev_count)
    for k, v in BASE_LOAD.items():
        m &= raw[k] == v
    m &= raw["repeat_idx"] >= HELD_OUT_MIN_REPEAT[ev_count]
    rows = raw[m]
    return float(rows["accuracy"].mean()), int(len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-raw", type=Path, default=ABLATION_RAW)
    parser.add_argument("--two-prefix-raw", type=Path, default=TWO_PREFIX_RAW)
    args = parser.parse_args()

    ablation_only = pd.read_csv(args.ablation_raw)
    two_prefix = pd.read_csv(args.two_prefix_raw)
    canonical = pd.read_csv(CANONICAL_RAW)

    # Gate 3: the two-prefix run must replay the committed ablation scenarios exactly
    # (same scenario_hash for every (EV, repeat)), so its rows pair with every series.
    ref_hash = (
        ablation_only[ablation_only["algorithm_id"] == "single_only"]
        .set_index(["ev_count", "repeat_idx"])["scenario_hash"]
    )
    for (ev_count, repeat_idx), grp in two_prefix.groupby(["ev_count", "repeat_idx"]):
        hashes = set(grp["scenario_hash"])
        if len(hashes) != 1 or hashes != {ref_hash.loc[(int(ev_count), int(repeat_idx))]}:
            raise AssertionError(
                f"two-prefix run EV={ev_count} repeat={repeat_idx}: scenario_hash differs from the committed ablation run"
            )
    ablation = pd.concat([ablation_only, two_prefix], ignore_index=True)

    # Gate 1: protocol repeat counts for every plotted cell.
    for algorithm_id, *_rest in SERIES:
        for ev_count in EV_COUNTS:
            _, n = _protocol_cell(ablation, algorithm_id, ev_count)
            if n != PROTOCOL_RUNS[ev_count]:
                raise AssertionError(
                    f"{algorithm_id} EV={ev_count}: runs={n} != {PROTOCOL_RUNS[ev_count]}"
                )

    # Gate 2: full-design rows must reproduce the canonical protocol means
    # exactly, proving the ablation run replayed the canonical scenario seeds.
    for algorithm_id in FULL_ANCHORS:
        for ev_count in EV_COUNTS:
            got, _ = _protocol_cell(ablation, algorithm_id, ev_count)
            want, _ = _protocol_cell(canonical, algorithm_id, ev_count)
            if abs(got - want) > 1e-9:
                raise AssertionError(
                    f"{algorithm_id} EV={ev_count}: ablation mean {got} != canonical {want}"
                )

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["matcher_family", "variant", "marker", "linestyle", "color",
                        "alpha", "dodge", "markerfill", "ev_count", "accuracy"],
        )
        writer.writeheader()
        for algorithm_id, family, variant, marker, linestyle, dodge, markerfill in SERIES:
            alpha = 1.0 if variant.endswith("(full)") else 0.85
            for ev_count in EV_COUNTS:
                acc, _ = _protocol_cell(ablation, algorithm_id, ev_count)
                writer.writerow(
                    dict(
                        matcher_family=family,
                        variant=variant,
                        marker=marker,
                        linestyle=linestyle,
                        color=FAMILY_COLORS[family],
                        alpha=alpha,
                        dodge=dodge,
                        markerfill=markerfill,
                        ev_count=ev_count,
                        accuracy=round(acc * 100.0, 2),
                    )
                )
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
