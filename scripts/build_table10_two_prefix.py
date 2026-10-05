#!/usr/bin/env python3
"""Tabulate the two-prefix baselines into `table10_two_prefix_ablation.csv` (TABLE 9 of the manuscript).

Reads the committed two-prefix run (``two_prefix_greedy``, ``two_prefix_hungarian``; produced by
``python scripts/reproduce_all.py --mode ablation --algorithms two_prefix_greedy,two_prefix_hungarian``
on the canonical base-load scenario seeds) together with the committed ablation run (A1 full,
A3 full, A3 without carry-over) and writes

    case_study/scalability_analysis/accuracy/table10_two_prefix_ablation.csv

one row per (variant, EV count) with the TABLE 5 protocol mean (EV 100/500: repeats 1–20;
EV 300: held-out repeats 6–20), its 95 % CI, the reference means of A1 / A3-without-carry-over /
A3, and the *paired* differences (same scenario realisations, same repeat index) against A3 and
A1 with their 95 % CIs. Gate: every (EV, repeat) of the two-prefix run must carry the same
``scenario_hash`` as the ablation run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
ABLATION_RAW = ACCURACY_DIR / "ablation_run_20260909_112655_937821" / "raw_runs.csv"
TWO_PREFIX_RAW = ACCURACY_DIR / "ablation_two_prefix_run_20260908_170955_253026" / "raw_runs.csv"
OUT_CSV = ACCURACY_DIR / "table10_two_prefix_ablation.csv"

EV_COUNTS = (100, 300, 500)
HELD_OUT_MIN_REPEAT = {100: 1, 300: 6, 500: 1}
BASE_LOAD = dict(ev_sample_s=30, matcher_delay_max_s=10.0, candidate_margin_s=120)
VARIANTS = ["two_prefix_greedy", "two_prefix_hungarian"]
REFERENCES = {"a1": "single_only", "a3_memory_off": "bayesian_windowed_memory_off", "a3": "bayesian_windowed"}
COLUMNS = [
    "variant", "ev_count", "runs", "accuracy_mean", "accuracy_ci95",
    "a1_mean", "a3_memory_off_mean", "a3_mean",
    "diff_vs_a3_mean", "diff_vs_a3_ci95", "diff_vs_a1_mean", "diff_vs_a1_ci95", "diff_vs_a3_memory_off_mean", "diff_vs_a3_memory_off_ci95",
]


def _cell(raw: pd.DataFrame, algorithm_id: str, ev_count: int) -> np.ndarray:
    m = (raw["algorithm_id"] == algorithm_id) & (raw["ev_count"] == ev_count)
    for k, v in BASE_LOAD.items():
        m &= raw[k] == v
    m &= raw["repeat_idx"] >= HELD_OUT_MIN_REPEAT[ev_count]
    rows = raw[m].sort_values("repeat_idx")
    return rows["accuracy"].to_numpy(dtype=float)


def _ci95(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-raw", type=Path, default=ABLATION_RAW)
    parser.add_argument("--two-prefix-raw", type=Path, default=TWO_PREFIX_RAW)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    ablation = pd.read_csv(args.ablation_raw)
    two_prefix = pd.read_csv(args.two_prefix_raw)
    ref_hash = ablation[ablation["algorithm_id"] == "single_only"].set_index(["ev_count", "repeat_idx"])["scenario_hash"]
    for r in two_prefix.itertuples():
        if ref_hash.loc[(int(r.ev_count), int(r.repeat_idx))] != r.scenario_hash:
            raise AssertionError(f"two-prefix run EV={r.ev_count} repeat={r.repeat_idx}: scenario_hash differs from the ablation run")

    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for ev_count in EV_COUNTS:
            x = _cell(two_prefix, variant, ev_count)
            refs = {k: _cell(ablation, aid, ev_count) for k, aid in REFERENCES.items()}
            if len(x) == 0 or any(len(v) != len(x) for v in refs.values()):
                raise AssertionError(f"{variant} EV={ev_count}: unequal or empty protocol cells")
            d3 = refs["a3"] - x
            d1 = x - refs["a1"]
            dm = x - refs["a3_memory_off"]
            rows.append(
                {
                    "variant": variant,
                    "ev_count": int(ev_count),
                    "runs": int(len(x)),
                    "accuracy_mean": round(float(x.mean()), 6),
                    "accuracy_ci95": round(_ci95(x), 6),
                    "a1_mean": round(float(refs["a1"].mean()), 6),
                    "a3_memory_off_mean": round(float(refs["a3_memory_off"].mean()), 6),
                    "a3_mean": round(float(refs["a3"].mean()), 6),
                    "diff_vs_a3_mean": round(float(d3.mean()), 6),
                    "diff_vs_a3_ci95": round(_ci95(d3), 6),
                    "diff_vs_a1_mean": round(float(d1.mean()), 6),
                    "diff_vs_a1_ci95": round(_ci95(d1), 6),
                    "diff_vs_a3_memory_off_mean": round(float(dm.mean()), 6),
                    "diff_vs_a3_memory_off_ci95": round(_ci95(dm), 6),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    for r in rows:
        print(
            f"  {r['variant']:22s} EV={r['ev_count']:4d} acc={r['accuracy_mean']:.4f} ± {r['accuracy_ci95']:.4f} | A1 {r['a1_mean']:.4f} "
            f"A3-memoff {r['a3_memory_off_mean']:.4f} A3 {r['a3_mean']:.4f} | A3 − variant = {r['diff_vs_a3_mean']:+.4f} ± {r['diff_vs_a3_ci95']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
