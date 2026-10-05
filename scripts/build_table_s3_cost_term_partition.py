#!/usr/bin/env python3
"""Build the cost-term partition table (TABLE S3 of the Supplementary Material).

Each matcher's cost is evaluated with one term removed, at three sensing-distortion levels, so the
component responsible for a matcher's behaviour under distortion can be named rather than inferred:

  Greedy   full = step-median + banded DTW · DTW-only (``single_only_step_off``) · step-only (``single_only_dtw_off``)
  Global   full = correlation x DTW        · no-corr  (``..._corr_off``)          · no-DTW  (``..._dtw_off``)

x1 comes from the base-load ablation run (EV = 500, the canonical representative cell); x4 and x8 come
from the two cost-term impairment runs on the same seeds. Writes
``case_study/scalability_analysis/accuracy/table_s3_cost_term_partition.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
OUT_CSV = ACCURACY_DIR / "table_s3_cost_term_partition.csv"

RUN_DIRS = {
    1.0: "ablation_run_20260909_112655_937821",
    4.0: "impairment_noise_x4_costterms_run_20260909_112756_275599",
    8.0: "impairment_noise_x8_costterms_run_20260909_112756_282217",
}
# the full-cost rows of the two distortion cells live in the plain noise-axis runs
FULL_COST_RUN_DIRS = {
    4.0: "impairment_noise_x4_run_20260908_194258_023727",
    8.0: "impairment_noise_x8_run_20260908_214145_238517",
}
VARIANTS = [
    ("single_only", "Greedy, full cost"),
    ("single_only_step_off", "Greedy, banded DTW only"),
    ("single_only_dtw_off", "Greedy, step median only"),
    ("nomura_original_interval_hungarian", "Global, full cost"),
    ("nomura_original_interval_hungarian_corr_off", "Global, without correlation"),
    ("nomura_original_interval_hungarian_dtw_off", "Global, without DTW"),
]
EV_COUNT = 500
COLUMNS = ["sensing_multiplier", "algorithm_id", "variant_label", "runs", "accuracy_mean", "accuracy_ci95", "run_id"]


def _cell(multiplier: float, algorithm_id: str) -> tuple[np.ndarray, str]:
    """Per-repeat accuracies of one variant at one distortion level, and the run they came from."""
    names = [RUN_DIRS[multiplier]]
    if multiplier in FULL_COST_RUN_DIRS:
        names.append(FULL_COST_RUN_DIRS[multiplier])
    for name in names:
        raw = pd.read_csv(ACCURACY_DIR / name / "raw_runs.csv")
        sel = raw[(raw["algorithm_id"] == algorithm_id) & (raw["ev_count"] == EV_COUNT)]
        if len(sel):
            return sel.sort_values("repeat_idx")["accuracy"].to_numpy(dtype=float), name.rsplit("_run_", 1)[-1]
    raise SystemExit(f"no rows for {algorithm_id} at sensing x{multiplier:g}")


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for multiplier in (1.0, 4.0, 8.0):
        for algorithm_id, label in VARIANTS:
            values, run_id = _cell(multiplier, algorithm_id)
            if len(values) != 20:
                raise AssertionError(f"x{multiplier:g} {algorithm_id}: {len(values)} repeats, expected 20")
            half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            rows.append({
                "sensing_multiplier": multiplier,
                "algorithm_id": algorithm_id,
                "variant_label": label,
                "runs": len(values),
                "accuracy_mean": float(values.mean()),
                "accuracy_ci95": half,
                "run_id": run_id,
            })
    # the two ablation arms must bracket the full cost at every level
    for multiplier in (1.0, 4.0, 8.0):
        at = {r["algorithm_id"]: r["accuracy_mean"] for r in rows if r["sensing_multiplier"] == multiplier}
        if not at["single_only_step_off"] > at["single_only"] > at["single_only_dtw_off"]:
            raise AssertionError(f"x{multiplier:g}: the Greedy arms do not bracket the full cost")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    df = pd.DataFrame(build_rows(), columns=COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} rows)")
    for multiplier in (1.0, 4.0, 8.0):
        sub = df[df["sensing_multiplier"] == multiplier]
        print(f"  x{multiplier:g}: " + " ".join(f"{r.accuracy_mean:.3f}" for r in sub.itertuples()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
