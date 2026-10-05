#!/usr/bin/env python3
"""Build the figure7 data table (manuscript Figure 5, impairment sweeps) from the committed impairment-sweep runs.

Each impairment cell is one committed single-cell compare run at the representative point
(EV = 500, 50 slots, Δt_EV 30 s, Δ 120 s; canonical scenario seeds 2187632432–2451; four
matchers; 20 paired repeats), produced by

    python scripts/reproduce_all.py --mode impairment-sweep --axis {default,loss,delay,noise}

The shared TABLE 3 default cell appears on every axis as its benign anchor. Output:

    case_study/scalability_analysis/accuracy/figure7_impairment_sweeps.csv

with one row per (axis, cell, matcher): x value (loss probability / mean delay in s /
sensing multiplier), the watermark bound, accuracy mean ± 95 % CI, requested-session accuracy,
p90 latency and the run id. Gates: every cell carries exactly the requested repeat count for
all four matchers; the default cell reproduces the canonical representative-point means
(0.7121 / 0.9732 / 0.9993 / 0.9838; p90 416.035) to 1e-6.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
OUT_CSV = ACCURACY_DIR / "figure7_impairment_sweeps.csv"
ALGORITHMS = ["time_only_baseline", "single_only", "nomura_original_interval_hungarian", "bayesian_windowed"]
CANONICAL = {"time_only_baseline": 0.7121, "single_only": 0.9732, "nomura_original_interval_hungarian": 0.9993, "bayesian_windowed": 0.9838}
CANONICAL_P90 = 416.035

# Committed run directories (folder names under ACCURACY_DIR), filled in as the runs land.
# The default cell is shared by the three axes.
IMPAIRMENT_RUN_DIRS: dict[str, str | None] = {
    "default": "impairment_default_run_20260910_091432_644397",
    "loss_0.05": "impairment_loss_0.05_run_20260907_111056_514449",
    "loss_0.15": "impairment_loss_0.15_run_20260907_132904_132323",
    "loss_0.30": "impairment_loss_0.30_run_20260907_153237_104004",
    "delay_5_2.5_15": "impairment_delay_5_2.5_15_run_20260907_111056_518615",
    "delay_10_5_30": "impairment_delay_10_5_30_run_20260907_132912_115378",
    "delay_20_10_60": "impairment_delay_20_10_60_run_20260907_153251_486330",
    "delay_20_10_60_dmax60": "impairment_delay_20_10_60_dmax60_run_20260907_174315_064624",
    "noise_x2": "impairment_noise_x2_run_20260908_170955_253038",
    "noise_x4": "impairment_noise_x4_run_20260908_194258_023727",
    "noise_x8": "impairment_noise_x8_run_20260908_214145_238517",
}
AXIS_OF_CELL = {
    "default": ("all", 0.0, False),
    "loss_0.05": ("loss", 0.05, False),
    "loss_0.15": ("loss", 0.15, False),
    "loss_0.30": ("loss", 0.30, False),
    "delay_5_2.5_15": ("delay", 5.0, False),
    "delay_10_5_30": ("delay", 10.0, False),
    "delay_20_10_60": ("delay", 20.0, False),
    "delay_20_10_60_dmax60": ("delay", 20.0, True),
    "noise_x2": ("noise", 2.0, False),
    "noise_x4": ("noise", 4.0, False),
    "noise_x8": ("noise", 8.0, False),
}
DEFAULT_X = {"loss": 0.005, "delay": 2.0, "noise": 1.0}
COLUMNS = [
    "axis", "cell", "x_value", "matched_dmax", "matcher_delay_max_s", "ev_ingest_loss_prob", "ev_ingest_delay_mean_s",
    "ev_ingest_delay_max_s", "sensing_multiplier", "algorithm_id", "runs", "accuracy_mean", "accuracy_ci95",
    "accuracy_requested_mean", "p90_latency_mean", "p90_latency_ci95", "run_id",
]


def _summary(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    return pd.read_csv(run_dir / "summary.csv"), json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def build_rows(run_dirs: dict[str, Path], expected_repeats: int, tol: float = 1e-6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell, run_dir in run_dirs.items():
        summary, cfg = _summary(run_dir)
        axis, x_value, matched = AXIS_OF_CELL[cell]
        multiplier = round(float(cfg["ev_sensor_noise_std_a"]) / 0.20, 6)
        # the default cell is copied onto every axis at that axis' default x
        targets = [(a, DEFAULT_X[a]) for a in ("loss", "delay", "noise")] if axis == "all" else [(axis, x_value)]
        for algorithm_id in ALGORITHMS:
            sel = summary[summary["algorithm_id"] == algorithm_id]
            if len(sel) != 1:
                raise AssertionError(f"{run_dir.name}: expected one summary row for {algorithm_id}, found {len(sel)}")
            r = sel.iloc[0]
            if int(r["runs"]) != int(expected_repeats):
                raise AssertionError(f"{run_dir.name} {algorithm_id}: runs={int(r['runs'])} != {expected_repeats}")
            if cell == "default":
                if abs(float(r["accuracy_mean"]) - CANONICAL[algorithm_id]) > tol or abs(float(r["p90_latency_mean"]) - CANONICAL_P90) > tol:
                    raise AssertionError(f"default cell does not reproduce the canonical representative point for {algorithm_id}")
            for a, x in targets:
                rows.append(
                    {
                        "axis": a,
                        "cell": cell,
                        "x_value": float(x),
                        "matched_dmax": bool(matched),
                        "matcher_delay_max_s": float(cfg["matcher_delay_max_values_s"][0]),
                        "ev_ingest_loss_prob": float(cfg["ev_ingest_loss_prob"]),
                        "ev_ingest_delay_mean_s": float(cfg["ev_ingest_delay_mean_s"]),
                        "ev_ingest_delay_max_s": float(cfg["ev_ingest_delay_max_s"]),
                        "sensing_multiplier": multiplier,
                        "algorithm_id": algorithm_id,
                        "runs": int(r["runs"]),
                        "accuracy_mean": round(float(r["accuracy_mean"]), 6),
                        "accuracy_ci95": round(float(r["accuracy_ci95"]), 6),
                        "accuracy_requested_mean": round(float(r["accuracy_requested_mean"]), 6),
                        "p90_latency_mean": round(float(r["p90_latency_mean"]), 3),
                        "p90_latency_ci95": round(float(r["p90_latency_ci95"]), 3),
                        "run_id": run_dir.name.rsplit("_run_", 1)[-1],
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="override: <cell>=<run_dir> (repeatable)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--allow-missing", action="store_true", help="tabulate only the cells whose runs exist")
    args = parser.parse_args()

    run_dirs: dict[str, Path] = {}
    for cell, name in IMPAIRMENT_RUN_DIRS.items():
        if name:
            run_dirs[cell] = ACCURACY_DIR / name
    for item in args.run:
        cell, _, path = item.partition("=")
        if cell not in AXIS_OF_CELL:
            raise SystemExit(f"unknown cell {cell!r}; known: {', '.join(AXIS_OF_CELL)}")
        run_dirs[cell] = Path(path)
    missing = [c for c in AXIS_OF_CELL if c not in run_dirs]
    if missing and not args.allow_missing:
        raise SystemExit(f"no run directory for cells: {missing} (pass --run <cell>=<dir> or fill IMPAIRMENT_RUN_DIRS)")

    rows = build_rows(run_dirs, int(args.repeats))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows, {len(run_dirs)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
