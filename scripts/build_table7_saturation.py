#!/usr/bin/env python3
"""Tabulate the saturation operating points into TABLE 7.

Reads the committed runs produced by ``python scripts/reproduce_all.py --mode saturation --point
{SA,SB,control}`` — S-A (EV 2000 requested sessions, 50 slots), S-B (EV 500, 15 slots) and the
control (EV 500, 50 slots = the canonical representative point; the same run serves as the impairment sweeps'
default cell) — and writes

    case_study/scalability_analysis/accuracy/table7_saturation.csv

with one row per (point, matcher): admitted-session accuracy, requested-session accuracy,
blocked ratio, fallback-finalised ratio, fallback-to-all activations, mean candidate-set size,
ground-truth-in-candidates ratio, p90 latency and total runtime (means ± 95 % CI over the 20
paired repeats). Gates: the control reproduces the canonical means (accuracy 0.7121 / 0.9732 /
0.9993 / 0.9838, p90 416.035) to 1e-6; S-A and S-B have blocked_ratio_mean > 0 and
accuracy_requested_mean < accuracy_mean for every matcher.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
OUT_CSV = ACCURACY_DIR / "table7_saturation.csv"
ALGORITHMS = ["time_only_baseline", "single_only", "nomura_original_interval_hungarian", "bayesian_windowed"]
CANONICAL = {"time_only_baseline": 0.7121, "single_only": 0.9732, "nomura_original_interval_hungarian": 0.9993, "bayesian_windowed": 0.9838}
POINT_SPEC = {"SA": (2000, 50), "SB": (500, 15), "control": (500, 50)}
POINT_LABEL = {"SA": "S-A: 2000 requested sessions, 50 slots", "SB": "S-B: 500 requested sessions, 15 slots", "control": "control: 500 sessions, 50 slots (TABLE 6 point)"}
# committed run directories (folder names under ACCURACY_DIR), filled in as the runs land
SATURATION_RUN_DIRS: dict[str, str | None] = {
    "SA": "saturation_SA_run_20260907_111056_513694",
    "SB": "saturation_SB_run_20260907_111056_525003",
    "control": "impairment_default_run_20260910_091432_644397",  # = impairment-sweep default cell (canonical representative point)
}
METRICS = [
    ("accuracy", True), ("accuracy_requested", True), ("requested_evs", False), ("admitted_evs", False), ("blocked_evs", False),
    ("blocked_ratio", True), ("fallback_finalized_ratio", True), ("session_end_fallback_ratio", False), ("timeline_end_fallback_ratio", False),
    ("fallback_to_all_rows_total", False), ("avg_candidates_per_ev", True), ("gt_in_candidates_ratio", False), ("p90_latency", True), ("runtime_s_total", True),
]
COLUMNS = ["point", "point_label", "ev_count", "evse_count", "algorithm_id", "runs"]
for name, with_ci in METRICS:
    COLUMNS.append(f"{name}_mean")
    if with_ci:
        COLUMNS.append(f"{name}_ci95")
COLUMNS.append("run_id")


def build_rows(run_dirs: dict[str, Path], expected_repeats: int, tol: float = 1e-6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point, run_dir in run_dirs.items():
        summary = pd.read_csv(run_dir / "summary.csv")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        ev_count, evse_count = POINT_SPEC[point]
        if list(cfg["ev_counts"]) != [ev_count] or int(cfg["evse_count"]) != evse_count:
            raise AssertionError(f"{run_dir.name}: config is not point {point} (EV {ev_count}, {evse_count} slots)")
        for algorithm_id in ALGORITHMS:
            sel = summary[summary["algorithm_id"] == algorithm_id]
            if len(sel) != 1:
                raise AssertionError(f"{run_dir.name}: expected one summary row for {algorithm_id}")
            r = sel.iloc[0]
            if int(r["runs"]) != int(expected_repeats):
                raise AssertionError(f"{run_dir.name} {algorithm_id}: runs={int(r['runs'])} != {expected_repeats}")
            if point == "control":
                if abs(float(r["accuracy_mean"]) - CANONICAL[algorithm_id]) > tol or abs(float(r["p90_latency_mean"]) - 416.035) > tol:
                    raise AssertionError(f"control point does not reproduce the canonical means for {algorithm_id}")
            else:
                if not float(r["blocked_ratio_mean"]) > 0.0:
                    raise AssertionError(f"{point} {algorithm_id}: no blocking — not a saturation point")
                if not float(r["accuracy_requested_mean"]) < float(r["accuracy_mean"]):
                    raise AssertionError(f"{point} {algorithm_id}: requested-session accuracy does not fall below admitted accuracy")
            row: dict[str, object] = {
                "point": point, "point_label": POINT_LABEL[point], "ev_count": ev_count, "evse_count": evse_count,
                "algorithm_id": algorithm_id, "runs": int(r["runs"]),
            }
            for name, with_ci in METRICS:
                row[f"{name}_mean"] = round(float(r[f"{name}_mean"]), 6)
                if with_ci:
                    row[f"{name}_ci95"] = round(float(r[f"{name}_ci95"]), 6)
            row["run_id"] = run_dir.name.rsplit("_run_", 1)[-1]
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="override: <point>=<run_dir> (repeatable; points SA, SB, control)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    run_dirs: dict[str, Path] = {p: ACCURACY_DIR / n for p, n in SATURATION_RUN_DIRS.items() if n}
    for item in args.run:
        point, _, path = item.partition("=")
        if point not in POINT_SPEC:
            raise SystemExit(f"unknown point {point!r}")
        run_dirs[point] = Path(path)
    missing = [p for p in POINT_SPEC if p not in run_dirs]
    if missing and not args.allow_missing:
        raise SystemExit(f"no run directory for points: {missing}")
    rows = build_rows(run_dirs, int(args.repeats))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['point']:8s} {r['algorithm_id']:36s} acc={r['accuracy_mean']:.4f} req={r['accuracy_requested_mean']:.4f} blocked={r['blocked_ratio_mean']:.4f} fb={r['fallback_finalized_ratio_mean']:.4f} cand={r['avg_candidates_per_ev_mean']:.1f} p90={r['p90_latency_mean']:.1f} rt={r['runtime_s_total_mean']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
