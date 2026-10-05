#!/usr/bin/env python3
"""Tabulate the static-limit sanity check into TABLE S2.

Reads the two committed static-limit runs produced by
``python scripts/reproduce_all.py --mode static-limit`` — the ideal-sensing cell
(all EV sensing-distortion parameters and the start-estimate jitter set to 0) and the
paper-sensing cell (Sec. V-A sensing defaults) — and writes

    case_study/scalability_analysis/accuracy/table_s2_static_limit.csv

with one row per (cell, matcher): accuracy mean ± 95 % CI (Student's t with n - 1 degrees of
freedom over the per-repeat accuracies of raw_runs.csv, the rule the manuscript uses for the
Supplementary tables), p90 decision latency,
blocked sessions, sessions ending before their watermark, and runtime. Both cells run
300 sessions arriving simultaneously on 300 slots with complete W = 360 s waveforms,
cloud ingestion disabled (zero delay / jitter / loss, watermark bound 0 s), the legacy
interval codebook with fixed slot→pattern binding (the setting of the prior static
study), 20 paired repeats.

Two integrity gates run before writing: every cell must carry exactly the requested
repeat count for every matcher, and no session may be blocked or finalised before its
watermark (otherwise the run is not a static-limit run).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
OUT_CSV = ACCURACY_DIR / "table_s2_static_limit.csv"

# Committed run directory per cell. Each run carries the three matchers on the same scenario
# and ingestion seeds, so they are paired repeat by repeat.
STATIC_LIMIT_RUN_DIRS: dict[str, list[str]] = {
    # One run per cell; each carries all three matchers on the same scenario seeds.
    "ideal_sensing": ["static_limit_ideal_sensing_run_20260908_170955_253099"],
    "paper_sensing": ["static_limit_paper_sensing_run_20260909_020513_080213"],
}
CELL_LABELS = {
    "ideal_sensing": "ST-ideal (sensing distortion off)",
    "paper_sensing": "ST-paper (TABLE 3 sensing defaults)",
}
MATCHER_ORDER = ["nomura_original_interval_hungarian", "single_only", "bayesian_windowed"]
MATCHER_LABELS = {
    "nomura_original_interval_hungarian": "A2 Global (Corr+DTW + Hungarian; prior static cost)",
    "single_only": "A1 Greedy",
    "bayesian_windowed": "A3 Posterior",
}
COLUMNS = [
    "cell",
    "cell_label",
    "algorithm_id",
    "matcher",
    "runs",
    "accuracy_mean",
    "accuracy_ci95",
    "p90_latency_mean",
    "blocked_evs_mean",
    "session_end_before_watermark_ratio_mean",
    "avg_candidates_per_ev_mean",
    "runtime_s_total_mean",
    "run_id",
]


def _ci95_t(x: np.ndarray) -> float:
    """Half-width of the Student-t 95 % interval of the mean (n - 1 degrees of freedom)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def _load_cell(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Concatenate the summary and raw rows of one cell's run directories; validate each config."""
    frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    run_id_by_algo: dict[str, str] = {}
    seeds: set[tuple[int, ...]] = set()
    for run_dir in run_dirs:
        summary = pd.read_csv(run_dir / "summary.csv")
        raw = pd.read_csv(run_dir / "raw_runs.csv")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if not bool(config.get("simultaneous_arrivals", False)) or bool(config.get("ingestion_enabled", True)):
            raise AssertionError(f"{run_dir.name}: not a static-limit run (simultaneous_arrivals / ingestion_enabled)")
        if int(config.get("evse_count", 0)) != 300 or list(config.get("ev_counts", [])) != [300]:
            raise AssertionError(f"{run_dir.name}: expected 300 sessions on 300 slots")
        seeds.add(tuple(sorted(int(s) for s in raw["scenario_seed"].unique())))
        for algorithm_id in summary["algorithm_id"].unique():
            run_id_by_algo[str(algorithm_id)] = run_dir.name.rsplit("_run_", 1)[-1]
        frames.append(summary)
        raw_frames.append(raw)
    if len(seeds) != 1:
        raise AssertionError("run directories of one cell do not share the same scenario seeds (not paired)")
    return pd.concat(frames, ignore_index=True), run_id_by_algo, pd.concat(raw_frames, ignore_index=True)


def build_rows(run_dirs: dict[str, list[Path]], expected_repeats: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell, dirs in run_dirs.items():
        summary, run_id_by_algo, raw = _load_cell(dirs)
        for algorithm_id in MATCHER_ORDER:
            sel = summary[summary["algorithm_id"] == algorithm_id]
            if len(sel) != 1:
                raise AssertionError(f"{cell}: expected one summary row for {algorithm_id}, found {len(sel)}")
            r = sel.iloc[0]
            run_dir_name = f"{cell}:{run_id_by_algo.get(algorithm_id, '?')}"
            if int(r["runs"]) != int(expected_repeats):
                raise AssertionError(f"{run_dir_name} {algorithm_id}: runs={int(r['runs'])} != {expected_repeats}")
            if float(r["blocked_evs_mean"]) != 0.0:
                raise AssertionError(f"{run_dir_name} {algorithm_id}: blocked sessions in the static limit")
            if float(r["session_end_before_watermark_ratio_mean"]) != 0.0:
                raise AssertionError(f"{run_dir_name} {algorithm_id}: sessions ended before their watermark")
            per_repeat = raw.loc[raw["algorithm_id"] == algorithm_id, "accuracy"].to_numpy(dtype=float)
            if len(per_repeat) != int(expected_repeats) or abs(float(per_repeat.mean()) - float(r["accuracy_mean"])) > 1e-9:
                raise AssertionError(f"{run_dir_name} {algorithm_id}: raw_runs.csv does not reproduce the summary mean")
            rows.append(
                {
                    "cell": cell,
                    "cell_label": CELL_LABELS[cell],
                    "algorithm_id": algorithm_id,
                    "matcher": MATCHER_LABELS[algorithm_id],
                    "runs": int(r["runs"]),
                    "accuracy_mean": round(float(r["accuracy_mean"]), 6),
                    "accuracy_ci95": round(_ci95_t(per_repeat), 6),
                    "p90_latency_mean": round(float(r["p90_latency_mean"]), 3),
                    "blocked_evs_mean": float(r["blocked_evs_mean"]),
                    "session_end_before_watermark_ratio_mean": float(r["session_end_before_watermark_ratio_mean"]),
                    "avg_candidates_per_ev_mean": round(float(r["avg_candidates_per_ev_mean"]), 3),
                    "runtime_s_total_mean": round(float(r["runtime_s_total_mean"]), 3),
                    "run_id": run_id_by_algo.get(algorithm_id, ""),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ideal-dir", type=Path, action="append", default=None, help="ideal-sensing run directory (repeatable, one per matcher)")
    parser.add_argument("--paper-dir", type=Path, action="append", default=None, help="paper-sensing run directory (repeatable, one per matcher)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    run_dirs: dict[str, list[Path]] = {}
    for cell, override in (("ideal_sensing", args.ideal_dir), ("paper_sensing", args.paper_dir)):
        if override:
            run_dirs[cell] = [Path(x) for x in override]
        elif STATIC_LIMIT_RUN_DIRS.get(cell):
            run_dirs[cell] = [ACCURACY_DIR / name for name in STATIC_LIMIT_RUN_DIRS[cell]]
        else:
            raise SystemExit(f"no run directory for cell {cell!r}: pass --{cell.split('_')[0]}-dir or set STATIC_LIMIT_RUN_DIRS")

    rows = build_rows(run_dirs, int(args.repeats))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    for row in rows:
        print(
            f"  {row['cell']:14s} {row['algorithm_id']:36s} acc={row['accuracy_mean']:.4f} ± {row['accuracy_ci95']:.4f} "
            f"p90={row['p90_latency_mean']} runs={row['runs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
