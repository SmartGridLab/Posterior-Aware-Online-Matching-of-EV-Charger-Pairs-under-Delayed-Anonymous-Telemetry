#!/usr/bin/env python3
"""Build the figure8 data table (manuscript Figure 6, response heterogeneity): accuracy vs EV-session count under
per-session response heterogeneity next to the nominal fitted model.

Reads the committed heterogeneous-response run (``python scripts/reproduce_all.py --mode
heterogeneous-response --repeats 20``; base-load grid, canonical base-load seeds 2187632392–2451)
and the canonical run, aggregates both under the TABLE 5 protocol (EV 100/500: repeats 1–20;
EV 300: repeats 6–20) and writes

    case_study/scalability_analysis/accuracy/figure8_heterogeneous_response.csv

with one row per (model, matcher, EV count): accuracy mean, 95 % CI, and for the heterogeneous
model the paired difference to the nominal model with its CI. Gate: the heterogeneous run must
carry the canonical scenario seeds for every (EV, repeat).
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
CANONICAL_RAW = ACCURACY_DIR / "canonical_run_20260907_111056_515561" / "raw_runs.csv"
OUT_CSV = ACCURACY_DIR / "figure8_heterogeneous_response.csv"
HETEROGENEOUS_RUN_DIR: str | None = "heterogeneous_response_run_20260908_170955_253072"
ALGORITHMS = ["time_only_baseline", "single_only", "nomura_original_interval_hungarian", "bayesian_windowed"]
EV_COUNTS = (100, 300, 500)
HELD_OUT_MIN_REPEAT = {100: 1, 300: 6, 500: 1}
BASE_LOAD = dict(ev_sample_s=30, matcher_delay_max_s=10.0, candidate_margin_s=120)
COLUMNS = ["model", "algorithm_id", "ev_count", "runs", "accuracy_mean", "accuracy_ci95", "delta_vs_nominal_mean", "delta_vs_nominal_ci95", "run_id"]


def _cell(raw: pd.DataFrame, aid: str, ev: int) -> pd.DataFrame:
    m = (raw["algorithm_id"] == aid) & (raw["ev_count"] == ev)
    for k, v in BASE_LOAD.items():
        m &= raw[k] == v
    m &= raw["repeat_idx"] >= HELD_OUT_MIN_REPEAT[ev]
    return raw[m].sort_values("repeat_idx")


def _ci95(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def build_rows(run_dir: Path) -> list[dict[str, object]]:
    het = pd.read_csv(run_dir / "raw_runs.csv")
    canon = pd.read_csv(CANONICAL_RAW)
    run_id = run_dir.name.rsplit("_run_", 1)[-1]
    rows: list[dict[str, object]] = []
    for aid in ALGORITHMS:
        for ev in EV_COUNTS:
            c = _cell(canon, aid, ev); h = _cell(het, aid, ev)
            if len(h) == 0 or len(h) != len(c):
                raise AssertionError(f"{aid} EV={ev}: heterogeneous cell has {len(h)} runs, canonical {len(c)}")
            if list(h["scenario_seed"]) != list(c["scenario_seed"]):
                raise AssertionError(f"{aid} EV={ev}: scenario seeds differ from the canonical run (not paired)")
            ca = c["accuracy"].to_numpy(dtype=float); ha = h["accuracy"].to_numpy(dtype=float)
            rows.append({"model": "nominal", "algorithm_id": aid, "ev_count": ev, "runs": int(len(ca)), "accuracy_mean": round(float(ca.mean()), 6), "accuracy_ci95": round(_ci95(ca), 6), "delta_vs_nominal_mean": 0.0, "delta_vs_nominal_ci95": 0.0, "run_id": "20260907_111056_515561 (canonical)"})
            d = ha - ca
            rows.append({"model": "heterogeneous", "algorithm_id": aid, "ev_count": ev, "runs": int(len(ha)), "accuracy_mean": round(float(ha.mean()), 6), "accuracy_ci95": round(_ci95(ha), 6), "delta_vs_nominal_mean": round(float(d.mean()), 6), "delta_vs_nominal_ci95": round(_ci95(d), 6), "run_id": run_id})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=(ACCURACY_DIR / HETEROGENEOUS_RUN_DIR) if HETEROGENEOUS_RUN_DIR else None)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    if args.run_dir is None:
        raise SystemExit("pass --run-dir or set HETEROGENEOUS_RUN_DIR")
    rows = build_rows(args.run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    for r in rows:
        if r["model"] == "heterogeneous":
            print(f"  {r['algorithm_id']:36s} EV={r['ev_count']:4d} heterogeneous {r['accuracy_mean']:.4f} ± {r['accuracy_ci95']:.4f}  Δ vs nominal {r['delta_vs_nominal_mean']:+.4f} ± {r['delta_vs_nominal_ci95']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
