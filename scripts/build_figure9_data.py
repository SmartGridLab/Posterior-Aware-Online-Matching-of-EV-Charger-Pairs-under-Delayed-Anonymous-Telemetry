#!/usr/bin/env python3
"""Build the figure9 data table (manuscript Figure 7, codebook band sweep).

Accuracy versus realised command-codebook separation at the representative point (EV = 500,
50 slots, Δt_EV = 30 s, d_max = 10 s, Δ = 120 s), for the Greedy (A1), Global (A2) and Posterior
(A3) matchers, obtained by narrowing the set-point band of the (unchanged, random-unique) codebook
generator: B30 = 6–30 A (canonical run, not re-run), B20 = 6–20 A, B16 = 8–16 A, B12 = 8–12 A.
Every band cell replays the canonical representative seeds (arrivals, slots, sensing and ingestion
draws identical; only the patterns differ), so the paired difference to the canonical cell is
reported with its 95 % CI. The realised separation per cell is the mean over repeats of the
``codebook_pairwise_l1_mean_a`` / ``codebook_pairwise_l1_min_a`` columns logged by the runtime; for
the canonical band the pool statistics come from the committed representative-point traces
(``codebook_realised_statistics.csv``, used patterns of the 20 canonical repeats).

Usage: python scripts/build_figure9_data.py [--run-dir B20=<dir> --run-dir B16=<dir> ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
CANONICAL_RAW = ACCURACY_DIR / "canonical_run_20260907_111056_515561" / "raw_runs.csv"
CANONICAL_CODEBOOK_STATS = ACCURACY_DIR / "codebook_realised_statistics.csv"
OUT_CSV = ACCURACY_DIR / "figure9_codebook_separation.csv"
BANDS: dict[str, tuple[int, int]] = {"B30": (6, 30), "B20": (6, 20), "B16": (8, 16), "B12": (8, 12)}
# committed band runs (filled by the coder after the full run); B30 is the canonical run
BAND_RUN_DIRS: dict[str, str | None] = {"B20": "codebook_band_B20_run_20260909_112529_182170", "B16": "codebook_band_B16_run_20260909_142401_231979", "B12": "codebook_band_B12_run_20260909_164615_270394"}
ALGORITHMS = ["single_only", "nomura_original_interval_hungarian", "bayesian_windowed"]
CELL = dict(ev_count=500, ev_sample_s=30, matcher_delay_max_s=10.0, candidate_margin_s=120)
COLUMNS = ["cell", "setpoint_lo_a", "setpoint_hi_a", "realised_l1_mean_a", "realised_l1_min_mean_a", "realised_l1_min_min_a", "internal_gap_min_a", "consec_step_min_a",
           "algorithm_id", "runs", "accuracy_mean", "accuracy_ci95", "delta_vs_canonical_mean", "delta_vs_canonical_ci95", "run_id"]


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def _cell(raw: pd.DataFrame, aid: str) -> pd.DataFrame:
    m = (raw["algorithm_id"] == aid)
    for k, v in CELL.items():
        m &= (raw[k] == v)
    return raw[m].sort_values("repeat_idx")


def build_rows(run_dirs: dict[str, Path]) -> list[dict[str, object]]:
    canon = pd.read_csv(CANONICAL_RAW)
    stats = pd.read_csv(CANONICAL_CODEBOOK_STATS)
    per_rep = stats[stats["repeat_idx"].astype(str) != "all"]
    rows: list[dict[str, object]] = []
    for aid in ALGORITHMS:
        ca = _cell(canon, aid)
        rows.append({"cell": "B30", "setpoint_lo_a": 6, "setpoint_hi_a": 30, "realised_l1_mean_a": round(float(per_rep["pairwise_l1_mean_a"].mean()), 3),
                     "realised_l1_min_mean_a": round(float(per_rep["pairwise_l1_min_a"].mean()), 3), "realised_l1_min_min_a": int(per_rep["pairwise_l1_min_a"].min()),
                     "internal_gap_min_a": int(per_rep["internal_gap_min_a"].min()), "consec_step_min_a": int(per_rep["consec_step_min_a"].min()),
                     "algorithm_id": aid, "runs": int(len(ca)), "accuracy_mean": round(float(ca["accuracy"].mean()), 6), "accuracy_ci95": round(_ci95(ca["accuracy"].to_numpy()), 6),
                     "delta_vs_canonical_mean": 0.0, "delta_vs_canonical_ci95": 0.0, "run_id": "20260907_111056_515561 (canonical)"})
    for label, run_dir in run_dirs.items():
        lo, hi = BANDS[label]
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert list(cfg["command_setpoint_range"]) == [lo, hi], f"{run_dir.name}: band mismatch {cfg['command_setpoint_range']} vs {(lo, hi)}"
        raw = pd.read_csv(run_dir / "raw_runs.csv")
        for aid in ALGORITHMS:
            a = _cell(raw, aid); ca = _cell(canon, aid)
            if len(a) == 0:
                continue
            if list(a["scenario_seed"]) != list(ca["scenario_seed"].head(len(a))):
                raise AssertionError(f"{run_dir.name} {aid}: scenario seeds differ from the canonical cell (not paired)")
            d = a["accuracy"].to_numpy(dtype=float) - ca["accuracy"].to_numpy(dtype=float)[: len(a)]
            assert bool(a["codebook_terminal_branch"].all()), f"{run_dir.name} {aid}: generator did not use the terminal branch in every repeat"
            rows.append({"cell": label, "setpoint_lo_a": lo, "setpoint_hi_a": hi, "realised_l1_mean_a": round(float(a["codebook_pairwise_l1_mean_a"].mean()), 3),
                         "realised_l1_min_mean_a": round(float(a["codebook_pairwise_l1_min_a"].mean()), 3), "realised_l1_min_min_a": int(a["codebook_pairwise_l1_min_a"].min()),
                         "internal_gap_min_a": int(a["codebook_internal_gap_min_a"].min()), "consec_step_min_a": int(a["codebook_consec_step_min_a"].min()),
                         "algorithm_id": aid, "runs": int(len(a)), "accuracy_mean": round(float(a["accuracy"].mean()), 6), "accuracy_ci95": round(_ci95(a["accuracy"].to_numpy()), 6),
                         "delta_vs_canonical_mean": round(float(d.mean()), 6), "delta_vs_canonical_ci95": round(_ci95(d), 6), "run_id": run_dir.name.rsplit("_run_", 1)[-1]})
    order = list(BANDS)
    rows.sort(key=lambda r: (order.index(str(r["cell"])), ALGORITHMS.index(str(r["algorithm_id"]))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", default=[], help="LABEL=<dir> (B20, B16, B12); overrides the committed registry")
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    run_dirs: dict[str, Path] = {k: ACCURACY_DIR / v for k, v in BAND_RUN_DIRS.items() if v}
    for item in args.run_dir:
        label, _, d = str(item).partition("=")
        if label not in BAND_RUN_DIRS or not d:
            raise SystemExit(f"--run-dir expects LABEL=<dir> with LABEL in {list(BAND_RUN_DIRS)} (got {item!r})")
        run_dirs[label] = Path(d)
    missing = [k for k in BAND_RUN_DIRS if k not in run_dirs]
    if missing:
        raise SystemExit(f"no run directory for band cell(s) {missing}; pass --run-dir LABEL=<dir> or fill BAND_RUN_DIRS")
    rows = build_rows(run_dirs)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(args.out, index=False)
    print(f"[build_figure9_data] wrote {args.out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['cell']} {r['setpoint_lo_a']:2d}-{r['setpoint_hi_a']:2d} A  l1 mean {r['realised_l1_mean_a']:5.1f} (min {r['realised_l1_min_mean_a']:4.1f})  {r['algorithm_id']:36s} acc={r['accuracy_mean']:.4f} ± {r['accuracy_ci95']:.4f}  Δ={r['delta_vs_canonical_mean']:+.4f} ± {r['delta_vs_canonical_ci95']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
