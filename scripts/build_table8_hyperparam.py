#!/usr/bin/env python3
"""Tabulate the Posterior matcher's hyperparameter sensitivity into `table8_hyperparam_sensitivity.csv` (Supplementary Table S4 of the manuscript).

Reads the committed one-knob-at-a-time runs produced by
``python scripts/reproduce_all.py --mode hyperparam-sweep [--cell-index i] --repeats 20``
(A3 only, representative point, canonical EV = 500 evaluation repeats 1–20) and writes

    case_study/scalability_analysis/accuracy/table8_hyperparam_sensitivity.csv

with one row per (knob, value): accuracy mean ± 95 % CI, the paired difference to the TABLE 2
default (the canonical representative cell of TABLE 6, same scenario seeds) with its 95 % CI, and
the Greedy (A1) reference of that same canonical cell. The TABLE 2 cell itself is taken from the
canonical run, not re-run. Gates: every cell carries the requested repeat count and its scenario
hashes equal the canonical cell's.
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
CANONICAL_RAW = ACCURACY_DIR / "canonical_run_20260907_111056_515561" / "raw_runs.csv"
OUT_CSV = ACCURACY_DIR / "table8_hyperparam_sensitivity.csv"
KNOB_SYMBOL = {"gamma_prev": "γ_prev", "eta_mix": "η_mix", "beta_curr": "β_curr"}
KNOB_KEY = {"gamma_prev": "a3_posterior_prev_power", "eta_mix": "a3_time_prior_mix", "beta_curr": "a3_current_like_beta"}
DEFAULTS = {"gamma_prev": 0.60, "eta_mix": 0.35, "beta_curr": 1.00}
# committed run directories (folder names under ACCURACY_DIR), filled in as the runs land
HYPERPARAM_RUN_DIRS: dict[str, str | None] = {
    "gamma_prev_0.3": "hyperparam_gamma_prev_0.3_run_20260909_112529_182056", "gamma_prev_0.45": "hyperparam_gamma_prev_0.45_run_20260909_122547_200269", "gamma_prev_0.75": "hyperparam_gamma_prev_0.75_run_20260909_132718_907265", "gamma_prev_0.9": "hyperparam_gamma_prev_0.9_run_20260909_142840_347159",
    "eta_mix_0.15": "hyperparam_eta_mix_0.15_run_20260909_152433_184981", "eta_mix_0.25": "hyperparam_eta_mix_0.25_run_20260909_160953_979895", "eta_mix_0.5": "hyperparam_eta_mix_0.5_run_20260909_165453_920798", "eta_mix_0.7": "hyperparam_eta_mix_0.7_run_20260909_173949_184148",
    "beta_curr_0.5": "hyperparam_beta_curr_0.5_run_20260909_182443_322414", "beta_curr_0.75": "hyperparam_beta_curr_0.75_run_20260909_190719_436127", "beta_curr_1.5": "hyperparam_beta_curr_1.5_run_20260909_194519_432942", "beta_curr_2": "hyperparam_beta_curr_2_run_20260909_202343_453722",
}
COLUMNS = ["knob", "symbol", "value", "is_default", "runs", "accuracy_mean", "accuracy_ci95", "delta_vs_default_mean", "delta_vs_default_ci95", "a1_reference", "run_id"]


def _ci95(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def canonical_cell(algorithm_id: str = "bayesian_windowed") -> pd.DataFrame:
    c = pd.read_csv(CANONICAL_RAW)
    m = (c["ev_count"] == 500) & (c["ev_sample_s"] == 30) & (c["matcher_delay_max_s"] == 10.0) & (c["candidate_margin_s"] == 120) & (c["algorithm_id"] == algorithm_id)
    return c[m].sort_values("repeat_idx")


def a1_reference() -> float:
    """Greedy (A1) mean accuracy of the canonical representative cell (TABLE 6)."""
    return round(float(canonical_cell("single_only")["accuracy"].mean()), 6)


def build_rows(run_dirs: dict[str, Path], expected_repeats: int) -> list[dict[str, object]]:
    ref = canonical_cell()
    ref_acc = ref["accuracy"].to_numpy(dtype=float)
    a1_ref = a1_reference()
    rows: list[dict[str, object]] = []
    for knob in ("gamma_prev", "eta_mix", "beta_curr"):
        rows.append({
            "knob": knob, "symbol": KNOB_SYMBOL[knob], "value": DEFAULTS[knob], "is_default": True, "runs": int(len(ref)),
            "accuracy_mean": round(float(ref_acc.mean()), 6), "accuracy_ci95": round(_ci95(ref_acc), 6),
            "delta_vs_default_mean": 0.0, "delta_vs_default_ci95": 0.0, "a1_reference": a1_ref, "run_id": "20260907_111056_515561 (canonical)",
        })
    for label, run_dir in run_dirs.items():
        knob = "gamma_prev" if label.startswith("gamma") else "eta_mix" if label.startswith("eta") else "beta_curr"
        raw = pd.read_csv(run_dir / "raw_runs.csv")
        raw = raw[raw["algorithm_id"] == "bayesian_windowed"].sort_values("repeat_idx")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        value = float(cfg[KNOB_KEY[knob]])
        if len(raw) != int(expected_repeats):
            raise AssertionError(f"{run_dir.name}: {len(raw)} runs != {expected_repeats}")
        if list(raw["scenario_hash"]) != list(ref["scenario_hash"].head(len(raw))):
            raise AssertionError(f"{run_dir.name}: scenario hashes differ from the canonical cell (not paired)")
        for k, other in KNOB_KEY.items():
            if k != knob and abs(float(cfg[other]) - DEFAULTS[k]) > 1e-12:
                raise AssertionError(f"{run_dir.name}: {other} is not at its TABLE 2 value (one knob at a time)")
        acc = raw["accuracy"].to_numpy(dtype=float)
        d = acc - ref_acc[: len(acc)]
        rows.append({
            "knob": knob, "symbol": KNOB_SYMBOL[knob], "value": value, "is_default": False, "runs": int(len(acc)),
            "accuracy_mean": round(float(acc.mean()), 6), "accuracy_ci95": round(_ci95(acc), 6),
            "delta_vs_default_mean": round(float(d.mean()), 6), "delta_vs_default_ci95": round(_ci95(d), 6),
            "a1_reference": a1_ref, "run_id": run_dir.name.rsplit("_run_", 1)[-1],
        })
    order = {"gamma_prev": 0, "eta_mix": 1, "beta_curr": 2}
    rows.sort(key=lambda r: (order[str(r["knob"])], float(r["value"])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="override: <label>=<run_dir> (repeatable)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    run_dirs: dict[str, Path] = {k: ACCURACY_DIR / v for k, v in HYPERPARAM_RUN_DIRS.items() if v}
    for item in args.run:
        label, _, path = item.partition("=")
        if label not in HYPERPARAM_RUN_DIRS:
            raise SystemExit(f"unknown cell {label!r}")
        run_dirs[label] = Path(path)
    missing = [k for k in HYPERPARAM_RUN_DIRS if k not in run_dirs]
    if missing and not args.allow_missing:
        raise SystemExit(f"no run directory for cells: {missing}")
    rows = build_rows(run_dirs, int(args.repeats))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['symbol']:7s} = {r['value']:<5g} acc={r['accuracy_mean']:.4f} ± {r['accuracy_ci95']:.4f}  Δ={r['delta_vs_default_mean']:+.4f} ± {r['delta_vs_default_ci95']:.4f}{'  (TABLE 2)' if r['is_default'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
