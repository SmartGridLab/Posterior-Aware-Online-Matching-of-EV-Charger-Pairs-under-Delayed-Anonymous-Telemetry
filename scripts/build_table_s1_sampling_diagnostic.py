#!/usr/bin/env python3
"""Tabulate the dense-sampling diagnostic into TABLE S1.

Rows: A1 and A3 accuracy at Δt_EV ∈ {5, 15, 30} s under
  (i)  the manuscript's sensing model and hold front end (canonical run, not re-run),
  (ii) the noise-free control (sample noise std 0; gain, bias, extra delay, start jitter kept), and
  (iii) the block-mean front end (dense samples averaged onto 30 s blocks before the forward-fill; 5 and 15 s),
  (iv)  the Eq. (7) term partition: the Greedy ablation variants DTW-only
        (`single_only_step_off`) and step-only (`single_only_dtw_off`) on the unchanged manuscript front end
        at 5, 15 and 30 s, paired with the canonical full Greedy cost (A1) of the same Δt_EV; the unpaired
        difference to the same variant at 30 s is reported as well (different Δt_EV cells use different seeds),
each with the 95 % CI over the 20 repeats and the paired difference to the canonical cell of the same
Δt_EV (identical scenario seeds/hashes are enforced). Produced by
``python scripts/reproduce_all.py --mode sampling-diagnostic [--cell-index i] --repeats 20``.
Output: case_study/scalability_analysis/accuracy/table_s1_sampling_diagnostic.csv
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
OUT_CSV = ACCURACY_DIR / "table_s1_sampling_diagnostic.csv"
ALGORITHMS = ["single_only", "bayesian_windowed"]
VARIANT_LABEL = {"canonical": "(i) TABLE 3 sensing, hold", "noise_free": "(ii) sample noise off, hold", "block_mean": "(iii) TABLE 3 sensing, 30 s block mean", "term_partition": "(iv) TABLE 3 sensing, hold — Eq. (6) term partition (A1 variants)", "alignment_control": "(v) TABLE 3 sensing, hold — EV meter offset fixed at the charger sampling period"}
TERM_PARTITION_ALGORITHMS = ["single_only_step_off", "single_only_dtw_off"]   # DTW-only, step-only
TERM_PARTITION_REFERENCE = "single_only"                                       # paired against the full Greedy cost
# (v) alignment control: the per-session meter offset is fixed at CHARGER_SAMPLE instead of drawn over
# 0-20 s, so every sample falls inside the banded-DTW window. Delta t_EV = 5 s only; each arm is paired
# against the same arm of the modeled-offset 5 s cell.
ALIGNMENT_ALGORITHMS = ["single_only", "bayesian_windowed", "nomura_original_interval_hungarian", "single_only_step_off"]
# committed run directories (folder names under ACCURACY_DIR), filled in as the runs land
SAMPLING_RUN_DIRS: dict[str, str | None] = {"noisefree_5": "sampling_diag_noisefree_5_run_20260909_112529_182567", "noisefree_15": "sampling_diag_noisefree_15_run_20260909_144709_767904", "noisefree_30": "sampling_diag_noisefree_30_run_20260909_172229_409746", "blockmean_5": "sampling_diag_blockmean_5_run_20260909_194237_669442", "blockmean_15": "sampling_diag_blockmean_15_run_20260909_214145_198227", "termpart_5": "sampling_diag_termpart_5_run_20260909_231451_372267", "termpart_15": "sampling_diag_termpart_15_run_20260910_004312_316315", "termpart_30": "sampling_diag_termpart_30_run_20260910_021212_788641", "offsetfree_5": "sampling_diag_offsetfree_5_run_20260910_180742_322660"}
COLUMNS = ["variant", "variant_label", "ev_sample_s", "algorithm_id", "runs", "accuracy_mean", "accuracy_ci95", "delta_vs_canonical_mean", "delta_vs_canonical_ci95", "delta_vs_30s_mean", "delta_vs_30s_ci95", "run_id"]


def _ci95(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def canonical_cell(canon: pd.DataFrame, dt: int, algorithm_id: str) -> pd.DataFrame:
    m = (canon["ev_count"] == 500) & (canon["ev_sample_s"] == dt) & (canon["matcher_delay_max_s"] == 10.0) & (canon["candidate_margin_s"] == 120) & (canon["algorithm_id"] == algorithm_id)
    return canon[m].sort_values("repeat_idx")


def build_rows(run_dirs: dict[str, Path], expected_repeats: int) -> list[dict[str, object]]:
    canon = pd.read_csv(CANONICAL_RAW)
    rows: list[dict[str, object]] = []
    for dt in (5, 15, 30):
        for aid in ALGORITHMS:
            ref = canonical_cell(canon, dt, aid)["accuracy"].to_numpy(dtype=float)
            rows.append({"variant": "canonical", "variant_label": VARIANT_LABEL["canonical"], "ev_sample_s": dt, "algorithm_id": aid, "runs": int(len(ref)),
                         "accuracy_mean": round(float(ref.mean()), 6), "accuracy_ci95": round(_ci95(ref), 6), "delta_vs_canonical_mean": 0.0, "delta_vs_canonical_ci95": 0.0, "run_id": "20260907_111056_515561 (canonical)"})
    term_cells: dict[tuple[str, int], np.ndarray] = {}
    for label, run_dir in run_dirs.items():
        variant = ("noise_free" if label.startswith("noisefree")
                   else "term_partition" if label.startswith("termpart")
                   else "alignment_control" if label.startswith("offsetfree")
                   else "block_mean")
        dt = int(label.rsplit("_", 1)[1])
        raw = pd.read_csv(run_dir / "raw_runs.csv")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        assert list(cfg["ev_sample_values_s"]) == [dt], f"{run_dir.name}: ev_sample mismatch"
        if variant == "noise_free":
            assert float(cfg["ev_sensor_noise_std_a"]) == 0.0 and cfg["ev_grid_aggregation"] == "hold", f"{run_dir.name}: not the noise-free control"
        elif variant == "term_partition":
            assert cfg["ev_grid_aggregation"] == "hold" and float(cfg["ev_sensor_noise_std_a"]) > 0.0, f"{run_dir.name}: not the manuscript front end"
        elif variant == "alignment_control":
            assert int(cfg["ev_sensor_extra_delay_max_s"]) == 0, f"{run_dir.name}: the meter offset is not fixed"
            assert cfg["ev_grid_aggregation"] == "hold" and float(cfg["ev_sensor_noise_std_a"]) > 0.0, f"{run_dir.name}: not the manuscript front end"
        else:
            assert cfg["ev_grid_aggregation"] == "block_mean" and float(cfg["ev_sensor_noise_std_a"]) > 0.0, f"{run_dir.name}: not the block-mean variant"
        arms = (TERM_PARTITION_ALGORITHMS if variant == "term_partition"
                else ALIGNMENT_ALGORITHMS if variant == "alignment_control"
                else ALGORITHMS)
        for aid in arms:
            a = raw[raw["algorithm_id"] == aid].sort_values("repeat_idx")
            if variant == "term_partition":
                refdf = canonical_cell(canon, dt, TERM_PARTITION_REFERENCE)
            elif variant == "alignment_control":
                # the modeled-offset cell of the same arm: the canonical grid carries the three full
                # matchers, and the term-partition run carries the DTW-only arm; both share its seeds
                refdf = canonical_cell(canon, dt, aid)
                if len(refdf) == 0:
                    modeled = pd.read_csv(run_dirs["termpart_5"] / "raw_runs.csv")
                    refdf = modeled[modeled["algorithm_id"] == aid].sort_values("repeat_idx")
            else:
                refdf = canonical_cell(canon, dt, aid)
            if len(a) != int(expected_repeats):
                raise AssertionError(f"{run_dir.name} {aid}: {len(a)} runs != {expected_repeats}")
            if list(a["scenario_seed"]) != list(refdf["scenario_seed"].head(len(a))):
                raise AssertionError(f"{run_dir.name} {aid}: scenario seeds differ from the canonical Δt_EV = {dt} s cell (not paired)")
            if variant in ("block_mean", "term_partition") and list(a["scenario_hash"]) != list(refdf["scenario_hash"].head(len(a))):
                raise AssertionError(f"{run_dir.name} {aid}: scenario hashes differ from the canonical cell")
            acc = a["accuracy"].to_numpy(dtype=float); ref = refdf["accuracy"].to_numpy(dtype=float)[: len(acc)]
            d = acc - ref
            if variant == "term_partition":
                term_cells[(aid, dt)] = acc
            rows.append({"variant": variant, "variant_label": VARIANT_LABEL[variant], "ev_sample_s": dt, "algorithm_id": aid, "runs": int(len(acc)),
                         "accuracy_mean": round(float(acc.mean()), 6), "accuracy_ci95": round(_ci95(acc), 6), "delta_vs_canonical_mean": round(float(d.mean()), 6), "delta_vs_canonical_ci95": round(_ci95(d), 6),
                         "delta_vs_30s_mean": "", "delta_vs_30s_ci95": "", "run_id": run_dir.name.rsplit("_run_", 1)[-1]})
    # unpaired difference of each term-partition cell to the same variant at 30 s (Welch CI; different seed blocks)
    for r in rows:
        if r["variant"] == "term_partition" and (str(r["algorithm_id"]), 30) in term_cells:
            a = term_cells[(str(r["algorithm_id"]), int(r["ev_sample_s"]))]; b = term_cells[(str(r["algorithm_id"]), 30)]
            se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            r["delta_vs_30s_mean"] = round(float(a.mean() - b.mean()), 6); r["delta_vs_30s_ci95"] = round(1.96 * se, 6)
    order = {"canonical": 0, "noise_free": 1, "block_mean": 2, "term_partition": 3, "alignment_control": 4}
    algo_order = ALGORITHMS + TERM_PARTITION_ALGORITHMS + ["nomura_original_interval_hungarian"]
    rows.sort(key=lambda r: (order[str(r["variant"])], int(r["ev_sample_s"]), algo_order.index(str(r["algorithm_id"]))))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="override: <label>=<run_dir> (repeatable)")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    run_dirs: dict[str, Path] = {k: ACCURACY_DIR / v for k, v in SAMPLING_RUN_DIRS.items() if v}
    for item in args.run:
        label, _, path = item.partition("=")
        if label not in SAMPLING_RUN_DIRS:
            raise SystemExit(f"unknown cell {label!r}")
        run_dirs[label] = Path(path)
    missing = [k for k in SAMPLING_RUN_DIRS if k not in run_dirs]
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
        extra = f"  Δ vs 30 s={r['delta_vs_30s_mean']:+.4f} ± {r['delta_vs_30s_ci95']:.4f}" if r.get("delta_vs_30s_mean") not in ("", None) else ""
        print(f"  {r['variant_label']:62s} Δt_EV={r['ev_sample_s']:2d} s {r['algorithm_id']:20s} acc={r['accuracy_mean']:.4f} ± {r['accuracy_ci95']:.4f} Δ={r['delta_vs_canonical_mean']:+.4f} ± {r['delta_vs_canonical_ci95']:.4f}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
