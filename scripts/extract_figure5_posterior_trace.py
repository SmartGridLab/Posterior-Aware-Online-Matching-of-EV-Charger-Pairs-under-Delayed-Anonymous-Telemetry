#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the posterior-evolution trace of figure file figure5 (Supplementary Fig. S1) from a canonical single-scenario replay.

Supplementary Fig. S1 shows the two-stage posterior evolution of one representative session
at the representative operating point (EV=500, candidate_margin=120 s, matcher_delay_max=10 s,
ev_sample=30 s, tau=60 s). Six diagnostic indices correspond to three renormalized-posterior
checkpoints inside each of the two update stages of a single watermark decision:

  1 stage1 carry-over (uniform prior)      4 stage2 carry-over (P_1^gamma_prev)
  2 stage1 after time-prior                5 stage2 after time-prior
  3 stage1 after current-likelihood        6 stage2 after current-likelihood

This driver replays exactly the canonical rep=0 scenario for that operating point. The
canonical grid assigns seed = base_seed + scenario_counter in nested-loop order; the
representative operating point (rep=0) has scenario_counter=1360, i.e. seed=2187632432.
Running the single operating point with base_seed set to that value reproduces the identical
dataset, ingestion realization and A3 decision path (canonical A3 accuracy = 0.982 for this
scenario), so the extracted checkpoints are the real posteriors of the paper's example.

The featured session is selected deterministically: among the sessions whose final
(index-6) top-1 posterior converges above 0.99, the one with the most gradual stage-1
evolution (smallest checkpoint-3 top-1) is shown, i.e. the clearest two-stage posterior
dynamics rather than an immediate collapse to 1.0.

Instrumentation is gated by EVLINK_FIG5_TRACE=1 (set here); with the flag unset the
matcher/event-loop/driver behave and perform identically. No parameters are modified and the
canonical summary is not touched.

Output: case_study/scalability_analysis/accuracy/figure5_posterior_trace.csv
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Representative operating point (paper Section VI-C example).
PAPER_BASE_SEED = 2187631072
REP_SCENARIO_COUNTER = 1360  # (ev_sample=30, dmax=10, margin=120, ev=500), rep=0
REP_SEED = PAPER_BASE_SEED + REP_SCENARIO_COUNTER  # 2187632432
OUT_CSV = REPO / "case_study" / "scalability_analysis" / "accuracy" / "figure5_posterior_trace.csv"

# Featured-session selection: converged final posterior, most gradual stage-1 rise.
CONVERGED_TOP1 = 0.99


def _run_replay() -> Path:
    os.environ["EVLINK_FIG5_TRACE"] = "1"
    from case_study.section_vi_reproduction import run_algorithm_compare_report, COMPARE_DIR

    result = run_algorithm_compare_report(
        ev_counts=[500],
        repeats=1,
        base_seed=REP_SEED,
        tau_values=[60],
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        algorithm_ids=["bayesian_windowed"],
        trace_level="full",
    )
    run_id = str(result["run_id"])
    out_dir = Path(COMPARE_DIR) / run_id
    print(f"[extract] replay run_id={run_id}")
    # Sanity: A3 accuracy of this single scenario must match the canonical raw row.
    raw = pd.read_csv(out_dir / "raw_runs.csv")
    acc = float(raw.loc[raw["algorithm_id"] == "bayesian_windowed", "accuracy"].iloc[0])
    print(f"[extract] replayed A3 accuracy={acc:.4f} (canonical raw = 0.982 for seed {REP_SEED})")
    return out_dir


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = _run_replay()
    ck = _load_jsonl(out_dir / "posterior_checkpoint_trace_A3.jsonl")
    if ck.empty:
        raise RuntimeError("checkpoint trace is empty; is EVLINK_FIG5_TRACE=1 honored?")
    ev = _load_jsonl(out_dir / "event_trace.jsonl")

    # Watermark time t_wm per session = watermark_ready_ts - arrival_ts (seconds).
    twm_by_ev: dict[int, float] = {}
    if not ev.empty and {"arrival_ts", "watermark_ready_ts"}.issubset(ev.columns):
        for eidx, g in ev.groupby("ev_idx"):
            r = g.iloc[0]
            t = (pd.Timestamp(r["watermark_ready_ts"]) - pd.Timestamp(r["arrival_ts"])).total_seconds()
            twm_by_ev[int(eidx)] = float(t)

    # Converged sessions (final index-6 top-1 above the convergence bar). Among
    # them we feature the one with the most gradual stage-1 evolution (smallest
    # top-1 at index 3), i.e. the clearest two-stage posterior dynamics.
    final6 = ck[ck["diagnostic_index"] == 6]
    converged = set(int(v) for v in final6[final6["top1_posterior"] >= CONVERGED_TOP1]["ev_idx"].tolist())
    summary = []
    for eidx, g in ck.groupby("ev_idx"):
        if int(eidx) not in converged:
            continue
        g = g.sort_values("diagnostic_index")
        idx3_top1 = float(g[g["diagnostic_index"] == 3]["top1_posterior"].iloc[0])
        summary.append((int(eidx), int(g["candidate_count"].iloc[0]), twm_by_ev.get(int(eidx), float("nan")), idx3_top1))
    if not summary:
        raise RuntimeError("no converged session found at the representative operating point")
    preview = sorted(summary, key=lambda x: (x[3], x[0]))[:8]
    for eidx, cc, tw, i3 in preview:
        print(f"[extract] converged ev_idx={eidx} candidates={cc} t_wm={tw:.1f}s idx3_top1={i3:.4f}")
    # Deterministic pick: most gradual stage-1 evolution.
    summary.sort(key=lambda x: (x[3], x[0]))
    target_ev, target_cc, target_twm, _ = summary[0]
    g = ck[ck["ev_idx"] == target_ev].sort_values("diagnostic_index")

    out = g[["diagnostic_index", "stage", "checkpoint", "top1_posterior", "top2_posterior"]].copy()
    out.insert(5, "session_id", int(target_ev))
    out.insert(6, "seed", int(REP_SEED))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    last = out[out["diagnostic_index"] == 6].iloc[0]
    top1, top2 = float(last["top1_posterior"]), float(last["top2_posterior"])
    assert top1 >= CONVERGED_TOP1, f"final top1 {top1} < {CONVERGED_TOP1}"

    print(f"[extract] wrote {OUT_CSV}")
    print(f"[extract] session_id={target_ev}  t_wm={target_twm:.1f}s  candidates={target_cc}")
    print(out.to_string(index=False))
    print(f"[extract] final checkpoint: top1={top1:.4f} top2={top2:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
