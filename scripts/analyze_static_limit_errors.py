#!/usr/bin/env python3
"""Static-limit anchor diagnostic, trace check (static-limit anchor, TABLE S2).

Reads one or more full-trace replays of static-limit repeats (``--mode static-limit --cell ideal
--algorithms nomura_original_interval_hungarian --repeats 1 --base-seed <scenario seed> --trace-level
full``) and lists every misidentified session with the ℓ1 set-point distance between the true
pattern and the chosen pattern, the true pattern's distance to its nearest other pattern in the
(fixed-binding) codebook, and whether the confused pair is a near-duplicate (ℓ1 ≤ 8 A).

Outputs ``static_limit_a2_ideal_error_pairs.csv`` (one row per error) and a JSON summary line.

Usage: python scripts/analyze_static_limit_errors.py --run-dir <dir> [--run-dir <dir> ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NEAR_DUP_A = 8


def _pattern(s: str) -> list[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip() != ""]


def _l1(p: list[int], q: list[int]) -> int:
    return int(sum(abs(int(a) - int(b)) for a, b in zip(p[1:], q[1:])))


def analyse(run_dir: Path) -> pd.DataFrame:
    events = []
    with (run_dir / "event_trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                events.append({k: r.get(k) for k in ("algorithm_id", "scenario_seed", "repeat_idx", "ev_idx", "gt_slot", "pred_slot", "candidate_count")})
    ev = pd.DataFrame(events)
    ss = pd.read_csv(run_dir / "slot_sessions.csv")
    slot_pattern = {int(s): str(p) for s, p in zip(ss["slot"], ss["pattern"])}
    pats = {s: _pattern(p) for s, p in slot_pattern.items()}
    rows = []
    for r in ev.itertuples(index=False):
        if int(r.gt_slot) == int(r.pred_slot):
            continue
        tp, cp = pats[int(r.gt_slot)], pats[int(r.pred_slot)]
        others = sorted((_l1(tp, q), s) for s, q in pats.items() if s != int(r.gt_slot))
        rows.append({
            "run_id": run_dir.name, "algorithm_id": r.algorithm_id, "scenario_seed": int(r.scenario_seed), "ev_idx": int(r.ev_idx),
            "gt_slot": int(r.gt_slot), "pred_slot": int(r.pred_slot), "true_pattern": slot_pattern[int(r.gt_slot)], "chosen_pattern": slot_pattern[int(r.pred_slot)],
            "l1_true_vs_chosen_a": _l1(tp, cp), "l1_true_to_nearest_other_a": int(others[0][0]), "chosen_is_nearest_other": int(others[0][1]) == int(r.pred_slot),
            "chosen_rank_by_l1": int(next(i for i, (_, s) in enumerate(others, start=1) if s == int(r.pred_slot))),
            "near_duplicate_pair": _l1(tp, cp) <= NEAR_DUP_A,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=Path("case_study/scalability_analysis/accuracy/static_limit_a2_ideal_error_pairs.csv"))
    args = ap.parse_args()
    df = pd.concat([analyse(d) for d in args.run_dir], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(df[["scenario_seed", "ev_idx", "gt_slot", "pred_slot", "l1_true_vs_chosen_a", "l1_true_to_nearest_other_a", "chosen_is_nearest_other", "chosen_rank_by_l1"]].to_string(index=False))
    summ = {"errors": int(len(df)), "near_duplicate_pairs_le_8a": int(df["near_duplicate_pair"].sum()),
            "l1_true_vs_chosen_median_a": float(df["l1_true_vs_chosen_a"].median()), "l1_true_vs_chosen_min_a": int(df["l1_true_vs_chosen_a"].min()),
            "l1_true_vs_chosen_max_a": int(df["l1_true_vs_chosen_a"].max()), "chosen_is_nearest_other": int(df["chosen_is_nearest_other"].sum()),
            "chosen_rank_by_l1_median": float(df["chosen_rank_by_l1"].median()), "per_seed": df.groupby("scenario_seed").size().to_dict()}
    print(json.dumps(summ, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
