#!/usr/bin/env python3
"""Codebook-separation evidence from committed traces (codebook-confound evidence).

Zero-run analysis of a full-trace run directory (``--mode trace-representative``):

1. ``codebook_realised_statistics.csv`` — per repeat, the pool of command patterns actually issued
   (unique patterns in ``slot_sessions.csv``): pairwise ℓ1 set-point distance (min / mean; the
   ``_l1`` definition of ``matching_core``: sum over the five non-zero steps), the minimum
   within-pattern set-point gap, the minimum consecutive-step change (between successive non-zero
   set-points), and the share of patterns violating the 4 A within-pattern / 8 A consecutive-step
   rules stated for the generator. The last row aggregates over repeats.
2. ``codebook_candidate_separation.csv`` — per matcher and separation bin: for every decision, the ℓ1
   distance between the true session's pattern and the nearest *other* pattern inside the gated
   candidate set (patterns resolved at decision time, largest overlap with the decision window);
   sessions, errors and the conditional error rate per bin {≤ 10, 11–20, 21–30, 31–40, > 40 A}.

Usage: python scripts/analyze_codebook_separation.py --run-dir <trace run dir> [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [(0, 10, "<= 10"), (11, 20, "11-20"), (21, 30, "21-30"), (31, 40, "31-40"), (41, 10**9, "> 40")]
INTERNAL_GAP_RULE_A = 4
CONSEC_GAP_RULE_A = 8


def _pattern(s: str) -> list[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip() != ""]


def _l1(p: list[int], q: list[int]) -> int:
    return int(sum(abs(int(a) - int(b)) for a, b in zip(p[1:], q[1:])))


def _bin_label(d: float) -> str:
    for lo, hi, label in BINS:
        if lo <= d <= hi:
            return label
    return "> 40"


def pool_statistics(slot_sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rep, g in slot_sessions.groupby("repeat_idx"):
        pats = sorted({str(p) for p in g["pattern"]})
        P = [_pattern(p) for p in pats]
        pair = [_l1(a, b) for a, b in itertools.combinations(P, 2)]
        internal = [min(abs(a - b) for a, b in itertools.combinations(p[1:], 2)) for p in P]
        consec = [min(abs(p[k + 1] - p[k]) for k in range(1, len(p) - 1)) for p in P]
        rows.append({
            "repeat_idx": int(rep),
            "patterns_used": len(P),
            "pairwise_l1_min_a": int(min(pair)),
            "pairwise_l1_mean_a": float(np.mean(pair)),
            "pairs_below_40a": int(sum(1 for d in pair if d < 40)),
            "pairs_total": len(pair),
            "internal_gap_min_a": int(min(internal)),
            "consec_step_min_a": int(min(consec)),
            "patterns_violating_internal_4a": int(sum(1 for v in internal if v < INTERNAL_GAP_RULE_A)),
            "patterns_violating_consec_8a": int(sum(1 for v in consec if v < CONSEC_GAP_RULE_A)),
        })
    df = pd.DataFrame(rows).sort_values("repeat_idx")
    total = {
        "repeat_idx": "all",
        "patterns_used": int(df["patterns_used"].sum()),
        "pairwise_l1_min_a": int(df["pairwise_l1_min_a"].min()),
        "pairwise_l1_mean_a": float(df["pairwise_l1_mean_a"].mean()),
        "pairs_below_40a": int(df["pairs_below_40a"].sum()),
        "pairs_total": int(df["pairs_total"].sum()),
        "internal_gap_min_a": int(df["internal_gap_min_a"].min()),
        "consec_step_min_a": int(df["consec_step_min_a"].min()),
        "patterns_violating_internal_4a": int(df["patterns_violating_internal_4a"].sum()),
        "patterns_violating_consec_8a": int(df["patterns_violating_consec_8a"].sum()),
    }
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _load_jsonl(path: Path, keep_keys: list[str]) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({k: r.get(k) for k in keep_keys})
    return pd.DataFrame(rows)


def candidate_separation(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = _load_jsonl(run_dir / "event_trace.jsonl", ["algorithm_id", "repeat_idx", "ev_idx", "gt_slot", "pred_slot", "start_est_ts", "effective_end_ts", "candidate_count", "gt_in_candidates"])
    costs = _load_jsonl(run_dir / "cost_trace.jsonl", ["algorithm_id", "repeat_idx", "ev_idx", "slot"])
    ev_meta = pd.read_csv(run_dir / "ev_meta.csv")[["repeat_idx", "ev_idx", "pattern"]]
    ss = pd.read_csv(run_dir / "slot_sessions.csv")
    ss["start_ts"] = pd.to_datetime(ss["start_ts"]); ss["end_ts"] = pd.to_datetime(ss["end_ts"])
    ss_by_key = {k: g for k, g in ss.groupby(["repeat_idx", "slot"])}
    events["start_est_ts"] = pd.to_datetime(events["start_est_ts"]); events["effective_end_ts"] = pd.to_datetime(events["effective_end_ts"])
    events = events.merge(ev_meta, on=["repeat_idx", "ev_idx"], how="left", validate="many_to_one")
    cand = costs.groupby(["algorithm_id", "repeat_idx", "ev_idx"])["slot"].apply(list).to_dict()

    def _slot_pattern(rep: int, slot: int, t0, t1) -> str:
        g = ss_by_key.get((int(rep), int(slot)))
        if g is None or len(g) == 0:
            return ""
        overlap = (np.minimum(g["end_ts"], t1) - np.maximum(g["start_ts"], t0)).dt.total_seconds()
        if float(overlap.max()) <= 0:
            return ""
        return str(g.iloc[int(np.argmax(overlap.to_numpy()))]["pattern"])

    recs = []
    for r in events.itertuples(index=False):
        slots = cand.get((r.algorithm_id, int(r.repeat_idx), int(r.ev_idx)), [])
        true_p = _pattern(r.pattern)
        others = []
        for s in slots:
            if int(s) == int(r.gt_slot):
                continue
            q = _slot_pattern(r.repeat_idx, s, r.start_est_ts, r.effective_end_ts)
            if q:
                others.append(_l1(true_p, _pattern(q)))
        recs.append({
            "algorithm_id": r.algorithm_id, "repeat_idx": int(r.repeat_idx), "ev_idx": int(r.ev_idx),
            "candidate_count": int(r.candidate_count), "other_candidates": len(others),
            "nearest_other_l1_a": (min(others) if others else np.nan),
            "misidentified": int(r.gt_slot) != int(r.pred_slot),
        })
    per = pd.DataFrame(recs)
    per["separation_bin"] = per["nearest_other_l1_a"].map(lambda d: "no other candidate" if pd.isna(d) else _bin_label(float(d)))
    order = [b[2] for b in BINS] + ["no other candidate"]
    rows = []
    for algo, g in per.groupby("algorithm_id"):
        for label in order:
            h = g[g["separation_bin"] == label]
            if len(h) == 0:
                continue
            rows.append({"algorithm_id": algo, "separation_bin": label, "sessions": int(len(h)), "errors": int(h["misidentified"].sum()),
                         "conditional_error_rate": float(h["misidentified"].mean()), "share_of_sessions": float(len(h) / len(g))})
        rows.append({"algorithm_id": algo, "separation_bin": "all", "sessions": int(len(g)), "errors": int(g["misidentified"].sum()),
                     "conditional_error_rate": float(g["misidentified"].mean()), "share_of_sessions": 1.0})
    return pd.DataFrame(rows), per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("case_study/scalability_analysis/accuracy"))
    args = ap.parse_args()
    ss = pd.read_csv(args.run_dir / "slot_sessions.csv")
    stats = pool_statistics(ss)
    stats.to_csv(args.out_dir / "codebook_realised_statistics.csv", index=False)
    table, per = candidate_separation(args.run_dir)
    table.to_csv(args.out_dir / "codebook_candidate_separation.csv", index=False)
    print(stats.to_string(index=False))
    print(table.to_string(index=False))
    print(json.dumps({"run_dir": str(args.run_dir), "decisions": int(len(per)), "errors": int(per["misidentified"].sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
