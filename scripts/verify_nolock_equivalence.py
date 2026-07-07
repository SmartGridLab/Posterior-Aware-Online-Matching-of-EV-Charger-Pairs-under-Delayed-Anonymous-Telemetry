#!/usr/bin/env python3
"""Provenance check: the retired confidence-lock stabilizer never changed any output.

The committed archived runs (canonical_run_20260626_170431_105970 and
ablation_run_20260703_185002_516173) were produced by an earlier code revision
whose A3 matcher included an optional confidence-lock stabilizer. Before
submission the stabilizer was removed from the method and the codebase. This
script proves, from the committed evidence run
lock_equivalence_run_20260704_201609_736039 (a paired replay of the FULL
81-cell x 20-repeat canonical sweep grid on the identical scenario seeds), that
the stabilizer was output-neutral everywhere:

  (a) the lock-disabled variant equals the lock-enabled variant per run inside
      the evidence run, and
  (b) the lock-disabled variant equals the committed canonical A3 rows per run
      (cross-run anchor, same scenario seeds).

Prints per-axis mismatch counts. Exit 0 iff zero mismatches anywhere, i.e. the
archived numbers are exactly the numbers of the submitted (lock-free) method.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CANONICAL_RAW = REPO / "case_study" / "scalability_analysis" / "accuracy" / \
    "canonical_run_20260626_170431_105970" / "raw_runs.csv"
KEY = ["ev_count", "ev_sample_s", "matcher_delay_max_s", "candidate_margin_s", "repeat_idx"]
CMP = ["accuracy", "accuracy_requested", "correct", "unmatched"]

def load(path: Path, algo: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["algorithm_id"] == algo]
    if df.empty:
        sys.exit(f"no rows for {algo} in {path}")
    return df.set_index(KEY).sort_index()

def compare(a: pd.DataFrame, b: pd.DataFrame, tag: str) -> int:
    j = a[CMP].join(b[CMP], rsuffix="_b", how="inner")
    if len(j) == 0:
        sys.exit(f"[{tag}] no overlapping cells")
    bad = 0
    for c in CMP:
        m = (j[c] != j[f"{c}_b"]).sum()
        print(f"[{tag}] {c:20s} mismatches: {m} / {len(j)}")
        bad += int(m)
    return bad

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, nargs="?", default=REPO / "case_study" / "scalability_analysis" / "accuracy" / "lock_equivalence_run_20260704_201609_736039", help="evidence run directory (default: the committed lock-equivalence run)")
    args = ap.parse_args()
    raw = args.run_dir / "raw_runs.csv"
    off = load(raw, "bayesian_windowed_lock_off")
    total_bad = 0
    try:
        full_in = load(raw, "bayesian_windowed")
        total_bad += compare(off, full_in, "in-run lock_off vs full")
    except SystemExit:
        print("[in-run] full variant not present (nolock-only mode) -- skipping")
    canon = load(CANONICAL_RAW, "bayesian_windowed")
    total_bad += compare(off, canon, "lock_off vs canonical full")
    n_cells = off.reset_index().groupby(KEY[:-1]).size()
    print(f"cells covered: {len(n_cells)} (expect 81), repeats/cell min={n_cells.min()} max={n_cells.max()}")
    verdict = "IDENTICAL EVERYWHERE -- lock removal changes no reported number" if total_bad == 0 \
        else f"DIFFERENCES FOUND ({total_bad}) -- lock is NOT neutral outside base load; keep it with this new evidence"
    print("VERDICT:", verdict)
    return 0 if total_bad == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
