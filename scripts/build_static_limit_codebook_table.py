#!/usr/bin/env python3
"""Static-limit anchor diagnostic (static-limit anchor, TABLE S2).

Regenerates, per repeat, the legacy interval codebook that the static-limit runs used (the prior
study's generator with fixed slot binding: ``_legacy_interval_generate_command_patterns`` seeded
from the repeat's scenario seed exactly as the harness does) and joins its near-duplicate
statistics with the per-repeat accuracy of the prior correlation–DTW cost (A2) in the ideal-sensing
cell. Output: ``case_study/scalability_analysis/accuracy/static_limit_a2_ideal_by_repeat.csv`` and a
Spearman rank correlation between the number of near-duplicate pairs and the accuracy.

Usage: python scripts/build_static_limit_codebook_table.py --run-dir <static_limit_ideal_sensing_nomura_* dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from case_study.section_vi_reproduction import _legacy_interval_generate_command_patterns  # noqa: E402

LEGACY_SEED_XOR = 0x1D872B41


def _l1(p, q) -> int:
    return int(sum(abs(int(a) - int(b)) for a, b in zip(p[1:], q[1:])))


def codebook_row(seed: int, n_patterns: int, step_count: int) -> dict:
    pats = _legacy_interval_generate_command_patterns(num_patterns=n_patterns, seed=int(seed) ^ LEGACY_SEED_XOR, command_step_count=step_count)
    P = [[int(v) for v in p] for p in pats]
    pair = [_l1(a, b) for a, b in itertools.combinations(P, 2)]
    return {
        "codebook_patterns": len(P), "codebook_distinct_patterns": len({tuple(p) for p in P}),
        "pairwise_l1_min_a": int(min(pair)), "pairwise_l1_mean_a": float(np.mean(pair)),
        "pairs_l1_le_4a": int(sum(1 for d in pair if d <= 4)), "pairs_l1_le_8a": int(sum(1 for d in pair if d <= 8)),
        "pairs_l1_le_12a": int(sum(1 for d in pair if d <= 12)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("case_study/scalability_analysis/accuracy/static_limit_a2_ideal_by_repeat.csv"))
    args = ap.parse_args()
    raw = pd.read_csv(args.run_dir / "raw_runs.csv")
    cfg = json.loads((args.run_dir / "config.json").read_text())
    n_slots = int(cfg.get("evse_count", 300)); steps = int(cfg.get("command_step_count", 6))
    rows = []
    for r in raw.sort_values("repeat_idx").itertuples(index=False):
        rec = {"repeat_idx": int(r.repeat_idx), "scenario_seed": int(r.scenario_seed), "algorithm_id": str(r.algorithm_id),
               "accuracy": float(r.accuracy), "errors": int(round((1.0 - float(r.accuracy)) * int(r.admitted_evs))), "sessions": int(r.admitted_evs)}
        rec.update(codebook_row(int(r.scenario_seed), n_slots, steps))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    from scipy.stats import spearmanr
    out = {"rows": int(len(df)), "distinct_patterns_min": int(df["codebook_distinct_patterns"].min()),
           "pairwise_l1_min_range_a": [int(df["pairwise_l1_min_a"].min()), int(df["pairwise_l1_min_a"].max())],
           "pairs_le_4a_range": [int(df["pairs_l1_le_4a"].min()), int(df["pairs_l1_le_4a"].max())],
           "spearman_pairs_le_4a_vs_accuracy": [float(x) for x in spearmanr(df["pairs_l1_le_4a"], df["accuracy"])],
           "spearman_pairs_le_8a_vs_accuracy": [float(x) for x in spearmanr(df["pairs_l1_le_8a"], df["accuracy"])],
           "spearman_pairs_le_12a_vs_accuracy": [float(x) for x in spearmanr(df["pairs_l1_le_12a"], df["accuracy"])],
           "spearman_l1_min_vs_accuracy": [float(x) for x in spearmanr(df["pairwise_l1_min_a"], df["accuracy"])]}
    print(df.sort_values("accuracy")[["repeat_idx", "scenario_seed", "accuracy", "errors", "codebook_distinct_patterns", "pairwise_l1_min_a", "pairs_l1_le_4a", "pairs_l1_le_8a", "pairs_l1_le_12a"]].to_string(index=False))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
