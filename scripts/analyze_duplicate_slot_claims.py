#!/usr/bin/env python3
"""Duplicate slot claims across decision batches, from a full-trace compare run
(Section VII-E of the manuscript).

The runtime enforces the one-to-one constraint of Eq. (5) among the sessions of one
matured batch. A slot finalised in an earlier batch is not reserved, so a later batch can
finalise the same slot while the earlier session is still connected. This script counts
how often that happens in a traced run, for every algorithm in the trace.

Input: a run directory written with ``--trace-level full`` (e.g. by
``python scripts/reproduce_all.py --mode trace-representative --trace-level full``),
containing ``event_trace.jsonl`` with one row per finalised session.

A *duplicate claim* is a pair of sessions (i, j) of the same repeat with the same predicted
slot, decided at different ticks (decision_ts_i < decision_ts_j), for which the earlier
session is still connected at the later decision (session_end_ts_i > decision_ts_j).
Because a slot hosts one session at a time, at least one session of such a pair is
misidentified; the script reports that split as well.

Output (``--out-dir``, default ``case_study/scalability_analysis/accuracy/``):
``duplicate_slot_claims.csv`` — one row per algorithm with the decision and error counts,
the number of duplicate-claim pairs, the number of distinct later decisions involved, the
split by how many sessions of the pair are misidentified, the number of pairs decided
inside one batch (zero by construction), and how many misidentified sessions appear in at
least one pair.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"

FIELDS = ("algorithm_id", "repeat_idx", "ev_idx", "gt_slot", "pred_slot",
          "decision_ts", "session_end_ts")


def load_events(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: r.get(k) for k in FIELDS})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no event-trace rows; run with --trace-level full")
    for col in ("decision_ts", "session_end_ts"):
        df[col] = pd.to_datetime(df[col])
    return df


def count_duplicates(df: pd.DataFrame) -> dict:
    pairs = same_batch = one_wrong = both_wrong = 0
    later_decisions: set = set()
    involved: set = set()
    for (repeat, slot), group in df[df["pred_slot"] >= 0].groupby(["repeat_idx", "pred_slot"]):
        rows = group.sort_values("decision_ts").to_dict("records")
        for a_i in range(len(rows)):
            for b_i in range(a_i + 1, len(rows)):
                a, b = rows[a_i], rows[b_i]
                if a["decision_ts"] == b["decision_ts"]:
                    same_batch += 1
                    continue
                if a["session_end_ts"] <= b["decision_ts"]:
                    continue
                pairs += 1
                later_decisions.add((repeat, b["ev_idx"]))
                wrong = [r for r in (a, b) if r["pred_slot"] != r["gt_slot"]]
                if len(wrong) == 2:
                    both_wrong += 1
                elif len(wrong) == 1:
                    one_wrong += 1
                for r in wrong:
                    involved.add((repeat, r["ev_idx"]))
    return dict(decisions=int(len(df)),
                errors=int((df["pred_slot"] != df["gt_slot"]).sum()),
                duplicate_pairs=pairs,
                later_decisions=len(later_decisions),
                pairs_one_misidentified=one_wrong,
                pairs_both_misidentified=both_wrong,
                same_batch_pairs=same_batch,
                misidentified_sessions_involved=len(involved))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ACCURACY_DIR)
    parser.add_argument("--table-name", default="duplicate_slot_claims.csv")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    events = load_events(args.run_dir / "event_trace.jsonl")
    records = []
    for algorithm, group in events.groupby("algorithm_id"):
        record = {"algorithm_id": algorithm}
        record.update(count_duplicates(group))
        records.append(record)
    table = pd.DataFrame(records).sort_values("algorithm_id").reset_index(drop=True)
    assert (table["same_batch_pairs"] == 0).all(), "the one-to-one constraint failed inside a batch"
    assert (table["duplicate_pairs"] == table["pairs_one_misidentified"] + table["pairs_both_misidentified"]).all(), \
        "every duplicate claim must contain a misidentified session"
    print(table.to_string(index=False))
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_dir / args.table_name, index=False)
        print(f"wrote {args.out_dir / args.table_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
