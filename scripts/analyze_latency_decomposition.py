#!/usr/bin/env python3
"""Decompose the decision latency of a full-trace compare run (Section VI-A latency decomposition).

For every finalised session the shared runtime (``pair_identification/runtime_event_loop.py``)
fixes

    decision_ts = first evaluation tick >= watermark_ready_ts
    watermark_ready_ts = effective_end_ts + d_max
    effective_end_ts   = min(start_est_ts + W, session_end_ts)

and reports ``decision_latency_s = decision_ts - arrival_ts`` (true arrival). The latency
therefore decomposes exactly into four terms:

    start_offset_s   = start_est_ts - arrival_ts        (EV-side start estimate vs true arrival)
    window_s         = effective_end_ts - start_est_ts  (W = 360 s, shorter only for truncated sessions)
    watermark_s      = watermark_ready_ts - effective_end_ts   (= d_max)
    tick_residual_s  = decision_ts - watermark_ready_ts  (0 .. evaluation interval)

This script reads ``event_trace.jsonl`` of a run directory, verifies the identity per session,
recomputes the per-repeat 90th percentile exactly as the harness does
(``numpy.quantile(latencies, 0.90)``, linear interpolation) and checks it against
``raw_runs.csv``. Output: a component table on stdout and, with ``--out``, a CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy" / "latency_decomposition_representative.csv"
COMPONENTS = ["start_offset_s", "window_s", "watermark_s", "tick_residual_s", "decision_latency_s"]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def decompose(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    for col in ("arrival_ts", "start_est_ts", "effective_end_ts", "watermark_ready_ts", "decision_ts"):
        df[col] = pd.to_datetime(df[col])
    sec = lambda a, b: (df[a] - df[b]).dt.total_seconds()  # noqa: E731
    df["start_offset_s"] = sec("start_est_ts", "arrival_ts")
    df["window_s"] = sec("effective_end_ts", "start_est_ts")
    df["watermark_s"] = sec("watermark_ready_ts", "effective_end_ts")
    df["tick_residual_s"] = sec("decision_ts", "watermark_ready_ts")
    df["reconstructed_latency_s"] = df["start_offset_s"] + df["window_s"] + df["watermark_s"] + df["tick_residual_s"]
    df["identity_error_s"] = (df["reconstructed_latency_s"] - df["decision_latency_s"].astype(float)).abs()
    df["truncated"] = df["window_s"] < df["window_s"].max() - 1e-9
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="compare run directory with event_trace.jsonl and raw_runs.csv")
    parser.add_argument("--algorithm", default="bayesian_windowed")
    parser.add_argument("--out", type=Path, default=None, help="write the per-repeat component table as CSV")
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    events = load_jsonl(args.run_dir / "event_trace.jsonl")
    if events.empty:
        raise SystemExit("event_trace.jsonl is empty — run the replay with --trace-level full")
    events = events[events["algorithm_id"] == args.algorithm]
    raw = pd.read_csv(args.run_dir / "raw_runs.csv")
    raw = raw[raw["algorithm_id"] == args.algorithm].set_index("repeat_idx")

    df = decompose(events)
    max_identity_error = float(df["identity_error_s"].max())

    per_repeat = []
    for repeat_idx, grp in df.groupby("repeat_idx"):
        p90_trace = float(np.quantile(grp["decision_latency_s"].astype(float).to_numpy(), 0.90))
        p90_raw = float(raw.loc[int(repeat_idx), "p90_latency_s"])
        row = {
            "repeat_idx": int(repeat_idx),
            "sessions": int(len(grp)),
            "p90_latency_trace_s": p90_trace,
            "p90_latency_raw_s": p90_raw,
            "p90_match": bool(abs(p90_trace - p90_raw) <= args.tol),
            "truncated_share": float(grp["truncated"].mean()),
        }
        for c in COMPONENTS:
            vals = grp[c].astype(float).to_numpy()
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_p90"] = float(np.quantile(vals, 0.90))
            row[f"{c}_min"] = float(vals.min())
            row[f"{c}_max"] = float(vals.max())
        per_repeat.append(row)
    table = pd.DataFrame(per_repeat).sort_values("repeat_idx")

    all_match = bool(table["p90_match"].all())
    summary = {
        "run_dir": str(args.run_dir),
        "algorithm": args.algorithm,
        "sessions": int(len(df)),
        "repeats": int(len(table)),
        "identity_holds": bool(max_identity_error <= args.tol),
        "max_identity_error_s": max_identity_error,
        "p90_matches_raw_runs_all_repeats": all_match,
        "p90_latency_mean_over_repeats_s": float(table["p90_latency_raw_s"].mean()),
        "truncated_share_overall": float(df["truncated"].mean()),
        "components_pooled": {
            c: {
                "mean": float(df[c].astype(float).mean()),
                "p90": float(np.quantile(df[c].astype(float).to_numpy(), 0.90)),
                "min": float(df[c].astype(float).min()),
                "max": float(df[c].astype(float).max()),
            }
            for c in COMPONENTS
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0 if (summary["identity_holds"] and all_match) else 1


if __name__ == "__main__":
    raise SystemExit(main())
