#!/usr/bin/env python3
"""Failure taxonomy and residual-error disaggregation from a full-trace compare run
(TABLE 8 of the manuscript; the data files keep their `table9_` names).

Inputs (a run directory written with ``--trace-level full``, e.g. by
``python scripts/reproduce_all.py --mode trace-representative``):

* ``event_trace.jsonl``  — one row per finalised session (ground-truth vs predicted slot,
  candidate-set size, prefix/shared sample counts, start estimate, effective end, decision).
* ``ev_meta.csv``        — per-session metadata: MCCT pattern and the realised sensing
  distortion (extra sampling delay, gain, bias).
* ``slot_sessions.csv``  — slot-side session intervals with their patterns (reference plane).
* ``cost_trace.jsonl``   — per (session, candidate) assignment cost / posterior (used for the
  cost and posterior of the true vs the chosen slot; streamed, only misidentified sessions kept).

Every misidentified session receives exactly one primary cause by the documented priority
rule (first match wins):

1. ``truncation``  — the decision window was cut by the session end (t_eff < W);
2. ``ingestion``   — at least one required EV sample was late or lost at the decision tick
   (``prefix_samples - shared_points >= 1``);
3. ``sensing``     — the realised sensing distortion of the session exceeds 2 σ of the
   sensing model: extra sampling delay ≥ 15 s (the DTW band), |gain − 1| ≥ 0.06 or
   |bias| ≥ 0.7 A;
4. ``overlap``     — remainder: the chosen slot's MCCT pattern was confused with the true
   one; the ℓ1 set-point distance between the two patterns (five non-zero steps) is reported.

Outputs (``--out-dir``, default ``case_study/scalability_analysis/accuracy/``):

* ``table9_failure_taxonomy.csv`` — long table: breakdown ∈ {candidate_set_size,
  window, steps_observed, cause}, bin, sessions, errors, error_rate, share_of_errors.
* ``failure_cases_representative.csv`` — one row per misidentified session with every
  feature used above.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"

# 2 σ of the sensing model in matching_core (gain std 0.03, bias std 0.35 A); the extra
# sampling delay threshold is the Sakoe–Chiba band (BAND_SEC = 15 s).
SENSING_DELAY_THRESH_S = 15
SENSING_GAIN_THRESH = 0.06
SENSING_BIAS_THRESH_A = 0.7
CAUSES = ["truncation", "ingestion", "sensing", "overlap"]
CANDIDATE_BINS = [(0, 10, "≤ 10"), (11, 25, "11–25"), (26, 40, "26–40"), (41, 10**9, "> 40")]


def load_jsonl(path: Path, keep=None) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if keep is None or keep(r):
                rows.append(r)
    return pd.DataFrame(rows)


def _pattern(s: str) -> list[int]:
    return [int(x) for x in str(s).split("|") if str(x).strip() != ""]


def _l1(p: list[int], q: list[int]) -> float:
    return float(sum(abs(int(a) - int(b)) for a, b in zip(p[1:], q[1:])))


def _bin_label(n: int) -> str:
    for lo, hi, label in CANDIDATE_BINS:
        if lo <= int(n) <= hi:
            return label
    return "> 40"


def build_features(events: pd.DataFrame, ev_meta: pd.DataFrame, slot_sessions: pd.DataFrame) -> pd.DataFrame:
    df = events.merge(
        ev_meta[
            ["repeat_idx", "ev_idx", "pattern", "session_len_s", "ev_sensor_delay_s", "ev_sensor_gain", "ev_sensor_bias_a"]
        ],
        on=["repeat_idx", "ev_idx"],
        how="left",
        validate="one_to_one",
    )
    for col in ("arrival_ts", "start_est_ts", "effective_end_ts", "decision_end_ts", "decision_ts"):
        df[col] = pd.to_datetime(df[col])
    df["t_eff_s"] = (df["effective_end_ts"] - df["start_est_ts"]).dt.total_seconds()
    df["window_s"] = (df["decision_end_ts"] - df["start_est_ts"]).dt.total_seconds()
    df["truncated"] = df["t_eff_s"] < df["window_s"] - 1e-9
    tau = float(df["tau_s"].iloc[0]) if "tau_s" in df.columns else 60.0
    n_steps = int(round(float(df["window_s"].max()) / tau))
    df["steps_observed"] = np.clip(np.ceil(df["t_eff_s"] / tau - 1e-9).astype(int), 1, n_steps)
    df["late_or_lost_samples"] = (df["prefix_samples"].astype(int) - df["shared_points"].astype(int)).clip(lower=0)
    df["sensing_2sigma"] = (
        (df["ev_sensor_delay_s"].astype(float) >= SENSING_DELAY_THRESH_S)
        | ((df["ev_sensor_gain"].astype(float) - 1.0).abs() >= SENSING_GAIN_THRESH)
        | (df["ev_sensor_bias_a"].astype(float).abs() >= SENSING_BIAS_THRESH_A)
    )
    df["misidentified"] = df["gt_slot"].astype(int) != df["pred_slot"].astype(int)
    df["candidate_bin"] = df["candidate_count"].astype(int).map(_bin_label)

    # chosen slot's pattern during the EV window (largest overlap with [start_est, effective_end])
    ss = slot_sessions.copy()
    ss["start_ts"] = pd.to_datetime(ss["start_ts"])
    ss["end_ts"] = pd.to_datetime(ss["end_ts"])
    ss_by_key = {k: g for k, g in ss.groupby(["repeat_idx", "slot"])}

    def _chosen_pattern(row) -> str:
        g = ss_by_key.get((int(row["repeat_idx"]), int(row["pred_slot"])))
        if g is None or len(g) == 0:
            return ""
        overlap = (np.minimum(g["end_ts"], row["effective_end_ts"]) - np.maximum(g["start_ts"], row["start_est_ts"])).dt.total_seconds()
        if float(overlap.max()) <= 0:
            return ""
        return str(g.iloc[int(np.argmax(overlap.to_numpy()))]["pattern"])

    df["chosen_pattern"] = df.apply(_chosen_pattern, axis=1)
    df["l1_true_vs_chosen_a"] = [
        _l1(_pattern(p), _pattern(q)) if (p and q and p == p and q == q) else float("nan")
        for p, q in zip(df["pattern"], df["chosen_pattern"])
    ]

    def _cause(row) -> str:
        if not bool(row["misidentified"]):
            return ""
        if bool(row["truncated"]):
            return "truncation"
        if int(row["late_or_lost_samples"]) >= 1:
            return "ingestion"
        if bool(row["sensing_2sigma"]):
            return "sensing"
        return "overlap"

    df["primary_cause"] = df.apply(_cause, axis=1)
    return df


def attach_costs(df: pd.DataFrame, run_dir: Path, algorithm: str) -> pd.DataFrame:
    wrong = df[df["misidentified"]]
    keys = set(zip(wrong["repeat_idx"].astype(int), wrong["ev_idx"].astype(int)))
    if not keys:
        return df
    ct = load_jsonl(
        run_dir / "cost_trace.jsonl",
        keep=lambda r: r.get("algorithm_id") == algorithm and (int(r["repeat_idx"]), int(r["ev_idx"])) in keys,
    )
    if ct.empty:
        return df
    gt_map = {(int(r.repeat_idx), int(r.ev_idx)): int(r.gt_slot) for r in wrong.itertuples()}
    pred_map = {(int(r.repeat_idx), int(r.ev_idx)): int(r.pred_slot) for r in wrong.itertuples()}
    ct["key"] = list(zip(ct["repeat_idx"].astype(int), ct["ev_idx"].astype(int)))
    ct["is_gt"] = [int(s) == gt_map.get(k, -1) for s, k in zip(ct["slot"], ct["key"])]
    ct["is_pred"] = [int(s) == pred_map.get(k, -1) for s, k in zip(ct["slot"], ct["key"])]
    def _num(frame: pd.DataFrame, col: str) -> float:
        # JSON null (e.g. no posterior for the non-Bayesian matchers) -> NaN
        if len(frame) == 0 or col not in frame.columns:
            return float("nan")
        v = pd.to_numeric(pd.Series([frame[col].iloc[0]]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) else float("nan")

    feats = {}
    for key, g in ct.groupby("key"):
        gt = g[g["is_gt"]]
        pr = g[g["is_pred"]]
        feats[key] = {
            "cost_true_slot": _num(gt, "assignment_cost"),
            "cost_chosen_slot": _num(pr, "assignment_cost"),
            "posterior_true_slot": _num(gt, "posterior"),
            "posterior_chosen_slot": _num(pr, "posterior"),
            "true_slot_candidate_position": int(gt["candidate_rank"].iloc[0]) if len(gt) else -1,
        }
    for col in ("cost_true_slot", "cost_chosen_slot", "posterior_true_slot", "posterior_chosen_slot", "true_slot_candidate_position"):
        df[col] = [feats.get((int(r), int(e)), {}).get(col, float("nan")) for r, e in zip(df["repeat_idx"], df["ev_idx"])]
    return df


def tabulate(df: pd.DataFrame) -> pd.DataFrame:
    total_err = int(df["misidentified"].sum())
    rows = []

    def _add(breakdown: str, label: str, mask: pd.Series, errors_mask: pd.Series | None = None) -> None:
        sessions = int(mask.sum())
        errors = int((df["misidentified"] & mask).sum()) if errors_mask is None else int(errors_mask.sum())
        rows.append(
            {
                "breakdown": breakdown,
                "bin": label,
                "sessions": sessions,
                "errors": errors,
                "error_rate": (errors / sessions) if sessions > 0 else float("nan"),
                "share_of_errors": (errors / total_err) if total_err > 0 else float("nan"),
            }
        )

    for _lo, _hi, label in CANDIDATE_BINS:
        _add("candidate_set_size", label, df["candidate_bin"] == label)
    _add("window", "t_eff = W", ~df["truncated"])
    _add("window", "t_eff < W (truncated)", df["truncated"])
    for k in sorted(df["steps_observed"].unique()):
        _add("steps_observed", str(int(k)), df["steps_observed"] == k)
    # cause rows: sessions = sessions exposed to the condition; errors = errors attributed to the cause
    _add("cause", "truncation", df["truncated"], df["primary_cause"] == "truncation")
    _add("cause", "ingestion (late/lost sample before watermark)", df["late_or_lost_samples"] >= 1, df["primary_cause"] == "ingestion")
    _add("cause", "sensing (2σ distortion)", df["sensing_2sigma"], df["primary_cause"] == "sensing")
    _add("cause", "step-signature overlap (remainder)", pd.Series(True, index=df.index), df["primary_cause"] == "overlap")
    _add("all", "all sessions", pd.Series(True, index=df.index))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--algorithm", default="bayesian_windowed")
    parser.add_argument("--out-dir", type=Path, default=ACCURACY_DIR)
    parser.add_argument("--table-name", default="table9_failure_taxonomy.csv")
    parser.add_argument("--cases-name", default="failure_cases_representative.csv")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    events = load_jsonl(args.run_dir / "event_trace.jsonl", keep=lambda r: r.get("algorithm_id") == args.algorithm)
    if events.empty:
        raise SystemExit("no event-trace rows for this algorithm; run with --trace-level full")
    ev_meta = pd.read_csv(args.run_dir / "ev_meta.csv")
    slot_sessions = pd.read_csv(args.run_dir / "slot_sessions.csv")
    df = build_features(events, ev_meta, slot_sessions)
    df = attach_costs(df, args.run_dir, args.algorithm)
    table = tabulate(df)

    wrong = df[df["misidentified"]].copy()
    assert (wrong["primary_cause"].isin(CAUSES)).all(), "every misidentified session needs exactly one cause"
    summary = {
        "run_dir": str(args.run_dir),
        "algorithm": args.algorithm,
        "repeats": int(df["repeat_idx"].nunique()),
        "sessions": int(len(df)),
        "errors": int(len(wrong)),
        "error_rate": float(len(wrong) / len(df)),
        "accuracy": float(1.0 - len(wrong) / len(df)),
        "gt_in_candidates_all": bool(df["gt_in_candidates"].all()),
        "cause_counts": {c: int((wrong["primary_cause"] == c).sum()) for c in CAUSES},
        "cause_shares": {c: float((wrong["primary_cause"] == c).mean()) if len(wrong) else float("nan") for c in CAUSES},
        "l1_true_vs_chosen_a": {
            "mean": float(wrong["l1_true_vs_chosen_a"].mean()) if len(wrong) else float("nan"),
            "median": float(wrong["l1_true_vs_chosen_a"].median()) if len(wrong) else float("nan"),
            "min": float(wrong["l1_true_vs_chosen_a"].min()) if len(wrong) else float("nan"),
        },
        "median_candidate_count_errors": float(wrong["candidate_count"].median()) if len(wrong) else float("nan"),
        "median_candidate_count_all": float(df["candidate_count"].median()),
    }
    print(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))

    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_dir / args.table_name, index=False)
        cols = [
            "repeat_idx", "ev_idx", "gt_slot", "pred_slot", "primary_cause", "candidate_count", "candidate_bin",
            "true_slot_candidate_position", "selected_slot_rank", "prefix_samples", "shared_points", "late_or_lost_samples",
            "t_eff_s", "steps_observed", "truncated", "session_len_s", "decision_latency_s",
            "ev_sensor_delay_s", "ev_sensor_gain", "ev_sensor_bias_a", "sensing_2sigma",
            "pattern", "chosen_pattern", "l1_true_vs_chosen_a",
            "cost_true_slot", "cost_chosen_slot", "posterior_true_slot", "posterior_chosen_slot",
        ]
        cols = [c for c in cols if c in wrong.columns]
        wrong[cols].sort_values(["repeat_idx", "ev_idx"]).to_csv(args.out_dir / args.cases_name, index=False)
        print(f"wrote {args.out_dir / args.table_name} and {args.out_dir / args.cases_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
