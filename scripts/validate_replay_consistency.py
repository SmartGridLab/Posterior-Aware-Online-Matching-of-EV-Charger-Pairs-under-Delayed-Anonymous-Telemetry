#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick reproducibility validator for algo-compare replay inputs.

Runs two tiny compare jobs with identical config/seed and checks that
scenario/ingestion realization signatures are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GENERATED_OUTPUT_DIR = REPO_ROOT / ".generated"

import pandas as pd

from case_study.section_vi_reproduction import run_algorithm_compare_report


def _raw_csv_path(result: dict) -> Path:
    rel = str((result.get("plot_paths") or {}).get("raw_csv") or "").strip()
    if rel == "":
        raise RuntimeError("raw_csv path missing in result")
    return (GENERATED_OUTPUT_DIR / rel).resolve()


def _signature_digest(raw_csv: Path) -> str:
    df = pd.read_csv(raw_csv)
    cols = [
        "scenario_seed",
        "repeat_idx",
        "ev_count",
        "tau_s",
        "charger_sample_s",
        "ev_sample_s",
        "matcher_delay_max_s",
        "candidate_margin_s",
        "scenario_hash",
        "ingestion_signature_sha256",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"raw CSV missing required columns: {missing}")

    d = (
        df[cols]
        .copy()
        .sort_values(cols)
        .reset_index(drop=True)
    )
    payload = d.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_once(base_seed: int, ev_count: int, repeats: int) -> dict:
    return run_algorithm_compare_report(
        ev_counts=[int(ev_count)],
        repeats=int(repeats),
        error_mode="ci95",
        base_seed=int(base_seed),
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=["time_only_baseline", "bayesian_windowed"],
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[5],
        ingestion_enabled=True,
        ev_ingest_delay_mean_s=2.0,
        ev_ingest_delay_jitter_s=1.5,
        ev_ingest_delay_max_s=10.0,
        ev_ingest_loss_prob=0.02,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate deterministic replay of scenario+ingestion for algo-compare")
    ap.add_argument("--base-seed", type=int, default=2187631072)
    ap.add_argument("--ev-count", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    r1 = _run_once(args.base_seed, args.ev_count, args.repeats)
    r2 = _run_once(args.base_seed, args.ev_count, args.repeats)

    raw1 = _raw_csv_path(r1)
    raw2 = _raw_csv_path(r2)

    h1 = _signature_digest(raw1)
    h2 = _signature_digest(raw2)

    rep1 = (r1.get("replay_checks") or {})
    rep2 = (r2.get("replay_checks") or {})

    ok = (h1 == h2) and int(rep1.get("ingestion_signature_mismatch_count", 0)) == 0 and int(rep2.get("ingestion_signature_mismatch_count", 0)) == 0

    out = {
        "ok": bool(ok),
        "base_seed": int(args.base_seed),
        "run_id_1": str(r1.get("run_id", "")),
        "run_id_2": str(r2.get("run_id", "")),
        "digest_1": str(h1),
        "digest_2": str(h2),
        "replay_checks_1": rep1,
        "replay_checks_2": rep2,
        "raw_csv_1": str(raw1),
        "raw_csv_2": str(raw2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
