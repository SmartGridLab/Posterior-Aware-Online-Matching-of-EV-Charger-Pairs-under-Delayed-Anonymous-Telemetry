#!/usr/bin/env python3
"""Realised share of lost EV measurements in the measurement-loss cells of the impairment sweep.

The runtime draws one transport delay and one loss per EV measurement
(``pair_identification.runtime_event_loop._simulate_ingest_times``) from the ingestion stream
seeded by ``ingest_seed`` (recorded per repeat in ``raw_runs.csv``), session by session in index
order, on the fixed decision grid of ``W / 5 s + 1`` points anchored at each session's start
estimate. The draws depend only on the seed, on the number of sessions and on the number of
measurements per session (``W / Δt_EV + 1``), not on the waveforms or on the matcher, so they are
replayed here with the runtime's own function on a grid of the same length, without re-running
the matchers.

Output: ``case_study/scalability_analysis/accuracy/impairment_realised_loss.csv`` — one row per
cell of the measurement-loss axis: nominal loss probability, repeats, sessions per repeat,
measurements per session, measurements, lost measurements, realised share (all repeats pooled)
and its per-repeat minimum and maximum, and the run id.

Usage: python scripts/analyze_realised_loss.py [--out <csv>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from pair_identification.runtime_event_loop import _LOST_TS, _simulate_ingest_times  # noqa: E402

ACCURACY_DIR = REPO_ROOT / "case_study" / "scalability_analysis" / "accuracy"
OUT_CSV = ACCURACY_DIR / "impairment_realised_loss.csv"
CELL_DIR = re.compile(r"^impairment_(default|loss_[0-9.]+)_run_(\d{8}_\d{6}_\d{6})$")
GRID_STEP_S = 5
COLUMNS = ["cell", "loss_prob", "repeats", "sessions_per_repeat", "measurements_per_session", "measurements",
           "lost_measurements", "realised_share", "repeat_share_min", "repeat_share_max", "run_id"]


def replay_repeat(ingest_seed: int, sessions: int, cfg: dict) -> tuple[int, int]:
    """Replay the ingestion draws of one repeat; return (measurements, lost measurements)."""
    window_s = int(cfg["window_values_s"][0])
    period_s = int(cfg["ev_sample_values_s"][0])
    times = pd.date_range("2000-01-01", periods=window_s // GRID_STEP_S + 1, freq=f"{GRID_STEP_S}s")
    first_copy = np.arange(0, len(times), period_s // GRID_STEP_S)  # grid index of each measurement
    rng = np.random.RandomState(int(ingest_seed))
    lost = 0
    for _ in range(int(sessions)):
        ingest = _simulate_ingest_times(
            times,
            delay_mean_s=float(cfg["ev_ingest_delay_mean_s"]),
            delay_jitter_s=float(cfg["ev_ingest_delay_jitter_s"]),
            delay_max_s=float(cfg["ev_ingest_delay_max_s"]),
            loss_prob=float(cfg["ev_ingest_loss_prob"]),
            rng=rng,
            measurement_period_s=period_s,
        )
        arr = ingest.to_numpy(dtype="datetime64[ns]")[first_copy]
        lost += int(np.sum(arr == _LOST_TS.to_datetime64()))
    return int(sessions) * len(first_copy), lost


def analyse_cell(run_dir: Path) -> dict[str, object]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(run_dir / "raw_runs.csv")
    reps = raw[["repeat_idx", "ingest_seed", "ev_count"]].drop_duplicates().sort_values("repeat_idx")
    if reps["repeat_idx"].duplicated().any():
        raise AssertionError(f"{run_dir.name}: one ingestion seed per repeat expected")
    shares, total, lost_total = [], 0, 0
    for r in reps.itertuples(index=False):
        n, lost = replay_repeat(int(r.ingest_seed), int(r.ev_count), cfg)
        shares.append(lost / n)
        total += n
        lost_total += lost
    m = CELL_DIR.match(run_dir.name)
    return {
        "cell": m.group(1), "loss_prob": float(cfg["ev_ingest_loss_prob"]), "repeats": int(len(reps)),
        "sessions_per_repeat": int(reps["ev_count"].iloc[0]),
        "measurements_per_session": int(cfg["window_values_s"][0]) // int(cfg["ev_sample_values_s"][0]) + 1,
        "measurements": int(total), "lost_measurements": int(lost_total),
        "realised_share": round(lost_total / total, 6), "repeat_share_min": round(min(shares), 6),
        "repeat_share_max": round(max(shares), 6), "run_id": m.group(2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    args = parser.parse_args()
    cells = sorted((d for d in ACCURACY_DIR.iterdir() if d.is_dir() and CELL_DIR.match(d.name)),
                   key=lambda d: json.loads((d / "config.json").read_text(encoding="utf-8"))["ev_ingest_loss_prob"])
    rows = [analyse_cell(d) for d in cells]
    rows = [r for r in rows if r["cell"] == "default" or str(r["cell"]).startswith("loss_")]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    for r in rows:
        print(f"  p = {r['loss_prob']:<6g} realised {100 * r['realised_share']:.2f} % "
              f"(repeats {100 * r['repeat_share_min']:.2f}–{100 * r['repeat_share_max']:.2f} %), {r['lost_measurements']}/{r['measurements']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
