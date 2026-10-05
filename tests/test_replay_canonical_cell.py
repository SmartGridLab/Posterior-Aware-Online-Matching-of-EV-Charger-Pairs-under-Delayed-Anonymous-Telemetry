"""The harness plumbing must leave the configuration of the first public release bit-for-bit reproducible.

Replays one canonical base-load scenario (EV = 100, repeat 1, Δt_EV = 30 s, d_max = 10 s,
Δ = 120 s; scenario seed 2187632392) through the compare harness with default kwargs and
checks that every matcher's per-run accuracy and the scenario hash equal the committed
canonical raw_runs.csv row. Runtime ≈ 10 s.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_study.section_vi_reproduction import run_algorithm_compare_report  # noqa: E402

CANONICAL_RAW = ROOT / "case_study" / "scalability_analysis" / "accuracy" / "canonical_run_20260907_111056_515561" / "raw_runs.csv"
GENERATED_DIR = ROOT / ".generated"
ALGORITHMS = ["time_only_baseline", "single_only", "nomura_original_interval_hungarian", "bayesian_windowed"]
BASE_LOAD_EV100_SEED = 2187632392  # PAPER_BASE_SEED + 1320


def test_default_kwargs_replay_canonical_ev100_repeat1():
    result = run_algorithm_compare_report(
        ev_counts=[100],
        repeats=1,
        error_mode="ci95",
        base_seed=BASE_LOAD_EV100_SEED,
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        pattern_unique_count=50,
        command_step_count=6,
        tau_values=[60],
        algorithm_ids=list(ALGORITHMS),
        charger_sample_values=[5],
        ev_sample_values=[30],
        matcher_delay_max_values_s=[10.0],
        candidate_margin_values_s=[120],
        ingestion_enabled=True,
        ev_ingest_delay_mean_s=2.0,
        ev_ingest_delay_jitter_s=1.0,
        ev_ingest_delay_max_s=10.0,
        ev_ingest_loss_prob=0.005,
        trace_level="summary",
    )
    run_dir = GENERATED_DIR / "algorithm_compare" / str(result["run_id"])
    try:
        raw = pd.read_csv(run_dir / "raw_runs.csv")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        canonical = pd.read_csv(CANONICAL_RAW)
        ref = canonical[
            (canonical["ev_count"] == 100)
            & (canonical["repeat_idx"] == 1)
            & (canonical["ev_sample_s"] == 30)
            & (canonical["matcher_delay_max_s"] == 10.0)
            & (canonical["candidate_margin_s"] == 120)
        ]
        assert len(ref) == len(ALGORITHMS)
        assert len(raw) == len(ALGORITHMS)
        assert set(raw["scenario_seed"]) == {BASE_LOAD_EV100_SEED}
        assert set(ref["scenario_seed"]) == {BASE_LOAD_EV100_SEED}
        assert set(raw["scenario_hash"]) == set(ref["scenario_hash"])
        assert set(raw["evse_count"]) == {50}
        for algorithm_id in ALGORITHMS:
            new = raw[raw["algorithm_id"] == algorithm_id].iloc[0]
            old = ref[ref["algorithm_id"] == algorithm_id].iloc[0]
            assert new["accuracy"] == pytest.approx(old["accuracy"], abs=1e-12), algorithm_id
            assert new["p90_latency_s"] == pytest.approx(old["p90_latency_s"], abs=1e-9), algorithm_id
            assert new["ingestion_signature_sha256"] == old["ingestion_signature_sha256"], algorithm_id
        # Plumbing defaults recorded in config.json equal the configuration of the first public release.
        assert config["evse_count"] == 50
        assert config["simultaneous_arrivals"] is False
        assert config["ev_sensor_gain_std"] == pytest.approx(0.03)
        assert config["ev_sensor_bias_std_a"] == pytest.approx(0.35)
        assert config["ev_sensor_noise_std_a"] == pytest.approx(0.20)
        assert config["ev_sensor_extra_delay_max_s"] == 20
        assert config["ev_start_est_jitter_s"] == 20
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
