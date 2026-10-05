from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _summary() -> pd.DataFrame:
    return pd.read_csv(ROOT / "case_study" / "scalability_analysis" / "accuracy" / "canonical_run_20260907_111056_515561" / "summary.csv")


def _row(summary: pd.DataFrame, **filters):
    mask = pd.Series(True, index=summary.index)
    for key, value in filters.items():
        mask &= summary[key] == value
    rows = summary[mask]
    assert len(rows) == 1
    return rows.iloc[0]


def test_paper_summary_grid_shape():
    summary = _summary()
    assert summary.shape == (324, 128)
    assert set(summary["algorithm_id"]) == {
        "time_only_baseline",
        "single_only",
        "nomura_original_interval_hungarian",
        "bayesian_windowed",
    }
    assert summary["candidate_query_lead_max_s_mean"].max() <= 0.0
    assert summary["candidate_future_query_violation_count_mean"].max() == 0.0


def test_representative_500_ev_claims():
    summary = _summary()
    filters = {
        "ev_count": 500,
        "ev_sample_s": 30,
        "matcher_delay_max_s": 10.0,
        "candidate_margin_s": 120,
    }
    expected_accuracy = {
        "time_only_baseline": 0.7121,
        "single_only": 0.9732,
        "nomura_original_interval_hungarian": 0.9993,
        "bayesian_windowed": 0.9838,
    }
    for algorithm_id, expected in expected_accuracy.items():
        row = _row(summary, algorithm_id=algorithm_id, **filters)
        assert row["runs"] == 20
        assert row["accuracy_mean"] == pytest.approx(expected)
        assert row["p90_latency_mean"] == pytest.approx(416.035)


def test_a3_base_load_and_sampling_claims():
    summary = _summary()
    for ev_count, expected in [(100, 0.9975), (300, 0.990667), (500, 0.9838)]:
        row = _row(
            summary,
            algorithm_id="bayesian_windowed",
            ev_count=ev_count,
            ev_sample_s=30,
            matcher_delay_max_s=10.0,
            candidate_margin_s=120,
        )
        assert row["accuracy_mean"] == pytest.approx(expected, abs=5e-6)

    for ev_sample_s, expected in [(5, 0.9495), (15, 0.9683), (30, 0.9838)]:
        row = _row(
            summary,
            algorithm_id="bayesian_windowed",
            ev_count=500,
            ev_sample_s=ev_sample_s,
            matcher_delay_max_s=10.0,
            candidate_margin_s=120,
        )
        assert row["accuracy_mean"] == pytest.approx(expected)
