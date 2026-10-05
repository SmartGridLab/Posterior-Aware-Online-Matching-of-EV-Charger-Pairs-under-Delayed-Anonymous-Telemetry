"""Per-session response heterogeneity draws (lag ~ U, ramp ~ U) — plumbing and pairing."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pair_identification import matching_core as algo  # noqa: E402
from case_study import section_vi_reproduction as harness  # noqa: E402
from case_study.site_simulation import SimulationParameters, override_algo_config  # noqa: E402
import reproduce_all as ra  # noqa: E402


def _build(seed: int, lag_range, ramp_range, n_evs: int = 12, n_evses: int = 6):
    params = SimulationParameters(ev_count=n_evs, evse_count=n_evses, sim_hours=12.0, session_min_minutes=10.0, session_max_minutes=60.0, charger_sample=5, ev_sample=30)
    old = (algo.RESPONSE_LAG_RANGE_S, algo.RESPONSE_RAMP_RANGE_A_PER_S)
    algo.RESPONSE_LAG_RANGE_S, algo.RESPONSE_RAMP_RANGE_A_PER_S = lag_range, ramp_range
    try:
        with override_algo_config(params):
            np.random.seed(seed); import random; random.seed(seed)
            return algo.build_dataset_with_random_arrivals(base_initial_ts=algo.BASE_INITIAL, n_evs=n_evs, n_evses=n_evses, tau=algo.TAU, window=algo.WINDOW, arrival_span_max=params.arrival_span_max)
    finally:
        algo.RESPONSE_LAG_RANGE_S, algo.RESPONSE_RAMP_RANGE_A_PER_S = old


def test_no_ranges_means_fitted_model_and_unchanged_scenario():
    ev_meta, sessions, series, _pats, diag = _build(11, None, None)
    assert all("response" not in s for slots in sessions.values() for s in slots)
    assert all(m["response_first_step_lag_s"] == 15.0 and np.isnan(m["response_ramp_rate_a_per_s"]) for m in ev_meta)
    assert "response_heterogeneity" not in diag


def test_ranges_draw_per_session_and_leave_the_scenario_realisation_unchanged():
    base_meta, base_sessions, base_series, _p, _d = _build(11, None, None)
    het_meta, het_sessions, het_series, _p2, diag = _build(11, (2.0, 30.0), (1.0, 10.0))
    # same arrivals, slots, patterns, sensor draws (dedicated RNG stream for the heterogeneity draws)
    for a, b in zip(base_meta, het_meta):
        assert a["arrival_ts"] == b["arrival_ts"] and a["evse_idx_gt"] == b["evse_idx_gt"] and a["pattern"] == b["pattern"]
        assert a["ev_sensor_gain"] == b["ev_sensor_gain"] and a["ev_sensor_bias_a"] == b["ev_sensor_bias_a"] and a["ev_sensor_delay_s"] == b["ev_sensor_delay_s"]
    lags = [m["response_first_step_lag_s"] for m in het_meta]; ramps = [m["response_ramp_rate_a_per_s"] for m in het_meta]
    assert all(2.0 <= v <= 30.0 and float(v).is_integer() for v in lags) and len(set(lags)) > 1
    assert all(1.0 <= v <= 10.0 for v in ramps) and len(set(ramps)) > 1
    assert all("response" in s and set(s["response"]) == {"first_step_lag_s", "ramp_rate_a_per_s"} for slots in het_sessions.values() for s in slots)
    assert diag["response_heterogeneity"]["sessions"] == len(het_meta)
    # the perturbed response changes the EV-side waveform of at least one session
    assert any(not np.array_equal(base_series[i].to_numpy(), het_series[i].to_numpy()) for i in range(len(het_meta)))


def test_draws_are_reproducible_from_the_seed():
    a = _build(23, (2.0, 30.0), (1.0, 10.0))[0]; b = _build(23, (2.0, 30.0), (1.0, 10.0))[0]
    assert [m["response_first_step_lag_s"] for m in a] == [m["response_first_step_lag_s"] for m in b]
    assert [m["response_ramp_rate_a_per_s"] for m in a] == [m["response_ramp_rate_a_per_s"] for m in b]


def test_harness_validation_and_registration():
    assert harness._validate_range(None, field="x") is None
    assert harness._validate_range((2, 30), field="x") == [2.0, 30.0]
    with pytest.raises(ValueError):
        harness._validate_range((30, 2), field="x")
    names = {f.name for f in dataclasses.fields(harness.ExperimentConfig)}
    assert {"response_lag_range_s", "response_ramp_range_a_per_s"} <= names
    with harness._override_response_model([2.0, 30.0], None):
        assert algo.RESPONSE_LAG_RANGE_S == (2.0, 30.0) and algo.RESPONSE_RAMP_RANGE_A_PER_S is None
    assert algo.RESPONSE_LAG_RANGE_S is None
    assert ra.HETEROGENEITY_LAG_RANGE_S == (2.0, 30.0) and ra.HETEROGENEITY_RAMP_RANGE_A_PER_S == (1.0, 10.0)
    assert callable(ra.MODES["heterogeneous-response"])
