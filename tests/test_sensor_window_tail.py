"""The EV meter must read the physical slot current for every delayed sample.

The sensor samples the slot with its own delay (CHARGER_SAMPLE + a per-session extra delay of
0-20 s), so the sample taken for the last timestamps of the observation window falls past the
window. Those samples have to come from the slot's physical current, with gain, bias, noise and
the 1 A ceiling applied exactly once. Reading back the previous *output* instead compounds the
distortion and makes the trace ramp away from the commanded set-points.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pair_identification import matching_core as algo

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RAW = ROOT / "case_study" / "scalability_analysis" / "accuracy" / "canonical_run_20260907_111056_515561" / "raw_runs.csv"
REPRESENTATIVE_SEED = 2187632432  # EV = 500, repeat 1 of the canonical base-load grid


def _constant_slot_series(value):
    def _fake(slot, t_start, t_end, sessions_by_slot):
        times = pd.date_range(start=t_start, end=t_end, freq="1s")
        return pd.Series(np.full(len(times), float(value)), index=times, name="charger_current")
    return _fake


def test_constant_current_gives_a_constant_trace_including_the_tail(monkeypatch):
    """A constant physical current must read back constant over the whole window.

    Gain 1, no noise, a positive bias and a 19 s sampling delay on a 60 s window: every sample
    is ceil(10 + bias) = 11 A. Reading back the previous output made the tail ramp 11, 12, 13 ...
    """
    monkeypatch.setattr(algo, "_slot_series_1s_for_window", _constant_slot_series(10.0))
    monkeypatch.setattr(algo, "EV_SENSOR_GAIN_STD", 0.0)
    monkeypatch.setattr(algo, "EV_SENSOR_NOISE_STD_A", 0.0)
    monkeypatch.setattr(algo, "EV_SENSOR_BIAS_STD_A", 0.045)
    monkeypatch.setattr(algo, "EV_SENSOR_EXTRA_DELAY_MAX_S", 14)  # + CHARGER_SAMPLE = up to 19 s
    monkeypatch.setenv("EVLINK_SEED", "20260906")

    algo.init_seeds()
    ev_meta, _sessions, ev_series, _patterns, _diag = algo.build_dataset_with_random_arrivals(
        base_initial_ts=algo.BASE_INITIAL, n_evs=8, n_evses=4, tau=algo.TAU, window=60,
        arrival_span_max=3600,
    )
    assert len(ev_series) > 0
    saw_positive_bias = False
    for meta in ev_meta:
        series = ev_series[int(meta["ev_idx"])].to_numpy(dtype=float)
        assert len(np.unique(series)) == 1, (
            f"EV {meta['ev_idx']} trace is not constant under a constant physical current: "
            f"{series[:5]} ... {series[-5:]}"
        )
        expected = np.ceil(10.0 * float(meta["ev_sensor_gain"]) + float(meta["ev_sensor_bias_a"]))
        assert series[0] == pytest.approx(expected)
        if float(meta["ev_sensor_bias_a"]) > 0.0:
            saw_positive_bias = True
            assert series[0] == pytest.approx(11.0)
    assert saw_positive_bias, "fixture did not exercise the positive-bias case that made the tail ramp"


def _representative_dataset(seed):
    os.environ["EVLINK_SEED"] = str(seed)
    algo.init_seeds()
    return algo.build_dataset_with_random_arrivals(
        base_initial_ts=algo.BASE_INITIAL, n_evs=500, n_evses=50, tau=algo.TAU,
        window=algo.WINDOW, arrival_span_max=algo.ARRIVAL_SPAN_MAX,
    )


def test_representative_traces_stay_within_the_physical_envelope():
    """No trace of the canonical EV = 500 scenario may run away from its own set-points.

    The measured current is the commanded set-point shaped by the fitted response, a gain of at
    most 1.20 and a bias, then ceiled: max(set-point) * 1.20 + 1.5 A bounds every sample. The
    compounding tail broke this bound by up to 4x on this scenario.
    """
    ev_meta, _sessions, ev_series, _patterns, _diag = _representative_dataset(REPRESENTATIVE_SEED)
    for meta in ev_meta:
        setpoints = algo._pattern_to_int_list(meta["pattern"])
        if not setpoints:
            continue
        bound = max(setpoints) * 1.20 + 1.5
        series = ev_series[int(meta["ev_idx"])].to_numpy(dtype=float)
        assert series.max() <= bound, (
            f"EV {meta['ev_idx']} reaches {series.max():.0f} A with set-points up to "
            f"{max(setpoints)} A (bound {bound:.1f} A)"
        )


def test_scenario_realisation_of_the_canonical_seeds_is_unchanged():
    """The fix must not move a single scenario draw: same arrivals, slots, patterns, sensor draws.

    The committed canonical rows carry the scenario hash of each realisation; it is computed from
    the EV metadata and the session table, which the sensor loop does not feed, so it pins the
    realisation independently of the measured traces.
    """
    canonical = pd.read_csv(CANONICAL_RAW)
    for offset in (0, 1):
        seed = REPRESENTATIVE_SEED + offset
        ev_meta, sessions_by_slot, _series, _patterns, _diag = _representative_dataset(seed)
        from case_study.section_vi_reproduction import _scenario_signature_hash

        reference = canonical[
            (canonical["ev_count"] == 500)
            & (canonical["scenario_seed"] == seed)
            & (canonical["ev_sample_s"] == 30)
            & (canonical["matcher_delay_max_s"] == 10.0)
            & (canonical["candidate_margin_s"] == 120)
        ]
        assert len(reference) > 0
        assert _scenario_signature_hash(ev_meta, sessions_by_slot) == reference["scenario_hash"].iloc[0], seed
