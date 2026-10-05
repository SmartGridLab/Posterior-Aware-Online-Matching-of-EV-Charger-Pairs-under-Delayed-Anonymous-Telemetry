"""Sweep driver modes (impairment sweeps, saturation): cell tables and kwargs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import reproduce_all as ra  # noqa: E402
from pair_identification import matching_core as algo  # noqa: E402


def test_impairment_cell_tables():
    assert set(ra.IMPAIRMENT_CELLS) == {"default", "loss", "delay", "noise"}
    assert [c["ev_ingest_loss_prob"] for c in ra.IMPAIRMENT_CELLS["loss"]] == [0.05, 0.15, 0.30]
    delays = [(c["ev_ingest_delay_mean_s"], c["ev_ingest_delay_jitter_s"], c["ev_ingest_delay_max_s"], c["matcher_delay_max_s"]) for c in ra.IMPAIRMENT_CELLS["delay"]]
    assert delays == [(5.0, 2.5, 15.0, 10.0), (10.0, 5.0, 30.0, 10.0), (20.0, 10.0, 60.0, 10.0), (20.0, 10.0, 60.0, 60.0)]
    assert [c["sensing_multiplier"] for c in ra.IMPAIRMENT_CELLS["noise"]] == [2.0, 4.0, 8.0]
    labels = [c["label"] for cells in ra.IMPAIRMENT_CELLS.values() for c in cells]
    assert len(labels) == len(set(labels)) == 11


def test_default_cell_kwargs_equal_the_manuscript_configuration():
    kw = ra._representative_point_kwargs(ra.IMPAIRMENT_DEFAULT_CELL, repeats=20, base_seed=ra.PAPER_REPRESENTATIVE_BASE_SEED, trace_level="summary")
    for k, v in ra.PAPER_INGESTION_DEFAULTS.items():
        assert kw[k] == v
    assert kw["ev_counts"] == [500] and kw["evse_count"] == 50 and kw["pattern_unique_count"] == 50
    assert kw["matcher_delay_max_values_s"] == [10.0] and kw["candidate_margin_values_s"] == [120]
    assert kw["algorithm_ids"] == ra.PAPER_ALGORITHMS
    assert kw["ev_sensor_noise_std_a"] == pytest.approx(float(algo.EV_SENSOR_NOISE_STD_A))
    assert kw["ev_sensor_gain_std"] == pytest.approx(float(algo.EV_SENSOR_GAIN_STD))
    assert kw["ev_sensor_bias_std_a"] == pytest.approx(float(algo.EV_SENSOR_BIAS_STD_A))
    assert kw["base_seed"] == 2187632432


def test_noise_multiplier_scales_the_three_sensing_stds_jointly():
    cell = ra.IMPAIRMENT_CELLS["noise"][2]
    kw = ra._representative_point_kwargs(cell, repeats=1, base_seed=1, trace_level="summary")
    assert kw["ev_sensor_noise_std_a"] == pytest.approx(8.0 * float(algo.EV_SENSOR_NOISE_STD_A))
    assert kw["ev_sensor_gain_std"] == pytest.approx(8.0 * float(algo.EV_SENSOR_GAIN_STD))
    assert kw["ev_sensor_bias_std_a"] == pytest.approx(8.0 * float(algo.EV_SENSOR_BIAS_STD_A))
    # ingestion untouched on the noise axis
    assert kw["ev_ingest_loss_prob"] == 0.005 and kw["matcher_delay_max_values_s"] == [10.0]


def test_saturation_points():
    assert ra.SATURATION_POINTS == {"SA": {"ev_count": 2000, "evse_count": 50}, "SB": {"ev_count": 500, "evse_count": 15}, "control": {"ev_count": 500, "evse_count": 50}}
    kw = ra._representative_point_kwargs(ra.IMPAIRMENT_DEFAULT_CELL, repeats=20, base_seed=ra.PAPER_REPRESENTATIVE_BASE_SEED, trace_level="summary", evse_count=15, ev_count=500)
    assert kw["evse_count"] == 15 and kw["pattern_unique_count"] == 15 and kw["ev_counts"] == [500]
    kw = ra._representative_point_kwargs(ra.IMPAIRMENT_DEFAULT_CELL, repeats=20, base_seed=ra.PAPER_REPRESENTATIVE_BASE_SEED, trace_level="summary", evse_count=50, ev_count=2000)
    assert kw["ev_counts"] == [2000] and kw["pattern_unique_count"] == 50


def test_modes_registered():
    for mode in ("impairment-sweep", "saturation", "static-limit", "trace-representative"):
        assert callable(ra.MODES[mode])


def test_hyperparam_cells_one_knob_at_a_time():
    cells = ra.HYPERPARAM_CELLS
    assert len(cells) == 12
    for c in cells:
        keys = [k for k in ("a3_posterior_prev_power", "a3_time_prior_mix", "a3_current_like_beta") if k in c]
        assert len(keys) == 1 and c[keys[0]] == c["value"]
    assert sorted(c["value"] for c in cells if c["knob"] == "gamma_prev") == [0.30, 0.45, 0.75, 0.90]
    assert sorted(c["value"] for c in cells if c["knob"] == "eta_mix") == [0.15, 0.25, 0.50, 0.70]
    assert sorted(c["value"] for c in cells if c["knob"] == "beta_curr") == [0.50, 0.75, 1.50, 2.00]
    assert callable(ra.MODES["hyperparam-sweep"])
