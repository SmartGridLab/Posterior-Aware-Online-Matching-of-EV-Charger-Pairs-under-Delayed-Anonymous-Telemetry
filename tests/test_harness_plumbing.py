"""Harness plumbing: per-run EVSE count, arrival model and sensor-model overrides must
default to the manuscript configuration and restore module state."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pair_identification import matching_core as algo  # noqa: E402
from case_study import section_vi_reproduction as harness  # noqa: E402
from case_study.site_simulation import MAX_EVSE_COUNT, SimulationParameters  # noqa: E402


def test_sensor_model_defaults_resolve_to_module_globals():
    cfg = harness._resolve_sensor_model()
    assert cfg == {
        "ev_sensor_extra_delay_max_s": int(algo.EV_SENSOR_EXTRA_DELAY_MAX_S),
        "ev_sensor_gain_std": float(algo.EV_SENSOR_GAIN_STD),
        "ev_sensor_bias_std_a": float(algo.EV_SENSOR_BIAS_STD_A),
        "ev_sensor_noise_std_a": float(algo.EV_SENSOR_NOISE_STD_A),
        "ev_start_est_jitter_s": int(algo.EV_START_EST_JITTER_S),
    }
    # Sec. V-A sensing layer of the manuscript.
    assert cfg["ev_sensor_gain_std"] == pytest.approx(0.03)
    assert cfg["ev_sensor_bias_std_a"] == pytest.approx(0.35)
    assert cfg["ev_sensor_noise_std_a"] == pytest.approx(0.20)
    assert cfg["ev_sensor_extra_delay_max_s"] == 20
    assert cfg["ev_start_est_jitter_s"] == 20


def test_sensor_model_override_applies_and_restores():
    before = {attr: getattr(algo, attr) for attr in harness._SENSOR_MODEL_GLOBALS.values()}
    cfg = harness._resolve_sensor_model(ev_sensor_noise_std_a=0.0, ev_sensor_gain_std=0.06, ev_start_est_jitter_s=0)
    with harness._override_sensor_model(cfg):
        assert algo.EV_SENSOR_NOISE_STD_A == 0.0
        assert algo.EV_SENSOR_GAIN_STD == pytest.approx(0.06)
        assert algo.EV_START_EST_JITTER_S == 0
        assert algo.EV_SENSOR_BIAS_STD_A == before["EV_SENSOR_BIAS_STD_A"]
    after = {attr: getattr(algo, attr) for attr in harness._SENSOR_MODEL_GLOBALS.values()}
    assert after == before


def test_sensor_model_rejects_negative_and_unknown():
    with pytest.raises(ValueError):
        harness._resolve_sensor_model(ev_sensor_noise_std_a=-0.1)
    with pytest.raises(ValueError):
        harness._resolve_sensor_model(not_a_knob=1.0)


def test_evse_count_validation():
    assert harness._validate_evse_count(15) == 15
    assert harness._validate_evse_count(50) == 50
    assert harness._validate_evse_count(MAX_EVSE_COUNT) == MAX_EVSE_COUNT
    with pytest.raises(ValueError):
        harness._validate_evse_count(0)
    with pytest.raises(ValueError):
        harness._validate_evse_count(MAX_EVSE_COUNT + 1)


def test_simulation_parameters_accept_static_limit_capacity():
    params = SimulationParameters(
        ev_count=300,
        evse_count=300,
        sim_hours=12.0,
        session_min_minutes=10.0,
        session_max_minutes=60.0,
        charger_sample=5,
        ev_sample=30,
    )
    params.validate()
    params.evse_count = MAX_EVSE_COUNT + 1
    with pytest.raises(ValueError):
        params.validate()


def test_experiment_config_records_plumbing_fields():
    names = {f.name for f in dataclasses.fields(harness.ExperimentConfig)}
    for key in (
        "evse_count",
        "simultaneous_arrivals",
        "ev_sensor_extra_delay_max_s",
        "ev_sensor_gain_std",
        "ev_sensor_bias_std_a",
        "ev_sensor_noise_std_a",
        "ev_start_est_jitter_s",
    ):
        assert key in names


def test_driver_mode_registry():
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import reproduce_all  # noqa: E402

    for mode in ("verify-existing", "figures", "smoke", "ablation", "full"):
        assert callable(reproduce_all.MODES[mode])
    assert reproduce_all.PAPER_REPRESENTATIVE_BASE_SEED == 2187632432
    assert reproduce_all.PAPER_ABLATION_BASE_SEED == 2187632392
    assert reproduce_all.PAPER_INGESTION_DEFAULTS["ev_ingest_loss_prob"] == pytest.approx(0.005)


# ---------------------------------------------------------------- A3 hyperparameter overrides
def test_a3_hyperparam_defaults_resolve_to_table2_values():
    cfg = harness._resolve_a3_hyperparams()
    assert cfg == {
        "a3_posterior_prev_power": float(algo.POSTERIOR_PREV_POWER),
        "a3_time_prior_mix": float(algo.TIME_PRIOR_MIX),
        "a3_current_like_beta": float(algo.CURRENT_LIKE_BETA),
    }
    assert cfg["a3_posterior_prev_power"] == pytest.approx(0.60)
    assert cfg["a3_time_prior_mix"] == pytest.approx(0.35)
    assert cfg["a3_current_like_beta"] == pytest.approx(1.0)


def test_a3_hyperparam_override_applies_and_restores():
    before = {attr: getattr(algo, attr) for attr in harness._A3_HYPERPARAM_GLOBALS.values()}
    cfg = harness._resolve_a3_hyperparams(a3_posterior_prev_power=0.3, a3_current_like_beta=2.0)
    with harness._override_a3_hyperparams(cfg):
        assert algo.POSTERIOR_PREV_POWER == pytest.approx(0.3)
        assert algo.CURRENT_LIKE_BETA == pytest.approx(2.0)
        assert algo.TIME_PRIOR_MIX == before["TIME_PRIOR_MIX"]
    assert {attr: getattr(algo, attr) for attr in harness._A3_HYPERPARAM_GLOBALS.values()} == before
    with pytest.raises(ValueError):
        harness._resolve_a3_hyperparams(a3_time_prior_mix=-0.1)
    names = {f.name for f in dataclasses.fields(harness.ExperimentConfig)}
    assert {"a3_posterior_prev_power", "a3_time_prior_mix", "a3_current_like_beta"} <= names
