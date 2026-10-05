"""The response model's new lag/ramp parameters default to the fitted model bit-for-bit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from charging_current_generation.command_charger_current import generate_charger_current, get_command_data  # noqa: E402

PATTERNS = [[0, 8, 18, 14, 19, 24], [0, 30, 6, 24, 12, 18], [0, 21, 9, 30, 10, 7]]


def _df(pattern):
    return get_command_data(pd.Timestamp("2024-05-23 16:00:00"), pattern, tau=60)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_defaults_reproduce_the_fitted_waveform_exactly(pattern):
    base = generate_charger_current(_df(pattern))
    explicit = generate_charger_current(_df(pattern), first_step_lag_s=15.0, step_lag_s=5.0, ramp_rate_a_per_s=None)
    assert base.index.equals(explicit.index)
    assert np.array_equal(base["charger_current"].to_numpy(), explicit["charger_current"].to_numpy())
    assert len(base) == 360


def test_first_step_lag_shifts_the_first_rise():
    pattern = PATTERNS[0]
    ref = generate_charger_current(_df(pattern))["charger_current"].to_numpy()
    lagged = generate_charger_current(_df(pattern), first_step_lag_s=30.0)["charger_current"].to_numpy()
    # fitted model: current is 0 for 60 s (zero step) + 15 s lag; with a 30 s lag the rise starts 15 s later
    first_ref = int(np.argmax(ref > 0)); first_lag = int(np.argmax(lagged > 0))
    assert first_ref == 75 and first_lag == 90


def test_ramp_rate_replaces_the_fitted_rise_rate_and_caps_the_fall():
    pattern = [0, 24, 6, 24, 6, 24]
    slow = generate_charger_current(_df(pattern), ramp_rate_a_per_s=1.0)["charger_current"].to_numpy()
    fast = generate_charger_current(_df(pattern), ramp_rate_a_per_s=10.0)["charger_current"].to_numpy()
    # rise slope: first non-zero step (24 A command -> ~23.2 A target): 1 A/s vs 10 A/s (rounded to 0.1 A)
    rise_slow = np.diff(slow[75:100]); rise_fast = np.diff(fast[75:80])
    assert np.isclose(rise_slow[0], 1.0, atol=0.15) and np.isclose(rise_fast[0], 10.0, atol=0.15)
    # fall from 24 to 6 A: fitted 4 A/s cap; ramp 1 A/s caps the fall at 1 A/s
    fall_idx = 60 * 2 + 5  # second step starts at 120 s + 5 s lag
    fall_slow = -np.diff(slow[fall_idx:fall_idx + 5]); fall_fast = -np.diff(fast[fall_idx:fall_idx + 5])
    assert np.isclose(fall_slow[0], 1.0, atol=0.15) and np.isclose(fall_fast[0], 4.0, atol=0.15)
