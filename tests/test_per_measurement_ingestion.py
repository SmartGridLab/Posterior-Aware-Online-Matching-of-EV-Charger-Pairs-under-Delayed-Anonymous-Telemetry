"""Ingestion delay and loss apply to an EV measurement, not to its grid copies.

The matcher grid is finer than the EV telemetry, so one measurement is held on several grid
points. Transporting each copy independently made a measurement survive unless every copy was
dropped (at a 30 % loss and a 30 s sampling period, 0.30^6 of measurements) and hid a late copy
behind the next one. One draw per measurement, inherited by its copies, is what the sensing model
of the manuscript describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pair_identification.runtime_event_loop import _LOST_TS, _observed_ev_prefix, _simulate_ingest_times

GRID_S = 5
WINDOW_S = 360
CELL = dict(delay_mean_s=2.0, delay_jitter_s=1.0, delay_max_s=10.0)


def _grid(period_s: int) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp("2024-05-23 16:00:00"), periods=WINDOW_S // GRID_S + 1, freq=f"{GRID_S}s")


def test_lost_fraction_is_the_measurement_loss_and_copies_share_their_draw():
    times = _grid(30)
    block = ((times - times[0]).total_seconds().to_numpy() // 30).astype(int)
    rng = np.random.RandomState(20260907)
    lost_meas = seen_meas = 0
    for _ in range(2000):
        ingest = _simulate_ingest_times(times, loss_prob=0.30, rng=rng, measurement_period_s=30, **CELL)
        lost = pd.DatetimeIndex(ingest).to_numpy(dtype="datetime64[ns]") == _LOST_TS.to_datetime64()
        for k in np.unique(block):
            copies = lost[block == k]
            # every copy of a measurement shares its measurement's loss flag ...
            assert copies.all() or (~copies).all(), f"measurement {k} is partly lost"
            lost_meas += int(copies[0]); seen_meas += 1
        # ... and its ingestion instant, up to the causal max(t, published)
        ok = ~lost
        for k in np.unique(block):
            sel = (block == k) & ok
            if sel.sum() > 1:
                arr = pd.DatetimeIndex(ingest[sel])
                assert (arr == np.maximum(times[sel].to_numpy(dtype="datetime64[ns]"), arr.max().to_datetime64())).all() or arr.is_monotonic_increasing
    frac = lost_meas / seen_meas
    assert 0.28 <= frac <= 0.32, f"per-measurement loss fraction {frac:.4f} is not 0.30 +/- 0.02"


def test_period_equal_to_the_grid_step_reproduces_one_draw_per_point():
    times = _grid(GRID_S)
    a = _simulate_ingest_times(times, loss_prob=0.15, rng=np.random.RandomState(7), measurement_period_s=GRID_S, **CELL)
    b = _simulate_ingest_times(times, loss_prob=0.15, rng=np.random.RandomState(7), measurement_period_s=0, **CELL)
    assert list(pd.DatetimeIndex(a)) == list(pd.DatetimeIndex(b))


def test_a_lost_measurement_leaves_the_previous_one_held():
    times = _grid(30)
    values = pd.Series(np.arange(1.0, 14.0), index=pd.date_range(times[0], periods=13, freq="30s")) \
        .reindex(times, method="pad").to_numpy()
    rng = np.random.RandomState(11)
    ingest = _simulate_ingest_times(times, loss_prob=0.30, rng=rng, measurement_period_s=30, **CELL)
    observed = _observed_ev_prefix(values, ingest, len(times), times[-1] + pd.Timedelta(hours=1))
    lost = pd.DatetimeIndex(ingest).to_numpy(dtype="datetime64[ns]") == _LOST_TS.to_datetime64()
    assert lost.any(), "fixture drew no loss"
    # a lost measurement never contributes its own value; the trace only ever holds earlier values
    for i in np.where(lost)[0]:
        if not np.isnan(observed[i]):
            assert observed[i] != values[i]
            assert observed[i] in set(values[:i])


def test_a_late_measurement_is_not_hidden_by_its_own_copies():
    """One 50 s delay must hide the whole measurement, not just its first copy."""
    times = _grid(30)
    ingest = _simulate_ingest_times(
        times, delay_mean_s=50.0, delay_jitter_s=1e-6, delay_max_s=60.0, loss_prob=0.0,
        rng=np.random.RandomState(3), measurement_period_s=30,
    )
    t0 = times[0]
    # 40 s after the first measurement instant nothing of it has arrived (it is published at +50 s)
    visible = pd.DatetimeIndex(ingest) <= t0 + pd.Timedelta(seconds=40)
    assert not visible[:6].any(), "a copy of the first measurement was visible before the measurement arrived"
    visible = pd.DatetimeIndex(ingest) <= t0 + pd.Timedelta(seconds=55)
    assert visible[:6].all(), "the measurement's copies did not become visible together"
