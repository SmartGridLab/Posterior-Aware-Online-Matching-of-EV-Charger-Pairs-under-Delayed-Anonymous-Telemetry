"""EV-grid front end (dense-sampling diagnostic): block-mean aggregation vs the manuscript's hold (forward-fill)."""

from __future__ import annotations

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
from pair_identification import runtime_event_loop as loop  # noqa: E402
from case_study import section_vi_reproduction as harness  # noqa: E402
import reproduce_all as ra  # noqa: E402


def _series(values_1s: np.ndarray, t0: pd.Timestamp) -> pd.Series:
    idx = pd.date_range(start=t0, periods=len(values_1s), freq="1s")
    return pd.Series(np.asarray(values_1s, dtype=float), index=idx)


def _grid(agg: str, ev_sample: int, values_1s: np.ndarray, window: int = 360):
    t0 = pd.Timestamp("2024-05-23 16:00:00")
    meta = [{"arrival_ts": t0, "start_est_ts": t0}]
    old = (algo.EV_GRID_AGGREGATION, algo.EV_SAMPLE)
    algo.EV_GRID_AGGREGATION, algo.EV_SAMPLE = agg, ev_sample
    try:
        times, vals = algo._build_ev_grid_for_match(meta, {0: _series(values_1s, t0)}, 0, window=window)
    finally:
        algo.EV_GRID_AGGREGATION, algo.EV_SAMPLE = old
    return times, np.asarray(vals, dtype=float)


def test_block_mean_equals_hold_at_30s_sampling():
    rng = np.random.RandomState(0)
    values = rng.uniform(0, 30, size=361)
    t_hold, v_hold = _grid("hold", 30, values)
    t_block, v_block = _grid("block_mean", 30, values)
    assert list(t_hold) == list(t_block) and len(v_hold) == 73
    assert v_block == pytest.approx(v_hold)


def test_block_mean_averages_the_dense_samples_inside_each_30s_block():
    # 1 s series with a ramp; at Δt_EV = 5 s each 30 s block holds six samples (t = 0,5,...,25 within the block)
    values = np.arange(361, dtype=float)
    _t, v_hold = _grid("hold", 5, values)
    _t, v_block = _grid("block_mean", 5, values)
    assert v_hold[:6] == pytest.approx([0, 5, 10, 15, 20, 25])
    assert v_block[:6] == pytest.approx([np.mean([0, 5, 10, 15, 20, 25])] * 6)  # block 0 mean = 12.5
    assert v_block[6:12] == pytest.approx([np.mean([30, 35, 40, 45, 50, 55])] * 6)
    # the final grid point (t = 360) forms its own block and equals its single sample
    assert v_block[-1] == pytest.approx(360.0)


def test_blockify_ingest_times_uses_the_latest_received_sample_of_each_block():
    t0 = pd.Timestamp("2024-05-23 16:00:00")
    times = pd.date_range(start=t0, periods=13, freq="5s")  # two full blocks + 1
    delays = np.array([1, 2, 30, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=float)
    ingest = pd.DatetimeIndex(times + pd.to_timedelta(delays, unit="s"))
    ingest = pd.DatetimeIndex(np.where(np.arange(13) == 9, loop._LOST_TS.to_datetime64(), ingest.to_numpy()))
    out = loop._blockify_ingest_times(times, ingest, 30)
    block0_max = (times[2] + pd.Timedelta(seconds=30))
    assert all(pd.Timestamp(x) == block0_max for x in out[:6])
    block1_max = max(times[i] + pd.Timedelta(seconds=delays[i]) for i in (6, 7, 8, 10, 11))  # index 9 lost
    assert all(pd.Timestamp(x) == block1_max for x in out[6:12])
    assert pd.Timestamp(out[12]) == times[12] + pd.Timedelta(seconds=13)


def test_harness_validates_and_records_the_front_end():
    with pytest.raises(ValueError):
        harness.run_algorithm_compare_report(ev_counts=[1], repeats=1, ev_grid_aggregation="median")
    assert "ev_grid_aggregation" in {f.name for f in __import__("dataclasses").fields(harness.ExperimentConfig)}
    old = algo.EV_GRID_AGGREGATION
    with harness._override_ev_grid_aggregation("block_mean"):
        assert algo.EV_GRID_AGGREGATION == "block_mean"
    assert algo.EV_GRID_AGGREGATION == old


def test_sampling_diagnostic_cells_and_seeds():
    assert [c["label"] for c in ra.SAMPLING_DIAG_CELLS] == ["noisefree_5", "noisefree_15", "noisefree_30", "blockmean_5", "blockmean_15", "termpart_5", "termpart_15", "termpart_30", "offsetfree_5"]
    assert [c["variant"] for c in ra.SAMPLING_DIAG_CELLS[5:]] == ["manuscript"] * 4   # Eq. (7) term partition on the unchanged front end
    assert ra.SAMPLING_DIAG_ALGORITHMS == ["single_only", "bayesian_windowed", "nomura_original_interval_hungarian", "single_only_step_off", "single_only_dtw_off"]
    # canonical scenario counter of the (Δt_EV, d_max 10 s, Δ 120 s) cells: (idx_dt*9 + 1*3 + 1)*60, plus 40 for EV = 500
    assert ra.SAMPLING_CELL_BASE_SEED == {5: 2187631352, 15: 2187631892, 30: 2187632432}
    assert callable(ra.MODES["sampling-diagnostic"])
