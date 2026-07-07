from __future__ import annotations

import numpy as np
import pandas as pd

from pair_identification import matching_core as algo


def test_reindex_at_times_does_not_backfill_from_future_sample():
    series = pd.Series(
        [7.0, 9.0],
        index=pd.DatetimeIndex(["2024-01-01 00:00:10", "2024-01-01 00:00:20"]),
    )
    times = pd.DatetimeIndex(["2024-01-01 00:00:00", "2024-01-01 00:00:10", "2024-01-01 00:00:15"])

    out = algo.reindex_at_times(series, times)

    np.testing.assert_allclose(out, np.asarray([0.0, 7.0, 7.0]))


def test_build_ev_grid_keeps_pre_arrival_prefix_unobserved_zero():
    arrival = pd.Timestamp("2024-01-01 00:00:10")
    ev_meta = [
        {
            "arrival_ts": arrival,
            "start_est_ts": arrival - pd.Timedelta(seconds=10),
        }
    ]
    ev_series = {
        0: pd.Series(
            [5.0, 6.0],
            index=pd.DatetimeIndex([arrival, arrival + pd.Timedelta(seconds=30)]),
        )
    }

    times_5s, ev_5s = algo._build_ev_grid_for_match(ev_meta, ev_series, 0, window=60)

    assert times_5s[0] == arrival - pd.Timedelta(seconds=10)
    np.testing.assert_allclose(ev_5s[:6], np.zeros(6))
    assert ev_5s[6] == 5.0
