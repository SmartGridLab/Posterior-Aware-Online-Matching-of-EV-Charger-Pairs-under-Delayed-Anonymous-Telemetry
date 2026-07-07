# -*- coding: utf-8 -*-
"""Online prefix-length utility."""

from __future__ import annotations


def prefix_len(
    arrival_ts,
    now_ts,
    sample_seconds: int,
    max_len: int,
) -> int:
    dt_s = int((now_ts - arrival_ts).total_seconds())
    n = dt_s // int(sample_seconds) + 1
    if n < 1:
        n = 1
    if n > int(max_len):
        n = int(max_len)
    return int(n)
