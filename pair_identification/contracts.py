# -*- coding: utf-8 -*-
"""Runtime contracts for online EV-EVSE identification."""

from __future__ import annotations

from dataclasses import dataclass

from pair_identification.ingestion_reference import (
    DEFAULT_EV_INGEST_DELAY_JITTER_S,
    DEFAULT_EV_INGEST_DELAY_MAX_S,
    DEFAULT_EV_INGEST_DELAY_MEAN_S,
    DEFAULT_EV_INGEST_LOSS_PROB,
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration for online active-set assignment."""

    switch_penalty: float = 0.0
    evaluation_interval_s: int = 5
    min_shared_points: int = 4
    ev_ingest_delay_mean_s: float = float(DEFAULT_EV_INGEST_DELAY_MEAN_S)
    ev_ingest_delay_jitter_s: float = float(DEFAULT_EV_INGEST_DELAY_JITTER_S)
    ev_ingest_delay_max_s: float = float(DEFAULT_EV_INGEST_DELAY_MAX_S)
    # Optional watermark-delay override (for d_max mis-spec studies).
    # If None, watermark uses ev_ingest_delay_max_s.
    watermark_delay_max_s: float | None = None
    ev_ingest_loss_prob: float = float(DEFAULT_EV_INGEST_LOSS_PROB)
    ev_ingest_seed: int | None = None
    event_driven_recompute: bool = True
    candidate_time_margin_s: int = 120
    time_prior_alpha: float = 1.0
    current_like_beta: float = 1.0
    posterior_eps: float = 1e-9
    time_prior_mix: float = 0.35
    posterior_prev_power: float = 0.60
    current_cost_weight: float = 0.15
    # EV-grid front end (dense-sampling diagnostic): "hold" (manuscript) or "block_mean".
    ev_grid_aggregation: str = "hold"
    ev_grid_block_s: int = 30
    # EV telemetry sampling period: ingestion delay and loss are drawn once per measurement,
    # and the measurement's grid copies inherit that draw.
    ev_measurement_period_s: int = 30
