# -*- coding: utf-8 -*-
"""Simulation utilities for EV-EVSE MCCT matching (fixed 60s/6-step profile)."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pair_identification import matching_core as algo
from pair_identification.ingestion_reference import (
    default_ingestion_values,
    ingestion_reference_payload,
)
from pair_identification.matching_core import (
    build_dataset_with_random_arrivals,
    run_timeline_and_match,
    init_seeds,
)
from case_study.diagnostic_ev_logs import write_ev_mcct_log

MCCT_TAU_SECONDS = 60
MCCT_STEPS_AFTER_ZERO = 5
MCCT_TOTAL_STEPS = 1 + MCCT_STEPS_AFTER_ZERO
MCCT_WINDOW_SECONDS = MCCT_TAU_SECONDS * MCCT_TOTAL_STEPS
# Upper bound on simulated EVSE slots. Raised from 200 to 300 for the
# static-limit sanity check (300 simultaneous sessions, one slot each).
MAX_EVSE_COUNT = 300
_DEFAULT_INGESTION = default_ingestion_values()


@dataclass
class SimulationParameters:
    ev_count: int
    evse_count: int
    sim_hours: float
    session_min_minutes: float
    session_max_minutes: float
    charger_sample: int
    ev_sample: int
    ingestion_enabled: bool = True
    ev_ingest_delay_mean_s: float = float(_DEFAULT_INGESTION["ev_ingest_delay_mean_s"])
    ev_ingest_delay_jitter_s: float = float(_DEFAULT_INGESTION["ev_ingest_delay_jitter_s"])
    ev_ingest_delay_max_s: float = float(_DEFAULT_INGESTION["ev_ingest_delay_max_s"])
    # Optional override for watermark delay bound used by matcher policy.
    # None means use ev_ingest_delay_max_s.
    matcher_delay_max_s: float | None = None
    ev_ingest_loss_prob: float = float(_DEFAULT_INGESTION["ev_ingest_loss_prob"])

    def validate(self) -> None:
        if self.ev_count <= 0:
            raise ValueError("EV count must be positive")
        if self.evse_count <= 0:
            raise ValueError("EVSE count must be positive")
        if self.ev_count > 2000:
            raise ValueError("EV count is too large (max 2000)")
        if self.evse_count > MAX_EVSE_COUNT:
            raise ValueError(f"EVSE count is too large (max {MAX_EVSE_COUNT})")
        if self.sim_hours <= 0:
            raise ValueError("Simulation hours must be positive")
        if self.session_min_minutes <= 0 or self.session_max_minutes <= 0:
            raise ValueError("Session durations must be positive")
        if self.session_min_minutes > self.session_max_minutes:
            raise ValueError("Session minimum duration must be <= maximum duration")
        if self.charger_sample <= 0:
            raise ValueError("EVSE current sample interval must be positive")
        if self.ev_sample <= 0:
            raise ValueError("EV current sample interval must be positive")
        if MCCT_TAU_SECONDS % self.charger_sample != 0:
            raise ValueError(f"EVSE sample interval must divide fixed MCCT tau ({MCCT_TAU_SECONDS}s)")
        if (MCCT_TAU_SECONDS // self.charger_sample) < 2:
            raise ValueError("Fixed tau/grid combination is too small for step median computation")
        for key, value in self._ingestion_dict().items():
            if float(value) < 0.0:
                raise ValueError(f"{key} must be >= 0")
        if self.matcher_delay_max_s is not None and float(self.matcher_delay_max_s) < 0.0:
            raise ValueError("matcher_delay_max_s must be >= 0 when provided")
        if self.ev_ingest_loss_prob > 1.0:
            raise ValueError("ev_ingest_loss_prob must be <= 1")

    @property
    def session_min_seconds(self) -> int:
        return int(math.ceil(self.session_min_minutes * 60))

    @property
    def session_max_seconds(self) -> int:
        return int(math.ceil(self.session_max_minutes * 60))

    @property
    def total_steps(self) -> int:
        return MCCT_TOTAL_STEPS

    @property
    def window_seconds(self) -> int:
        return MCCT_WINDOW_SECONDS

    @property
    def arrival_span_max(self) -> int:
        if self.sim_hours <= 1:
            return max(MCCT_TAU_SECONDS, 1)
        span = int((self.sim_hours - 1) * 3600)
        return max(span, MCCT_TAU_SECONDS)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            tau=MCCT_TAU_SECONDS,
            steps_after_zero=MCCT_STEPS_AFTER_ZERO,
            total_steps=MCCT_TOTAL_STEPS,
            window_seconds=self.window_seconds,
            session_min_seconds=self.session_min_seconds,
            session_max_seconds=self.session_max_seconds,
        )
        data["ingestion_effective"] = self.effective_ingestion_dict()
        data["watermark_delay_max_effective_s"] = self.effective_watermark_delay_max_s()
        data["ingestion_reference"] = ingestion_reference_payload()
        data["ingestion_source_note"] = (
            "Defaults follow the evidence-informed central-console baseline; site-level calibration is still required."
        )
        return data

    def _ingestion_dict(self) -> Dict[str, float]:
        return {
            "ev_ingest_delay_mean_s": float(self.ev_ingest_delay_mean_s),
            "ev_ingest_delay_jitter_s": float(self.ev_ingest_delay_jitter_s),
            "ev_ingest_delay_max_s": float(self.ev_ingest_delay_max_s),
            "ev_ingest_loss_prob": float(self.ev_ingest_loss_prob),
        }

    def effective_ingestion_dict(self) -> Dict[str, float]:
        values = self._ingestion_dict()
        if bool(self.ingestion_enabled):
            return values
        return {k: 0.0 for k in values}

    def effective_watermark_delay_max_s(self) -> float:
        if not bool(self.ingestion_enabled):
            return 0.0
        if self.matcher_delay_max_s is None:
            return float(self.ev_ingest_delay_max_s)
        return float(self.matcher_delay_max_s)


@contextmanager
def override_algo_config(params: SimulationParameters):
    """Temporarily apply simulation parameters to the global algo module."""

    overrides = {
        "N_EV": params.ev_count,
        "N_EVSE": params.evse_count,
        "TAU": MCCT_TAU_SECONDS,
        "N_STEPS": params.total_steps,
        "WINDOW": params.window_seconds,
        "CHARGER_SAMPLE": params.charger_sample,
        "EV_SAMPLE": params.ev_sample,
        "SIM_HOURS": params.sim_hours,
        "SIM_HORIZON_SEC": int(params.sim_hours * 3600),
        "ARRIVAL_SPAN_MAX": params.arrival_span_max,
        "SESSION_MIN_SEC": params.session_min_seconds,
        "SESSION_MAX_SEC": params.session_max_seconds,
    }

    old_values: Dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(algo, key):
            raise AttributeError(f"evlink_algo has no attribute {key}")
        old_values[key] = getattr(algo, key)
        setattr(algo, key, value)

    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(algo, key, value)


@contextmanager
def override_ingestion_config(params: SimulationParameters):
    """Temporarily apply cloud ingestion delay/loss configuration to algo globals."""

    values = params.effective_ingestion_dict()
    overrides = {
        "EV_INGEST_DELAY_MEAN_S": float(values["ev_ingest_delay_mean_s"]),
        "EV_INGEST_DELAY_JITTER_S": float(values["ev_ingest_delay_jitter_s"]),
        "EV_INGEST_DELAY_MAX_S": float(values["ev_ingest_delay_max_s"]),
        "EV_WATERMARK_DELAY_MAX_S": float(params.effective_watermark_delay_max_s()),
        "EV_INGEST_LOSS_PROB": float(values["ev_ingest_loss_prob"]),
    }

    old_values: Dict[str, Any] = {}
    for key, value in overrides.items():
        if not hasattr(algo, key):
            raise AttributeError(f"evlink_algo has no attribute {key}")
        old_values[key] = getattr(algo, key)
        setattr(algo, key, value)

    try:
        yield
    finally:
        for key, value in old_values.items():
            setattr(algo, key, value)


simulation_lock = Lock()


def _build_latency_histogram(latencies: list[int]) -> list[dict[str, int]]:
    bins = [
        ("0-60", 0, 60),
        ("60-120", 60, 120),
        ("120-180", 120, 180),
        ("180-240", 180, 240),
        ("240-300", 240, 300),
        ("300-360", 300, 360),
        ("360+", 360, None),
    ]
    out: list[dict[str, int]] = []
    arr = [int(x) for x in latencies]
    for label, lo, hi in bins:
        if hi is None:
            cnt = sum(1 for x in arr if x >= int(lo))
        else:
            cnt = sum(1 for x in arr if int(lo) <= x < int(hi))
        out.append({"label": str(label), "count": int(cnt)})
    return out


def run_simulation(params: SimulationParameters) -> Dict[str, Any]:
    """Run the MCCT simulation with the provided parameters."""

    params.validate()

    with simulation_lock:
        with override_algo_config(params):
            with override_ingestion_config(params):
                seed_used = init_seeds()
                base_ts = pd.Timestamp(algo.BASE_INITIAL)

                ev_meta, sessions_by_slot, ev_series, _patterns, _ = build_dataset_with_random_arrivals(
                    base_initial_ts=algo.BASE_INITIAL,
                    n_evs=params.ev_count,
                    n_evses=params.evse_count,
                    tau=MCCT_TAU_SECONDS,
                    window=params.window_seconds,
                    arrival_span_max=params.arrival_span_max,
                )

                runtime_diag: Dict[str, Any] = {}
                links = run_timeline_and_match(
                    ev_meta,
                    sessions_by_slot,
                    ev_series,
                    base_ts=base_ts,
                    tau=MCCT_TAU_SECONDS,
                    window=params.window_seconds,
                    diagnostics_out=runtime_diag,
                )

            rows: List[Dict[str, Any]] = []
            correct = 0
            costs = []
            latencies = []
            requested_total = int(params.ev_count)
            admitted_total = int(len(ev_meta))
            blocked_total = int(max(0, requested_total - admitted_total))
            for ei, meta in enumerate(ev_meta):
                gt = int(meta['evse_idx_gt'])
                link = links.get(ei, {})
                pred = int(link.get('slot', -1))
                ok = int(pred == gt)
                correct += ok
                cost = float(link.get('cost', float('inf')))
                if np.isfinite(cost):
                    costs.append(cost)
                decision_ts = pd.Timestamp(link.get('decision_ts', meta.get('session_end_ts', meta['arrival_ts'])))
                latency = int((decision_ts - meta['arrival_ts']).total_seconds())
                latencies.append(latency)
                t0 = meta['arrival_ts']
                n5 = params.window_seconds // params.charger_sample + 1
                times_5s = pd.date_range(start=t0, periods=n5, freq=f"{params.charger_sample}s")

                arrival_time = meta['arrival_ts'].strftime("%H:%M:%S") if isinstance(meta['arrival_ts'], pd.Timestamp) else ""
                decision_time = decision_ts.strftime("%H:%M:%S") if isinstance(decision_ts, pd.Timestamp) else ""
                session_end_ts = meta.get('session_end_ts')
                session_end_time = session_end_ts.strftime("%H:%M:%S") if isinstance(session_end_ts, pd.Timestamp) else ""

                ev_grid = algo.reindex_at_times(ev_series[ei], times_5s, method="pad").astype(float)
                ev_steps = algo.step_medians(ev_grid, tau=MCCT_TAU_SECONDS, grid=params.charger_sample, n_steps=params.total_steps)

                if pred >= 0:
                    slot_series_pred = algo._slot_series_1s_for_window(
                        pred,
                        t0,
                        t0 + pd.Timedelta(seconds=params.window_seconds),
                        sessions_by_slot,
                    )
                    evse_pred_grid = algo.reindex_at_times(slot_series_pred, times_5s, method="pad").astype(float)
                    evse_pred_steps = algo.step_medians(evse_pred_grid, tau=MCCT_TAU_SECONDS, grid=params.charger_sample, n_steps=params.total_steps)
                    command_pred = algo.slot_pattern_for_window(
                        pred,
                        t0,
                        t0 + pd.Timedelta(seconds=params.window_seconds),
                        sessions_by_slot,
                    )
                else:
                    evse_pred_steps = np.full(params.total_steps, np.nan, dtype=float)
                    command_pred = []

                slot_series_gt = algo._slot_series_1s_for_window(
                    gt,
                    t0,
                    t0 + pd.Timedelta(seconds=params.window_seconds),
                    sessions_by_slot,
                )
                evse_gt_grid = algo.reindex_at_times(slot_series_gt, times_5s, method="pad").astype(float)
                evse_gt_steps = algo.step_medians(evse_gt_grid, tau=MCCT_TAU_SECONDS, grid=params.charger_sample, n_steps=params.total_steps)

                command_gt = algo.slot_pattern_for_window(
                    gt,
                    t0,
                    t0 + pd.Timedelta(seconds=params.window_seconds),
                    sessions_by_slot,
                )

                rows.append({
                    "ev_token": meta['token'],
                    "pred_slot": pred,
                    "gt_slot": gt,
                    "cost": cost,
                    "correct": bool(ok),
                    "latency_s": latency,
                    "arrival_time": arrival_time,
                    "decision_time": decision_time,
                    "session_end_time": session_end_time,
                    "ev_steps": [float(x) if np.isfinite(x) else float('nan') for x in ev_steps.tolist()],
                    "evse_pred_steps": [float(x) if np.isfinite(x) else float('nan') for x in evse_pred_steps.tolist()],
                    "evse_gt_steps": [float(x) if np.isfinite(x) else float('nan') for x in evse_gt_steps.tolist()],
                    "command_pred_steps": [int(x) for x in command_pred] if isinstance(command_pred, list) else [],
                    "command_gt_steps": [int(x) for x in command_gt] if isinstance(command_gt, list) else [],
                    "decision_reason": str(link.get("decision_reason", "")),
                    "prefix_samples": int(link.get("prefix_samples", 0)),
                })

            total = int(admitted_total)
            accuracy = float(correct) / total if total else 0.0
            accuracy_requested = float(correct) / float(requested_total) if requested_total > 0 else 0.0
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
            p50_latency = float(np.quantile(latencies, 0.50)) if latencies else 0.0
            p90_latency = float(np.quantile(latencies, 0.90)) if latencies else 0.0
            p99_latency = float(np.quantile(latencies, 0.99)) if latencies else 0.0
            avg_cost = float(np.mean(costs)) if costs else 0.0
            max_cost = float(max(costs)) if costs else 0.0
            final_wrong_rate = float(1.0 - accuracy) if total > 0 else 0.0
            latency_hist = _build_latency_histogram(latencies)

            # Persist per-run CSV for offline inspection
            logs_dir = str((Path(__file__).parent / "logs").resolve())
            try:
                out_csv = write_ev_mcct_log(
                    ev_meta,
                    sessions_by_slot,
                    ev_series,
                    links,
                    MCCT_TAU_SECONDS,
                    params.window_seconds,
                    params.charger_sample,
                    logs_dir=logs_dir,
                )
            except Exception:
                out_csv = ""

            return {
                "seed": int(seed_used),
                "requested_evs": int(requested_total),
                "admitted_evs": int(admitted_total),
                "blocked_evs": int(blocked_total),
                "blocked_ratio": float(blocked_total / requested_total) if requested_total > 0 else 0.0,
                "total_evs": total,
                "correct": int(correct),
                "accuracy": accuracy,
                "accuracy_requested": float(accuracy_requested),
                "avg_latency_s": avg_latency,
                "p50_latency_s": p50_latency,
                "p90_latency_s": p90_latency,
                "p99_latency_s": p99_latency,
                "avg_cost": avg_cost,
                "max_cost": max_cost,
                "final_wrong_rate": float(final_wrong_rate),
                "decision_latency_hist": latency_hist,
                "ingestion_model": runtime_diag.get("ingestion_model", {}),
                "ingestion_observed": runtime_diag.get("ingestion_observed", {}),
                "runtime_loop": runtime_diag.get("runtime_loop", {}),
                "ingestion_note": (
                    "EV cloud ingestion delay/loss is runtime simulation input. "
                    "Defaults follow an evidence-informed central-console baseline (not site-calibrated). "
                    "EVSE side is treated as console-known reference without uplink delay/loss modeling."
                ),
                "ingestion_enabled": bool(params.ingestion_enabled),
                "ingestion_effective": params.effective_ingestion_dict(),
                "refined_cost_weights": {
                    "dtw": float(getattr(algo, "REFINED_DTW_WEIGHT", 1.0)),
                    "step_signature": float(getattr(algo, "REFINED_STEP_WEIGHT", 0.40)),
                    "mean_gap": float(getattr(algo, "REFINED_MEAN_WEIGHT", 0.0)),
                },
                "ingestion_reference": ingestion_reference_payload(),
                "per_ev": rows,
                "csv_path": out_csv,
            }


class HistoryStore:
    """Simple JSON-backed storage for simulation results."""

    def __init__(self, path: Path, limit: int = 50):
        self.path = path
        self.limit = limit
        self._lock = Lock()
        self._cache: List[Dict[str, Any]] | None = None

    @staticmethod
    def _format_timestamp(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text == "":
            return value
        try:
            normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
            dt = datetime.fromisoformat(normalized)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return value

    def _load(self) -> List[Dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._cache = []
            return self._cache
        with self.path.open("r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                data = []
        if not isinstance(data, list):
            data = []
        modified = False
        for entry in data:
            if isinstance(entry, dict) and "note" not in entry:
                entry["note"] = ""
                modified = True
            if isinstance(entry, dict) and "timestamp" in entry:
                new_ts = self._format_timestamp(entry.get("timestamp"))
                if new_ts != entry.get("timestamp"):
                    entry["timestamp"] = new_ts
                    modified = True
        if modified:
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        self._cache = data
        return self._cache

    def all_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = list(self._load())
        return data

    @staticmethod
    def _normalize_note(note: Any) -> str:
        if note is None:
            return ""
        text = str(note).strip()
        # Keep notes compact for list rendering and storage.
        if len(text) > 300:
            text = text[:300]
        return text

    def add_entry(self, params: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(params, "to_dict"):
            params_dict = params.to_dict()
        elif isinstance(params, dict):
            params_dict = params
        else:
            raise TypeError("params must be a SimulationParameters instance or dict")

        now = datetime.now()
        base_id = now.strftime("%Y-%m-%d-%H-%M-%S")
        display_ts = now.strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            data = self._load()
            existing_ids = {entry.get("id") for entry in data}
            unique_id = base_id
            suffix = 1
            while unique_id in existing_ids:
                suffix += 1
                unique_id = f"{base_id}-{suffix}"

            entry = {
                "id": unique_id,
                "timestamp": display_ts,
                "params": params_dict,
                "result": result,
                "note": "",
            }
            data.insert(0, entry)
            if len(data) > self.limit:
                del data[self.limit:]
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        return entry

    def get_entry(self, entry_id: str) -> Dict[str, Any] | None:
        with self._lock:
            for entry in self._load():
                if entry.get("id") == entry_id:
                    return entry
        return None

    def clear(self) -> None:
        with self._lock:
            self._cache = []
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump([], fh, ensure_ascii=False, indent=2)

    def update_note(self, entry_id: str, note: Any) -> Dict[str, Any] | None:
        normalized_note = self._normalize_note(note)
        with self._lock:
            data = self._load()
            for entry in data:
                if entry.get("id") != entry_id:
                    continue
                entry["note"] = normalized_note
                with self.path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                return entry
        return None
