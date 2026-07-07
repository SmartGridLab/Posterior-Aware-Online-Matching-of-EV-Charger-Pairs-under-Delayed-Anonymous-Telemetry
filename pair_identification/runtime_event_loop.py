# -*- coding: utf-8 -*-
"""Online window-watermark event loop for EV-EVSE matching.

EV stream is treated as cloud-ingested (delay/loss modeled), while EVSE stream
is treated as console-known reference (no EVSE uplink delay/loss model).
"""

from __future__ import annotations

from typing import Any, Callable

import hashlib
import os
import time

import numpy as np
import pandas as pd

from pair_identification.contracts import RuntimeConfig
from pair_identification.runtime_assignment import assign_sparse_hungarian
from pair_identification.runtime_candidates import prefix_len

_LOST_TS = pd.Timestamp("2262-04-11 00:00:00")


def _new_ingest_accumulator(max_delay_s: float, bin_width_s: float = 0.5) -> dict[str, Any]:
    max_delay = max(1.0, float(max_delay_s))
    bw = max(0.1, float(bin_width_s))
    n_bins = int(np.ceil(max_delay / bw)) + 1
    return {
        "total": 0,
        "received": 0,
        "lost": 0,
        "delay_sum_s": 0.0,
        "delay_max_s": 0.0,
        "max_delay_s_cfg": float(max_delay),
        "bin_width_s": float(bw),
        "delay_hist": np.zeros(n_bins, dtype=np.int64),
    }


def _accumulate_ingest_stats(
    acc: dict[str, Any],
    source_ts: pd.DatetimeIndex,
    ingest_ts: pd.DatetimeIndex,
) -> None:
    src = pd.DatetimeIndex(source_ts).to_numpy(dtype="datetime64[ns]")
    ing = pd.DatetimeIndex(ingest_ts).to_numpy(dtype="datetime64[ns]")
    if src.size == 0 or ing.size == 0:
        return

    n = int(min(src.size, ing.size))
    src = src[:n]
    ing = ing[:n]
    lost_mask = ing == _LOST_TS.to_datetime64()
    recv_mask = ~lost_mask

    lost_n = int(np.sum(lost_mask))
    recv_n = int(n - lost_n)
    acc["total"] = int(acc["total"] + n)
    acc["lost"] = int(acc["lost"] + lost_n)
    acc["received"] = int(acc["received"] + recv_n)

    if recv_n <= 0:
        return

    delays_s = (ing[recv_mask] - src[recv_mask]).astype("timedelta64[ns]").astype(np.int64).astype(np.float64) / 1e9
    if delays_s.size == 0:
        return
    max_cfg = float(acc["max_delay_s_cfg"])
    delays_s = np.clip(delays_s, 0.0, max_cfg)

    acc["delay_sum_s"] = float(acc["delay_sum_s"] + float(np.sum(delays_s)))
    acc["delay_max_s"] = float(max(float(acc["delay_max_s"]), float(np.max(delays_s))))

    bw = float(acc["bin_width_s"])
    bins = np.asarray(acc["delay_hist"], dtype=np.int64)
    idx = np.floor(delays_s / bw).astype(np.int64)
    idx = np.clip(idx, 0, bins.size - 1)
    np.add.at(bins, idx, 1)
    acc["delay_hist"] = bins


def _hist_quantile(hist: np.ndarray, q: float, bin_width_s: float) -> float:
    counts = np.asarray(hist, dtype=np.int64)
    total = int(np.sum(counts))
    if total <= 0:
        return float("nan")
    qq = float(np.clip(float(q), 0.0, 1.0))
    target = int(np.ceil(qq * total))
    target = max(1, target)
    csum = np.cumsum(counts)
    idx = int(np.searchsorted(csum, target, side="left"))
    idx = max(0, min(idx, counts.size - 1))
    return float((idx + 0.5) * float(bin_width_s))


def _finalize_ingest_stats(acc: dict[str, Any]) -> dict[str, Any]:
    total = int(acc["total"])
    received = int(acc["received"])
    lost = int(acc["lost"])
    loss_ratio = float(lost / total) if total > 0 else float("nan")
    recv_ratio = float(received / total) if total > 0 else float("nan")
    delay_mean = float(acc["delay_sum_s"] / received) if received > 0 else float("nan")
    return {
        "samples_total": int(total),
        "samples_received": int(received),
        "samples_lost": int(lost),
        "received_ratio": float(recv_ratio),
        "loss_ratio": float(loss_ratio),
        "delay_mean_s": float(delay_mean),
        "delay_p50_s": _hist_quantile(np.asarray(acc["delay_hist"]), 0.50, float(acc["bin_width_s"])),
        "delay_p90_s": _hist_quantile(np.asarray(acc["delay_hist"]), 0.90, float(acc["bin_width_s"])),
        "delay_p99_s": _hist_quantile(np.asarray(acc["delay_hist"]), 0.99, float(acc["bin_width_s"])),
        "delay_max_s": float(acc["delay_max_s"]) if received > 0 else float("nan"),
    }


def _empty_ingest_stats() -> dict[str, Any]:
    return {
        "samples_total": 0,
        "samples_received": 0,
        "samples_lost": 0,
        "received_ratio": float("nan"),
        "loss_ratio": float("nan"),
        "delay_mean_s": float("nan"),
        "delay_p50_s": float("nan"),
        "delay_p90_s": float("nan"),
        "delay_p99_s": float("nan"),
        "delay_max_s": float("nan"),
    }


def _build_timeline(
    ev_meta: list[dict[str, Any]],
    base_ts: pd.Timestamp,
    tick_s: int,
) -> list[tuple[pd.Timestamp, list[int], list[int]]]:
    event_map: dict[pd.Timestamp, dict[str, list[int]]] = {}
    start_ts: pd.Timestamp | None = None
    end_ts: pd.Timestamp | None = None

    for ei, meta in enumerate(ev_meta):
        arr = pd.Timestamp(meta["arrival_ts"])
        leave = pd.Timestamp(meta.get("session_end_ts", arr))
        event_map.setdefault(arr, {"arrive": [], "leave": []})["arrive"].append(int(ei))
        event_map.setdefault(leave, {"arrive": [], "leave": []})["leave"].append(int(ei))
        if start_ts is None or arr < start_ts:
            start_ts = arr
        if end_ts is None or leave > end_ts:
            end_ts = leave

    if start_ts is None or end_ts is None:
        return []

    tick = max(1, int(tick_s))
    base = pd.Timestamp(base_ts)
    rel_start = int((start_ts - base).total_seconds())
    rel_end = int((end_ts - base).total_seconds())
    tick_start_rel = (rel_start // tick) * tick
    tick_end_rel = ((rel_end + tick - 1) // tick) * tick
    tick_start = base + pd.Timedelta(seconds=int(tick_start_rel))
    tick_end = base + pd.Timedelta(seconds=int(tick_end_rel))

    for ts in pd.date_range(start=tick_start, end=tick_end, freq=f"{tick}s"):
        event_map.setdefault(pd.Timestamp(ts), {"arrive": [], "leave": []})

    out: list[tuple[pd.Timestamp, list[int], list[int]]] = []
    for ts in sorted(event_map.keys()):
        out.append((ts, sorted(event_map[ts]["arrive"]), sorted(event_map[ts]["leave"])))
    return out


def _build_slot_intervals(sessions_by_slot: dict[int, list[dict[str, Any]]]) -> dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    out: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for slot, sessions in sessions_by_slot.items():
        intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for sess in sessions:
            st = pd.Timestamp(sess.get("start_ts", sess.get("arrival_ts")))
            et = pd.Timestamp(sess.get("end_ts", sess.get("session_end_ts", st)))
            intervals.append((st, et))
        intervals.sort(key=lambda x: x[0])
        out[int(slot)] = intervals
    return out


def _active_slots_at(
    ts: pd.Timestamp,
    slot_intervals: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> list[int]:
    now = pd.Timestamp(ts)
    active: list[int] = []
    for slot, intervals in slot_intervals.items():
        for st, et in intervals:
            if st <= now <= et:
                active.append(int(slot))
                break
            if st > now:
                break
    return sorted(active)


def _interval_distance_s(ts: pd.Timestamp, st: pd.Timestamp, et: pd.Timestamp) -> float:
    now = pd.Timestamp(ts)
    a = pd.Timestamp(st)
    b = pd.Timestamp(et)
    if a <= now <= b:
        return 0.0
    if now < a:
        return float((a - now).total_seconds())
    return float((now - b).total_seconds())


def _slot_time_only_cost(
    slot: int,
    decision_end_ts: pd.Timestamp,
    start_est_ts: pd.Timestamp,
    slot_intervals: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> float:
    intervals = slot_intervals.get(int(slot), [])
    if len(intervals) == 0:
        return float("inf")
    end_dist = min(_interval_distance_s(decision_end_ts, st, et) for st, et in intervals)
    start_dist = min(abs(float((pd.Timestamp(st) - pd.Timestamp(start_est_ts)).total_seconds())) for st, _et in intervals)
    return float(end_dist + 0.10 * start_dist)


def _slot_end_distance_s(
    slot: int,
    decision_end_ts: pd.Timestamp,
    slot_intervals: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> float:
    intervals = slot_intervals.get(int(slot), [])
    if len(intervals) == 0:
        return float("inf")
    return float(min(_interval_distance_s(decision_end_ts, st, et) for st, et in intervals))


def _simulate_ingest_times(
    times: pd.DatetimeIndex,
    *,
    delay_mean_s: float,
    delay_jitter_s: float,
    delay_max_s: float,
    loss_prob: float,
    rng: np.random.RandomState,
) -> pd.DatetimeIndex:
    n = int(len(times))
    if n <= 0:
        return pd.DatetimeIndex([])

    jitter = max(1e-6, float(delay_jitter_s))
    max_delay = max(0.0, float(delay_max_s))
    delays = rng.normal(loc=float(delay_mean_s), scale=jitter, size=n)
    delays = np.clip(delays, 0.0, max_delay)
    ingest = pd.DatetimeIndex(times + pd.to_timedelta(delays, unit="s"))

    p_loss = float(np.clip(float(loss_prob), 0.0, 1.0))
    if p_loss > 0.0:
        lost = rng.random(size=n) < p_loss
        if np.any(lost):
            # copy(): pandas >= 3.0 returns a read-only view from to_numpy()
            arr = ingest.to_numpy(dtype="datetime64[ns]").copy()
            arr[lost] = _LOST_TS.to_datetime64()
            ingest = pd.DatetimeIndex(arr)
    return ingest


def _observed_ev_prefix(
    values_full: np.ndarray,
    ingest_ts_full: pd.DatetimeIndex,
    pref_n: int,
    now_ts: pd.Timestamp,
) -> np.ndarray:
    n = max(0, int(pref_n))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    vals = np.asarray(values_full[:n], dtype=np.float64)
    ingest = pd.DatetimeIndex(ingest_ts_full[:n]).to_numpy(dtype="datetime64[ns]")
    now = pd.Timestamp(now_ts).to_datetime64()

    visible = (ingest <= now) & np.isfinite(vals)
    if not np.any(visible):
        return np.full(n, np.nan, dtype=np.float64)

    idx = np.where(visible, np.arange(n, dtype=np.int64), -1)
    idx = np.maximum.accumulate(idx)
    out = np.full(n, np.nan, dtype=np.float64)
    valid = idx >= 0
    if np.any(valid):
        out[valid] = vals[idx[valid]]
    return out


def _slot_prefix(values_full: np.ndarray, pref_n: int) -> np.ndarray:
    n = max(0, int(pref_n))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(values_full, dtype=np.float64)
    return np.asarray(arr[:n], dtype=np.float64)


def _safe_pair_cost(
    e_obs: np.ndarray,
    s_obs: np.ndarray,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
    min_shared_points: int,
) -> float:
    mask = np.isfinite(e_obs) & np.isfinite(s_obs)
    if int(np.sum(mask)) < int(min_shared_points):
        return float("inf")
    v = float(cost_fn(np.asarray(e_obs[mask], dtype=np.float64), np.asarray(s_obs[mask], dtype=np.float64)))
    return v if np.isfinite(v) else float("inf")


def _fallback_single(
    ev_idx: int,
    ts: pd.Timestamp,
    candidate_slots: list[int],
    full_cache: dict[int, dict[str, Any]],
    refined_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    charger_sample_s: int,
    min_shared_points: int,
    slot_grid_getter: Callable[[int, int], np.ndarray],
) -> tuple[int, float, int, dict[int, float]]:
    slots = [int(s) for s in candidate_slots]
    if len(slots) == 0:
        return -1, float("inf"), 1, {}

    cache = full_cache[ev_idx]
    arrival_ts = pd.Timestamp(cache["arrival_ts"])
    ev_grid_full = np.asarray(cache["ev_grid"], dtype=np.float64)
    ev_ingest_full = pd.DatetimeIndex(cache["ev_ingest_ts"])

    pref_n = prefix_len(arrival_ts, ts, charger_sample_s, len(ev_grid_full))
    e_pref = _observed_ev_prefix(ev_grid_full, ev_ingest_full, pref_n, ts)

    cost_by_slot: dict[int, float] = {}
    best_slot = -1
    best_cost = float("inf")
    for cj, slot in enumerate(slots):
        s_full = slot_grid_getter(int(ev_idx), int(slot))
        s_pref = _slot_prefix(s_full, pref_n)
        cost = _safe_pair_cost(e_pref, s_pref, refined_cost_fn, int(min_shared_points))
        if np.isfinite(cost):
            cost = float(cost + 1e-6 * (0.01 * cj))
            cost_by_slot[int(slot)] = cost
            if cost < best_cost:
                best_cost = cost
                best_slot = int(slot)

    if best_slot < 0:
        return -1, float("inf"), int(pref_n), cost_by_slot

    return int(best_slot), float(best_cost), int(pref_n), cost_by_slot


def run_online_active_set(
    ev_meta: list[dict[str, Any]],
    sessions_by_slot: dict[int, list[dict[str, Any]]],
    ev_series: dict[int, pd.Series],
    *,
    base_ts: pd.Timestamp,
    window: int,
    charger_sample_s: int,
    cfg: RuntimeConfig,
    build_ev_grid_for_match: Callable[..., tuple[pd.DatetimeIndex, np.ndarray]],
    slot_grid_for_times: Callable[..., np.ndarray],
    coarse_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    refined_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    logger: Callable[[str], None] | None = None,
    diagnostics_out: dict[str, Any] | None = None,
    assignment_fn: Callable[..., tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]] | None = None,
    trace_level: str = "full",
) -> dict[int, dict[str, Any]]:
    del coarse_cost_fn  # no coarse-stage pruning in the console-known EVSE architecture

    def _log(msg: str) -> None:
        if logger is not None:
            logger(msg)

    if len(ev_meta) == 0:
        return {}

    loop_t0 = time.perf_counter()
    trace_mode = str(trace_level).strip().lower()
    if trace_mode not in {"full", "summary"}:
        trace_mode = "full"
    collect_full_trace = bool(trace_mode == "full")
    assert_causal_candidates = os.getenv("EVLINK_ASSERT_CAUSAL_CANDIDATES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    slots_list = sorted(int(s) for s in sessions_by_slot.keys())
    if len(slots_list) == 0:
        return {}

    ingest_seed = cfg.ev_ingest_seed
    if ingest_seed is None:
        rng = np.random.RandomState(int(np.random.randint(0, 2**31 - 1)))
    else:
        rng = np.random.RandomState(int(ingest_seed))
    ev_ingest_acc = _new_ingest_accumulator(float(cfg.ev_ingest_delay_max_s))

    window_sec = max(1, int(window))
    delay_grace_cfg = cfg.watermark_delay_max_s if getattr(cfg, "watermark_delay_max_s", None) is not None else cfg.ev_ingest_delay_max_s
    delay_grace_s = max(0.0, float(delay_grace_cfg))

    full_cache: dict[int, dict[str, Any]] = {}
    for ei, _meta in enumerate(ev_meta):
        times_5s, e_grid_raw = build_ev_grid_for_match(ev_meta, ev_series, ei, window=window_sec)
        ev_grid = np.asarray(e_grid_raw, dtype=np.float64)
        ev_ingest_ts = _simulate_ingest_times(
            times_5s,
            delay_mean_s=float(cfg.ev_ingest_delay_mean_s),
            delay_jitter_s=float(cfg.ev_ingest_delay_jitter_s),
            delay_max_s=float(cfg.ev_ingest_delay_max_s),
            loss_prob=float(cfg.ev_ingest_loss_prob),
            rng=rng,
        )
        _accumulate_ingest_stats(ev_ingest_acc, times_5s, ev_ingest_ts)

        arrival_ts = pd.Timestamp(ev_meta[ei].get("start_est_ts", ev_meta[ei]["arrival_ts"]))
        session_end_ts = pd.Timestamp(ev_meta[ei].get("session_end_ts", arrival_ts))
        decision_end_ts = arrival_ts + pd.Timedelta(seconds=int(window_sec))
        effective_end_ts = min(decision_end_ts, session_end_ts)
        watermark_ready_ts = effective_end_ts + pd.Timedelta(seconds=float(delay_grace_s))

        full_cache[int(ei)] = {
            "times_5s": times_5s,
            "arrival_ts": arrival_ts,
            "session_end_ts": session_end_ts,
            "decision_end_ts": decision_end_ts,
            "effective_end_ts": effective_end_ts,
            "watermark_ready_ts": watermark_ready_ts,
            "ev_grid": ev_grid,
            "ev_ingest_ts": ev_ingest_ts,
            "slot_grids": {},
        }

    ingest_sig = hashlib.sha256()
    for eidx in sorted(full_cache.keys()):
        arr = pd.DatetimeIndex(full_cache[int(eidx)]["ev_ingest_ts"]).to_numpy(dtype="datetime64[ns]").astype("int64")
        ingest_sig.update(int(eidx).to_bytes(4, "little", signed=False))
        ingest_sig.update(arr.tobytes())
    ingestion_signature_sha256 = ingest_sig.hexdigest()

    def _slot_grid_cached(eidx: int, slot: int) -> np.ndarray:
        cache = full_cache[int(eidx)]["slot_grids"]
        key = int(slot)
        if key not in cache:
            times_5s = pd.DatetimeIndex(full_cache[int(eidx)]["times_5s"])
            s_raw = slot_grid_for_times(int(key), times_5s, sessions_by_slot)
            cache[key] = np.asarray(s_raw, dtype=np.float64)
        return np.asarray(cache[key], dtype=np.float64)

    slot_intervals = _build_slot_intervals(sessions_by_slot)
    timeline = _build_timeline(ev_meta, base_ts=pd.Timestamp(base_ts), tick_s=int(cfg.evaluation_interval_s))
    active: set[int] = set()
    links: dict[int, dict[str, Any]] = {}
    ticks_total = 0
    ticks_recomputed = 0
    ticks_skipped = 0
    decision_rows = 0
    decision_batches = 0

    stage_candidate_build_s = 0.0
    stage_assignment_s = 0.0
    stage_fallback_s = 0.0
    fallback_invocations = 0

    candidate_rows_total = 0
    candidate_slots_total = 0
    gt_in_candidates_count = 0
    no_candidate_rows_total = 0
    fallback_to_all_rows_total = 0
    gt_recovered_by_fallback_count = 0
    candidate_query_lead_max_s = float("-inf")
    candidate_future_query_violation_count = 0

    decision_context_by_ev: dict[int, dict[str, Any]] = {}
    event_trace_rows: list[dict[str, Any]] = []
    cost_trace_rows: list[dict[str, Any]] = []
    posterior_trace_rows: list[dict[str, Any]] = []
    posterior_checkpoint_trace_rows: list[dict[str, Any]] = []

    def _safe_float_nan(value: Any) -> float:
        try:
            v = float(value)
        except Exception:
            return float("nan")
        return float(v) if np.isfinite(v) else float("nan")

    def _slot_rank(candidates: list[int], slot: int) -> int:
        try:
            return int(candidates.index(int(slot)) + 1)
        except ValueError:
            return -1

    def _normalize_detail_map(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                out[str(k)] = _safe_float_nan(v)
            else:
                out[str(k)] = v
        return out

    def _slot_detail_for_row(
        details_by_row: dict[int, dict[int, dict[str, Any]]],
        ri: int,
        slot: int,
    ) -> dict[str, Any]:
        row_map = details_by_row.get(int(ri), {})
        if not isinstance(row_map, dict):
            return {}
        detail = row_map.get(int(slot))
        return _normalize_detail_map(detail)

    def _finalize(eidx: int, ts: pd.Timestamp, pred: dict[str, Any], reason: str) -> None:
        decision_ts = pd.Timestamp(ts)
        cache = full_cache.get(int(eidx), {})
        meta = ev_meta[int(eidx)]
        ctx = decision_context_by_ev.get(int(eidx), {})
        slot = int(pred.get("slot", -1))
        cost = float(pred.get("cost", float("inf")))
        prefix_samples = int(pred.get("prefix_samples", 0))
        decision_latency_s = float((decision_ts - pd.Timestamp(meta["arrival_ts"])).total_seconds())
        fallback_triggered = bool(
            bool(ctx.get("assignment_fallback_triggered", False))
            or bool(ctx.get("fallback_to_all_triggered", False))
        )
        links[eidx] = {
            "slot": int(slot),
            "cost": float(cost),
            "arrival_ts": ev_meta[eidx]["arrival_ts"],
            "decision_ts": decision_ts,
            "prefix_samples": int(prefix_samples),
            "switch_count": int(pred.get("switch_count", 0)),
            "decision_reason": str(reason),
            "candidate_count": int(ctx.get("candidate_count", 0)),
            "gt_in_candidates": bool(ctx.get("gt_in_candidates", False)),
            "fallback_triggered": bool(fallback_triggered),
        }

        if collect_full_trace:
            event_trace_rows.append(
                {
                    "ev_idx": int(eidx),
                    "ev_token": str(meta.get("token", f"EV#{int(eidx)}")),
                    "gt_slot": int(meta.get("evse_idx_gt", -1)),
                    "pred_slot": int(slot),
                    "arrival_ts": pd.Timestamp(meta.get("arrival_ts")).isoformat(),
                    "start_est_ts": pd.Timestamp(meta.get("start_est_ts", meta.get("arrival_ts"))).isoformat(),
                    "session_end_ts": pd.Timestamp(meta.get("session_end_ts", decision_ts)).isoformat(),
                    "decision_end_ts": pd.Timestamp(cache.get("decision_end_ts", decision_ts)).isoformat(),
                    "effective_end_ts": pd.Timestamp(cache.get("effective_end_ts", decision_ts)).isoformat(),
                    "watermark_ready_ts": pd.Timestamp(cache.get("watermark_ready_ts", decision_ts)).isoformat(),
                    "candidate_gate_upper_ts": str(ctx.get("candidate_gate_upper_ts", "")),
                    "candidate_query_min_ts": str(ctx.get("candidate_query_min_ts", "")),
                    "candidate_query_max_ts": str(ctx.get("candidate_query_max_ts", "")),
                    "candidate_query_upper_minus_decision_s": _safe_float_nan(
                        ctx.get("candidate_query_upper_minus_decision_s", float("nan"))
                    ),
                    "candidate_future_query_violation": bool(ctx.get("candidate_future_query_violation", False)),
                    "last_required_sample_generation_ts": str(ctx.get("last_required_sample_generation_ts", "")),
                    "last_sample_ingestion_ts": str(ctx.get("last_sample_ingestion_ts", "")),
                    "evaluation_tick_ts": decision_ts.isoformat(),
                    "finalization_ts": decision_ts.isoformat(),
                    "decision_ts": decision_ts.isoformat(),
                    "decision_reason": str(reason),
                    "finalize_reason": str(reason),
                    "decision_latency_s": float(decision_latency_s),
                    "cost": float(cost),
                    "candidate_count": int(ctx.get("candidate_count", 0)),
                    "candidate_time_margin_s": int(getattr(cfg, "candidate_time_margin_s", 0)),
                    "gt_in_candidates": bool(ctx.get("gt_in_candidates", False)),
                    "gt_in_candidates_before_fallback": bool(ctx.get("gt_in_candidates_before_fallback", False)),
                    "no_candidate_before_fallback": bool(ctx.get("no_candidate_before_fallback", False)),
                    "fallback_to_all_triggered": bool(ctx.get("fallback_to_all_triggered", False)),
                    "assignment_fallback_triggered": bool(ctx.get("assignment_fallback_triggered", False)),
                    "fallback_triggered": bool(fallback_triggered),
                    "prefix_samples": int(prefix_samples),
                    "prefix_points": int(prefix_samples),
                    "shared_points": int(ctx.get("shared_points", 0)),
                    "selected_slot_rank": int(ctx.get("selected_slot_rank", -1)),
                }
            )

    def _candidate_slots_for_eidx(eidx: int, decision_ts: pd.Timestamp) -> tuple[list[int], bool, dict[str, Any]]:
        end_ts = pd.Timestamp(full_cache[int(eidx)]["effective_end_ts"])
        start_ts = pd.Timestamp(full_cache[int(eidx)]["arrival_ts"])
        wm_ts = pd.Timestamp(full_cache[int(eidx)]["watermark_ready_ts"])
        decision_boundary_ts = pd.Timestamp(decision_ts)
        margin_s = max(0, int(getattr(cfg, "candidate_time_margin_s", 0)))
        candidate_gate_upper_ts = min(end_ts + pd.Timedelta(seconds=int(margin_s)), decision_boundary_ts)
        query_times: list[pd.Timestamp] = []
        if margin_s <= 0:
            query_ts = min(end_ts, decision_boundary_ts)
            query_times = [pd.Timestamp(query_ts)]
            cand = _active_slots_at(query_ts, slot_intervals)
        else:
            ts0 = end_ts - pd.Timedelta(seconds=int(margin_s))
            ts1 = candidate_gate_upper_ts
            step = max(1, int(charger_sample_s))
            cand_set: set[int] = set()
            if ts1 < ts0:
                query_times = [pd.Timestamp(ts1)]
            else:
                query_times = [pd.Timestamp(tcur) for tcur in pd.date_range(start=ts0, end=ts1, freq=f"{step}s")]
            for tcur in query_times:
                if assert_causal_candidates:
                    assert pd.Timestamp(tcur) <= candidate_gate_upper_ts, (eidx, pd.Timestamp(tcur), candidate_gate_upper_ts)
                cand_set.update(_active_slots_at(pd.Timestamp(tcur), slot_intervals))
            cand = sorted(int(s) for s in cand_set)
        if query_times:
            query_min_ts = min(pd.Timestamp(t) for t in query_times)
            query_max_ts = max(pd.Timestamp(t) for t in query_times)
        else:
            query_min_ts = pd.NaT
            query_max_ts = pd.NaT
        if assert_causal_candidates and pd.notna(query_max_ts):
            assert pd.Timestamp(query_max_ts) <= candidate_gate_upper_ts, (eidx, pd.Timestamp(query_max_ts), candidate_gate_upper_ts)
        query_lead_s = (
            float((pd.Timestamp(query_max_ts) - decision_boundary_ts).total_seconds())
            if pd.notna(query_max_ts)
            else float("nan")
        )
        no_candidate_before_fallback = bool(len(cand) == 0)
        if no_candidate_before_fallback:
            cand = list(slots_list)
        cand = sorted(
            [int(s) for s in cand],
            key=lambda s: (
                _slot_time_only_cost(int(s), end_ts, start_ts, slot_intervals),
                int(s),
            ),
        )
        diag = {
            "watermark_ready_ts": wm_ts.isoformat(),
            "decision_boundary_ts": decision_boundary_ts.isoformat(),
            "candidate_gate_upper_ts": pd.Timestamp(candidate_gate_upper_ts).isoformat(),
            "candidate_query_min_ts": "" if pd.isna(query_min_ts) else pd.Timestamp(query_min_ts).isoformat(),
            "candidate_query_max_ts": "" if pd.isna(query_max_ts) else pd.Timestamp(query_max_ts).isoformat(),
            "candidate_query_upper_minus_decision_s": float(query_lead_s),
            "candidate_query_count": int(len(query_times)),
            "candidate_future_query_violation": bool(np.isfinite(query_lead_s) and query_lead_s > 1e-9),
        }
        return [int(s) for s in cand], bool(no_candidate_before_fallback), diag

    def _assign_rows(target_eidxs: list[int], now_ts: pd.Timestamp) -> dict[int, dict[str, Any]]:
        nonlocal stage_candidate_build_s, stage_assignment_s, stage_fallback_s
        nonlocal candidate_rows_total, candidate_slots_total, gt_in_candidates_count, fallback_invocations
        nonlocal no_candidate_rows_total, fallback_to_all_rows_total, gt_recovered_by_fallback_count
        nonlocal candidate_query_lead_max_s, candidate_future_query_violation_count

        if len(target_eidxs) == 0:
            return {}

        ev_rows_pref: list[np.ndarray] = []
        row_slot_pref: list[dict[int, np.ndarray]] = []
        candidate_slots_by_row: list[list[int]] = []
        time_costs_by_row: list[dict[int, float]] = []
        time_delta_by_row: list[dict[int, float]] = []
        row_context_by_row: dict[int, dict[str, Any]] = {}
        row_ev_ids: list[int] = []
        prefix_by_row: dict[int, int] = {}

        t_cand0 = time.perf_counter()
        for ri, eidx in enumerate(target_eidxs):
            cache = full_cache[int(eidx)]
            arr_ts = pd.Timestamp(cache["arrival_ts"])
            end_ts = pd.Timestamp(cache["effective_end_ts"])
            ev_full = np.asarray(cache["ev_grid"], dtype=np.float64)
            ev_ingest = pd.DatetimeIndex(cache["ev_ingest_ts"])
            row_ev_ids.append(int(eidx))

            pref_n = prefix_len(arr_ts, end_ts, int(charger_sample_s), len(ev_full))
            prefix_by_row[ri] = int(pref_n)
            e_pref = _observed_ev_prefix(ev_full, ev_ingest, pref_n, now_ts)
            ev_rows_pref.append(e_pref)

            candidates, no_candidate_before_fallback, cand_diag = _candidate_slots_for_eidx(int(eidx), pd.Timestamp(now_ts))
            query_lead_s = _safe_float_nan(cand_diag.get("candidate_query_upper_minus_decision_s", float("nan")))
            if np.isfinite(query_lead_s):
                candidate_query_lead_max_s = max(float(candidate_query_lead_max_s), float(query_lead_s))
                if query_lead_s > 1e-9:
                    candidate_future_query_violation_count += 1
            cand_set = set(int(s) for s in candidates)
            candidate_rows_total += 1
            candidate_slots_total += int(len(candidates))
            gt_slot = int(ev_meta[int(eidx)].get("evse_idx_gt", -1))
            gt_in_candidates = bool(int(gt_slot) in cand_set)
            gt_in_candidates_before_fallback = bool(False if no_candidate_before_fallback else gt_in_candidates)
            if gt_in_candidates:
                gt_in_candidates_count += 1
            if no_candidate_before_fallback:
                no_candidate_rows_total += 1
                fallback_to_all_rows_total += 1
                if gt_in_candidates and not gt_in_candidates_before_fallback:
                    gt_recovered_by_fallback_count += 1

            row_grids: dict[int, np.ndarray] = {}
            time_costs: dict[int, float] = {}
            time_deltas: dict[int, float] = {}
            for slot in candidates:
                row_grids[int(slot)] = _slot_prefix(_slot_grid_cached(int(eidx), int(slot)), pref_n)
                time_costs[int(slot)] = _slot_time_only_cost(int(slot), end_ts, arr_ts, slot_intervals)
                time_deltas[int(slot)] = _slot_end_distance_s(int(slot), end_ts, slot_intervals)
            candidate_slots_by_row.append(candidates)
            row_slot_pref.append(row_grids)
            time_costs_by_row.append(time_costs)
            time_delta_by_row.append(time_deltas)
            row_context_by_row[int(ri)] = {
                "candidate_count": int(len(candidates)),
                "gt_in_candidates": bool(gt_in_candidates),
                "gt_in_candidates_before_fallback": bool(gt_in_candidates_before_fallback),
                "no_candidate_before_fallback": bool(no_candidate_before_fallback),
                "fallback_to_all_triggered": bool(no_candidate_before_fallback),
                **cand_diag,
                "last_required_sample_generation_ts": (
                    pd.Timestamp(cache["times_5s"][max(0, min(int(pref_n), len(cache["times_5s"])) - 1)]).isoformat()
                    if int(pref_n) > 0 and len(cache["times_5s"]) > 0
                    else ""
                ),
                "last_sample_ingestion_ts": (
                    pd.Timestamp(cache["ev_ingest_ts"][max(0, min(int(pref_n), len(cache["ev_ingest_ts"])) - 1)]).isoformat()
                    if int(pref_n) > 0 and len(cache["ev_ingest_ts"]) > 0
                    else ""
                ),
            }

        def _refined_safe(e_obs: np.ndarray, s_obs: np.ndarray) -> float:
            return _safe_pair_cost(e_obs, s_obs, refined_cost_fn, int(cfg.min_shared_points))

        stage_candidate_build_s += float(time.perf_counter() - t_cand0)

        t_assign0 = time.perf_counter()
        if assignment_fn is None:
            assigned_by_row, row_costs = assign_sparse_hungarian(
                ev_rows=ev_rows_pref,
                row_slot_grids=row_slot_pref,
                slots_list=slots_list,
                candidate_slots_by_row=candidate_slots_by_row,
                refined_cost_fn=_refined_safe,
                switch_penalty=float(cfg.switch_penalty),
                previous_slot_by_row={},
            )
        else:
            assigned_by_row, row_costs = assignment_fn(
                ev_rows_pref=ev_rows_pref,
                row_slot_pref=row_slot_pref,
                slots_list=slots_list,
                candidate_slots_by_row=candidate_slots_by_row,
                time_costs_by_row=time_costs_by_row,
                previous_slot_by_row={},
                cfg=cfg,
                refined_cost_fn=_refined_safe,
                row_ev_ids=row_ev_ids,
                time_delta_by_row=time_delta_by_row,
                now_ts=pd.Timestamp(now_ts),
            )

        stage_assignment_s += float(time.perf_counter() - t_assign0)
        row_detail_by_row: dict[int, dict[int, dict[str, Any]]] = {}
        row_state_by_row: dict[int, dict[str, Any]] = {}
        if collect_full_trace and assignment_fn is not None:
            maybe_details = getattr(assignment_fn, "last_row_details", None)
            if isinstance(maybe_details, dict):
                row_detail_by_row = maybe_details
            maybe_states = getattr(assignment_fn, "last_row_states", None)
            if isinstance(maybe_states, dict):
                row_state_by_row = maybe_states

        out: dict[int, dict[str, Any]] = {}
        for ri, eidx in enumerate(target_eidxs):
            row_candidates = list(candidate_slots_by_row[ri])
            row_time_costs = dict(time_costs_by_row[ri]) if ri < len(time_costs_by_row) else {}
            row_assignment_costs = row_costs.get(ri, {})
            assignment_fallback_triggered = False
            if ri in assigned_by_row:
                slot, cost = assigned_by_row[ri]
                pref_n = int(prefix_by_row[ri])
                cost_by_slot = {
                    int(k): float(v)
                    for k, v in row_assignment_costs.items()
                    if np.isfinite(float(v))
                }
                if (not np.isfinite(float(cost))) or int(slot) < 0:
                    tf0 = time.perf_counter()
                    slot, cost, pref_n, cost_by_slot = _fallback_single(
                        int(eidx),
                        now_ts,
                        row_candidates if row_candidates else slots_list,
                        full_cache,
                        refined_cost_fn=refined_cost_fn,
                        charger_sample_s=int(charger_sample_s),
                        min_shared_points=int(cfg.min_shared_points),
                        slot_grid_getter=_slot_grid_cached,
                    )
                    assignment_fallback_triggered = True
                    stage_fallback_s += float(time.perf_counter() - tf0)
                    fallback_invocations += 1
            else:
                tf0 = time.perf_counter()
                slot, cost, pref_n, cost_by_slot = _fallback_single(
                    int(eidx),
                    now_ts,
                    row_candidates if row_candidates else slots_list,
                    full_cache,
                    refined_cost_fn=refined_cost_fn,
                    charger_sample_s=int(charger_sample_s),
                    min_shared_points=int(cfg.min_shared_points),
                    slot_grid_getter=_slot_grid_cached,
                )
                assignment_fallback_triggered = True
                stage_fallback_s += float(time.perf_counter() - tf0)
                fallback_invocations += 1

            chosen_slot = int(slot)
            chosen_rank = _slot_rank(row_candidates, chosen_slot)
            chosen_shared_points = 0
            chosen_grid = row_slot_pref[ri].get(chosen_slot)
            if chosen_grid is not None:
                e_obs = np.asarray(ev_rows_pref[ri], dtype=np.float64)
                s_obs = np.asarray(chosen_grid, dtype=np.float64)
                chosen_shared_points = int(np.sum(np.isfinite(e_obs) & np.isfinite(s_obs)))

            row_ctx = dict(row_context_by_row.get(int(ri), {}))
            row_ctx.update(
                {
                    "assignment_fallback_triggered": bool(assignment_fallback_triggered),
                    "selected_slot": int(chosen_slot),
                    "selected_slot_rank": int(chosen_rank),
                    "selected_cost": float(cost),
                    "prefix_samples": int(pref_n),
                    "shared_points": int(chosen_shared_points),
                }
            )
            decision_context_by_ev[int(eidx)] = row_ctx

            if collect_full_trace and len(row_candidates) > 0:
                for rank, cand_slot in enumerate(row_candidates, start=1):
                    trace_row = {
                        "ev_idx": int(eidx),
                        "slot": int(cand_slot),
                        "candidate_rank": int(rank),
                        "selected_slot": int(chosen_slot),
                        "is_selected": bool(int(cand_slot) == int(chosen_slot)),
                        "assignment_cost": _safe_float_nan(row_assignment_costs.get(int(cand_slot), float("nan"))),
                        "final_assignment_cost": _safe_float_nan(row_assignment_costs.get(int(cand_slot), float("nan"))),
                        "time_cost": _safe_float_nan(row_time_costs.get(int(cand_slot), float("nan"))),
                        "time_prior_cost": float("nan"),
                        "dtw_term": float("nan"),
                        "corr_term": float("nan"),
                        "step_signature_term": float("nan"),
                        "posterior": float("nan"),
                        "prefix_samples": int(pref_n),
                        "candidate_gate_upper_ts": str(row_ctx.get("candidate_gate_upper_ts", "")),
                        "candidate_query_min_ts": str(row_ctx.get("candidate_query_min_ts", "")),
                        "candidate_query_max_ts": str(row_ctx.get("candidate_query_max_ts", "")),
                        "candidate_query_upper_minus_decision_s": _safe_float_nan(
                            row_ctx.get("candidate_query_upper_minus_decision_s", float("nan"))
                        ),
                        "candidate_future_query_violation": bool(
                            row_ctx.get("candidate_future_query_violation", False)
                        ),
                        "no_candidate_before_fallback": bool(row_ctx.get("no_candidate_before_fallback", False)),
                        "fallback_to_all_triggered": bool(row_ctx.get("fallback_to_all_triggered", False)),
                        "assignment_fallback_triggered": bool(assignment_fallback_triggered),
                    }
                    trace_row.update(_slot_detail_for_row(row_detail_by_row, int(ri), int(cand_slot)))
                    cost_trace_rows.append(trace_row)
            row_state = row_state_by_row.get(int(ri), {}) if collect_full_trace else {}
            if collect_full_trace and isinstance(row_state, dict) and len(row_state) > 0:
                stage_trace = row_state.get("stage_trace", [])
                if isinstance(stage_trace, list) and len(stage_trace) > 0:
                    for st in stage_trace:
                        if not isinstance(st, dict):
                            continue
                        top_prob = _safe_float_nan(st.get("top_prob", float("nan")))
                        second_prob = _safe_float_nan(st.get("second_prob", float("nan")))
                        top_gap = (
                            float(top_prob - second_prob)
                            if np.isfinite(float(top_prob)) and np.isfinite(float(second_prob))
                            else float("nan")
                        )
                        posterior_trace_rows.append(
                            {
                                "ev_idx": int(eidx),
                                "decision_ts": pd.Timestamp(now_ts).isoformat(),
                                "stage_id": int(st.get("stage", -1)),
                                "prefix_ratio": _safe_float_nan(st.get("prefix_ratio", float("nan"))),
                                "top_slot": int(st.get("top_slot", -1) if st.get("top_slot") is not None else -1),
                                "top1_prob": float(top_prob),
                                "top2_prob": float(second_prob),
                                "top_gap": float(top_gap),
                                "final_top_slot": int(row_state.get("top_slot", -1)),
                            }
                        )
                checkpoint_trace = row_state.get("checkpoint_trace", [])
                if isinstance(checkpoint_trace, list) and len(checkpoint_trace) > 0:
                    for ck in checkpoint_trace:
                        if not isinstance(ck, dict):
                            continue
                        posterior_checkpoint_trace_rows.append(
                            {
                                "ev_idx": int(eidx),
                                "decision_ts": pd.Timestamp(now_ts).isoformat(),
                                "diagnostic_index": int(ck.get("diagnostic_index", -1)),
                                "stage": int(ck.get("stage", -1)),
                                "checkpoint": str(ck.get("checkpoint", "")),
                                "top_slot": int(ck.get("top_slot", -1) if ck.get("top_slot") is not None else -1),
                                "top1_posterior": _safe_float_nan(ck.get("top1_posterior", float("nan"))),
                                "top2_posterior": _safe_float_nan(ck.get("top2_posterior", float("nan"))),
                                "candidate_count": int(len(row_candidates)),
                            }
                        )

            out[int(eidx)] = {
                "slot": int(slot),
                "cost": float(cost),
                "prefix_samples": int(pref_n),
                "switch_count": 0,
                "cost_by_slot": cost_by_slot,
            }
        return out

    for ts, arrivals, leaves in timeline:
        ticks_total += 1
        rel = int((ts - pd.Timestamp(base_ts)).total_seconds())

        for ei in arrivals:
            active.add(int(ei))
            _log(f"[{rel:5d}s | {ts:%H:%M:%S}] {ev_meta[ei]['token']} arrived")

        ready = [
            int(ei)
            for ei in sorted(active)
            if int(ei) not in links and pd.Timestamp(ts) >= pd.Timestamp(full_cache[int(ei)]["watermark_ready_ts"])
        ]
        if ready:
            ticks_recomputed += 1
            decision_batches += 1
            preds = _assign_rows(ready, ts)
            decision_rows += int(len(ready))
            for eidx in ready:
                pred = preds.get(int(eidx), {"slot": -1, "cost": float("inf"), "prefix_samples": 0, "switch_count": 0})
                _finalize(int(eidx), ts, pred, reason="window_watermark")
                active.discard(int(eidx))
        else:
            ticks_skipped += 1

        unresolved_leaves = [int(ei) for ei in leaves if int(ei) not in links]
        if unresolved_leaves:
            ticks_recomputed += 1
            decision_batches += 1
            unresolved_sorted = sorted(set(unresolved_leaves))
            preds = _assign_rows(unresolved_sorted, ts)
            decision_rows += int(len(unresolved_sorted))
            for eidx in unresolved_sorted:
                pred = preds.get(int(eidx), {"slot": -1, "cost": float("inf"), "prefix_samples": 0, "switch_count": 0})
                reason = (
                    "session_end_before_watermark"
                    if pd.Timestamp(ts) < pd.Timestamp(full_cache[int(eidx)]["watermark_ready_ts"])
                    else "session_end_fallback"
                )
                _finalize(int(eidx), ts, pred, reason=reason)

        for ei in leaves:
            eidx = int(ei)
            token = ev_meta[eidx]["token"]
            gt_slot = int(ev_meta[eidx]["evse_idx_gt"])
            start_ts = pd.Timestamp(ev_meta[eidx]["arrival_ts"])
            end_ts = pd.Timestamp(ev_meta[eidx].get("session_end_ts", ts))
            dur_s = int((end_ts - start_ts).total_seconds()) if pd.notna(end_ts) else 0
            _log(
                f"[{rel:5d}s | {ts:%H:%M:%S}] Departed: {token} from slot {gt_slot:02d} | "
                f"started={start_ts:%H:%M:%S} ended={end_ts:%H:%M:%S} | duration={dur_s}s"
            )
            active.discard(eidx)

    unresolved = [int(ei) for ei in range(len(ev_meta)) if int(ei) not in links]
    if unresolved:
        final_ts = pd.Timestamp(timeline[-1][0]) if timeline else pd.Timestamp(base_ts)
        ticks_recomputed += 1
        decision_batches += 1
        preds = _assign_rows(unresolved, final_ts)
        decision_rows += int(len(unresolved))
        for eidx in unresolved:
            pred = preds.get(int(eidx), {"slot": -1, "cost": float("inf"), "prefix_samples": 0, "switch_count": 0})
            _finalize(int(eidx), final_ts, pred, reason="timeline_end_fallback")

    if diagnostics_out is not None:
        diagnostics_out["ingestion_model"] = {
            "ev": {
                "delay_mean_s": float(cfg.ev_ingest_delay_mean_s),
                "delay_jitter_s": float(cfg.ev_ingest_delay_jitter_s),
                "delay_max_s": float(cfg.ev_ingest_delay_max_s),
                "watermark_delay_max_s": float(delay_grace_s),
                "loss_prob": float(cfg.ev_ingest_loss_prob),
                "seed": None if cfg.ev_ingest_seed is None else int(cfg.ev_ingest_seed),
            },
            "evse": {
                "source": "console_known",
                "delay_mean_s": 0.0,
                "delay_jitter_s": 0.0,
                "delay_max_s": 0.0,
                "loss_prob": 0.0,
            },
        }
        diagnostics_out["ingestion_observed"] = {
            "ev": _finalize_ingest_stats(ev_ingest_acc),
            "evse": _empty_ingest_stats(),
        }
        total_wall_s = float(time.perf_counter() - loop_t0)
        stage_other_s = float(max(0.0, total_wall_s - stage_candidate_build_s - stage_assignment_s - stage_fallback_s))
        diagnostics_out["runtime_loop"] = {
            "mode": "window_watermark",
            "ticks_total": int(ticks_total),
            "ticks_recomputed": int(ticks_recomputed),
            "ticks_skipped": int(ticks_skipped),
            "recompute_ratio": float(ticks_recomputed / ticks_total) if ticks_total > 0 else float("nan"),
            "decision_batches": int(decision_batches),
            "decision_rows": int(decision_rows),
            "candidate_rows_total": int(candidate_rows_total),
            "candidate_slots_total": int(candidate_slots_total),
            "avg_candidates_per_ev": float(candidate_slots_total / candidate_rows_total) if candidate_rows_total > 0 else float("nan"),
            "gt_in_candidates_count": int(gt_in_candidates_count),
            "gt_in_candidates_ratio": float(gt_in_candidates_count / candidate_rows_total) if candidate_rows_total > 0 else float("nan"),
            "no_candidate_rows_total": int(no_candidate_rows_total),
            "fallback_to_all_rows_total": int(fallback_to_all_rows_total),
            "gt_recovered_by_fallback_count": int(gt_recovered_by_fallback_count),
            "no_candidate_rows_ratio": float(no_candidate_rows_total / candidate_rows_total) if candidate_rows_total > 0 else float("nan"),
            "fallback_to_all_rows_ratio": float(fallback_to_all_rows_total / candidate_rows_total) if candidate_rows_total > 0 else float("nan"),
            "gt_recovered_by_fallback_ratio": (
                float(gt_recovered_by_fallback_count / no_candidate_rows_total)
                if no_candidate_rows_total > 0
                else float("nan")
            ),
            "fallback_invocations": int(fallback_invocations),
            "candidate_query_lead_max_s": (
                float(candidate_query_lead_max_s)
                if np.isfinite(float(candidate_query_lead_max_s))
                else float("nan")
            ),
            "candidate_future_query_violation_count": int(candidate_future_query_violation_count),
            "candidate_causal_assertion_enabled": bool(assert_causal_candidates),
            "stage_candidate_build_s": float(stage_candidate_build_s),
            "stage_assignment_s": float(stage_assignment_s),
            "stage_fallback_s": float(stage_fallback_s),
            "stage_other_s": float(stage_other_s),
            "total_wall_s": float(total_wall_s),
            "trace_level": str(trace_mode),
        }
        diagnostics_out["event_trace"] = list(event_trace_rows) if collect_full_trace else []
        diagnostics_out["cost_trace"] = list(cost_trace_rows) if collect_full_trace else []
        diagnostics_out["posterior_trace"] = list(posterior_trace_rows) if collect_full_trace else []
        diagnostics_out["posterior_checkpoint_trace"] = list(posterior_checkpoint_trace_rows) if collect_full_trace else []
        diagnostics_out["ingestion_signature_sha256"] = str(ingestion_signature_sha256)

    return links
