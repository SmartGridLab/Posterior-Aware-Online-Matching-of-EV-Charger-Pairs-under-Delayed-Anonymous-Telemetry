# -*- coding: utf-8 -*-
"""Bayesian windowed assignment utilities for EV-EVSE matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import math
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _normalize_prob_map(values: dict[int, float], eps: float) -> dict[int, float]:
    if not values:
        return {}
    keys = [int(k) for k in values.keys()]
    arr = np.asarray([max(float(values[k]), float(eps)) for k in keys], dtype=np.float64)
    total = float(np.sum(arr))
    if not np.isfinite(total) or total <= 0.0:
        uni = 1.0 / float(len(keys))
        return {int(k): float(uni) for k in keys}
    arr = arr / total
    return {int(k): float(v) for k, v in zip(keys, arr)}


def compute_time_prior(
    candidate_slots: list[int],
    delta_t_by_slot: dict[int, float],
    alpha: float,
    eps: float,
    prior_mix: float = 0.35,
) -> dict[int, float]:
    slots = [int(s) for s in candidate_slots]
    if len(slots) == 0:
        return {}

    finite = [float(delta_t_by_slot.get(int(s), float("inf"))) for s in slots]
    finite = [d for d in finite if np.isfinite(d)]
    if len(finite) == 0:
        u = 1.0 / float(len(slots))
        return {int(s): float(u) for s in slots}

    scale = float(np.quantile(np.asarray(finite, dtype=np.float64), 0.75))
    if (not np.isfinite(scale)) or scale < 1.0:
        scale = 1.0
    alpha_eff = max(0.0, float(alpha))
    max_finite = float(np.max(np.asarray(finite, dtype=np.float64)))

    raw: dict[int, float] = {}
    for slot in slots:
        d = float(delta_t_by_slot.get(int(slot), max_finite + scale))
        if not np.isfinite(d):
            d = max_finite + scale
        z = float(np.clip(-alpha_eff * (max(0.0, d) / scale), -80.0, 80.0))
        raw[int(slot)] = float(math.exp(z))

    shaped = _normalize_prob_map(raw, float(eps))
    mix = float(np.clip(float(prior_mix), 0.0, 1.0))
    u = 1.0 / float(len(slots))
    blended = {int(s): float(mix * shaped.get(int(s), u) + (1.0 - mix) * u) for s in slots}
    return _normalize_prob_map(blended, float(eps))


def compute_current_likelihood(
    candidate_slots: list[int],
    cost_by_slot: dict[int, float],
    beta: float,
    eps: float,
) -> dict[int, float]:
    slots = [int(s) for s in candidate_slots]
    if len(slots) == 0:
        return {}

    finite_costs = [float(cost_by_slot.get(int(s), float("inf"))) for s in slots]
    finite_costs = [c for c in finite_costs if np.isfinite(c)]
    if len(finite_costs) == 0:
        u = 1.0 / float(len(slots))
        return {int(s): float(u) for s in slots}

    arr = np.asarray(finite_costs, dtype=np.float64)
    cmin = float(np.min(arr))
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = mad if (np.isfinite(mad) and mad > 1e-6) else float(np.std(arr))
    if (not np.isfinite(scale)) or scale <= 1e-6:
        span = float(np.max(arr) - np.min(arr))
        scale = span if span > 1e-6 else 1.0

    beta_eff = max(0.0, float(beta))
    raw: dict[int, float] = {}
    for slot in slots:
        c = float(cost_by_slot.get(int(slot), float("inf")))
        if not np.isfinite(c):
            raw[int(slot)] = float(eps)
            continue
        x = max(0.0, float((c - cmin) / scale))
        z = float(np.clip(-beta_eff * x, -80.0, 80.0))
        raw[int(slot)] = float(math.exp(z))
    return _normalize_prob_map(raw, float(eps))


def update_posterior(
    prev: dict[int, float],
    time_prior: dict[int, float],
    current_like: dict[int, float],
    eps: float,
    prev_power: float = 0.60,
    time_power: float = 1.00,
) -> dict[int, float]:
    keys = sorted(set(int(k) for k in prev) | set(int(k) for k in time_prior) | set(int(k) for k in current_like))
    if not keys:
        return {}
    log_scores = []
    for slot in keys:
        p_prev = max(float(prev.get(int(slot), eps)), float(eps))
        p_t = max(float(time_prior.get(int(slot), eps)), float(eps))
        p_c = max(float(current_like.get(int(slot), eps)), float(eps))
        log_scores.append(float(prev_power) * math.log(p_prev) + float(time_power) * math.log(p_t) + math.log(p_c))
    arr = np.asarray(log_scores, dtype=np.float64)
    arr = arr - float(np.max(arr))
    probs = np.exp(arr)
    s = float(np.sum(probs))
    if not np.isfinite(s) or s <= 0.0:
        uni = 1.0 / float(len(keys))
        return {int(k): float(uni) for k in keys}
    probs = probs / s
    return {int(k): float(v) for k, v in zip(keys, probs)}


def _combine_checkpoint(
    components: list[tuple[dict[int, float], float]],
    keys: list[int],
    eps: float,
) -> dict[int, float]:
    """Renormalized distribution from a log-linear combination of components.

    ``components`` is a list of ``(prob_map, exponent)`` pairs combined in log
    space over ``keys`` and renormalized to sum to 1. This mirrors the algebra of
    :func:`update_posterior` but lets a partial combination (e.g. carry-over only,
    or carry-over + time-prior before the likelihood) be observed. It is used
    exclusively to record the Supplementary Fig. S1 diagnostic checkpoints (figure file figure5) when
    ``EVLINK_FIG5_TRACE=1`` and never participates in the assignment path.
    """
    if not keys:
        return {}
    log_scores = []
    for slot in keys:
        acc = 0.0
        for dist, power in components:
            p = max(float(dist.get(int(slot), eps)), float(eps))
            acc += float(power) * math.log(p)
        log_scores.append(acc)
    arr = np.asarray(log_scores, dtype=np.float64)
    arr = arr - float(np.max(arr))
    probs = np.exp(arr)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        uni = 1.0 / float(len(keys))
        return {int(k): float(uni) for k in keys}
    probs = probs / total
    return {int(k): float(v) for k, v in zip(keys, probs)}


def _top2(posterior: dict[int, float]) -> tuple[int | None, float, float]:
    if not posterior:
        return None, 0.0, 0.0
    ranked = sorted(((int(k), float(v)) for k, v in posterior.items()), key=lambda kv: (-kv[1], kv[0]))
    top_slot, top_prob = ranked[0]
    second_prob = float(ranked[1][1]) if len(ranked) >= 2 else 0.0
    return int(top_slot), float(top_prob), float(second_prob)


def _hungarian_with_inf_penalty(cost_matrix: np.ndarray, switch_penalty: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.asarray(cost_matrix, dtype=np.float64).copy()
    finite_mask = np.isfinite(dense)
    if np.any(finite_mask):
        mx = float(np.max(dense[finite_mask]))
        penalty = mx + max(1.0, float(switch_penalty) * 100.0)
    else:
        penalty = 1e9
    dense[~finite_mask] = penalty
    row_ind, col_ind = linear_sum_assignment(dense)
    return dense, row_ind.astype(int), col_ind.astype(int)


@dataclass
class BayesianWindowedAssigner:
    """Sequential MAP-like assignment under window-watermark flow."""

    time_prior_alpha: float = 0.01
    current_like_beta: float = 1.0
    eps: float = 1e-9
    time_prior_mix: float = 0.35
    posterior_prev_power: float = 0.60
    current_cost_weight: float = 0.15
    history_limit: int = 20
    match_state: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Runtime diagnostics snapshots populated after each __call__.
    last_row_details: dict[int, dict[int, dict[str, Any]]] = field(default_factory=dict, init=False)
    last_row_states: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)

    def _uniform_prior(self, candidates: list[int]) -> dict[int, float]:
        if not candidates:
            return {}
        p = 1.0 / float(len(candidates))
        return {int(slot): float(p) for slot in candidates}

    def __call__(
        self,
        *,
        ev_rows_pref: list[np.ndarray],
        row_slot_pref: list[dict[int, np.ndarray]],
        slots_list: list[int],
        candidate_slots_by_row: list[list[int]],
        time_costs_by_row: list[dict[int, float]] | None = None,
        previous_slot_by_row: dict[int, int] | None = None,
        cfg: Any = None,
        refined_cost_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        row_ev_ids: list[int] | None = None,
        time_delta_by_row: list[dict[int, float]] | None = None,
        now_ts: pd.Timestamp | None = None,
        **_kwargs: Any,
    ) -> tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]:
        del time_costs_by_row

        if previous_slot_by_row is None:
            previous_slot_by_row = {}

        # Ablation variants instantiate this class with modified Bayesian parameters.
        # Those instance values must have precedence over runtime cfg defaults.
        alpha = float(self.time_prior_alpha)
        beta = float(self.current_like_beta)
        eps = max(float(self.eps), float(getattr(cfg, "posterior_eps", self.eps)))
        prior_mix = float(self.time_prior_mix)
        prev_power = float(self.posterior_prev_power)
        current_cost_weight = float(np.clip(float(getattr(cfg, "current_cost_weight", self.current_cost_weight)), 0.0, 1.0))
        min_shared = max(1, int(getattr(cfg, "min_shared_points", 1)))

        n_rows = int(len(ev_rows_pref))
        n_cols = int(len(slots_list))
        slot_to_col = {int(slot): int(cj) for cj, slot in enumerate(slots_list)}
        cost_matrix = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
        row_costs: dict[int, dict[int, float]] = {int(ri): {} for ri in range(n_rows)}
        row_details: dict[int, dict[int, dict[str, Any]]] = {}
        row_states: dict[int, dict[str, Any]] = {}

        for ri in range(n_rows):
            ev_id = int(row_ev_ids[ri]) if row_ev_ids is not None and ri < len(row_ev_ids) else int(ri)
            candidates = [int(s) for s in candidate_slots_by_row[ri] if int(s) in slot_to_col]
            if len(candidates) == 0:
                candidates = [int(s) for s in slots_list]
            e_obs = np.asarray(ev_rows_pref[ri], dtype=np.float64)

            delta_map = {}
            if time_delta_by_row is not None and ri < len(time_delta_by_row):
                delta_map = {int(k): float(v) for k, v in dict(time_delta_by_row[ri] or {}).items()}
            time_prior = compute_time_prior(
                candidates,
                delta_map,
                alpha=alpha,
                eps=eps,
                prior_mix=prior_mix,
            )

            prev_state = self.match_state.get(int(ev_id), {})
            prev_post = {}
            if isinstance(prev_state.get("posterior"), dict):
                prev_post = {
                    int(k): float(v)
                    for k, v in prev_state["posterior"].items()
                    if int(k) in set(candidates)
                }
            if len(prev_post) == 0:
                prev_post = self._uniform_prior(candidates)
            prev_post = _normalize_prob_map(prev_post, eps)

            def _stage_cost_map(prefix_ratio: float) -> dict[int, float]:
                if refined_cost_fn is None:
                    return {}
                n = int(len(e_obs))
                if n <= 0:
                    return {}
                use_n = max(int(min_shared), int(round(float(prefix_ratio) * float(n))))
                use_n = max(1, min(use_n, n))
                e_cut = np.asarray(e_obs[:use_n], dtype=np.float64)
                out: dict[int, float] = {}
                for slot in candidates:
                    s_obs_raw = row_slot_pref[ri].get(int(slot))
                    if s_obs_raw is None:
                        continue
                    s_obs = np.asarray(s_obs_raw, dtype=np.float64)
                    s_cut = np.asarray(s_obs[:use_n], dtype=np.float64)
                    mask = np.isfinite(e_cut) & np.isfinite(s_cut)
                    if int(np.sum(mask)) < int(min_shared):
                        continue
                    c = float(refined_cost_fn(e_cut, s_cut))
                    if np.isfinite(c):
                        out[int(slot)] = c
                return out

            posterior = dict(prev_post)
            current_like_last = {int(slot): 1.0 / float(max(1, len(candidates))) for slot in candidates}
            stage_trace: list[dict[str, Any]] = []
            # Supplementary Fig. S1 diagnostic (figure file figure5): when EVLINK_FIG5_TRACE=1, record the renormalized
            # posterior at three checkpoints inside each of the two update stages
            # (six checkpoints total). Purely observational; gated so the flag-off
            # assignment path and its performance are unchanged.
            fig5_trace = os.environ.get("EVLINK_FIG5_TRACE") == "1"
            checkpoint_trace: list[dict[str, Any]] = []
            ckpt_keys = sorted(int(s) for s in candidates)
            for stage_id, ratio, time_power in ((1, 0.60, 1.00), (2, 1.00, 0.20)):
                if fig5_trace:
                    prev_stage = dict(posterior)
                cost_by_slot = _stage_cost_map(prefix_ratio=float(ratio))
                current_like = compute_current_likelihood(candidates, cost_by_slot, beta=beta, eps=eps)
                current_like_last = dict(current_like)
                posterior = update_posterior(
                    posterior,
                    time_prior,
                    current_like,
                    eps=eps,
                    prev_power=prev_power,
                    time_power=float(time_power),
                )
                top_slot, top_prob, second_prob = _top2(posterior)
                if fig5_trace:
                    # Checkpoint A: carry-over. Stage 1 shows the raw carry-over (a
                    # uniform prior on the session's first watermark decision); stage 2
                    # shows the stage-1 posterior raised to gamma_prev.
                    if int(stage_id) == 1:
                        ck_a = _combine_checkpoint([(prev_stage, 1.0)], ckpt_keys, eps)
                        label_a = "carry_over_uniform_prior"
                    else:
                        ck_a = _combine_checkpoint([(prev_stage, prev_power)], ckpt_keys, eps)
                        label_a = "carry_over_prev_power"
                    # Checkpoint B: after the time-prior term (before the likelihood).
                    ck_b = _combine_checkpoint(
                        [(prev_stage, prev_power), (time_prior, float(time_power))], ckpt_keys, eps
                    )
                    # Checkpoint C: after the current-likelihood term + renormalize;
                    # equals the actual stage posterior.
                    ck_c = dict(posterior)
                    base_index = 3 * (int(stage_id) - 1)
                    for offset, (dist, label) in enumerate(
                        (
                            (ck_a, label_a),
                            (ck_b, "time_prior_applied"),
                            (ck_c, "current_likelihood_renormalized"),
                        ),
                        start=1,
                    ):
                        c_top, c_p1, c_p2 = _top2(dist)
                        checkpoint_trace.append(
                            {
                                "diagnostic_index": int(base_index + offset),
                                "stage": int(stage_id),
                                "checkpoint": str(label),
                                "top_slot": None if c_top is None else int(c_top),
                                "top1_posterior": float(c_p1),
                                "top2_posterior": float(c_p2),
                            }
                        )
                stage_trace.append(
                    {
                        "stage": int(stage_id),
                        "prefix_ratio": float(ratio),
                        "top_slot": None if top_slot is None else int(top_slot),
                        "top_prob": float(top_prob),
                        "second_prob": float(second_prob),
                    }
                )

            top_slot, top_prob, second_prob = _top2(posterior)

            posterior_used = dict(posterior)

            hist = list(prev_state.get("history", []))
            hist.append(
                {
                    "ts": None if now_ts is None else pd.Timestamp(now_ts).isoformat(),
                    "top_slot": None if top_slot is None else int(top_slot),
                    "top_prob": float(top_prob),
                    "second_prob": float(second_prob),
                    "stages": stage_trace,
                }
            )
            if len(hist) > int(self.history_limit):
                hist = hist[-int(self.history_limit) :]

            self.match_state[int(ev_id)] = {
                "posterior": {int(k): float(v) for k, v in posterior_used.items()},
                "confidence": float(top_prob),
                "history": hist,
            }
            stage1 = stage_trace[0] if len(stage_trace) >= 1 else {}
            stage2 = stage_trace[1] if len(stage_trace) >= 2 else {}
            top_slot_int = -1 if top_slot is None else int(top_slot)
            top_gap = float(top_prob - second_prob)
            row_states[int(ri)] = {
                "ev_id": int(ev_id),
                "top_slot": int(top_slot_int),
                "top1_prob": float(top_prob),
                "top2_prob": float(second_prob),
                "top_gap": float(top_gap),
                "stage1_top_slot": int(stage1.get("top_slot", -1) if stage1.get("top_slot") is not None else -1),
                "stage1_top_prob": float(stage1.get("top_prob", float("nan"))),
                "stage1_second_prob": float(stage1.get("second_prob", float("nan"))),
                "stage2_top_slot": int(stage2.get("top_slot", -1) if stage2.get("top_slot") is not None else -1),
                "stage2_top_prob": float(stage2.get("top_prob", float("nan"))),
                "stage2_second_prob": float(stage2.get("second_prob", float("nan"))),
                "stage_trace": list(stage_trace),
            }
            if fig5_trace:
                row_states[int(ri)]["checkpoint_trace"] = list(checkpoint_trace)
            row_details[int(ri)] = {}

            prev_slot = previous_slot_by_row.get(int(ri))
            for slot in candidates:
                p = float(posterior_used.get(int(slot), eps))
                c_post = float(-math.log(max(p, eps)))
                like_p = float(current_like_last.get(int(slot), eps))
                c_like = float(-math.log(max(like_p, eps)))
                c = float((1.0 - current_cost_weight) * c_post + current_cost_weight * c_like)
                if prev_slot is not None and int(prev_slot) != int(slot):
                    c += float(getattr(cfg, "switch_penalty", 0.0))
                cj = slot_to_col[int(slot)]
                c = float(c + 1e-6 * (ri + 0.01 * cj))
                cost_matrix[ri, cj] = c
                row_costs[int(ri)][int(slot)] = c
                row_details[int(ri)][int(slot)] = {
                    "ev_id": int(ev_id),
                    "time_prior": float(time_prior.get(int(slot), float("nan"))),
                    "time_prior_cost": float(-math.log(max(float(time_prior.get(int(slot), eps)), eps))),
                    "current_like": float(current_like_last.get(int(slot), float("nan"))),
                    "posterior": float(p),
                    "posterior_cost": float(c_post),
                    "current_like_cost": float(c_like),
                    "final_assignment_cost": float(c),
                    "dtw_term": float("nan"),
                    "corr_term": float("nan"),
                    "step_signature_term": float("nan"),
                    "top_slot": int(top_slot_int),
                    "top1_prob": float(top_prob),
                    "top2_prob": float(second_prob),
                    "top_gap": float(top_gap),
                }

        valid_mask = np.isfinite(cost_matrix)
        _dense, row_ind, col_ind = _hungarian_with_inf_penalty(cost_matrix, float(getattr(cfg, "switch_penalty", 0.0)))
        assigned_by_row: dict[int, tuple[int, float]] = {}
        for r, c in zip(row_ind, col_ind):
            ri = int(r)
            cj = int(c)
            slot = int(slots_list[cj])
            if not bool(valid_mask[ri, cj]):
                assigned_by_row[ri] = (slot, float("inf"))
                continue
            assigned_by_row[ri] = (slot, float(cost_matrix[ri, cj]))

        self.last_row_details = row_details
        self.last_row_states = row_states
        return assigned_by_row, row_costs
