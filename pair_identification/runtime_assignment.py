# -*- coding: utf-8 -*-
"""Sparse online assignment for active EV sets."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment


def assign_sparse_hungarian(
    ev_rows: list[np.ndarray],
    row_slot_grids: list[dict[int, np.ndarray]],
    slots_list: list[int],
    candidate_slots_by_row: list[list[int]],
    refined_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    switch_penalty: float = 0.0,
    previous_slot_by_row: dict[int, int] | None = None,
) -> tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]:
    """Return row->(slot,cost) assignment and row-wise evaluated costs."""

    n_rows = len(ev_rows)
    n_cols = len(slots_list)
    if n_rows == 0 or n_cols == 0:
        return {}, {}

    previous_slot_by_row = previous_slot_by_row or {}
    slot_to_col = {int(slot): int(cj) for cj, slot in enumerate(slots_list)}
    sparse = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    row_costs: dict[int, dict[int, float]] = {}

    for ri in range(n_rows):
        row_costs[ri] = {}
        allowed = set(int(s) for s in candidate_slots_by_row[ri])
        prev_slot = previous_slot_by_row.get(ri)
        if prev_slot is not None:
            allowed.add(int(prev_slot))
        if not allowed:
            allowed = set(int(s) for s in slots_list)

        for slot in allowed:
            if slot not in slot_to_col:
                continue
            cj = slot_to_col[int(slot)]
            base_cost = float(refined_cost_fn(ev_rows[ri], row_slot_grids[ri][int(slot)]))
            if prev_slot is not None and int(prev_slot) != int(slot):
                base_cost += float(switch_penalty)
            cost = float(base_cost + 1e-6 * (ri + 0.01 * cj))
            sparse[ri, cj] = cost
            row_costs[ri][int(slot)] = cost

    finite_mask = np.isfinite(sparse)
    if not np.any(finite_mask):
        return {}, row_costs

    dense = sparse.copy()
    finite_vals = dense[finite_mask]
    max_finite = float(np.max(finite_vals)) if finite_vals.size > 0 else 1e6
    penalty = max_finite + max(1.0, float(switch_penalty) * 100.0)
    dense[~finite_mask] = penalty

    row_ind, col_ind = linear_sum_assignment(dense)
    out: dict[int, tuple[int, float]] = {}
    for r, c in zip(row_ind, col_ind):
        ri = int(r)
        cj = int(c)
        slot = int(slots_list[cj])
        if not np.isfinite(sparse[ri, cj]):
            continue
        cost = float(sparse[ri, cj])
        out[ri] = (slot, cost)

    return out, row_costs
