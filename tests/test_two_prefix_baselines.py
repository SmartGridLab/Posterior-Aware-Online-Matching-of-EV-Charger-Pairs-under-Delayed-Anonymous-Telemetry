"""Two-prefix baselines: cost rule and assignment behaviour on synthetic rows."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_study import section_vi_reproduction as harness  # noqa: E402


def _l1_cost(e: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(e, dtype=float) - np.asarray(s, dtype=float))))


def _synthetic():
    # Two EV rows, three slots. Row 0 matches slot 10 on the full prefix but slot 11 on the
    # first 60 %; row 1 matches slot 12 everywhere. n = 10 -> n60 = 6, n100 = 10.
    e0 = np.array([1, 1, 1, 1, 1, 1, 5, 5, 5, 5], dtype=float)
    e1 = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float)
    slots = {
        10: np.array([1, 1, 1, 1, 1, 1, 5, 5, 5, 5], dtype=float),   # exact match of row 0 (cost 0 / 0)
        11: np.array([1, 1, 1, 1, 1, 1, 9, 9, 9, 9], dtype=float),   # matches row 0 on the 60 % prefix only
        12: np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float),   # exact match of row 1
    }
    cfg = SimpleNamespace(min_shared_points=4, switch_penalty=0.0)
    return dict(
        ev_rows_pref=[e0, e1],
        row_slot_pref=[dict(slots), dict(slots)],
        slots_list=[10, 11, 12],
        candidate_slots_by_row=[[10, 11, 12], [10, 11, 12]],
        previous_slot_by_row={},
        cfg=cfg,
        refined_cost_fn=_l1_cost,
    )


def test_specs_registered_in_ablation_family():
    for aid in ("two_prefix_greedy", "two_prefix_hungarian"):
        assert aid in harness.ALL_ALGO_SPEC_BY_ID
        assert aid in harness.PAPER_ABLATION_ALGORITHM_IDS
        assert aid in harness.PAPER_ABLATION_FAMILIES["A1"]
        assert harness._validate_algorithm_ids([aid])[0].id == aid
    assert harness.TWO_PREFIX_STAGE_RATIOS == (0.60, 1.00)


def test_two_prefix_cost_is_mean_of_stage_costs():
    kw = _synthetic()
    row_costs, details = harness._two_prefix_row_costs(**kw)
    # row 0 vs slot 11: 60 % prefix cost 0, full-prefix cost mean(|[0]*6 + [4]*4|) = 1.6 -> mean 0.8
    c = row_costs[0][11]
    assert c == pytest.approx(0.5 * (0.0 + 1.6), abs=1e-4)
    assert details[0][11]["stage_costs"] == pytest.approx([0.0, 1.6], abs=1e-9)
    assert row_costs[0][10] == pytest.approx(0.0, abs=1e-4)
    assert row_costs[1][12] == pytest.approx(0.0, abs=1e-4)
    # deterministic tie-break 1e-6*(row + 0.01*col): exactly 0 for (row 0, col 0), tiny and positive otherwise
    assert row_costs[0][10] == 0.0
    assert 0.0 < row_costs[1][12] < 1e-4
    assert row_costs[0][10] < row_costs[0][11]


def test_greedy_and_hungarian_agree_on_unambiguous_case():
    kw = _synthetic()
    g_assigned, g_costs = harness._online_assign_two_prefix_greedy(**kw)
    h_assigned, h_costs = harness._online_assign_two_prefix_hungarian(**kw)
    assert g_assigned[0][0] == 10 and g_assigned[1][0] == 12
    assert h_assigned[0][0] == 10 and h_assigned[1][0] == 12
    assert g_costs == h_costs
    assert isinstance(harness._online_assign_two_prefix_greedy.last_row_details, dict)
    assert isinstance(harness._online_assign_two_prefix_hungarian.last_row_details, dict)


def test_hungarian_resolves_conflict_globally_where_greedy_cannot():
    # Both rows prefer slot 20; the global optimum gives slot 20 to row 1 (row 0 has a cheap alternative).
    e0 = np.array([2.0] * 10)
    e1 = np.array([2.0] * 10)
    slots = {20: np.array([2.0] * 10), 21: np.array([2.5] * 10)}
    cfg = SimpleNamespace(min_shared_points=4, switch_penalty=0.0)
    kw = dict(
        ev_rows_pref=[e0, e1],
        row_slot_pref=[{20: slots[20], 21: slots[21]}, {20: slots[20], 21: np.array([9.0] * 10)}],
        slots_list=[20, 21],
        candidate_slots_by_row=[[20, 21], [20, 21]],
        previous_slot_by_row={},
        cfg=cfg,
        refined_cost_fn=_l1_cost,
    )
    h_assigned, _ = harness._online_assign_two_prefix_hungarian(**kw)
    assert h_assigned[1][0] == 20 and h_assigned[0][0] == 21
    g_assigned, _ = harness._online_assign_two_prefix_greedy(**kw)
    # greedy ranks rows by margin: row 1 has the larger margin (2.0 vs 7.0) and takes slot 20 first
    assert g_assigned[1][0] == 20 and g_assigned[0][0] == 21


def test_non_finite_stage_drops_candidate():
    kw = _synthetic()
    kw["row_slot_pref"][0][11] = np.array([np.nan] * 6 + [9.0] * 4)  # 60 % stage has no shared points
    row_costs, _ = harness._two_prefix_row_costs(**kw)
    assert 11 not in row_costs[0]
    assert 10 in row_costs[0]


def test_driver_algorithm_filter():
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import reproduce_all  # noqa: E402

    allowed = list(reproduce_all.PAPER_ABLATION_ALGORITHM_IDS)
    assert reproduce_all._parse_algorithm_filter(None, allowed) == allowed
    assert reproduce_all._parse_algorithm_filter("two_prefix_greedy, two_prefix_hungarian", allowed) == [
        "two_prefix_greedy",
        "two_prefix_hungarian",
    ]
    with pytest.raises(SystemExit):
        reproduce_all._parse_algorithm_filter("not_an_id", allowed)
