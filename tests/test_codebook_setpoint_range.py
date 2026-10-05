"""Codebook band-sweep plumbing: the set-point band kwarg narrows the codebook only; the default reproduces the canonical draw."""
import numpy as np
import pytest

from pair_identification import matching_core as mc
from case_study.section_vi_reproduction import _validate_setpoint_range


def test_default_band_is_bit_identical_to_explicit_canonical_band():
    a = mc.generate_command_patterns(num_patterns=8, seed=1234, internal_min_gap=4, consec_min_gap=8, min_pairwise_l1=40, candidate_pool=150)
    info_a = dict(mc.LAST_CODEBOOK_INFO)
    b = mc.generate_command_patterns(num_patterns=8, seed=1234, internal_min_gap=4, consec_min_gap=8, min_pairwise_l1=40, candidate_pool=150, setpoint_range=(6, 30))
    assert a == b
    assert info_a["setpoint_range"] == (6, 30)
    assert set(info_a) >= {"terminal_branch", "ladder_rungs", "setpoint_range"}


def test_narrow_band_confines_setpoints_and_is_reproducible():
    p = mc.generate_command_patterns(num_patterns=8, seed=77, internal_min_gap=4, consec_min_gap=8, min_pairwise_l1=40, candidate_pool=150, setpoint_range=(8, 16))
    q = mc.generate_command_patterns(num_patterns=8, seed=77, internal_min_gap=4, consec_min_gap=8, min_pairwise_l1=40, candidate_pool=150, setpoint_range=(8, 16))
    assert p == q and len(p) == 8
    vals = [v for pat in p for v in pat[1:]]
    assert min(vals) >= 8 and max(vals) <= 16
    assert all(pat[0] == 0 and len(set(pat[1:])) == 5 for pat in p)
    stats = mc.codebook_statistics(p)
    assert stats["unique_patterns"] == 8 and stats["pairwise_l1_min_a"] >= 0 and stats["internal_gap_min_a"] >= 1


def test_codebook_statistics_definition():
    pats = [[0, 6, 18, 28, 15, 10], [0, 8, 18, 14, 19, 24]]
    s = mc.codebook_statistics(pats)
    assert s["pairwise_l1_min_a"] == s["pairwise_l1_mean_a"] == float(2 + 0 + 14 + 4 + 14)
    assert s["internal_gap_min_a"] == 1.0   # 18/19 in the second pattern
    assert s["consec_step_min_a"] == 4.0    # 14 -> 18 ... min |Δ| over successive non-zero steps


def test_band_validation():
    assert _validate_setpoint_range(None) is None
    assert _validate_setpoint_range((8, 16)) == [8, 16]
    for bad in ((30, 6), (0, 10), (10, 12), "6-30"):
        with pytest.raises(ValueError):
            _validate_setpoint_range(bad)
    with pytest.raises(ValueError):
        mc.generate_command_patterns(num_patterns=3, seed=1, candidate_pool=50, setpoint_range=(10, 12))
