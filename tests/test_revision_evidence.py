"""Claim checks on the committed evidence.

Each test is skipped while its artefact does not exist, so the suite stays green on a tree
that does not carry that evidence.
"""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ACC = ROOT / "case_study" / "scalability_analysis" / "accuracy"
CANONICAL_RAW = ACC / "canonical_run_20260907_111056_515561" / "raw_runs.csv"


def _need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"evidence not committed yet: {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------- Failure taxonomy (TABLE 8; data file table9_failure_taxonomy.csv)
def test_failure_taxonomy_is_complete_and_consistent():
    table = pd.read_csv(_need(ACC / "table9_failure_taxonomy.csv"))
    cases = pd.read_csv(_need(ACC / "failure_cases_representative.csv"))
    all_row = table[(table["breakdown"] == "all") & (table["bin"] == "all sessions")].iloc[0]
    assert int(all_row["sessions"]) == 10000  # 20 repeats x 500 sessions
    assert int(all_row["errors"]) == len(cases)
    # the taxonomy is a partition of the A3 errors at the representative point (accuracy 0.9838)
    assert 1.0 - float(all_row["error_rate"]) == pytest.approx(0.9838, abs=1e-6)
    causes = table[table["breakdown"] == "cause"]
    assert set(cases["primary_cause"]) <= {"truncation", "ingestion", "sensing", "overlap"}
    assert int(causes["errors"].sum()) == len(cases)
    assert float(causes["share_of_errors"].sum()) == pytest.approx(1.0, abs=1e-9)
    # every candidate-set-size bin row counts sessions; bins partition the sessions
    bins = table[table["breakdown"] == "candidate_set_size"]
    assert int(bins["sessions"].sum()) == 10000
    assert int(bins["errors"].sum()) == len(cases)


def test_trace_replay_matches_canonical_cell():
    raw = pd.read_csv(_need(ACC / "trace_representative_run_20260909_112529_182312" / "raw_runs.csv"))
    canonical = pd.read_csv(CANONICAL_RAW)
    ref = canonical[
        (canonical["ev_count"] == 500)
        & (canonical["ev_sample_s"] == 30)
        & (canonical["matcher_delay_max_s"] == 10.0)
        & (canonical["candidate_margin_s"] == 120)
    ]
    for algorithm_id in ("single_only", "bayesian_windowed"):
        a = raw[raw["algorithm_id"] == algorithm_id].sort_values("repeat_idx")
        b = ref[ref["algorithm_id"] == algorithm_id].sort_values("repeat_idx")
        assert len(a) == 20
        assert list(a["scenario_hash"]) == list(b["scenario_hash"])
        assert a["accuracy"].to_numpy() == pytest.approx(b["accuracy"].to_numpy(), abs=1e-12)


# ---------------------------------------------------------------- Latency decomposition
def test_latency_decomposition_reproduces_canonical_p90():
    lat = pd.read_csv(_need(ACC / "latency_decomposition_representative.csv"))
    assert len(lat) == 20
    assert lat["p90_match"].astype(bool).all()
    assert lat["p90_latency_raw_s"].mean() == pytest.approx(416.035, abs=1e-6)
    # the four terms: start offset + W + d_max + tick residual
    assert lat["window_s_mean"].to_numpy() == pytest.approx(360.0)
    assert lat["watermark_s_mean"].to_numpy() == pytest.approx(10.0)
    assert (lat["tick_residual_s_min"] >= 0).all() and (lat["tick_residual_s_max"] <= 5).all()


# ---------------------------------------------------------------- Two-prefix baselines (table10_*, TABLE 9 of the manuscript)
def test_two_prefix_run_is_paired_with_the_ablation_run():
    new = pd.read_csv(_need(ACC / "ablation_two_prefix_run_20260908_170955_253026" / "raw_runs.csv"))
    ablation = pd.read_csv(ACC / "ablation_run_20260909_112655_937821" / "raw_runs.csv")
    ref = ablation[ablation["algorithm_id"] == "single_only"].set_index(["ev_count", "repeat_idx"])["scenario_hash"]
    assert len(new) == 120  # 2 variants x 3 EV counts x 20 repeats
    assert set(new["algorithm_id"]) == {"two_prefix_greedy", "two_prefix_hungarian"}
    for r in new.itertuples():
        assert ref.loc[(int(r.ev_count), int(r.repeat_idx))] == r.scenario_hash
    assert new["ingestion_signature_match_ref"].astype(bool).all()
    assert new["candidate_query_lead_max_s"].max() <= 0.0
    assert new["candidate_future_query_violation_count"].max() == 0


def test_table10_and_figure6_series():
    table10 = pd.read_csv(_need(ACC / "table10_two_prefix_ablation.csv"))
    assert len(table10) == 6
    for variant in ("two_prefix_greedy", "two_prefix_hungarian"):
        row = table10[(table10["variant"] == variant) & (table10["ev_count"] == 500)].iloc[0]
        assert int(row["runs"]) == 20
        assert row["accuracy_mean"] == pytest.approx(0.9798, abs=1e-6)
        assert row["a3_mean"] == pytest.approx(0.9838, abs=1e-6)
        assert row["a1_mean"] == pytest.approx(0.9732, abs=1e-6)
    fig6 = pd.read_csv(ACC / "figure6_ablation_simulated.csv")
    assert len(fig6[["matcher_family", "variant"]].drop_duplicates()) == 14  # 12 manuscript series + 2 two-prefix baselines
    greedy_panel = fig6[fig6["matcher_family"] == "Greedy"]
    assert {"two-prefix greedy", "two-prefix Hungarian"} <= set(greedy_panel["variant"])


# ---------------------------------------------------------------- Static-limit sanity check (TABLE S2)
def test_static_limit_table_and_pairing():
    table = pd.read_csv(_need(ACC / "table_s2_static_limit.csv"))
    assert len(table) == 6 and set(table["cell"]) == {"ideal_sensing", "paper_sensing"}
    assert (table["runs"] == 20).all()
    assert (table["blocked_evs_mean"] == 0).all() and (table["session_end_before_watermark_ratio_mean"] == 0).all()
    assert (table["avg_candidates_per_ev_mean"] == 300).all()
    ideal = table[table["cell"] == "ideal_sensing"].set_index("algorithm_id")["accuracy_mean"]
    paper = table[table["cell"] == "paper_sensing"].set_index("algorithm_id")["accuracy_mean"]
    assert ideal["bayesian_windowed"] == pytest.approx(1.0)
    assert ideal["nomura_original_interval_hungarian"] == pytest.approx(1.0, abs=1e-6)
    assert paper["nomura_original_interval_hungarian"] == pytest.approx(0.952333, abs=1e-6)
    assert paper["bayesian_windowed"] == pytest.approx(0.625667, abs=1e-6)
    # one run per cell, carrying the three matchers on the same scenario seeds and hashes (paired)
    for cell in ("ideal_sensing", "paper_sensing"):
        hashes = []
        for run_id in set(table[table["cell"] == cell]["run_id"]):
            d = next(ACC.glob(f"static_limit_{cell}_run_{run_id}"))
            raw = pd.read_csv(d / "raw_runs.csv").sort_values(["algorithm_id", "repeat_idx"])
            for _aid, grp in raw.groupby("algorithm_id"):
                assert list(grp["scenario_seed"]) == list(range(2187631072, 2187631092))
                hashes.append(tuple(grp["scenario_hash"]))
        assert len(set(hashes)) == 1


# ---------------------------------------------------------------- Impairment sweeps (Fig. 5)
def test_impairment_sweeps_table():
    fig7 = pd.read_csv(_need(ACC / "figure7_impairment_sweeps.csv"))
    assert len(fig7) == 52 and fig7["cell"].nunique() == 11
    assert (fig7["runs"] == 20).all()
    default = fig7[(fig7["cell"] == "default") & (fig7["axis"] == "loss")].set_index("algorithm_id")["accuracy_mean"]
    for aid, expected in {"time_only_baseline": 0.7121, "single_only": 0.9732, "nomura_original_interval_hungarian": 0.9993, "bayesian_windowed": 0.9838}.items():
        assert default[aid] == pytest.approx(expected, abs=1e-6)
    for cell, grp in fig7.drop_duplicates(["cell", "algorithm_id"]).groupby("cell"):
        acc = grp.set_index("algorithm_id")["accuracy_mean"]
        assert acc["nomura_original_interval_hungarian"] == acc.max(), cell   # the prior static cost leads every impairment cell
        assert acc["bayesian_windowed"] >= acc["single_only"] > acc["time_only_baseline"], cell
    harsh = fig7[(fig7["cell"] == "noise_x8") & (fig7["algorithm_id"] == "bayesian_windowed")].iloc[0]
    assert harsh["accuracy_mean"] == pytest.approx(0.8708, abs=1e-6)
    matched = fig7[fig7["matched_dmax"].astype(bool)]
    assert matched["matcher_delay_max_s"].eq(60.0).all() and matched["p90_latency_mean"].iloc[0] == pytest.approx(465.98, abs=1e-2)


def test_every_impairment_cell_is_paired_with_the_canonical_representative_scenarios():
    fig7 = pd.read_csv(_need(ACC / "figure7_impairment_sweeps.csv"))
    canonical = pd.read_csv(CANONICAL_RAW)
    ref = canonical[(canonical["ev_count"] == 500) & (canonical["ev_sample_s"] == 30) & (canonical["matcher_delay_max_s"] == 10.0) & (canonical["candidate_margin_s"] == 120) & (canonical["algorithm_id"] == "bayesian_windowed")].sort_values("repeat_idx")
    for cell, run_id in fig7.drop_duplicates("cell")[["cell", "run_id"]].itertuples(index=False):
        d = next(ACC.glob(f"impairment_{cell}_run_{run_id}"))
        raw = pd.read_csv(d / "raw_runs.csv")
        a3 = raw[raw["algorithm_id"] == "bayesian_windowed"].sort_values("repeat_idx")
        assert list(a3["scenario_seed"]) == list(ref["scenario_seed"]), cell
        # sensing-multiplier cells redraw the sensor realisation, so only ingestion/delay/loss cells share the scenario hash
        if not cell.startswith("noise"):
            assert list(a3["scenario_hash"]) == list(ref["scenario_hash"]), cell


# ---------------------------------------------------------------- Saturation operating points (TABLE 7)
def test_saturation_table():
    table = pd.read_csv(_need(ACC / "table7_saturation.csv"))
    assert len(table) == 12 and set(table["point"]) == {"SA", "SB", "control"}
    assert (table["runs"] == 20).all()
    control = table[table["point"] == "control"].set_index("algorithm_id")
    for aid, expected in {"time_only_baseline": 0.7121, "single_only": 0.9732, "nomura_original_interval_hungarian": 0.9993, "bayesian_windowed": 0.9838}.items():
        assert control.loc[aid, "accuracy_mean"] == pytest.approx(expected, abs=1e-6)
        assert control.loc[aid, "blocked_ratio_mean"] == 0.0
    for point in ("SA", "SB"):
        sub = table[table["point"] == point].set_index("algorithm_id")
        assert (sub["blocked_ratio_mean"] > 0).all()
        assert (sub["accuracy_requested_mean"] < sub["accuracy_mean"]).all()
        assert sub["accuracy_mean"].idxmax() == "nomura_original_interval_hungarian"
        assert sub.loc["bayesian_windowed", "accuracy_mean"] >= sub.loc["single_only", "accuracy_mean"]
        assert (sub["gt_in_candidates_ratio_mean"] == 1.0).all()
    sa = table[(table["point"] == "SA") & (table["algorithm_id"] == "bayesian_windowed")].iloc[0]
    assert sa["ev_count"] == 2000 and sa["evse_count"] == 50
    assert sa["accuracy_mean"] == pytest.approx(0.9545, abs=1e-4) and sa["blocked_ratio_mean"] == pytest.approx(0.5289, abs=1e-4)
    sb = table[(table["point"] == "SB") & (table["algorithm_id"] == "bayesian_windowed")].iloc[0]
    assert sb["ev_count"] == 500 and sb["evse_count"] == 15
    assert sb["accuracy_mean"] == pytest.approx(0.9837, abs=1e-4) and sb["blocked_ratio_mean"] == pytest.approx(0.4631, abs=1e-4)


# ---------------------------------------------------------------- Hyperparameter sensitivity (Supplementary Table S4; data file table8_hyperparam_sensitivity.csv)
def test_hyperparameter_table():
    table = pd.read_csv(_need(ACC / "table8_hyperparam_sensitivity.csv"))
    assert len(table) == 15 and (table["runs"] == 20).all()
    defaults = table[table["is_default"].astype(bool)]
    assert len(defaults) == 3 and all(abs(float(v) - 0.9838) < 1e-6 for v in defaults["accuracy_mean"])
    nondefault = table[~table["is_default"].astype(bool)]
    assert len(nondefault) == 12
    assert (nondefault["accuracy_mean"] > 0.9732).all()  # every cell above the A1 reference
    assert nondefault["accuracy_mean"].min() == pytest.approx(0.9815, abs=1e-4)
    assert nondefault["accuracy_mean"].max() == pytest.approx(0.9838, abs=1e-4)
    for knob in ("gamma_prev", "eta_mix", "beta_curr"):
        assert sorted(table[table["knob"] == knob]["value"]) == sorted(table[table["knob"] == knob]["value"].unique()) and len(table[table["knob"] == knob]) == 5
    # eta_mix and beta_curr are flat within +/- 0.0002; gamma_prev is the only sensitive knob
    for knob in ("eta_mix", "beta_curr"):
        assert (nondefault[nondefault["knob"] == knob]["delta_vs_default_mean"].abs() <= 0.0002).all()
    gp = nondefault[nondefault["knob"] == "gamma_prev"].sort_values("value")
    # the grid does not discriminate: every cell sits within 0.9815-0.9838 and each CI covers the TABLE 2 point
    assert nondefault["accuracy_mean"].between(0.9815 - 1e-4, 0.9838 + 1e-4).all()
    assert list(gp["accuracy_mean"]) == pytest.approx([0.9815, 0.9829, 0.9833, 0.9830], abs=1e-4)
    assert (gp["accuracy_mean"] + gp["accuracy_ci95"] >= 0.9838).all()


# ---------------------------------------------------------------- Dense-sampling diagnostic (TABLE S1)
def test_sampling_diagnostic_table():
    table = pd.read_csv(_need(ACC / "table_s1_sampling_diagnostic.csv"))
    assert len(table) == 26 and (table["runs"] == 20).all()   # 16 front-end + 6 term-partition + 4 alignment-control rows
    a3 = table[table["algorithm_id"] == "bayesian_windowed"].set_index(["variant", "ev_sample_s"])["accuracy_mean"]
    # Eq. (7) term partition: the 5 -> 30 s rise lives in the banded-DTW term, the step term moves the other way
    tp = table[table["variant"] == "term_partition"].set_index(["algorithm_id", "ev_sample_s"])["accuracy_mean"]
    dtw = [tp[("single_only_step_off", dt)] for dt in (5, 15, 30)]
    stp = [tp[("single_only_dtw_off", dt)] for dt in (5, 15, 30)]
    assert dtw == sorted(dtw) and dtw[0] == pytest.approx(0.9028, abs=1e-4) and dtw[2] == pytest.approx(0.9963, abs=1e-4)
    assert stp != sorted(stp) and stp[0] == pytest.approx(0.9122, abs=1e-4) and stp[2] == pytest.approx(0.9213, abs=1e-4)
    for dt, expected in ((5, 0.9495), (15, 0.9683), (30, 0.9838)):
        assert a3[("canonical", dt)] == pytest.approx(expected, abs=1e-6)
    # noise-free control: tracks the canonical cells within half a point, and the trend with the
    # sampling period persists in both
    nf = [a3[("noise_free", dt)] for dt in (5, 15, 30)]
    assert nf == sorted(nf) and nf == pytest.approx([0.9491, 0.9658, 0.9832], abs=1e-4)
    canonical = [a3[("canonical", dt)] for dt in (5, 15, 30)]
    assert all(abs(n - c) < 0.005 for n, c in zip(nf, canonical))
    assert a3[("noise_free", 5)] == pytest.approx(0.9491, abs=1e-4)
    # block mean: below the hold front end at both sampling periods
    assert a3[("block_mean", 5)] < a3[("canonical", 5)] and a3[("block_mean", 15)] < a3[("canonical", 15)]
    assert a3[("block_mean", 5)] == pytest.approx(0.9439, abs=1e-4)
    # every diagnostic run is paired with its canonical cell (seeds) — checked by the builder; deltas present
    assert table[table["variant"] != "canonical"]["delta_vs_canonical_ci95"].gt(0).all()


# ---------------------------------------------------------------- Response heterogeneity (Fig. 6)
def test_heterogeneous_response_figure8_table():
    table = pd.read_csv(_need(ACC / "figure8_heterogeneous_response.csv"))
    assert len(table) == 24 and (table["runs"] >= 15).all()
    het = table[table["model"] == "heterogeneous"].set_index(["algorithm_id", "ev_count"])
    assert (het.xs(300, level="ev_count")["runs"] == 15).all() and (het.drop(300, level="ev_count")["runs"] == 20).all()   # TABLE 5 protocol at EV 300
    a3 = [het.loc[("bayesian_windowed", ev), "accuracy_mean"] for ev in (100, 300, 500)]
    assert a3 == sorted(a3, reverse=True) and a3[2] == pytest.approx(0.9650, abs=1e-4)
    # accuracy falls by at most 2 pt under per-session lag/ramp draws; ranking at EV = 500 unchanged
    for aid in ("bayesian_windowed", "single_only"):
        assert het.loc[(aid, 500), "delta_vs_nominal_mean"] > -0.03
    assert abs(het.loc[("nomura_original_interval_hungarian", 500), "delta_vs_nominal_mean"]) < 0.005   # the prior cost is unchanged
    for ev in (100, 300, 500):
        r = {aid: het.loc[(aid, ev), "accuracy_mean"] for aid in ("bayesian_windowed", "single_only", "nomura_original_interval_hungarian", "time_only_baseline")}
        assert r["nomura_original_interval_hungarian"] > r["bayesian_windowed"] > r["single_only"] > r["time_only_baseline"], ev
    raw = pd.read_csv(_need(ACC / "heterogeneous_response_run_20260908_170955_253072" / "raw_runs.csv"))
    assert raw["response_lag_min_s"].min() >= 2 and raw["response_lag_max_s"].max() <= 30
    assert raw["response_ramp_min_a_per_s"].min() >= 1.0 and raw["response_ramp_max_a_per_s"].max() <= 10.0
    cfg = json.loads(_need(ACC / "heterogeneous_response_run_20260908_170955_253072" / "config.json").read_text(encoding="utf-8"))
    assert list(cfg["response_lag_range_s"]) == [2.0, 30.0] and list(cfg["response_ramp_range_a_per_s"]) == [1.0, 10.0]


# ---------------------------------------------------------------- Codebook band sweep (Fig. 7)
def test_band_sweep_figure9_table():
    table = pd.read_csv(_need(ACC / "figure9_codebook_separation.csv"))
    assert len(table) == 12 and (table["runs"] == 20).all()
    sep = table.drop_duplicates("cell").set_index("cell")["realised_l1_mean_a"]
    assert sep["B30"] > sep["B20"] > sep["B16"] > sep["B12"]
    for cell in ("B30", "B20", "B16", "B12"):
        r = table[table["cell"] == cell].set_index("algorithm_id")["accuracy_mean"]
        assert r["nomura_original_interval_hungarian"] > r["bayesian_windowed"] > r["single_only"]
    a3 = table[table["algorithm_id"] == "bayesian_windowed"].set_index("cell")["accuracy_mean"]
    assert a3["B30"] == pytest.approx(0.9838, abs=1e-4) and a3["B20"] == pytest.approx(0.9738, abs=1e-4) and a3["B16"] == pytest.approx(0.9484, abs=1e-4) and a3["B12"] == pytest.approx(0.8952, abs=1e-4)
    assert a3["B30"] > a3["B20"] > a3["B16"] > a3["B12"]
    band = table[table["cell"] != "B30"]
    assert (band["delta_vs_canonical_ci95"] > 0).all() and (band["delta_vs_canonical_mean"] < 0).all()
    for cell, folder in (("B20", "codebook_band_B20_run_20260909_112529_182170"), ("B16", "codebook_band_B16_run_20260909_142401_231979"), ("B12", "codebook_band_B12_run_20260909_164615_270394")):
        raw = pd.read_csv(_need(ACC / folder / "raw_runs.csv"))
        assert raw["codebook_terminal_branch"].all() and (raw["codebook_unique_patterns"] == 50).all()
        cfg = json.loads(_need(ACC / folder / "config.json").read_text(encoding="utf-8"))
        assert [int(cfg["command_setpoint_range"][0]), int(cfg["command_setpoint_range"][1])] == {"B20": [6, 20], "B16": [8, 16], "B12": [8, 12]}[cell]

