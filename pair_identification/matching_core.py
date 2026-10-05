# -*- coding: utf-8 -*-
import os, math, random, time
import numpy as np
import pandas as pd

from case_study.diagnostic_terminal_logs import _log
from pair_identification.contracts import RuntimeConfig
from pair_identification.ingestion_reference import (
    DEFAULT_EV_INGEST_DELAY_JITTER_S,
    DEFAULT_EV_INGEST_DELAY_MAX_S,
    DEFAULT_EV_INGEST_DELAY_MEAN_S,
    DEFAULT_EV_INGEST_LOSS_PROB,
)
from pair_identification.runtime_event_loop import run_online_active_set

# ---- User-provided module (kept as-is) ----
from charging_current_generation.command_charger_current import get_command_data, generate_charger_current

# ----------------------------- Parameters (same defaults as original) -----------------------------
N_EVSE = 50
N_EV   = 50
TAU = 60            # seconds per MCCT step
N_STEPS = 6         # 0 + 5 steps
WINDOW = TAU * N_STEPS

CHARGER_SAMPLE = 5  # EVSE log resolution (sec)
EV_SAMPLE = 30      # EV log resolution (sec) - base env fixed
BAND_SEC = 15       # Sakoe-Chiba band width (sec) for DTW

# 12-hour simulation horizon (arrival allowed only up to 11h)
SIM_HOURS = 12
SIM_HORIZON_SEC = SIM_HOURS * 3600  # 43200
ARRIVAL_SPAN_MAX = 11 * 3600        # last arrival before 11 hours
BASE_INITIAL = "2024-05-23 15:59:13"  # base timestamp

# EV session duration constraints (realistic stay)
SESSION_MIN_SEC = 10 * 60   # 10 minutes
SESSION_MAX_SEC = 60 * 60   # 60 minutes

# Online matching policy:
# - fixed decision window per EV: [arrival, arrival + WINDOW]
# - finalize when watermark reaches window end (+ EV ingest delay bound)
# - candidate EVSE set is limited to console-known EVSE slots active at window end
# - global 1:1 assignment runs per matured-window group (not per-tick full recompute)
def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        v = int(default)
    if min_value is not None:
        v = max(int(min_value), int(v))
    return int(v)


def _env_float(name: str, default: float, min_value: float | None = None) -> float:
    try:
        v = float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        v = float(default)
    if min_value is not None:
        v = max(float(min_value), float(v))
    return float(v)


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


SWITCH_PENALTY = _env_float("EVLINK_SWITCH_PENALTY", 0.0, min_value=0.0)

EVALUATION_INTERVAL_S = _env_int("EVLINK_EVAL_INTERVAL_S", CHARGER_SAMPLE, min_value=1)
MIN_SHARED_POINTS = _env_int("EVLINK_MIN_SHARED_POINTS", 4, min_value=1)

EV_INGEST_DELAY_MEAN_S = _env_float("EVLINK_EV_INGEST_DELAY_MEAN_S", float(DEFAULT_EV_INGEST_DELAY_MEAN_S), min_value=0.0)
EV_INGEST_DELAY_JITTER_S = _env_float("EVLINK_EV_INGEST_DELAY_JITTER_S", float(DEFAULT_EV_INGEST_DELAY_JITTER_S), min_value=0.0)
EV_INGEST_DELAY_MAX_S = _env_float("EVLINK_EV_INGEST_DELAY_MAX_S", float(DEFAULT_EV_INGEST_DELAY_MAX_S), min_value=0.0)
# Watermark delay bound can be decoupled from true ingestion d_max for mis-spec studies.
EV_WATERMARK_DELAY_MAX_S = _env_float("EVLINK_WATERMARK_DELAY_MAX_S", float(EV_INGEST_DELAY_MAX_S), min_value=0.0)
EV_INGEST_LOSS_PROB = _env_float("EVLINK_EV_INGEST_LOSS_PROB", float(DEFAULT_EV_INGEST_LOSS_PROB), min_value=0.0)
EV_INGEST_SEED = _env_optional_int("EVLINK_EV_INGEST_SEED", default=None)

# EV telemetry realism knobs (kept moderate by default):
# - additional per-EV sensing delay over charger reference sampling
# - sensor gain/bias/noise distortion
# - non-oracle EV start estimate jitter used for decision-window anchoring
EV_SENSOR_EXTRA_DELAY_MAX_S = _env_int("EVLINK_EV_SENSOR_EXTRA_DELAY_MAX_S", 20, min_value=0)
EV_SENSOR_GAIN_STD = _env_float("EVLINK_EV_SENSOR_GAIN_STD", 0.03, min_value=0.0)
EV_SENSOR_BIAS_STD_A = _env_float("EVLINK_EV_SENSOR_BIAS_STD_A", 0.35, min_value=0.0)
EV_SENSOR_NOISE_STD_A = _env_float("EVLINK_EV_SENSOR_NOISE_STD_A", 0.20, min_value=0.0)
EV_START_EST_JITTER_S = _env_int("EVLINK_EV_START_EST_JITTER_S", 20, min_value=0)
CANDIDATE_TIME_MARGIN_S = _env_int("EVLINK_CANDIDATE_TIME_MARGIN_S", 120, min_value=0)
# EV-grid front end (dense-sampling diagnostic): "hold" = the manuscript's causal forward-fill of
# the Δt_EV samples onto the 5 s charger grid; "block_mean" = the Δt_EV samples inside each
# EV_GRID_BLOCK_S block are averaged before the forward-fill (identical to "hold" at Δt_EV = 30 s).
EV_GRID_AGGREGATION = (os.getenv("EVLINK_EV_GRID_AGGREGATION", "hold").strip().lower() or "hold")
EV_GRID_BLOCK_S = _env_int("EVLINK_EV_GRID_BLOCK_S", 30, min_value=1)
# Per-session response heterogeneity: when set, every admitted
# session draws its first-step lag ~ U(lo, hi) s (1 s resolution) and its ramp rate ~ U(lo, hi) A/s
# from a dedicated RNG stream (the scenario's own draws are untouched); None = the fitted model.
RESPONSE_LAG_RANGE_S: tuple[float, float] | None = None
# Codebook band sweep: command set-point band of the codebook generator. The
# canonical band is {6, ..., 30} A; narrowing it forces similar commands
# with the generator otherwise unchanged. LAST_CODEBOOK_INFO records whether the generator's
# relaxation ladder was exhausted (terminal random-unique branch) for the last call.
COMMAND_SETPOINT_RANGE: tuple[int, int] = (6, 30)
LAST_CODEBOOK_INFO: dict = {}
RESPONSE_RAMP_RANGE_A_PER_S: tuple[float, float] | None = None

# Refined matching cost weights:
# - Keep DTW as base term.
# - Increase step-signature term for MCCT symbol discrimination.
# - Mean-gap term is disabled by default (weight=0.0).
REFINED_DTW_WEIGHT = _env_float("EVLINK_REFINED_DTW_WEIGHT", 1.0, min_value=0.0)
REFINED_STEP_WEIGHT = _env_float("EVLINK_REFINED_STEP_WEIGHT", 0.40, min_value=0.0)
REFINED_MEAN_WEIGHT = _env_float("EVLINK_REFINED_MEAN_WEIGHT", 0.0, min_value=0.0)

# Bayesian windowed assignment knobs (used by compare add-on algorithm)
TIME_PRIOR_ALPHA = _env_float("EVLINK_TIME_PRIOR_ALPHA", 1.0, min_value=0.0)
CURRENT_LIKE_BETA = _env_float("EVLINK_CURRENT_LIKE_BETA", 1.0, min_value=0.0)
POSTERIOR_EPS = _env_float("EVLINK_POSTERIOR_EPS", 1e-9, min_value=1e-12)
TIME_PRIOR_MIX = _env_float("EVLINK_TIME_PRIOR_MIX", 0.35, min_value=0.0)
POSTERIOR_PREV_POWER = _env_float("EVLINK_POSTERIOR_PREV_POWER", 0.60, min_value=0.0)
CURRENT_COST_WEIGHT = _env_float("EVLINK_CURRENT_COST_WEIGHT", 0.15, min_value=0.0)

# ----------------------------- RNG Init -----------------------------
def init_seeds():
    env = os.getenv("EVLINK_SEED", "").strip()
    if env:
        try:
            s = int(env)
        except ValueError:
            s = (hash(env) & 0xffffffff)
    else:
        s = (int(time.time_ns()) ^ os.getpid() ^ int.from_bytes(os.urandom(4), "little")) & 0xffffffff
    random.seed(s)
    np.random.seed(s)
    return s

# ----------------------------- Utils -----------------------------
def ceil_1a(x: float) -> float:
    return math.ceil(x - 1e-9)

# ----------------------------- Pattern generation (kept as original) -----------------------------
def generate_command_patterns(
    num_patterns: int = 10,
    seed: int | None = None,
    internal_min_gap: int = 4,
    consec_min_gap: int = 10,
    min_pairwise_l1: int = 45,
    candidate_pool: int = 3000,
    setpoint_range: tuple[int, int] | None = None,
):
    import numpy as _np
    rnd = _np.random.RandomState(seed if seed is not None else _np.random.randint(0, 2**31-1))
    lo, hi = (6, 30) if setpoint_range is None else (int(setpoint_range[0]), int(setpoint_range[1]))
    if lo < 1 or hi <= lo or (hi - lo + 1) < 5:
        raise ValueError(f"setpoint_range must span at least five integer set-points >= 1 A (got {(lo, hi)})")
    allowed = _np.arange(lo, hi + 1, dtype=int)
    ladder_rungs = 0

    def _valid_internal(vals: _np.ndarray, gap_all: int) -> bool:
        v = _np.asarray(vals, dtype=int)
        if v.size <= 1:
            return True
        diffs = _np.abs(v[:, None] - v[None, :])
        mask = ~_np.eye(len(v), dtype=bool)
        return (diffs[mask].min() >= gap_all)

    def _l1(p, q) -> int:
        ap = _np.asarray(p[1:], dtype=int)
        aq = _np.asarray(q[1:], dtype=int)
        return int(_np.sum(_np.abs(ap - aq)))

    def _pat_features(p) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray, float]:
        m = _np.asarray(p, dtype=float)
        mu = float(m.mean())
        sd = float(m.std())
        if not _np.isfinite(sd) or sd < 1e-6:
            z = _np.zeros_like(m)
        else:
            z = (m - mu) / sd
        d = _np.diff(m)  # length 5
        return m, z, d, mu

    def _proxy_dist(p_feat, q_feat) -> float:
        mp, zp, dp, mup = p_feat
        mq, zq, dq, muq = q_feat
        zdist = float(_np.linalg.norm(zp - zq))
        l1_m = float(_np.mean(_np.abs(mp - mq)))
        l1_d = float(_np.mean(_np.abs(dp - dq)))
        mean_gap = abs(mup - muq)
        return zdist + 0.05 * l1_m + 0.02 * l1_d + 0.01 * mean_gap

    def _build_candidates(gap_all: int, consec_gap: int, pool_n: int):
        cands = []; tries = 0
        while len(cands) < pool_n and tries < pool_n * 10:
            start = int(rnd.choice(allowed))
            seq = [start]; rem = allowed.tolist(); rem.remove(start)
            for _k in range(4):
                prev = seq[-1]
                rem_arr = _np.array(rem, dtype=int)
                d = _np.abs(rem_arr - prev)
                cand_arr = rem_arr[d >= consec_gap]
                if cand_arr.size == 0:
                    maxd = d.max()
                    cand_arr = rem_arr[d >= maxd]
                filtered = []
                for v in cand_arr.tolist():
                    if _valid_internal(_np.array(seq + [v], dtype=int), gap_all):
                        filtered.append(v)
                if len(filtered) == 0:
                    filtered = cand_arr.tolist()
                cand_d = _np.abs(_np.array(filtered) - prev)
                # Stable sort: equal distances keep their set-point order on every platform
                # (the default introsort may reorder ties differently under SIMD dispatch).
                order = _np.argsort(-cand_d, kind="stable")
                topK = max(1, min(3, len(order)))
                pick = int(filtered[int(order[rnd.randint(0, topK)])])
                seq.append(pick); rem.remove(pick)
            if not _valid_internal(_np.array(seq, dtype=int), gap_all):
                tries += 1; continue
            cands.append([0] + seq); tries += 1
        return cands

    def _pick_farthest_proxy(cands: list, need: int, min_l1_thr: int):
        if not cands:
            return []
        feats = [_pat_features(c) for c in cands]
        spreads = [_np.std(f[0][1:]) for f in feats]
        seed_idx = int(_np.argmax(spreads))
        sel_idx = [seed_idx]; used = _np.zeros(len(cands), dtype=bool); used[seed_idx] = True
        min_d_l1 = _np.array([_l1(c, cands[seed_idx]) for c in cands], dtype=float)
        min_d_proxy = _np.array([_proxy_dist(f, feats[seed_idx]) for f in feats], dtype=float)
        while len(sel_idx) < need:
            eligible = (~used) & (min_d_l1 >= float(min_l1_thr))
            if not _np.any(eligible):
                break
            cand_scores = _np.where(eligible, min_d_proxy, -1.0)
            idx = int(_np.argmax(cand_scores))
            sel_idx.append(idx); used[idx] = True
            new_l1 = _np.array([_l1(c, cands[idx]) for c in cands], dtype=float)
            min_d_l1 = _np.minimum(min_d_l1, new_l1)
            new_proxy = _np.array([_proxy_dist(f, feats[idx]) for f in feats], dtype=float)
            min_d_proxy = _np.minimum(min_d_proxy, new_proxy)
        return [cands[i] for i in sel_idx]

    gap = int(internal_min_gap)
    thr = int(min_pairwise_l1)

    while True:
        cands = _build_candidates(gap_all=gap, consec_gap=consec_min_gap, pool_n=candidate_pool)
        selected = _pick_farthest_proxy(cands, num_patterns, thr)
        if len(selected) >= num_patterns:
            LAST_CODEBOOK_INFO.clear()
            LAST_CODEBOOK_INFO.update({"terminal_branch": False, "ladder_rungs": int(ladder_rungs), "setpoint_range": (int(lo), int(hi))})
            return [list(p) for p in selected[:num_patterns]]
        ladder_rungs += 1
        if thr > 30:
            thr -= 2
        elif consec_min_gap > 6:
            consec_min_gap -= 1
        elif gap > 2:
            gap -= 1
            thr = int(min_pairwise_l1)
        else:
            patterns = set()
            max_attempts = num_patterns * 50
            attempts = 0
            while len(patterns) < num_patterns and attempts < max_attempts:
                vals = rnd.choice(allowed, size=5, replace=False)
                patterns.add(tuple([0] + vals.tolist()))
                attempts += 1
            LAST_CODEBOOK_INFO.clear()
            LAST_CODEBOOK_INFO.update({"terminal_branch": True, "ladder_rungs": int(ladder_rungs), "setpoint_range": (int(lo), int(hi))})
            return [list(p) for p in patterns]


def codebook_statistics(patterns) -> dict:
    """Realised separation of a command codebook (codebook band sweep).

    Pairwise ℓ1 set-point distance over the five non-zero steps (the ``_l1`` definition of the
    generator), the minimum within-pattern set-point gap and the minimum consecutive-step change
    between successive non-zero set-points, and the number of distinct patterns.
    """
    import itertools as _it
    P = [[int(v) for v in p] for p in patterns]
    uniq = {tuple(p) for p in P}
    pair = [sum(abs(a - b) for a, b in zip(p[1:], q[1:])) for p, q in _it.combinations(P, 2)]
    internal = [min(abs(a - b) for a, b in _it.combinations(p[1:], 2)) for p in P if len(p) > 2]
    consec = [min(abs(p[k + 1] - p[k]) for k in range(1, len(p) - 1)) for p in P if len(p) > 2]
    return {
        "unique_patterns": int(len(uniq)),
        "pairwise_l1_min_a": float(min(pair)) if pair else float("nan"),
        "pairwise_l1_mean_a": float(sum(pair) / len(pair)) if pair else float("nan"),
        "internal_gap_min_a": float(min(internal)) if internal else float("nan"),
        "consec_step_min_a": float(min(consec)) if consec else float("nan"),
    }

# ----------------------------- Series helpers -----------------------------
def reindex_at_times(series: pd.Series, times: pd.DatetimeIndex, method="pad", fill_value=0.0):
    """Causal forward-fill onto ``times`` without backfilling from future samples."""
    if str(method).lower() not in {"pad", "ffill", "forward_fill"}:
        raise ValueError("reindex_at_times only supports causal forward-fill")
    df = series.to_frame("v")
    expanded = df.reindex(df.index.union(times)).sort_index()
    filled = expanded.ffill()
    out = filled.loc[times, "v"].fillna(float(fill_value)).to_numpy()
    return out

def znorm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    m = np.nanmean(a); s = np.nanstd(a)
    if not np.isfinite(s) or s < 1e-6:
        return np.zeros_like(a)
    return (a - m) / s

def cdtw_l2(x: np.ndarray, y: np.ndarray, band: int) -> float:
    n = min(len(x), len(y))
    x = x[:n]; y = y[:n]
    INF = 1e18
    dp = np.full((n+1, n+1), INF, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n+1):
        j0 = max(1, i-band); j1 = min(n, i+band)
        xi = x[i-1]
        for j in range(j0, j1+1):
            dy = xi - y[j-1]
            cost = dy*dy
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return float(dp[n, n])

def step_medians(vals: np.ndarray, tau=60, grid=5, n_steps=6) -> np.ndarray:
    seg_len = tau // grid  # e.g., 60/5 = 12
    meds = []
    for k in range(n_steps):
        s = k * seg_len
        e = min((k + 1) * seg_len, len(vals))
        seg = vals[s:e] if e > s else vals
        meds.append(np.nanmedian(seg))
    return np.array(meds, dtype=float)

# ----------------------------- Pattern distance helpers -----------------------------
def _pat_features(p: list[int]):
    m = np.asarray(p, dtype=float)
    mu = float(np.mean(m))
    sd = float(np.std(m))
    z = (m - mu) / sd if sd >= 1e-6 else np.zeros_like(m)
    d = np.diff(m)  # length 5
    return m, z, d, mu

def _proxy_dist(p_feat, q_feat) -> float:
    mp, zp, dp, mup = p_feat
    mq, zq, dq, muq = q_feat
    zdist = float(np.linalg.norm(zp - zq))
    l1_m = float(np.mean(np.abs(mp - mq)))
    l1_d = float(np.mean(np.abs(dp - dq)))
    mean_gap = abs(mup - muq)
    return zdist + 0.05 * l1_m + 0.02 * l1_d + 0.01 * mean_gap

def _pick_pattern_maximin(available_idxs: list[int], active_feats: list, feats_pool: list) -> int:
    if not available_idxs:
        raise RuntimeError("No available pattern indices")
    if not active_feats:
        return int(available_idxs[0])
    best_idx = int(available_idxs[0])
    best_score = -1.0
    for idx in available_idxs:
        f = feats_pool[idx]
        mind = min(_proxy_dist(f, af) for af in active_feats) if active_feats else 1e9
        if mind > best_score:
            best_score = mind
            best_idx = int(idx)
    return best_idx

# ----------------------------- Session/Slot series -----------------------------
def _session_series_1s(session: dict, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.Series:
    s0 = session['start_ts']; s1 = session['end_ts']; pat = session['pattern']
    a = max(t_start, s0); b = min(t_end, s1)
    if b < a:
        return pd.Series([], dtype=float)
    out = []
    k0 = int(np.floor((a - s0).total_seconds() / WINDOW))
    k1 = int(np.floor((b - s0).total_seconds() / WINDOW))
    for k in range(k0, k1 + 1):
        t_blk = s0 + pd.Timedelta(seconds=k * WINDOW)
        df_cmd = get_command_data(t_blk, pat, tau=TAU)
        df_chg = generate_charger_current(df_cmd, **(session.get('response') or {}))
        ser = df_chg['charger_current']
        out.append(ser)
    if not out:
        return pd.Series([], dtype=float)
    full = pd.concat(out).sort_index()
    if full.index.has_duplicates:
        full = full[~full.index.duplicated(keep="last")]
    full = full[(full.index >= a) & (full.index <= b)]
    return full

def _slot_series_1s_for_window(slot: int, t_start: pd.Timestamp, t_end: pd.Timestamp, sessions_by_slot: dict[int, list[dict]]) -> pd.Series:
    times = pd.date_range(start=t_start, end=t_end, freq="1s")
    if slot not in sessions_by_slot or len(sessions_by_slot[slot]) == 0:
        return pd.Series(np.zeros(len(times)), index=times, name="charger_current")
    base = pd.Series(np.zeros(len(times)), index=times, name="charger_current")
    for sess in sessions_by_slot[slot]:
        s = _session_series_1s(sess, t_start, t_end)
        if s.size == 0:
            continue
        s = s.reindex(s.index.union(base.index)).sort_index().loc[base.index]
        base = base.where(s.isna(), s)
    return base


def _pattern_to_int_list(pattern_obj) -> list[int]:
    if isinstance(pattern_obj, np.ndarray):
        return [int(x) for x in pattern_obj.tolist()]
    if isinstance(pattern_obj, (list, tuple)):
        return [int(x) for x in pattern_obj]
    return []


def slot_pattern_for_window(
    slot: int,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    sessions_by_slot: dict[int, list[dict]],
) -> list[int]:
    """
    Return the command pattern representative for a slot in [t_start, t_end].
    Priority:
    1) session active at t_start
    2) session with maximum overlap over the window
    """
    sessions = sessions_by_slot.get(slot, [])
    if not sessions:
        return []

    for sess in sessions:
        s0 = sess.get("start_ts")
        s1 = sess.get("end_ts")
        if s0 is None or s1 is None:
            continue
        if s0 <= t_start <= s1:
            return _pattern_to_int_list(sess.get("pattern"))

    best_overlap = -1.0
    best_pattern: list[int] = []
    for sess in sessions:
        s0 = sess.get("start_ts")
        s1 = sess.get("end_ts")
        if s0 is None or s1 is None:
            continue
        if s1 < t_start or s0 > t_end:
            continue
        ov_start = max(s0, t_start)
        ov_end = min(s1, t_end)
        overlap = float((ov_end - ov_start).total_seconds())
        if overlap > best_overlap:
            best_overlap = overlap
            best_pattern = _pattern_to_int_list(sess.get("pattern"))

    return best_pattern

# ----------------------------- Data Generation -----------------------------
def build_dataset_with_random_arrivals(
    base_initial_ts: str,
    n_evs=N_EV,
    n_evses=N_EVSE,
    tau=TAU,
    window=WINDOW,
    arrival_span_max=ARRIVAL_SPAN_MAX,
    arrival_schedule_sec=None,
    slot_fixed_patterns=None,
):
    n_evs = int(n_evs)
    n_evses = int(n_evses)
    window = int(window)
    arrival_span_max = int(arrival_span_max)

    if n_evs <= 0 or n_evses <= 0:
        sessions_empty = {slot: [] for slot in range(max(n_evses, 0))}
        return [], sessions_empty, {}, {}, {}

    rng = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))
    # Response-heterogeneity draws use their own stream so the scenario realisation (arrivals,
    # slots, codebook, sensor draws, noise) is identical with and without heterogeneity.
    heterogeneous = RESPONSE_LAG_RANGE_S is not None or RESPONSE_RAMP_RANGE_A_PER_S is not None
    rng_resp = np.random.default_rng(int(np.random.randint(0, 2**31 - 1)) ^ 0x5E5510A5) if heterogeneous else None
    response_draws: list[dict] = []
    base_ts = pd.Timestamp(base_initial_ts)
    t0_all = time.perf_counter()
    _log(f"[BUILD] base_ts={base_ts} | horizon=12h | last_arrival<=11h | slots={n_evses}")

    t0_pat = time.perf_counter()
    if slot_fixed_patterns is None:
        seed_pat = int(rng.integers(0, 2**31 - 1))
        _log(f"[BUILD] Generating MCCT patterns for {n_evses} EVSE slots ...")
        patterns_list = generate_command_patterns(
            num_patterns=n_evses,
            seed=seed_pat,
            internal_min_gap=4,
            consec_min_gap=8,
            min_pairwise_l1=40,
            candidate_pool=3000,
            setpoint_range=tuple(COMMAND_SETPOINT_RANGE),
        )
        codebook_diag = dict(codebook_statistics(patterns_list))
        codebook_diag["terminal_branch"] = bool(LAST_CODEBOOK_INFO.get("terminal_branch", False))
        codebook_diag["ladder_rungs"] = int(LAST_CODEBOOK_INFO.get("ladder_rungs", 0))
        codebook_diag["setpoint_lo_a"] = int(COMMAND_SETPOINT_RANGE[0]); codebook_diag["setpoint_hi_a"] = int(COMMAND_SETPOINT_RANGE[1])
        if codebook_diag["terminal_branch"]:
            _log(f"[BUILD] codebook: relaxation ladder exhausted after {codebook_diag['ladder_rungs']} rungs -> terminal random-unique branch (no pairwise-l1 floor, no gap constraint); realised min pairwise l1 {codebook_diag['pairwise_l1_min_a']:.0f} A, mean {codebook_diag['pairwise_l1_mean_a']:.1f} A, band {COMMAND_SETPOINT_RANGE[0]}-{COMMAND_SETPOINT_RANGE[1]} A")
        patterns = {i: patterns_list[i] for i in range(n_evses)}
        feats_pool = [_pat_features(patterns[i]) for i in range(n_evses)]
        fixed_mode = False
    else:
        arr_patterns = [list(p) for p in slot_fixed_patterns]
        if len(arr_patterns) < n_evses:
            raise ValueError("slot_fixed_patterns must have at least n_evses entries")
        codebook_diag = dict(codebook_statistics(arr_patterns[:n_evses]))
        codebook_diag["terminal_branch"] = False; codebook_diag["ladder_rungs"] = 0
        codebook_diag["setpoint_lo_a"] = int(min(v for p in arr_patterns[:n_evses] for v in p[1:])); codebook_diag["setpoint_hi_a"] = int(max(v for p in arr_patterns[:n_evses] for v in p[1:]))
        patterns = {i: arr_patterns[i] for i in range(n_evses)}
        feats_pool = []
        fixed_mode = True
        _log(f"[BUILD] Using fixed EVSE slot patterns for {n_evses} slots")
    _log(f"[BUILD] Patterns ready in {time.perf_counter() - t0_pat:.3f}s")

    t0_sched = time.perf_counter()
    if arrival_schedule_sec is None:
        arrivals_sec = np.sort(rng.integers(0, max(arrival_span_max, 0) + 1, size=n_evs).astype(int))
    else:
        arr = np.asarray(arrival_schedule_sec, dtype=int).reshape(-1)
        if arr.size == 0:
            raise ValueError("arrival_schedule_sec is empty")
        if arr.size < n_evs:
            raise ValueError("arrival_schedule_sec must contain at least n_evs entries")
        arr = np.clip(arr[:n_evs], 0, max(arrival_span_max, 0)).astype(int)
        arrivals_sec = np.sort(arr)
    durations_sec = rng.integers(SESSION_MIN_SEC, SESSION_MAX_SEC + 1, size=n_evs).astype(int)

    busy_until: dict[int, pd.Timestamp] = {slot: base_ts for slot in range(n_evses)}
    sessions_by_slot: dict[int, list[dict]] = {slot: [] for slot in range(n_evses)}

    import heapq
    active_heap: list[tuple[pd.Timestamp, int]] = []
    active_pat_idxs: set[int] = set()

    ev_meta = []
    ev_series: dict[int, pd.Series] = {}
    ev_idx_assigned = 0
    blocked_sessions: list[dict] = []

    for idx in range(n_evs):
        arrival_ts = base_ts + pd.Timedelta(seconds=int(arrivals_sec[idx]))
        session_len = int(durations_sec[idx])
        end_ts = arrival_ts + pd.Timedelta(seconds=session_len)
        if end_ts > base_ts + pd.Timedelta(seconds=SIM_HORIZON_SEC):
            end_ts = base_ts + pd.Timedelta(seconds=SIM_HORIZON_SEC)

        while active_heap and active_heap[0][0] <= arrival_ts:
            _ended_ts, pat_idx_done = heapq.heappop(active_heap)
            if pat_idx_done in active_pat_idxs:
                active_pat_idxs.remove(pat_idx_done)

        free_slots = [s for s, tfree in busy_until.items() if tfree <= arrival_ts]
        if not free_slots:
            active_remaining_s = [
                float((pd.Timestamp(tfree) - pd.Timestamp(arrival_ts)).total_seconds())
                for tfree in busy_until.values()
                if pd.Timestamp(tfree) > pd.Timestamp(arrival_ts)
            ]
            active_sessions = int(len(active_remaining_s))
            long_thresh_s = float(0.75 * SESSION_MAX_SEC)
            long_active = int(sum(1 for x in active_remaining_s if float(x) >= long_thresh_s))
            long_ratio = float(long_active / active_sessions) if active_sessions > 0 else 0.0
            overlap_ratio = float(active_sessions / max(1, int(n_evses)))
            blocked_reason_code = "arrival_slot_full"
            if long_ratio >= 0.6:
                blocked_reason_code = "long_session_occupancy_domination"
            elif overlap_ratio >= 0.95:
                blocked_reason_code = "session_overlap_concentration"
            blocked_sessions.append(
                {
                    "request_idx": int(idx),
                    "arrival_ts": pd.Timestamp(arrival_ts).tz_localize(None),
                    "active_sessions_at_arrival": int(active_sessions),
                    "requested_duration_s": int(session_len),
                    "blocked_reason_code": str(blocked_reason_code),
                    "capacity_shortage": True,
                    "overlap_ratio": float(overlap_ratio),
                    "long_session_dominance_ratio": float(long_ratio),
                }
            )
            continue

        slot = int(rng.choice(free_slots))
        if fixed_mode:
            chosen_pattern = patterns[slot]
            chosen_pat_idx = None
        else:
            available_idxs = [i for i in range(n_evses) if i not in active_pat_idxs]
            active_feats = [feats_pool[i] for i in active_pat_idxs]
            chosen_pat_idx = _pick_pattern_maximin(available_idxs, active_feats, feats_pool)
            chosen_pattern = patterns[chosen_pat_idx]
        busy_until[slot] = end_ts
        session_rec = dict(start_ts=arrival_ts, end_ts=end_ts, pattern=chosen_pattern)
        if heterogeneous:
            resp = {}
            if RESPONSE_LAG_RANGE_S is not None:
                lo, hi = float(RESPONSE_LAG_RANGE_S[0]), float(RESPONSE_LAG_RANGE_S[1])
                resp["first_step_lag_s"] = float(int(round(rng_resp.uniform(lo, hi))))
            if RESPONSE_RAMP_RANGE_A_PER_S is not None:
                lo, hi = float(RESPONSE_RAMP_RANGE_A_PER_S[0]), float(RESPONSE_RAMP_RANGE_A_PER_S[1])
                resp["ramp_rate_a_per_s"] = float(rng_resp.uniform(lo, hi))
            session_rec["response"] = resp
            response_draws.append(resp)
        sessions_by_slot[slot].append(session_rec)
        if not fixed_mode and chosen_pat_idx is not None:
            heapq.heappush(active_heap, (end_ts, chosen_pat_idx))
            active_pat_idxs.add(chosen_pat_idx)

        sensor_extra_delay_s = int(rng.integers(0, max(0, int(EV_SENSOR_EXTRA_DELAY_MAX_S)) + 1))
        sensor_sample_dt = pd.Timedelta(seconds=int(CHARGER_SAMPLE + sensor_extra_delay_s))
        sensor_gain = float(np.clip(rng.normal(1.0, float(EV_SENSOR_GAIN_STD)), 0.80, 1.20))
        sensor_bias = float(rng.normal(0.0, float(EV_SENSOR_BIAS_STD_A)))
        sensor_noise_std = float(EV_SENSOR_NOISE_STD_A)

        times_1s = pd.date_range(start=arrival_ts, periods=window + 1, freq="1s")
        # The EV meter reads the slot with its own delay, so the sample taken for the last
        # timestamp of the window falls up to CHARGER_SAMPLE + EV_SENSOR_EXTRA_DELAY_MAX_S
        # seconds past the window. Build the physical slot current over that longer segment so
        # every sample reads a real current: the session re-issues its pattern for its whole
        # duration, and _session_series_1s continues it into that range. Gain, bias, noise and
        # the 1 A ceiling are then applied exactly once per sample, tail included.
        sensor_seg_end = arrival_ts + pd.Timedelta(
            seconds=window + int(CHARGER_SAMPLE) + int(EV_SENSOR_EXTRA_DELAY_MAX_S)
        )
        slot_seg = _slot_series_1s_for_window(slot, arrival_ts, sensor_seg_end, sessions_by_slot)
        results = []
        for t in times_1s:
            t_sample = t + sensor_sample_dt
            v = float(slot_seg.at[t_sample])
            meas = v * sensor_gain + sensor_bias
            if sensor_noise_std > 0.0:
                meas += float(rng.normal(0.0, sensor_noise_std))
            cur = ceil_1a(max(0.0, meas))
            results.append({'timestamp': t, 'ev_current': cur})
        df_ev = pd.DataFrame(results).set_index('timestamp')
        ev_series[ev_idx_assigned] = df_ev['ev_current'].copy()

        active_obs = df_ev.index[df_ev['ev_current'] >= 1.0]
        if len(active_obs) > 0:
            start_est_ts = pd.Timestamp(active_obs[0])
        else:
            start_est_ts = pd.Timestamp(arrival_ts)
        if int(EV_START_EST_JITTER_S) > 0:
            j = int(rng.integers(-int(EV_START_EST_JITTER_S), int(EV_START_EST_JITTER_S) + 1))
            start_est_ts = start_est_ts + pd.Timedelta(seconds=j)
        lower = pd.Timestamp(arrival_ts) - pd.Timedelta(seconds=max(1, int(EV_SAMPLE)))
        upper = pd.Timestamp(arrival_ts) + pd.Timedelta(seconds=max(int(EV_SAMPLE), int(window // 2)))
        if start_est_ts < lower:
            start_est_ts = lower
        if start_est_ts > upper:
            start_est_ts = upper

        ev_meta.append(dict(
            token=f"EV#{ev_idx_assigned:02d}",
            ev_idx=ev_idx_assigned,
            evse_idx_gt=slot,
            arrival_ts=arrival_ts.tz_localize(None),
            start_est_ts=start_est_ts.tz_localize(None),
            decision_ts=(arrival_ts + pd.Timedelta(seconds=window)).tz_localize(None),
            session_end_ts=end_ts.tz_localize(None),
            session_len_s=session_len,
            pattern=chosen_pattern,
            ev_sensor_delay_s=int(sensor_extra_delay_s),
            ev_sensor_gain=float(sensor_gain),
            ev_sensor_bias_a=float(sensor_bias),
            ev_sensor_noise_std_a=float(sensor_noise_std),
            response_first_step_lag_s=float(session_rec.get("response", {}).get("first_step_lag_s", 15.0)),
            response_ramp_rate_a_per_s=(float(session_rec["response"]["ramp_rate_a_per_s"]) if session_rec.get("response", {}).get("ramp_rate_a_per_s") is not None else float("nan")),
        ))
        ev_idx_assigned += 1

    blocked = max(0, n_evs - len(ev_meta))
    _log(f"[BUILD] Scheduling complete: requested={n_evs}, admitted={len(ev_meta)}, blocked={blocked} in {time.perf_counter() - t0_sched:.3f}s")
    _log(f"[BUILD] TOTAL dataset build time: {time.perf_counter() - t0_all:.3f}s")
    dataset_diag = {
        "requested_evs": int(n_evs),
        "admitted_evs": int(len(ev_meta)),
        "blocked_evs": int(blocked),
        "blocked_ratio": float(blocked / n_evs) if int(n_evs) > 0 else float("nan"),
        "blocked_sessions": list(blocked_sessions),
        "codebook": dict(codebook_diag),
    }
    if heterogeneous and response_draws:
        lags = [r["first_step_lag_s"] for r in response_draws if "first_step_lag_s" in r]
        ramps = [r["ramp_rate_a_per_s"] for r in response_draws if "ramp_rate_a_per_s" in r]
        dataset_diag["response_heterogeneity"] = {
            "sessions": int(len(response_draws)),
            "lag_min_s": float(min(lags)) if lags else float("nan"), "lag_mean_s": float(np.mean(lags)) if lags else float("nan"), "lag_max_s": float(max(lags)) if lags else float("nan"),
            "ramp_min_a_per_s": float(min(ramps)) if ramps else float("nan"), "ramp_mean_a_per_s": float(np.mean(ramps)) if ramps else float("nan"), "ramp_max_a_per_s": float(max(ramps)) if ramps else float("nan"),
        }
    return ev_meta, sessions_by_slot, ev_series, patterns, dataset_diag

# ----------------------------- Matching (online / incremental) -----------------------------
def _build_ev_grid_for_match(ev_meta, ev_series, ev_idx, window=WINDOW):
    meta = ev_meta[ev_idx]
    t0 = pd.Timestamp(meta.get('start_est_ts', meta['arrival_ts']))
    n5 = window // CHARGER_SAMPLE + 1
    times_5s = pd.date_range(start=t0, periods=n5, freq=f"{CHARGER_SAMPLE}s")
    times_30s = pd.date_range(start=t0,
                               end=t0 + pd.Timedelta(seconds=window),
                               freq=f"{EV_SAMPLE}s")
    ev_1s = ev_series[ev_idx]
    vals_30 = reindex_at_times(ev_1s, times_30s, method="pad")
    s30 = pd.Series(vals_30, index=times_30s, name='ev30')
    if str(EV_GRID_AGGREGATION) == "block_mean":
        # Average the Δt_EV samples inside each EV_GRID_BLOCK_S block (blocks start at the EV
        # start estimate) and hold the block mean from the block start; at Δt_EV = block length
        # this is the plain forward-fill.
        block_s = max(1, int(EV_GRID_BLOCK_S))
        offsets = ((times_30s - t0).total_seconds() // block_s).astype(int)
        block_means = pd.Series(np.asarray(vals_30, dtype=float)).groupby(np.asarray(offsets)).mean()
        block_times = pd.DatetimeIndex([t0 + pd.Timedelta(seconds=int(k) * block_s) for k in block_means.index])
        s30 = pd.Series(block_means.to_numpy(dtype=float), index=block_times, name='ev30')
    ev_5s_vals = reindex_at_times(s30, times_5s, method="pad").astype(float)
    return times_5s, ev_5s_vals

def _slot_grid_for_times(slot_id: int, times_5s: pd.DatetimeIndex, sessions_by_slot: dict[int, list[dict]]) -> np.ndarray:
    t_start = times_5s[0]
    t_end = times_5s[-1]
    series_1s = _slot_series_1s_for_window(slot_id, t_start, t_end, sessions_by_slot)
    return reindex_at_times(series_1s, times_5s, method="pad").astype(float)


def _coarse_cost_from_grids(e_grid_raw: np.ndarray, s_grid_raw: np.ndarray) -> float:
    e_sig = step_medians(e_grid_raw, tau=TAU, grid=CHARGER_SAMPLE, n_steps=N_STEPS)
    s_sig = step_medians(s_grid_raw, tau=TAU, grid=CHARGER_SAMPLE, n_steps=N_STEPS)
    step_cost = float(np.nanmean(np.abs(e_sig - s_sig)))
    mean_gap = abs(float(np.nanmean(e_grid_raw) - np.nanmean(s_grid_raw)))
    return float(0.95 * step_cost + 0.05 * mean_gap)


def _refined_cost_from_grids(e_grid_raw: np.ndarray, s_grid_raw: np.ndarray) -> float:
    e_grid = znorm(e_grid_raw)
    s_grid = znorm(s_grid_raw)
    band = max(1, BAND_SEC // CHARGER_SAMPLE)
    d_dtw = cdtw_l2(s_grid, e_grid, band=band) / float(len(e_grid))
    e_sig = step_medians(e_grid_raw, tau=TAU, grid=CHARGER_SAMPLE, n_steps=N_STEPS)
    s_sig = step_medians(s_grid_raw, tau=TAU, grid=CHARGER_SAMPLE, n_steps=N_STEPS)
    sig_cost = float(np.nanmean(np.abs(e_sig - s_sig)))
    mean_gap = abs(float(np.nanmean(e_grid_raw) - np.nanmean(s_grid_raw)))
    return float(
        float(REFINED_DTW_WEIGHT) * d_dtw
        + float(REFINED_STEP_WEIGHT) * sig_cost
        + float(REFINED_MEAN_WEIGHT) * mean_gap
    )


# ----------------------------- Timeline-driven matching -----------------------------
def run_timeline_and_match(
    ev_meta,
    sessions_by_slot,
    ev_series,
    base_ts: pd.Timestamp,
    tau=TAU,
    window=WINDOW,
    diagnostics_out: dict | None = None,
    assignment_fn=None,
    coarse_cost_fn=None,
    refined_cost_fn=None,
    ingest_seed: int | None = None,
    trace_level: str = "full",
):
    _log("[SIM] Online window-watermark matching started")
    cfg = RuntimeConfig(
        switch_penalty=float(SWITCH_PENALTY),
        evaluation_interval_s=max(1, int(EVALUATION_INTERVAL_S)),
        min_shared_points=max(1, int(MIN_SHARED_POINTS)),
        ev_ingest_delay_mean_s=float(EV_INGEST_DELAY_MEAN_S),
        ev_ingest_delay_jitter_s=float(EV_INGEST_DELAY_JITTER_S),
        ev_ingest_delay_max_s=float(EV_INGEST_DELAY_MAX_S),
        watermark_delay_max_s=float(EV_WATERMARK_DELAY_MAX_S),
        ev_ingest_loss_prob=float(EV_INGEST_LOSS_PROB),
        ev_ingest_seed=EV_INGEST_SEED if ingest_seed is None else int(ingest_seed),
        event_driven_recompute=True,
        candidate_time_margin_s=max(0, int(CANDIDATE_TIME_MARGIN_S)),
        time_prior_alpha=float(TIME_PRIOR_ALPHA),
        current_like_beta=float(CURRENT_LIKE_BETA),
        posterior_eps=max(float(POSTERIOR_EPS), 1e-12),
        time_prior_mix=max(0.0, float(TIME_PRIOR_MIX)),
        posterior_prev_power=max(0.0, float(POSTERIOR_PREV_POWER)),
        current_cost_weight=float(np.clip(float(CURRENT_COST_WEIGHT), 0.0, 1.0)),
        ev_grid_aggregation=str(EV_GRID_AGGREGATION),
        ev_grid_block_s=int(EV_GRID_BLOCK_S),
        ev_measurement_period_s=max(1, int(EV_SAMPLE)),
    )
    coarse_fn = _coarse_cost_from_grids if coarse_cost_fn is None else coarse_cost_fn
    refined_fn = _refined_cost_from_grids if refined_cost_fn is None else refined_cost_fn
    return run_online_active_set(
        ev_meta=ev_meta,
        sessions_by_slot=sessions_by_slot,
        ev_series=ev_series,
        base_ts=base_ts,
        window=int(window),
        charger_sample_s=int(CHARGER_SAMPLE),
        cfg=cfg,
        build_ev_grid_for_match=_build_ev_grid_for_match,
        slot_grid_for_times=_slot_grid_for_times,
        coarse_cost_fn=coarse_fn,
        refined_cost_fn=refined_fn,
        logger=_log,
        diagnostics_out=diagnostics_out,
        assignment_fn=assignment_fn,
        trace_level=str(trace_level),
    )
