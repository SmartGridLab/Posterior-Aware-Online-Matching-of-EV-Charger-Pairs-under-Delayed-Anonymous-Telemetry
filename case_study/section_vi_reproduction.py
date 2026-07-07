# -*- coding: utf-8 -*-
"""Algorithm comparison runner for EV-Link MCCT matching."""

from __future__ import annotations

import json
import os
import time
import shutil
import re
import random
import hashlib
import subprocess
import tempfile
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_CACHE = Path(os.environ.get("EVLINK_CACHE_DIR", str(Path(tempfile.gettempdir()) / "evlink_cache")))
_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_LOCAL_CACHE))

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
try:
    import mpljapan  # type: ignore  # noqa: F401
except Exception:
    mpljapan = None
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "DejaVu Sans",
]
from matplotlib.legend import Legend
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as mpl_font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, MaxNLocator, ScalarFormatter
from matplotlib import transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pair_identification import matching_core as algo
from pair_identification.runtime_bayesian import BayesianWindowedAssigner
from pair_identification.ingestion_reference import ingestion_reference_payload
from case_study.site_simulation import (
    SimulationParameters,
    override_algo_config,
    override_ingestion_config,
    simulation_lock,
)


DEFAULT_EV_COUNTS = [100, 300, 500]
DEFAULT_REPEATS = 10
MAX_REPEATS = 50
DEFAULT_ERROR_MODE = "ci95"
VALID_ERROR_MODES = {"ci95", "std"}
FIXED_EVSE_COUNT = 50
DEFAULT_PATTERN_UNIQUE_COUNT = FIXED_EVSE_COUNT
MIN_PATTERN_UNIQUE_COUNT = 1
MAX_PATTERN_UNIQUE_COUNT = FIXED_EVSE_COUNT
DEFAULT_COMMAND_STEP_COUNT = 6
MIN_COMMAND_STEP_COUNT = 2
MAX_COMMAND_STEP_COUNT = 26
DEFAULT_TAU_VALUES = [60]
MIN_TAU_SECONDS = 10
MAX_TAU_SECONDS = 300
CHARGER_SAMPLE_SECONDS = 5
DEFAULT_CHARGER_SAMPLE_VALUES = [5]
DEFAULT_EV_SAMPLE_VALUES = [30]
DEFAULT_MATCHER_DELAY_MAX_VALUES_S = [float(algo.EV_INGEST_DELAY_MAX_S)]
DEFAULT_CANDIDATE_MARGIN_VALUES_S = [int(getattr(algo, "CANDIDATE_TIME_MARGIN_S", 120))]
CURRENT_GREEDY_TOPK = 6

DEFAULT_PATTERN_MODE = "current"
VALID_PATTERN_MODES = {"current", "legacy", "both"}
PATTERN_MODE_LABELS = {
    "current": "Optimized Pattern Generator",
    "legacy": "Legacy Random Unique (Existing)",
}

DEFAULT_PATTERN_ASSIGNMENT_MODE = "dynamic_session"
VALID_PATTERN_ASSIGNMENT_MODES = {"dynamic_session", "legacy_slot_fixed"}
PATTERN_ASSIGNMENT_MODE_LABELS = {
    "dynamic_session": "Dynamic Session Pattern Assignment",
    "legacy_slot_fixed": "Legacy Fixed EVSE Slot Pattern Assignment",
}

GENERATED_OUTPUT_DIR = REPO_ROOT / ".generated"
COMPARE_DIR = GENERATED_OUTPUT_DIR / "algorithm_compare"
AUTO_PRUNE_COMPARE_REPORTS = False
AUTO_PRUNE_KEEP_PER_MODE = 100


@dataclass(frozen=True)
class AlgoSpec:
    id: str
    label: str
    description: str
    is_current: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    run_type: str
    run_started: str
    base_seed: int
    repeats: int
    ev_counts: list[int]
    evse_count: int
    sim_hours: float
    session_min_minutes: float
    session_max_minutes: float
    pattern_mode: str
    pattern_modes: list[str]
    pattern_assignment_mode: str
    pattern_unique_count: int
    command_step_count: int
    tau_values: list[int]
    window_values_s: list[int]
    charger_sample_values_s: list[int]
    ev_sample_values_s: list[int]
    matcher_delay_max_values_s: list[float]
    candidate_margin_values_s: list[int]
    trace_level: str
    algorithm_ids: list[str]
    ingestion_enabled: bool
    ev_ingest_delay_mean_s: float
    ev_ingest_delay_jitter_s: float
    ev_ingest_delay_max_s: float
    ev_ingest_loss_prob: float


ALGO_SPECS: list[AlgoSpec] = [
    AlgoSpec(
        id="time_only_baseline",
        label="A0) Time Prior + Hungarian (Baseline)",
        description="Waveform features are not used. Uses only timing-based candidate costs with global 1:1 Hungarian.",
    ),
    AlgoSpec(
        id="single_only",
        label="A1) Refined Cost + Greedy 1:1",
        description="Refined cost evaluated per EV-slot pair, followed by greedy unique 1:1 assignment.",
        is_current=True,
    ),
    AlgoSpec(
        id="nomura_original_interval_hungarian",
        label="A2) Corr+DTW + Hungarian",
        description="Nomura original correlation + DTW scoring with dense global Hungarian as reference baseline.",
    ),
    AlgoSpec(
        id="bayesian_windowed",
        label="A3) Two-Stage Bayesian Posterior + Hungarian",
        description="Time prior + current likelihood + two-stage posterior accumulation with posterior-Hungarian assignment.",
    ),
]

ALGO_SPEC_BY_ID: dict[str, AlgoSpec] = {s.id: s for s in ALGO_SPECS}
EXTRA_ALGO_SPECS: list[AlgoSpec] = [
    AlgoSpec(
        id="single_only_step_off",
        label="A1 Ablation) Step Signature Off",
        description="A1 refined-cost variant with step-signature term disabled (DTW-only + mean-gap).",
        is_current=False,
    ),
    AlgoSpec(
        id="single_only_dtw_off",
        label="A1 Ablation) DTW Off",
        description="A1 refined-cost variant with DTW term disabled (step-signature + mean-gap).",
        is_current=False,
    ),
    AlgoSpec(
        id="single_only_all_off",
        label="A1 Ablation) All Waveform Terms Off",
        description="A1 variant with waveform terms removed; falls back to timing-only assignment.",
        is_current=False,
    ),
    AlgoSpec(
        id="nomura_original_interval_hungarian_dtw_off",
        label="A2 Ablation) DTW Off (Corr-only)",
        description="A2 variant using only correlation term (DTW contribution removed).",
        is_current=False,
    ),
    AlgoSpec(
        id="nomura_original_interval_hungarian_corr_off",
        label="A2 Ablation) Corr Off (DTW-only)",
        description="A2 variant using only DTW term (correlation contribution removed).",
        is_current=False,
    ),
    AlgoSpec(
        id="nomura_original_interval_hungarian_all_off",
        label="A2 Ablation) All Waveform Terms Off",
        description="A2 variant with waveform terms removed; falls back to timing-only assignment.",
        is_current=False,
    ),
    AlgoSpec(
        id="bayesian_windowed_memory_off",
        label="A3 Ablation) Posterior Memory Off",
        description="A3 variant with posterior memory term removed (prev-power=0).",
        is_current=False,
    ),
    AlgoSpec(
        id="bayesian_windowed_timeprior_off",
        label="A3 Ablation) Time Prior Off",
        description="A3 variant with timing-prior influence removed.",
        is_current=False,
    ),
    AlgoSpec(
        id="bayesian_windowed_all_off",
        label="A3 Ablation) Time Prior + Memory Off",
        description="A3 variant removing the time-prior and posterior-memory terms (current-likelihood only).",
        is_current=False,
    ),
]
EXTRA_ALGO_SPEC_BY_ID: dict[str, AlgoSpec] = {s.id: s for s in EXTRA_ALGO_SPECS}
ALL_ALGO_SPEC_BY_ID: dict[str, AlgoSpec] = {**ALGO_SPEC_BY_ID, **EXTRA_ALGO_SPEC_BY_ID}
DEFAULT_ALGORITHM_IDS = [
    "time_only_baseline",
    "single_only",
    "nomura_original_interval_hungarian",
    "bayesian_windowed",
]
AVAILABLE_ALGORITHM_IDS = [s.id for s in ALGO_SPECS]
PAPER_ABLATION_ALGORITHM_IDS = [
    "single_only",
    "single_only_dtw_off",
    "single_only_step_off",
    "single_only_all_off",
    "nomura_original_interval_hungarian",
    "nomura_original_interval_hungarian_dtw_off",
    "nomura_original_interval_hungarian_corr_off",
    "nomura_original_interval_hungarian_all_off",
    "bayesian_windowed",
    "bayesian_windowed_memory_off",
    "bayesian_windowed_timeprior_off",
    "bayesian_windowed_all_off",
]
PAPER_ABLATION_FAMILIES: dict[str, list[str]] = {
    "A1": ["single_only", "single_only_dtw_off", "single_only_step_off", "single_only_all_off"],
    "A2": [
        "nomura_original_interval_hungarian",
        "nomura_original_interval_hungarian_dtw_off",
        "nomura_original_interval_hungarian_corr_off",
        "nomura_original_interval_hungarian_all_off",
    ],
    "A3": [
        "bayesian_windowed",
            "bayesian_windowed_memory_off",
        "bayesian_windowed_timeprior_off",
        "bayesian_windowed_all_off",
    ],
}

_PAPER_ALGO_COLORS: dict[str, str] = {
    "time_only_baseline": "#808080",
    "single_only": "#1F77B4",
    "nomura_original_interval_hungarian": "#2CA02C",
    "bayesian_windowed": "#D62728",
    "single_only_step_off": "#0ea5e9",
    "single_only_dtw_off": "#0284c7",
    "single_only_all_off": "#0369a1",
    "nomura_original_interval_hungarian_dtw_off": "#14b8a6",
    "nomura_original_interval_hungarian_corr_off": "#10b981",
    "nomura_original_interval_hungarian_all_off": "#059669",
    "bayesian_windowed_memory_off": "#f97316",
    "bayesian_windowed_timeprior_off": "#dc2626",
    "bayesian_windowed_all_off": "#7f1d1d",
}

def _detect_ieee_serif_fonts() -> tuple[str, ...]:
    try:
        available = {str(f.name) for f in mpl_font_manager.fontManager.ttflist}
    except Exception:
        return ("DejaVu Serif",)
    preferred = ["Times New Roman", "Times", "DejaVu Serif"]
    picked = [f for f in preferred if f in available]
    if len(picked) == 0:
        return ("DejaVu Serif",)
    return tuple(picked)


_IEEE_SERIF_FALLBACKS: tuple[str, ...] = _detect_ieee_serif_fonts()
_PAPER_FIG_W = 7.16
_PAPER_FIG_H = 4.6
# IEEE Access typography split: outer axis labels vs inner elements.
_PAPER_MAIN_LABEL_SIZE = 10.0
_PAPER_INNER_TEXT_SIZE = 8.0
_PAPER_TITLE_SIZE = _PAPER_INNER_TEXT_SIZE
_PAPER_LABEL_SIZE = _PAPER_MAIN_LABEL_SIZE
_PAPER_TICK_SIZE = _PAPER_INNER_TEXT_SIZE
_PAPER_LEGEND_SIZE = 9.0
_PAPER_NOTE_SIZE = 8.5
_PAPER_LEGEND_MARKER_SIZE = 6.0

_LEGACY_ALGO_META: dict[str, dict[str, Any]] = {
    "legacy_corr_dtw_hungarian": {
        "label": "A2L) Corr+DTW + Hungarian Legacy Cost",
        "description": "Legacy Corr+DTW cost with dense global Hungarian (kept for historical report compatibility).",
        "is_current": False,
    },
}


def _algo_meta_for_id(aid: str) -> dict[str, Any]:
    key = str(aid).strip()
    spec = ALL_ALGO_SPEC_BY_ID.get(key)
    if spec is not None:
        return {
            "id": str(spec.id),
            "label": str(spec.label),
            "description": str(spec.description),
            "is_current": bool(spec.is_current),
        }
    legacy = _LEGACY_ALGO_META.get(key)
    if legacy is not None:
        return {
            "id": str(key),
            "label": str(legacy.get("label", key)),
            "description": str(legacy.get("description", "")),
            "is_current": bool(legacy.get("is_current", False)),
        }
    return {
        "id": str(key),
        "label": str(key),
        "description": "",
        "is_current": False,
    }


def _paper_algo_short_label(aid: str) -> str:
    key = str(aid).strip()
    fixed = {
        "time_only_baseline": "A0) Baseline",
        "single_only": "A1",
        "single_only_step_off": "A1-step off",
        "single_only_dtw_off": "A1-DTW off",
        "single_only_all_off": "A1-all off",
        "nomura_original_interval_hungarian": "A2",
        "nomura_original_interval_hungarian_dtw_off": "A2-DTW off",
        "nomura_original_interval_hungarian_corr_off": "A2-corr off",
        "nomura_original_interval_hungarian_all_off": "A2-all off",
        "bayesian_windowed": "A3",
        "bayesian_windowed_memory_off": "A3-memory off",
        "bayesian_windowed_timeprior_off": "A3-timeprior off",
        "bayesian_windowed_all_off": "A3-all off",
        "legacy_corr_dtw_hungarian": "A2L",
    }
    if key in fixed:
        return str(fixed[key])
    return str(key)


def _validate_repeats(value: int) -> int:
    n = int(value)
    if n <= 0:
        raise ValueError("repeats must be positive")
    if n > MAX_REPEATS:
        raise ValueError(f"repeats is too large (max {MAX_REPEATS})")
    return n


def _validate_error_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in VALID_ERROR_MODES:
        raise ValueError(f"error_mode must be one of {sorted(VALID_ERROR_MODES)}")
    return mode


def _validate_pattern_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in VALID_PATTERN_MODES:
        raise ValueError(f"pattern_mode must be one of {sorted(VALID_PATTERN_MODES)}")
    return mode


def _validate_pattern_assignment_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in VALID_PATTERN_ASSIGNMENT_MODES:
        raise ValueError(f"pattern_assignment_mode must be one of {sorted(VALID_PATTERN_ASSIGNMENT_MODES)}")
    return mode


def _validate_pattern_unique_count(value: int) -> int:
    n = int(value)
    if n < MIN_PATTERN_UNIQUE_COUNT:
        raise ValueError(f"pattern_unique_count must be >= {MIN_PATTERN_UNIQUE_COUNT}")
    if n > MAX_PATTERN_UNIQUE_COUNT:
        raise ValueError(f"pattern_unique_count must be <= {MAX_PATTERN_UNIQUE_COUNT}")
    return n


def _validate_command_step_count(value: int) -> int:
    n = int(value)
    if n < MIN_COMMAND_STEP_COUNT:
        raise ValueError(f"command_step_count must be >= {MIN_COMMAND_STEP_COUNT}")
    if n > MAX_COMMAND_STEP_COUNT:
        raise ValueError(f"command_step_count must be <= {MAX_COMMAND_STEP_COUNT}")
    return n


def _validate_tau_values(values: list[int] | None) -> list[int]:
    if values is None:
        values = list(DEFAULT_TAU_VALUES)
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("tau_values must be a non-empty array")

    out: list[int] = []
    for v in values:
        n = int(v)
        if n < MIN_TAU_SECONDS:
            raise ValueError(f"tau_values must be >= {MIN_TAU_SECONDS}")
        if n > MAX_TAU_SECONDS:
            raise ValueError(f"tau_values must be <= {MAX_TAU_SECONDS}")
        if n % CHARGER_SAMPLE_SECONDS != 0:
            raise ValueError(f"tau_values must be divisible by {CHARGER_SAMPLE_SECONDS}s")
        if n not in out:
            out.append(n)
    out.sort()
    return out


def _validate_sample_values(values: list[int] | None, *, field: str, default: list[int], min_value: int = 1, max_value: int = 3600) -> list[int]:
    if values is None:
        values = list(default)
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{field} must be a non-empty array")
    out: list[int] = []
    for v in values:
        n = int(v)
        if n < int(min_value):
            raise ValueError(f"{field} must be >= {min_value}")
        if n > int(max_value):
            raise ValueError(f"{field} must be <= {max_value}")
        if n not in out:
            out.append(int(n))
    out.sort()
    return out


def _validate_float_values(values: list[float] | None, *, field: str, default: list[float], min_value: float = 0.0, max_value: float | None = None) -> list[float]:
    if values is None:
        values = list(default)
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{field} must be a non-empty array")
    out: list[float] = []
    for v in values:
        x = float(v)
        if not np.isfinite(x):
            raise ValueError(f"{field} must contain finite numbers")
        if x < float(min_value):
            raise ValueError(f"{field} must be >= {min_value}")
        if max_value is not None and x > float(max_value):
            raise ValueError(f"{field} must be <= {max_value}")
        xr = float(round(x, 9))
        if xr not in out:
            out.append(xr)
    out.sort()
    return out


def _validate_candidate_margin_values(values: list[int] | None) -> list[int]:
    return _validate_sample_values(
        values,
        field="candidate_margin_values_s",
        default=list(DEFAULT_CANDIDATE_MARGIN_VALUES_S),
        min_value=0,
        max_value=3600,
    )


def _validate_ingestion_inputs(
    *,
    ingestion_enabled: bool,
    ev_ingest_delay_mean_s: float,
    ev_ingest_delay_jitter_s: float,
    ev_ingest_delay_max_s: float,
    ev_ingest_loss_prob: float,
) -> dict[str, float | bool]:
    enabled = bool(ingestion_enabled)
    mean = float(ev_ingest_delay_mean_s)
    jitter = float(ev_ingest_delay_jitter_s)
    max_delay = float(ev_ingest_delay_max_s)
    loss = float(ev_ingest_loss_prob)

    values = {
        "ev_ingest_delay_mean_s": mean,
        "ev_ingest_delay_jitter_s": jitter,
        "ev_ingest_delay_max_s": max_delay,
        "ev_ingest_loss_prob": loss,
    }
    for key, val in values.items():
        if float(val) < 0.0:
            raise ValueError(f"{key} must be >= 0")
    if float(loss) > 1.0:
        raise ValueError("ev_ingest_loss_prob must be <= 1")

    return {
        "ingestion_enabled": enabled,
        "ev_ingest_delay_mean_s": float(mean),
        "ev_ingest_delay_jitter_s": float(jitter),
        "ev_ingest_delay_max_s": float(max_delay),
        "ev_ingest_loss_prob": float(loss),
    }


def list_available_algorithms() -> list[dict[str, Any]]:
    return [
        {
            "id": s.id,
            "label": s.label,
            "description": s.description,
            "is_current": bool(s.is_current),
        }
        for s in ALGO_SPECS
    ]


def list_paper_ablation_algorithms() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family, ids in PAPER_ABLATION_FAMILIES.items():
        for aid in ids:
            meta = _algo_meta_for_id(aid)
            out.append(
                {
                    "id": str(meta["id"]),
                    "label": str(meta["label"]),
                    "description": str(meta["description"]),
                    "family": str(family),
                }
            )
    return out


def _validate_algorithm_ids(values: list[str] | None) -> list[AlgoSpec]:
    if values is None:
        values = list(DEFAULT_ALGORITHM_IDS)
    if not isinstance(values, list):
        raise ValueError("algorithm_ids must be a list")
    if len(values) == 0:
        raise ValueError("Select at least one algorithm")

    out: list[AlgoSpec] = []
    seen: set[str] = set()
    legacy_alias = {
        "proposed_full": "single_only",
    }
    for raw in values:
        aid = str(raw).strip()
        aid = str(legacy_alias.get(aid, aid))
        if aid == "" or aid in seen:
            continue
        spec = ALL_ALGO_SPEC_BY_ID.get(aid)
        if spec is None:
            raise ValueError(f"Unsupported algorithm id: {aid}")
        out.append(spec)
        seen.add(aid)

    if len(out) == 0:
        raise ValueError("Select at least one valid algorithm")
    return out


def _resolve_pattern_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["current", "legacy"]
    return [mode]


def _validate_ev_counts(values: list[int] | None) -> list[int]:
    if values is None:
        values = list(DEFAULT_EV_COUNTS)
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError("ev_counts must be a non-empty array")
    out: list[int] = []
    for v in values:
        n = int(v)
        if n <= 0:
            raise ValueError("ev_counts must contain positive integers")
        if n > 2000:
            raise ValueError("ev_count is too large (max 2000)")
        if n not in out:
            out.append(n)
    out.sort()
    return out


def _validate_run_id(run_id: str) -> str:
    rid = str(run_id).strip()
    if rid == "":
        raise ValueError("run_id is empty")
    if "/" in rid or "\\" in rid or ".." in rid:
        raise ValueError("invalid run_id")
    return rid


def _to_rel_plot(path: Path, plots_base: Path) -> str:
    p_resolved = path.resolve()
    base_resolved = plots_base.resolve()

    # Normal case: output file physically under the generated-output root.
    try:
        return str(p_resolved.relative_to(base_resolved))
    except Exception:
        pass

    # Symlink-aware fallback:
    # If the output root contains symlink children (for example, to an external
    # volume),
    # map the resolved absolute path back to "symlink_name/..." relative form.
    try:
        for child in plots_base.iterdir():
            try:
                child_target = child.resolve()
            except Exception:
                continue
            try:
                rem = p_resolved.relative_to(child_target)
            except Exception:
                continue
            return str(Path(child.name) / rem)
    except Exception:
        pass

    raise ValueError(f"{p_resolved} is not in the subpath of {base_resolved}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _mean_std_ci95(values: list[float]) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if len(values) <= 1:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    ci95 = float(1.96 * std / np.sqrt(len(arr)))
    return mean, std, ci95


@contextmanager
def _silence_algo_logs():
    old_log = algo._log
    algo._log = lambda _m: None
    try:
        yield
    finally:
        algo._log = old_log


def _legacy_random_generate_command_patterns(
    num_patterns: int = 10,
    seed: int | None = 42,
    command_step_count: int = DEFAULT_COMMAND_STEP_COUNT,
    **_kwargs: Any,
) -> list[list[int]]:
    """Legacy method generalized for selected command-step length."""
    if seed is None:
        seed = 42
    rnd = np.random.RandomState(int(seed))
    command_step_count = int(command_step_count)
    nonzero_steps = command_step_count - 1
    if nonzero_steps <= 0:
        raise ValueError("command_step_count must be >= 2")
    if nonzero_steps > 25:
        raise ValueError("command_step_count is too large for unique 6..30 current values")

    patterns: set[tuple[int, ...]] = set()
    max_attempts = max(int(num_patterns) * 200, 200)
    attempts = 0
    while len(patterns) < int(num_patterns) and attempts < max_attempts:
        pattern = [0] + rnd.choice(range(6, 31), size=nonzero_steps, replace=False).tolist()
        patterns.add(tuple(int(x) for x in pattern))
        attempts += 1

    if len(patterns) < int(num_patterns):
        raise ValueError("Could not generate enough unique legacy patterns")
    return [list(p) for p in sorted(patterns)]


def _legacy_interval_generate_command_patterns(
    num_patterns: int = 10,
    seed: int | None = 42,
    command_step_count: int = DEFAULT_COMMAND_STEP_COUNT,
) -> list[list[int]]:
    """Legacy generator used in Correlation-DTW_EV_interval.py style."""
    if seed is None:
        seed = 42
    rnd = np.random.RandomState(int(seed))
    steps = int(command_step_count)
    nonzero_steps = steps - 1
    if nonzero_steps <= 0:
        raise ValueError("command_step_count must be >= 2")
    if nonzero_steps > 25:
        raise ValueError("command_step_count is too large for unique 6..30 current values")

    patterns: set[tuple[int, ...]] = set()
    max_attempts = int(num_patterns) * 10
    attempts = 0
    while len(patterns) < int(num_patterns) and attempts < max_attempts:
        pattern = [0] + rnd.choice(range(6, 31), size=nonzero_steps, replace=False).tolist()
        patterns.add(tuple(int(x) for x in pattern))
        attempts += 1
    if len(patterns) < int(num_patterns):
        raise ValueError("Could not generate enough unique legacy patterns (interval style)")
    # Keep original style (set iteration order) intentionally.
    return [list(p) for p in patterns]


def _max_unique_sequence_count_for_steps(command_step_count: int) -> int:
    nonzero_steps = int(command_step_count) - 1
    if nonzero_steps <= 0:
        return 0
    if nonzero_steps == 1:
        return 25
    # With >=2 non-zero values, permutations exceed EVSE count; cap at EVSE for runtime stability.
    return FIXED_EVSE_COUNT


def _random_pattern_for_steps(
    command_step_count: int,
    rng: np.random.RandomState,
) -> list[int]:
    nonzero_steps = int(command_step_count) - 1
    if nonzero_steps <= 0:
        raise ValueError("command_step_count must be >= 2")
    if nonzero_steps > 25:
        raise ValueError("command_step_count is too large for unique 6..30 current values")
    vals = rng.choice(np.arange(6, 31, dtype=int), size=nonzero_steps, replace=False)
    return [0] + [int(x) for x in vals.tolist()]


def _reshape_single_pattern_to_steps(
    pattern: list[int] | tuple[int, ...] | np.ndarray,
    command_step_count: int,
    rng: np.random.RandomState,
) -> list[int]:
    target = int(command_step_count)
    nonzero_target = target - 1
    if nonzero_target <= 0:
        raise ValueError("command_step_count must be >= 2")
    if nonzero_target > 25:
        raise ValueError("command_step_count is too large for unique 6..30 current values")

    arr = list(pattern.tolist()) if isinstance(pattern, np.ndarray) else list(pattern)
    base: list[int] = []
    for x in arr[1:]:
        xi = int(x)
        if 6 <= xi <= 30 and xi not in base:
            base.append(xi)
        if len(base) >= nonzero_target:
            break

    if len(base) < nonzero_target:
        need = nonzero_target - len(base)
        allowed = [v for v in range(6, 31) if v not in base]
        add = rng.choice(np.asarray(allowed, dtype=int), size=need, replace=False).tolist()
        base.extend(int(v) for v in add)

    return [0] + [int(v) for v in base[:nonzero_target]]


def _reshape_patterns_to_steps(
    base_patterns: list[list[int]],
    target_unique_count: int,
    command_step_count: int,
    seed: int | None,
) -> list[list[int]]:
    rng = np.random.RandomState(42 if seed is None else int(seed))
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    for pattern in base_patterns:
        reshaped = _reshape_single_pattern_to_steps(pattern, command_step_count, rng)
        key = tuple(reshaped)
        if key in seen:
            continue
        seen.add(key)
        out.append(reshaped)
        if len(out) >= target_unique_count:
            return out

    max_attempts = max(target_unique_count * 200, 400)
    attempts = 0
    while len(out) < target_unique_count and attempts < max_attempts:
        attempts += 1
        reshaped = _random_pattern_for_steps(command_step_count, rng)
        key = tuple(reshaped)
        if key in seen:
            continue
        seen.add(key)
        out.append(reshaped)

    if len(out) < target_unique_count:
        raise ValueError("Could not generate enough patterns for selected command_step_count")
    return out


def _expand_pattern_pool(
    base_patterns: list[list[int]],
    target_count: int,
    seed: int | None,
) -> list[list[int]]:
    if target_count <= 0:
        return []
    if len(base_patterns) >= target_count:
        return [list(p) for p in base_patterns[:target_count]]
    if len(base_patterns) == 0:
        raise ValueError("base_patterns is empty")

    rng = np.random.RandomState(42 if seed is None else int(seed))
    out: list[list[int]] = []
    idxs = np.arange(len(base_patterns), dtype=int)

    while len(out) < target_count:
        rng.shuffle(idxs)
        for bi in idxs:
            out.append(list(base_patterns[int(bi)]))
            if len(out) >= target_count:
                break
    return out


def _build_pattern_generator(
    base_fn: Callable[..., list[list[int]]],
    pattern_unique_count: int,
    command_step_count: int,
) -> Callable[..., list[list[int]]]:
    unique_count = max(1, int(pattern_unique_count))
    step_count = int(command_step_count)

    def _wrapped(num_patterns: int = 10, seed: int | None = None, **kwargs: Any) -> list[list[int]]:
        target_count = int(num_patterns)
        if target_count <= 0:
            return []
        max_unique = _max_unique_sequence_count_for_steps(step_count)
        requested_unique = min(unique_count, target_count, max_unique)
        if requested_unique <= 0:
            raise ValueError("No feasible unique pattern for selected command_step_count")

        try:
            base_patterns = base_fn(
                num_patterns=requested_unique,
                seed=seed,
                command_step_count=step_count,
                **kwargs,
            )
        except TypeError:
            base_patterns = base_fn(num_patterns=requested_unique, seed=seed, **kwargs)
        base_patterns = [list(p) for p in base_patterns]
        if len(base_patterns) == 0:
            raise ValueError("pattern generator returned empty result")

        reshaped_unique = _reshape_patterns_to_steps(
            base_patterns=base_patterns,
            target_unique_count=requested_unique,
            command_step_count=step_count,
            seed=seed,
        )
        return _expand_pattern_pool(reshaped_unique, target_count, seed)

    return _wrapped


@contextmanager
def _override_pattern_generator(pattern_mode: str, pattern_unique_count: int, command_step_count: int):
    old_fn = algo.generate_command_patterns
    base_fn: Callable[..., list[list[int]]]
    if pattern_mode == "legacy":
        base_fn = _legacy_random_generate_command_patterns
    else:
        base_fn = old_fn
    algo.generate_command_patterns = _build_pattern_generator(base_fn, pattern_unique_count, command_step_count)
    try:
        yield
    finally:
        algo.generate_command_patterns = old_fn


@contextmanager
def _override_candidate_margin(candidate_margin_s: int):
    old_margin = int(getattr(algo, "CANDIDATE_TIME_MARGIN_S", 0))
    algo.CANDIDATE_TIME_MARGIN_S = int(max(0, int(candidate_margin_s)))
    try:
        yield
    finally:
        algo.CANDIDATE_TIME_MARGIN_S = int(old_margin)


@contextmanager
def _override_mcct_timing(tau_seconds: int, command_step_count: int):
    old_tau = int(algo.TAU)
    old_steps = int(algo.N_STEPS)
    old_window = int(algo.WINDOW)
    new_tau = int(tau_seconds)
    new_steps = int(command_step_count)
    new_window = int(new_tau * new_steps)
    algo.TAU = new_tau
    algo.N_STEPS = new_steps
    algo.WINDOW = new_window
    try:
        yield
    finally:
        algo.TAU = old_tau
        algo.N_STEPS = old_steps
        algo.WINDOW = old_window


def _nomura_calculate_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Original Correlation-DTW_EV_interval.py DTW with path-length normalization."""
    x = np.asarray(seq1, dtype=np.float32)
    y = np.asarray(seq2, dtype=np.float32)
    n = int(len(x))
    m = int(len(y))
    if n == 0 or m == 0:
        return float("inf")

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        xi = float(x[i - 1])
        for j in range(1, m + 1):
            cost = abs(xi - float(y[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    total_cost = float(dtw[n, m])

    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            if dtw[i - 1, j - 1] == min_val:
                i -= 1
                j -= 1
            elif dtw[i - 1, j] == min_val:
                i -= 1
            else:
                j -= 1

    if path_len <= 0:
        return float("inf")
    return float(total_cost / float(path_len))


def _shared_observed_arrays(
    e_obs: np.ndarray,
    s_obs: np.ndarray,
    min_shared_points: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mask = np.isfinite(e_obs) & np.isfinite(s_obs)
    if int(np.sum(mask)) < int(min_shared_points):
        return None, None
    return np.asarray(e_obs[mask], dtype=np.float64), np.asarray(s_obs[mask], dtype=np.float64)


def _refined_term_decomposition(
    e_obs: np.ndarray,
    s_obs: np.ndarray,
    *,
    min_shared_points: int,
) -> dict[str, float]:
    x, y = _shared_observed_arrays(
        np.asarray(e_obs, dtype=np.float64),
        np.asarray(s_obs, dtype=np.float64),
        int(min_shared_points),
    )
    if x is None or y is None:
        return {
            "dtw_term": float("nan"),
            "corr_term": float("nan"),
            "step_signature_term": float("nan"),
        }

    xz = algo.znorm(np.asarray(x, dtype=np.float64))
    yz = algo.znorm(np.asarray(y, dtype=np.float64))
    band = max(1, int(getattr(algo, "BAND_SEC", 30)) // max(1, int(getattr(algo, "CHARGER_SAMPLE", 5))))
    dtw_term = float(algo.cdtw_l2(yz, xz, band=band) / float(max(1, len(xz))))
    e_sig = algo.step_medians(
        np.asarray(x, dtype=np.float64),
        tau=int(getattr(algo, "TAU", 60)),
        grid=int(getattr(algo, "CHARGER_SAMPLE", 5)),
        n_steps=int(getattr(algo, "N_STEPS", 6)),
    )
    s_sig = algo.step_medians(
        np.asarray(y, dtype=np.float64),
        tau=int(getattr(algo, "TAU", 60)),
        grid=int(getattr(algo, "CHARGER_SAMPLE", 5)),
        n_steps=int(getattr(algo, "N_STEPS", 6)),
    )
    step_term = float(np.nanmean(np.abs(e_sig - s_sig)))
    corr_term = float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if len(x) > 1 and len(y) > 1 and sx > 1e-8 and sy > 1e-8:
        cv = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(cv):
            corr_term = float(cv)
    return {
        "dtw_term": float(dtw_term),
        "corr_term": float(corr_term),
        "step_signature_term": float(step_term),
    }


def _hungarian_with_inf_penalty(
    sparse_cost: np.ndarray,
    switch_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.asarray(sparse_cost, dtype=np.float64).copy()
    finite_mask = np.isfinite(dense)
    if not np.any(finite_mask):
        dense[:, :] = 1e9
    else:
        max_finite = float(np.max(dense[finite_mask]))
        penalty = max_finite + max(1.0, float(switch_penalty) * 100.0)
        dense[~finite_mask] = penalty
    row_ind, col_ind = linear_sum_assignment(dense)
    return dense, row_ind.astype(int), col_ind.astype(int)


def _greedy_unique_assign_from_row_costs(
    row_costs: dict[int, dict[int, float]],
    candidate_slots_by_row: list[list[int]],
    slots_list: list[int],
) -> dict[int, tuple[int, float]]:
    n_rows = int(len(candidate_slots_by_row))
    slot_pool = set(int(s) for s in slots_list)

    ranked_rows: list[tuple[float, float, int]] = []
    ranked_slots_by_row: dict[int, list[tuple[int, float]]] = {}
    for ri in range(n_rows):
        cands = [int(s) for s in candidate_slots_by_row[ri] if int(s) in slot_pool]
        if len(cands) == 0:
            cands = [int(s) for s in slots_list]

        row_map = row_costs.get(int(ri), {})
        scored: list[tuple[int, float]] = []
        for slot in cands:
            val = float(row_map.get(int(slot), float("inf")))
            if np.isfinite(val):
                scored.append((int(slot), val))
        if len(scored) == 0:
            scored = [(int(slot), float("inf")) for slot in cands]
        scored.sort(key=lambda x: (x[1], x[0]))
        ranked_slots_by_row[int(ri)] = scored

        best = float(scored[0][1])
        second = float(scored[1][1]) if len(scored) >= 2 else (best + 1e6)
        margin = float(second - best) if np.isfinite(best) and np.isfinite(second) else 0.0
        ranked_rows.append((-margin, best, int(ri)))

    ranked_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    available = set(int(s) for s in slots_list)
    assigned: dict[int, tuple[int, float]] = {}

    for _neg_margin, _best, ri in ranked_rows:
        picked: tuple[int, float] | None = None
        for slot, cost in ranked_slots_by_row.get(int(ri), []):
            if int(slot) in available:
                picked = (int(slot), float(cost))
                break
        if picked is None:
            continue
        slot, cost = picked
        assigned[int(ri)] = (int(slot), float(cost))
        available.discard(int(slot))

    for ri in range(n_rows):
        if int(ri) in assigned:
            continue
        choices = ranked_slots_by_row.get(int(ri), [])
        if len(choices) == 0:
            assigned[int(ri)] = (-1, float("inf"))
            continue
        picked: tuple[int, float] | None = None
        for slot, cost in choices:
            if int(slot) in available:
                picked = (int(slot), float(cost))
                break
        if picked is None:
            assigned[int(ri)] = (-1, float("inf"))
            continue
        slot, cost = picked
        assigned[int(ri)] = (int(slot), float(cost))
        available.discard(int(slot))

    return assigned


def _online_assign_single_only(
    *,
    ev_rows_pref: list[np.ndarray],
    row_slot_pref: list[dict[int, np.ndarray]],
    slots_list: list[int],
    candidate_slots_by_row: list[list[int]],
    time_costs_by_row: list[dict[int, float]] | None = None,
    previous_slot_by_row: dict[int, int],
    cfg: Any,
    refined_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    **_kwargs: Any,
) -> tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]:
    del time_costs_by_row
    n_rows = len(ev_rows_pref)
    slot_to_col = {int(slot): int(cj) for cj, slot in enumerate(slots_list)}
    row_costs: dict[int, dict[int, float]] = {int(ri): {} for ri in range(n_rows)}
    row_details: dict[int, dict[int, dict[str, Any]]] = {int(ri): {} for ri in range(n_rows)}
    min_shared = max(1, int(getattr(cfg, "min_shared_points", 1)))

    for ri in range(n_rows):
        prev_slot = previous_slot_by_row.get(int(ri))
        candidates = [int(s) for s in candidate_slots_by_row[ri] if int(s) in slot_to_col]
        if len(candidates) == 0:
            candidates = [int(s) for s in slots_list]
        costs: dict[int, float] = {}
        e_obs = np.asarray(ev_rows_pref[ri], dtype=np.float64)

        for slot in candidates:
            s_obs_raw = row_slot_pref[ri].get(int(slot))
            if s_obs_raw is None:
                continue
            s_obs = np.asarray(s_obs_raw, dtype=np.float64)
            cost = float(refined_cost_fn(np.asarray(ev_rows_pref[ri], dtype=np.float64), s_obs))
            if not np.isfinite(cost):
                continue
            if prev_slot is not None and int(prev_slot) != int(slot):
                cost += float(cfg.switch_penalty)
            cj = slot_to_col[int(slot)]
            cost = float(cost + 1e-6 * (ri + 0.01 * cj))
            costs[int(slot)] = cost
            terms = _refined_term_decomposition(e_obs, s_obs, min_shared_points=min_shared)
            row_details[int(ri)][int(slot)] = {
                "time_prior_cost": float("nan"),
                "dtw_term": float(terms.get("dtw_term", float("nan"))),
                "corr_term": float(terms.get("corr_term", float("nan"))),
                "step_signature_term": float(terms.get("step_signature_term", float("nan"))),
                "posterior": float("nan"),
                "final_assignment_cost": float(cost),
            }

        if len(costs) == 0:
            fallback_slot = int(candidates[0]) if len(candidates) > 0 else int(slots_list[0])
            if prev_slot is not None and int(prev_slot) in slot_to_col and int(prev_slot) in candidates:
                fallback_slot = int(prev_slot)
            cj = slot_to_col[fallback_slot]
            costs[fallback_slot] = float(1e9 + 1e-6 * (ri + 0.01 * cj))
            row_details[int(ri)][int(fallback_slot)] = {
                "time_prior_cost": float("nan"),
                "dtw_term": float("nan"),
                "corr_term": float("nan"),
                "step_signature_term": float("nan"),
                "posterior": float("nan"),
                "final_assignment_cost": float(costs[fallback_slot]),
            }
        row_costs[int(ri)] = costs

    assigned_by_row = _greedy_unique_assign_from_row_costs(row_costs, candidate_slots_by_row, slots_list)
    _online_assign_single_only.last_row_details = row_details
    _online_assign_single_only.last_row_states = {}
    return assigned_by_row, row_costs


def _online_assign_time_only_baseline(
    *,
    ev_rows_pref: list[np.ndarray],
    row_slot_pref: list[dict[int, np.ndarray]],
    slots_list: list[int],
    candidate_slots_by_row: list[list[int]],
    time_costs_by_row: list[dict[int, float]] | None = None,
    previous_slot_by_row: dict[int, int],
    cfg: Any,
    refined_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    **_kwargs: Any,
) -> tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]:
    """Timing-only control group: no current waveform cost, timing candidate cost only."""
    del ev_rows_pref, row_slot_pref, refined_cost_fn

    n_rows = len(candidate_slots_by_row)
    n_cols = len(slots_list)
    slot_to_col = {int(slot): int(cj) for cj, slot in enumerate(slots_list)}
    cost_matrix = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    row_costs: dict[int, dict[int, float]] = {int(ri): {} for ri in range(n_rows)}
    row_details: dict[int, dict[int, dict[str, Any]]] = {int(ri): {} for ri in range(n_rows)}

    for ri in range(n_rows):
        prev_slot = previous_slot_by_row.get(int(ri))
        candidates = [int(s) for s in candidate_slots_by_row[ri] if int(s) in slot_to_col]
        if len(candidates) == 0:
            candidates = [int(s) for s in slots_list]

        tc = {}
        if time_costs_by_row is not None and ri < len(time_costs_by_row):
            tc = dict(time_costs_by_row[ri] or {})

        for rank, slot in enumerate(candidates):
            cj = slot_to_col[int(slot)]
            base = float(tc.get(int(slot), float(rank)))
            if not np.isfinite(base):
                base = float(rank)
            if prev_slot is not None and int(prev_slot) != int(slot):
                base += float(cfg.switch_penalty)
            base = float(base + 1e-6 * (ri + 0.01 * cj))
            cost_matrix[ri, cj] = base
            row_costs[int(ri)][int(slot)] = base
            row_details[int(ri)][int(slot)] = {
                "time_prior_cost": float(base),
                "dtw_term": float("nan"),
                "corr_term": float("nan"),
                "step_signature_term": float("nan"),
                "posterior": float("nan"),
                "final_assignment_cost": float(base),
            }

    valid_mask = np.isfinite(cost_matrix)
    _dense, row_ind, col_ind = _hungarian_with_inf_penalty(cost_matrix, float(cfg.switch_penalty))
    assigned_by_row: dict[int, tuple[int, float]] = {}
    for r, c in zip(row_ind, col_ind):
        ri = int(r)
        cj = int(c)
        slot = int(slots_list[cj])
        if not bool(valid_mask[ri, cj]):
            assigned_by_row[ri] = (slot, float("inf"))
            continue
        assigned_by_row[ri] = (slot, float(cost_matrix[ri, cj]))
    _online_assign_time_only_baseline.last_row_details = row_details
    _online_assign_time_only_baseline.last_row_states = {}
    return assigned_by_row, row_costs


def _online_assign_nomura_original_hungarian(
    *,
    ev_rows_pref: list[np.ndarray],
    row_slot_pref: list[dict[int, np.ndarray]],
    slots_list: list[int],
    candidate_slots_by_row: list[list[int]],
    time_costs_by_row: list[dict[int, float]] | None = None,
    previous_slot_by_row: dict[int, int],
    cfg: Any,
    refined_cost_fn: Callable[[np.ndarray, np.ndarray], float],
    corr_weight: float = 1.0,
    dtw_weight: float = 1.0,
    **_kwargs: Any,
) -> tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]:
    del time_costs_by_row, refined_cost_fn
    n_rows = len(ev_rows_pref)
    n_cols = len(slots_list)
    correlations = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    dtw_distances = np.full((n_rows, n_cols), np.inf, dtype=np.float64)
    slot_to_col = {int(slot): int(cj) for cj, slot in enumerate(slots_list)}

    for ri in range(n_rows):
        e_obs = np.asarray(ev_rows_pref[ri], dtype=np.float64)
        allowed = [int(s) for s in candidate_slots_by_row[ri] if int(s) in slot_to_col]
        if len(allowed) == 0:
            allowed = [int(s) for s in slots_list]
        for slot in allowed:
            cj = slot_to_col[int(slot)]
            s_obs_raw = row_slot_pref[ri].get(int(slot))
            if s_obs_raw is None:
                continue
            s_obs = np.asarray(s_obs_raw, dtype=np.float64)
            x, y = _shared_observed_arrays(e_obs, s_obs, int(cfg.min_shared_points))
            if x is None or y is None:
                continue

            corr = 0.0
            if len(x) > 1 and len(y) > 1:
                sx = float(np.std(x))
                sy = float(np.std(y))
                if sx > 1e-8 and sy > 1e-8:
                    corr = float(np.corrcoef(x, y)[0, 1])
                    if not np.isfinite(corr):
                        corr = 0.0
            correlations[ri, cj] = float(np.clip(corr, -1.0, 1.0))
            dtw_distances[ri, cj] = float(_nomura_calculate_dtw_distance(x, y))

    corr_filled = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    corr_norm = (corr_filled + 1.0) / 2.0

    finite_dtw = dtw_distances[np.isfinite(dtw_distances)]
    dtw_norm = np.zeros_like(dtw_distances, dtype=np.float64)
    if finite_dtw.size > 0:
        dmin = float(np.min(finite_dtw))
        dmax = float(np.max(finite_dtw))
        if dmax != dmin:
            dtw_norm = (dtw_distances - dmin) / (dmax - dmin)
            dtw_norm[~np.isfinite(dtw_norm)] = 0.0
        dtw_norm = 1.0 - dtw_norm
        dtw_norm = np.clip(dtw_norm, 0.0, 1.0)

    cw = max(0.0, float(corr_weight))
    dw = max(0.0, float(dtw_weight))
    if cw <= 0.0 and dw <= 0.0:
        eval_matrix = np.zeros_like(corr_norm, dtype=np.float64)
    elif cw <= 0.0:
        eval_matrix = np.asarray(dtw_norm, dtype=np.float64)
    elif dw <= 0.0:
        eval_matrix = np.asarray(corr_norm, dtype=np.float64)
    else:
        # Keep original A2 baseline behavior when both terms are enabled.
        eval_matrix = corr_norm * dtw_norm
    cost_matrix = 1.0 - eval_matrix
    cost_matrix[~np.isfinite(dtw_distances)] = np.inf

    row_costs: dict[int, dict[int, float]] = {int(ri): {} for ri in range(n_rows)}
    row_details: dict[int, dict[int, dict[str, Any]]] = {int(ri): {} for ri in range(n_rows)}
    for ri in range(n_rows):
        prev_slot = previous_slot_by_row.get(int(ri))
        for cj, slot in enumerate(slots_list):
            base = float(cost_matrix[ri, cj])
            if not np.isfinite(base):
                continue
            if prev_slot is not None and int(prev_slot) != int(slot):
                base += float(cfg.switch_penalty)
            base = float(base + 1e-6 * (ri + 0.01 * cj))
            cost_matrix[ri, cj] = base
            row_costs[int(ri)][int(slot)] = base
            row_details[int(ri)][int(slot)] = {
                "time_prior_cost": float("nan"),
                "dtw_term": float(dtw_distances[ri, cj]) if np.isfinite(dtw_distances[ri, cj]) else float("nan"),
                "corr_term": float(correlations[ri, cj]) if np.isfinite(correlations[ri, cj]) else float("nan"),
                "step_signature_term": float("nan"),
                "posterior": float("nan"),
                "final_assignment_cost": float(base),
            }

    valid_mask = np.isfinite(cost_matrix)
    dense, row_ind, col_ind = _hungarian_with_inf_penalty(cost_matrix, float(cfg.switch_penalty))
    assigned_by_row: dict[int, tuple[int, float]] = {}
    for r, c in zip(row_ind, col_ind):
        ri = int(r)
        cj = int(c)
        slot = int(slots_list[cj])
        if not bool(valid_mask[ri, cj]):
            assigned_by_row[ri] = (slot, float("inf"))
            continue
        cost = float(cost_matrix[ri, cj])
        assigned_by_row[ri] = (slot, cost)
        if slot not in row_costs[ri]:
            row_costs[ri][slot] = cost
    _online_assign_nomura_original_hungarian.last_row_details = row_details
    _online_assign_nomura_original_hungarian.last_row_states = {}
    return assigned_by_row, row_costs


def _run_one_algorithm(
    algo_id: str,
    ev_meta: list[dict[str, Any]],
    sessions_by_slot: dict[int, list[dict[str, Any]]] | None = None,
    ev_series: dict[int, pd.Series] | None = None,
    window: int | None = None,
    ingest_seed: int | None = None,
    trace_level: str = "full",
) -> tuple[dict[int, dict[str, Any]], float, dict[str, Any]]:
    if sessions_by_slot is None or ev_series is None or window is None:
        raise ValueError(f"{algo_id} requires sessions_by_slot, ev_series, and window")

    base_ts = min(pd.Timestamp(meta["arrival_ts"]) for meta in ev_meta) if ev_meta else pd.Timestamp(algo.BASE_INITIAL)
    assignment_fn = None
    refined_cost_fn = algo._refined_cost_from_grids
    algo_key = str(algo_id).strip()

    def _make_refined_cost_fn(*, dtw_w: float, step_w: float, mean_w: float) -> Callable[[np.ndarray, np.ndarray], float]:
        def _refined(e_grid_raw: np.ndarray, s_grid_raw: np.ndarray) -> float:
            e_grid = algo.znorm(e_grid_raw)
            s_grid = algo.znorm(s_grid_raw)
            band = max(1, int(getattr(algo, "BAND_SEC", 30)) // int(getattr(algo, "CHARGER_SAMPLE", 5)))
            d_dtw = algo.cdtw_l2(s_grid, e_grid, band=band) / float(max(1, len(e_grid)))
            e_sig = algo.step_medians(
                e_grid_raw,
                tau=int(getattr(algo, "TAU", 60)),
                grid=int(getattr(algo, "CHARGER_SAMPLE", 5)),
                n_steps=int(getattr(algo, "N_STEPS", 6)),
            )
            s_sig = algo.step_medians(
                s_grid_raw,
                tau=int(getattr(algo, "TAU", 60)),
                grid=int(getattr(algo, "CHARGER_SAMPLE", 5)),
                n_steps=int(getattr(algo, "N_STEPS", 6)),
            )
            sig_cost = float(np.nanmean(np.abs(e_sig - s_sig)))
            mean_gap = abs(float(np.nanmean(e_grid_raw) - np.nanmean(s_grid_raw)))
            return float(float(dtw_w) * d_dtw + float(step_w) * sig_cost + float(mean_w) * mean_gap)

        return _refined

    def _build_bayesian_assigner(
        *,
        time_prior_alpha: float | None = None,
        time_prior_mix: float | None = None,
        posterior_prev_power: float | None = None,
    ) -> BayesianWindowedAssigner:
        return BayesianWindowedAssigner(
            time_prior_alpha=float(getattr(algo, "TIME_PRIOR_ALPHA", 1.0) if time_prior_alpha is None else float(time_prior_alpha)),
            current_like_beta=float(getattr(algo, "CURRENT_LIKE_BETA", 1.0)),
            eps=max(float(getattr(algo, "POSTERIOR_EPS", 1e-9)), 1e-12),
            time_prior_mix=max(0.0, float(getattr(algo, "TIME_PRIOR_MIX", 0.35) if time_prior_mix is None else float(time_prior_mix))),
            posterior_prev_power=max(0.0, float(getattr(algo, "POSTERIOR_PREV_POWER", 0.60) if posterior_prev_power is None else float(posterior_prev_power))),
            current_cost_weight=float(np.clip(float(getattr(algo, "CURRENT_COST_WEIGHT", 0.15)), 0.0, 1.0)),
        )

    def _make_nomura_assigner(*, corr_weight: float, dtw_weight: float) -> Callable[..., tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]]:
        def _assign(**kwargs: Any) -> tuple[dict[int, tuple[int, float]], dict[int, dict[int, float]]]:
            return _online_assign_nomura_original_hungarian(
                corr_weight=float(corr_weight),
                dtw_weight=float(dtw_weight),
                **kwargs,
            )

        return _assign

    if algo_key == "time_only_baseline":
        assignment_fn = _online_assign_time_only_baseline
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "single_only":
        assignment_fn = _online_assign_single_only
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "single_only_step_off":
        assignment_fn = _online_assign_single_only
        refined_cost_fn = _make_refined_cost_fn(
            dtw_w=float(getattr(algo, "REFINED_DTW_WEIGHT", 1.0)),
            step_w=0.0,
            mean_w=float(getattr(algo, "REFINED_MEAN_WEIGHT", 0.0)),
        )
    elif algo_key == "single_only_dtw_off":
        assignment_fn = _online_assign_single_only
        refined_cost_fn = _make_refined_cost_fn(
            dtw_w=0.0,
            step_w=float(getattr(algo, "REFINED_STEP_WEIGHT", 0.40)),
            mean_w=float(getattr(algo, "REFINED_MEAN_WEIGHT", 0.0)),
        )
    elif algo_key == "single_only_all_off":
        assignment_fn = _online_assign_time_only_baseline
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "nomura_original_interval_hungarian":
        assignment_fn = _online_assign_nomura_original_hungarian
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "nomura_original_interval_hungarian_dtw_off":
        assignment_fn = _make_nomura_assigner(corr_weight=1.0, dtw_weight=0.0)
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "nomura_original_interval_hungarian_corr_off":
        assignment_fn = _make_nomura_assigner(corr_weight=0.0, dtw_weight=1.0)
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "nomura_original_interval_hungarian_all_off":
        assignment_fn = _online_assign_time_only_baseline
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "bayesian_windowed":
        assignment_fn = _build_bayesian_assigner()
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "bayesian_windowed_memory_off":
        assignment_fn = _build_bayesian_assigner(
            posterior_prev_power=0.0,
        )
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "bayesian_windowed_timeprior_off":
        assignment_fn = _build_bayesian_assigner(
            time_prior_alpha=0.0,
            time_prior_mix=0.0,
        )
        refined_cost_fn = algo._refined_cost_from_grids
    elif algo_key == "bayesian_windowed_all_off":
        assignment_fn = _build_bayesian_assigner(
            time_prior_alpha=0.0,
            time_prior_mix=0.0,
            posterior_prev_power=0.0,
        )
        refined_cost_fn = algo._refined_cost_from_grids
    else:
        raise ValueError(f"Unsupported algorithm id: {algo_key}")

    t0 = time.perf_counter()
    runtime_diag: dict[str, Any] = {}
    links = algo.run_timeline_and_match(
        ev_meta,
        sessions_by_slot,
        ev_series,
        base_ts=base_ts,
        tau=algo.TAU,
        window=int(window),
        assignment_fn=assignment_fn,
        refined_cost_fn=refined_cost_fn,
        ingest_seed=None if ingest_seed is None else int(ingest_seed),
        diagnostics_out=runtime_diag,
        trace_level=str(trace_level),
    )
    return links, float(time.perf_counter() - t0), runtime_diag


def _evaluate_links(
    ev_meta: list[dict[str, Any]],
    links: dict[int, dict[str, Any]],
    runtime_s: float,
    requested_evs: int | None = None,
) -> dict[str, float | int]:
    total = len(ev_meta)
    requested = int(total if requested_evs is None else requested_evs)
    correct = 0
    unmatched = 0
    costs: list[float] = []
    latencies: list[float] = []
    decision_reason_counts: dict[str, int] = defaultdict(int)

    for ei, meta in enumerate(ev_meta):
        pred = links.get(ei)
        if pred is None:
            unmatched += 1
            continue
        gt = int(meta["evse_idx_gt"])
        slot = int(pred["slot"])
        if slot == gt:
            correct += 1
        costs.append(float(pred["cost"]))
        lat = float((pred["decision_ts"] - meta["arrival_ts"]).total_seconds())
        latencies.append(lat)
        reason = str(pred.get("decision_reason", "")).strip()
        if reason != "":
            decision_reason_counts[reason] += 1

    accuracy = float(correct / total) if total > 0 else float("nan")
    blocked = int(max(0, requested - total))
    blocked_ratio = float(blocked / requested) if requested > 0 else float("nan")
    accuracy_requested = float(correct / requested) if requested > 0 else float("nan")
    avg_cost = float(np.mean(costs)) if costs else float("nan")
    max_cost = float(np.max(costs)) if costs else float("nan")
    avg_latency = float(np.mean(latencies)) if latencies else float("nan")
    p90_latency = float(np.quantile(latencies, 0.90)) if latencies else float("nan")

    reason_keys = {
        "window_watermark": "window_watermark",
        "session_end_before_watermark": "session_end_before_watermark",
        "session_end_fallback": "session_end_fallback",
        "timeline_end_fallback": "timeline_end_fallback",
    }
    reason_counts: dict[str, int] = {
        out_key: int(decision_reason_counts.get(raw_key, 0))
        for raw_key, out_key in reason_keys.items()
    }
    fallback_finalized_count = (
        int(reason_counts["session_end_before_watermark"])
        + int(reason_counts["session_end_fallback"])
        + int(reason_counts["timeline_end_fallback"])
    )
    denom = int(total)
    reason_ratios: dict[str, float] = {
        f"{k}_ratio": (float(v) / float(denom)) if denom > 0 else float("nan")
        for k, v in reason_counts.items()
    }
    fallback_finalized_ratio = (
        float(fallback_finalized_count) / float(denom)
        if denom > 0
        else float("nan")
    )

    return {
        "requested_evs": int(requested),
        "admitted_evs": int(total),
        "blocked_evs": int(blocked),
        "blocked_ratio": float(blocked_ratio),
        "total_evs": int(total),
        "correct": int(correct),
        "unmatched": int(unmatched),
        "accuracy": float(accuracy),
        "accuracy_requested": float(accuracy_requested),
        "avg_cost": float(avg_cost),
        "max_cost": float(max_cost),
        "avg_latency_s": float(avg_latency),
        "p90_latency_s": float(p90_latency),
        "assignment_runtime_s": float(runtime_s),
        "window_watermark_count": int(reason_counts["window_watermark"]),
        "session_end_before_watermark_count": int(reason_counts["session_end_before_watermark"]),
        "session_end_fallback_count": int(reason_counts["session_end_fallback"]),
        "timeline_end_fallback_count": int(reason_counts["timeline_end_fallback"]),
        "fallback_finalized_count": int(fallback_finalized_count),
        "window_watermark_ratio": float(reason_ratios["window_watermark_ratio"]),
        "session_end_before_watermark_ratio": float(reason_ratios["session_end_before_watermark_ratio"]),
        "session_end_fallback_ratio": float(reason_ratios["session_end_fallback_ratio"]),
        "timeline_end_fallback_ratio": float(reason_ratios["timeline_end_fallback_ratio"]),
        "fallback_finalized_ratio": float(fallback_finalized_ratio),
    }


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_json_safe(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(str(text).encode("utf-8"))
    return str(h.hexdigest())


def _derive_algo_seed(seed: int, algo_id: str) -> int:
    digest = hashlib.sha256(str(algo_id).encode("utf-8")).digest()
    algo_part = int.from_bytes(digest[:4], "little", signed=False)
    return int((int(seed) ^ int(algo_part)) & 0xFFFFFFFF)


def _scenario_signature_hash(
    ev_meta: list[dict[str, Any]],
    sessions_by_slot: dict[int, list[dict[str, Any]]],
) -> str:
    ev_rows: list[dict[str, Any]] = []
    for m in ev_meta:
        ev_rows.append(
            {
                "ev_idx": int(m.get("ev_idx", -1)),
                "evse_idx_gt": int(m.get("evse_idx_gt", -1)),
                "arrival_ts": str(pd.Timestamp(m.get("arrival_ts"))),
                "session_end_ts": str(pd.Timestamp(m.get("session_end_ts"))),
                "start_est_ts": str(pd.Timestamp(m.get("start_est_ts"))),
                "pattern": [int(x) for x in list(m.get("pattern", []) or [])],
                "session_len_s": int(m.get("session_len_s", 0)),
            }
        )

    slots_obj: dict[str, Any] = {}
    for slot, sessions in sorted(sessions_by_slot.items(), key=lambda kv: int(kv[0])):
        rows: list[dict[str, Any]] = []
        for s in sessions:
            rows.append(
                {
                    "start_ts": str(pd.Timestamp(s.get("start_ts"))),
                    "end_ts": str(pd.Timestamp(s.get("end_ts"))),
                    "pattern": [int(x) for x in list(s.get("pattern", []) or [])],
                }
            )
        slots_obj[str(int(slot))] = rows

    payload = {
        "ev_meta": ev_rows,
        "sessions_by_slot": slots_obj,
    }
    return _sha256_text(_stable_json_dumps(payload))


def _safe_git_revision(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir.resolve()),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        text = str(out).strip()
        if text != "":
            return text
    except Exception:
        pass
    return "unknown"


def _aggregate_rows(
    raw_rows: list[dict[str, Any]],
    ev_counts: list[int],
    tau_values: list[int],
    pattern_modes: list[str],
    pattern_unique_count: int,
    command_step_count: int,
    algo_specs: list[AlgoSpec],
    pattern_assignment_mode: str,
    charger_sample_values: list[int],
    ev_sample_values: list[int],
    matcher_delay_max_values_s: list[float],
    candidate_margin_values_s: list[int],
    config_hash: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def _f_eq(a: Any, b: float) -> bool:
        try:
            x = float(a)
        except Exception:
            return False
        return bool(np.isfinite(x) and abs(x - float(b)) <= 1e-9)

    for tau_s in tau_values:
        for mode in pattern_modes:
            for charger_sample_s in charger_sample_values:
                for ev_sample_s in ev_sample_values:
                    for matcher_delay_max_s in matcher_delay_max_values_s:
                        for candidate_margin_s in candidate_margin_values_s:
                            for spec in algo_specs:
                                for ev_count in ev_counts:
                                    rows = [
                                        r
                                        for r in raw_rows
                                        if r["algorithm_id"] == spec.id
                                        and r.get("pattern_mode") == mode
                                        and str(r.get("pattern_assignment_mode", DEFAULT_PATTERN_ASSIGNMENT_MODE)) == pattern_assignment_mode
                                        and int(r.get("tau_s", 60)) == int(tau_s)
                                        and int(r.get("ev_count", 0)) == int(ev_count)
                                        and int(r.get("charger_sample_s", DEFAULT_CHARGER_SAMPLE_VALUES[0])) == int(charger_sample_s)
                                        and int(r.get("ev_sample_s", DEFAULT_EV_SAMPLE_VALUES[0])) == int(ev_sample_s)
                                        and _f_eq(r.get("matcher_delay_max_s", float("nan")), float(matcher_delay_max_s))
                                        and int(r.get("candidate_margin_s", DEFAULT_CANDIDATE_MARGIN_VALUES_S[0])) == int(candidate_margin_s)
                                    ]

                                    acc_vals = [float(r.get("accuracy", float("nan"))) for r in rows]
                                    cost_vals = [float(r.get("avg_cost", float("nan"))) for r in rows]
                                    lat_vals = [float(r.get("p90_latency_s", float("nan"))) for r in rows]
                                    rt_vals = [float(r.get("assignment_runtime_s", float("nan"))) for r in rows]
                                    un_vals = [float(r.get("unmatched", float("nan"))) for r in rows]
                                    acc_req_vals = [float(r.get("accuracy_requested", float("nan"))) for r in rows]
                                    req_vals = [float(r.get("requested_evs", float("nan"))) for r in rows]
                                    adm_vals = [float(r.get("admitted_evs", float("nan"))) for r in rows]
                                    blk_vals = [float(r.get("blocked_evs", float("nan"))) for r in rows]
                                    blk_ratio_vals = [float(r.get("blocked_ratio", float("nan"))) for r in rows]

                                    wm_cnt_vals = [float(r.get("window_watermark_count", float("nan"))) for r in rows]
                                    sebw_cnt_vals = [float(r.get("session_end_before_watermark_count", float("nan"))) for r in rows]
                                    sef_cnt_vals = [float(r.get("session_end_fallback_count", float("nan"))) for r in rows]
                                    tef_cnt_vals = [float(r.get("timeline_end_fallback_count", float("nan"))) for r in rows]
                                    fb_cnt_vals = [float(r.get("fallback_finalized_count", float("nan"))) for r in rows]
                                    wm_ratio_vals = [float(r.get("window_watermark_ratio", float("nan"))) for r in rows]
                                    sebw_ratio_vals = [float(r.get("session_end_before_watermark_ratio", float("nan"))) for r in rows]
                                    sef_ratio_vals = [float(r.get("session_end_fallback_ratio", float("nan"))) for r in rows]
                                    tef_ratio_vals = [float(r.get("timeline_end_fallback_ratio", float("nan"))) for r in rows]
                                    fb_ratio_vals = [float(r.get("fallback_finalized_ratio", float("nan"))) for r in rows]

                                    gt_in_cand_ratio_vals = [float(r.get("gt_in_candidates_ratio", float("nan"))) for r in rows]
                                    avg_candidates_vals = [float(r.get("avg_candidates_per_ev", float("nan"))) for r in rows]
                                    no_candidate_rows_total_vals = [float(r.get("no_candidate_rows_total", float("nan"))) for r in rows]
                                    fallback_to_all_rows_total_vals = [float(r.get("fallback_to_all_rows_total", float("nan"))) for r in rows]
                                    gt_recovered_by_fallback_count_vals = [float(r.get("gt_recovered_by_fallback_count", float("nan"))) for r in rows]
                                    no_candidate_rows_ratio_vals = [float(r.get("no_candidate_rows_ratio", float("nan"))) for r in rows]
                                    fallback_to_all_rows_ratio_vals = [float(r.get("fallback_to_all_rows_ratio", float("nan"))) for r in rows]
                                    gt_recovered_by_fallback_ratio_vals = [float(r.get("gt_recovered_by_fallback_ratio", float("nan"))) for r in rows]
                                    candidate_query_lead_max_vals = [float(r.get("candidate_query_lead_max_s", float("nan"))) for r in rows]
                                    candidate_future_query_violation_count_vals = [
                                        float(r.get("candidate_future_query_violation_count", float("nan"))) for r in rows
                                    ]
                                    rt_cand_vals = [float(r.get("runtime_s_candidate_build", float("nan"))) for r in rows]
                                    rt_assign_vals = [float(r.get("runtime_s_assignment", float("nan"))) for r in rows]
                                    rt_fb_vals = [float(r.get("runtime_s_fallback", float("nan"))) for r in rows]
                                    rt_other_vals = [float(r.get("runtime_s_other", float("nan"))) for r in rows]
                                    rt_total_vals = [float(r.get("runtime_s_total", float("nan"))) for r in rows]
                                    rt_cost_vals = [float(r.get("runtime_s_cost", float("nan"))) for r in rows]
                                    rt_loop_vals = [float(r.get("runtime_s_loop", float("nan"))) for r in rows]

                                    acc_mean, acc_std, acc_ci95 = _mean_std_ci95(acc_vals)
                                    cost_mean, cost_std, cost_ci95 = _mean_std_ci95(cost_vals)
                                    lat_mean, lat_std, lat_ci95 = _mean_std_ci95(lat_vals)
                                    rt_mean, rt_std, rt_ci95 = _mean_std_ci95(rt_vals)
                                    un_mean, un_std, un_ci95 = _mean_std_ci95(un_vals)
                                    acc_req_mean, acc_req_std, acc_req_ci95 = _mean_std_ci95(acc_req_vals)
                                    req_mean, req_std, req_ci95 = _mean_std_ci95(req_vals)
                                    adm_mean, adm_std, adm_ci95 = _mean_std_ci95(adm_vals)
                                    blk_mean, blk_std, blk_ci95 = _mean_std_ci95(blk_vals)
                                    blk_ratio_mean, blk_ratio_std, blk_ratio_ci95 = _mean_std_ci95(blk_ratio_vals)

                                    wm_cnt_mean, wm_cnt_std, wm_cnt_ci95 = _mean_std_ci95(wm_cnt_vals)
                                    sebw_cnt_mean, sebw_cnt_std, sebw_cnt_ci95 = _mean_std_ci95(sebw_cnt_vals)
                                    sef_cnt_mean, sef_cnt_std, sef_cnt_ci95 = _mean_std_ci95(sef_cnt_vals)
                                    tef_cnt_mean, tef_cnt_std, tef_cnt_ci95 = _mean_std_ci95(tef_cnt_vals)
                                    fb_cnt_mean, fb_cnt_std, fb_cnt_ci95 = _mean_std_ci95(fb_cnt_vals)
                                    wm_ratio_mean, wm_ratio_std, wm_ratio_ci95 = _mean_std_ci95(wm_ratio_vals)
                                    sebw_ratio_mean, sebw_ratio_std, sebw_ratio_ci95 = _mean_std_ci95(sebw_ratio_vals)
                                    sef_ratio_mean, sef_ratio_std, sef_ratio_ci95 = _mean_std_ci95(sef_ratio_vals)
                                    tef_ratio_mean, tef_ratio_std, tef_ratio_ci95 = _mean_std_ci95(tef_ratio_vals)
                                    fb_ratio_mean, fb_ratio_std, fb_ratio_ci95 = _mean_std_ci95(fb_ratio_vals)

                                    gt_in_cand_ratio_mean, gt_in_cand_ratio_std, gt_in_cand_ratio_ci95 = _mean_std_ci95(gt_in_cand_ratio_vals)
                                    avg_candidates_mean, avg_candidates_std, avg_candidates_ci95 = _mean_std_ci95(avg_candidates_vals)
                                    no_candidate_rows_total_mean, no_candidate_rows_total_std, no_candidate_rows_total_ci95 = _mean_std_ci95(no_candidate_rows_total_vals)
                                    fallback_to_all_rows_total_mean, fallback_to_all_rows_total_std, fallback_to_all_rows_total_ci95 = _mean_std_ci95(fallback_to_all_rows_total_vals)
                                    gt_recovered_by_fallback_count_mean, gt_recovered_by_fallback_count_std, gt_recovered_by_fallback_count_ci95 = _mean_std_ci95(gt_recovered_by_fallback_count_vals)
                                    no_candidate_rows_ratio_mean, no_candidate_rows_ratio_std, no_candidate_rows_ratio_ci95 = _mean_std_ci95(no_candidate_rows_ratio_vals)
                                    fallback_to_all_rows_ratio_mean, fallback_to_all_rows_ratio_std, fallback_to_all_rows_ratio_ci95 = _mean_std_ci95(fallback_to_all_rows_ratio_vals)
                                    gt_recovered_by_fallback_ratio_mean, gt_recovered_by_fallback_ratio_std, gt_recovered_by_fallback_ratio_ci95 = _mean_std_ci95(gt_recovered_by_fallback_ratio_vals)
                                    candidate_query_lead_max_mean, candidate_query_lead_max_std, candidate_query_lead_max_ci95 = _mean_std_ci95(candidate_query_lead_max_vals)
                                    candidate_future_query_violation_count_mean, candidate_future_query_violation_count_std, candidate_future_query_violation_count_ci95 = _mean_std_ci95(
                                        candidate_future_query_violation_count_vals
                                    )
                                    rt_cand_mean, rt_cand_std, rt_cand_ci95 = _mean_std_ci95(rt_cand_vals)
                                    rt_assign_mean, rt_assign_std, rt_assign_ci95 = _mean_std_ci95(rt_assign_vals)
                                    rt_fb_mean, rt_fb_std, rt_fb_ci95 = _mean_std_ci95(rt_fb_vals)
                                    rt_other_mean, rt_other_std, rt_other_ci95 = _mean_std_ci95(rt_other_vals)
                                    rt_total_mean, rt_total_std, rt_total_ci95 = _mean_std_ci95(rt_total_vals)
                                    rt_cost_mean, rt_cost_std, rt_cost_ci95 = _mean_std_ci95(rt_cost_vals)
                                    rt_loop_mean, rt_loop_std, rt_loop_ci95 = _mean_std_ci95(rt_loop_vals)

                                    out.append(
                                        {
                                            "config_hash": str(config_hash),
                                            "tau_s": int(tau_s),
                                            "pattern_mode": mode,
                                            "pattern_mode_label": PATTERN_MODE_LABELS.get(mode, mode),
                                            "pattern_assignment_mode": pattern_assignment_mode,
                                            "pattern_assignment_mode_label": PATTERN_ASSIGNMENT_MODE_LABELS.get(pattern_assignment_mode, pattern_assignment_mode),
                                            "pattern_unique_count": int(pattern_unique_count),
                                            "command_step_count": int(command_step_count),
                                            "charger_sample_s": int(charger_sample_s),
                                            "ev_sample_s": int(ev_sample_s),
                                            "matcher_delay_max_s": float(matcher_delay_max_s),
                                            "candidate_margin_s": int(candidate_margin_s),
                                            "algorithm_id": spec.id,
                                            "algorithm_label": spec.label,
                                            "is_current": bool(spec.is_current),
                                            "ev_count": int(ev_count),
                                            "runs": len(rows),
                                            "accuracy_mean": float(acc_mean),
                                            "accuracy_std": float(acc_std),
                                            "accuracy_ci95": float(acc_ci95),
                                            "accuracy_requested_mean": float(acc_req_mean),
                                            "accuracy_requested_std": float(acc_req_std),
                                            "accuracy_requested_ci95": float(acc_req_ci95),
                                            "requested_evs_mean": float(req_mean),
                                            "requested_evs_std": float(req_std),
                                            "requested_evs_ci95": float(req_ci95),
                                            "admitted_evs_mean": float(adm_mean),
                                            "admitted_evs_std": float(adm_std),
                                            "admitted_evs_ci95": float(adm_ci95),
                                            "blocked_evs_mean": float(blk_mean),
                                            "blocked_evs_std": float(blk_std),
                                            "blocked_evs_ci95": float(blk_ci95),
                                            "blocked_ratio_mean": float(blk_ratio_mean),
                                            "blocked_ratio_std": float(blk_ratio_std),
                                            "blocked_ratio_ci95": float(blk_ratio_ci95),
                                            "avg_cost_mean": float(cost_mean),
                                            "avg_cost_std": float(cost_std),
                                            "avg_cost_ci95": float(cost_ci95),
                                            "p90_latency_mean": float(lat_mean),
                                            "p90_latency_std": float(lat_std),
                                            "p90_latency_ci95": float(lat_ci95),
                                            "assignment_runtime_mean": float(rt_mean),
                                            "assignment_runtime_std": float(rt_std),
                                            "assignment_runtime_ci95": float(rt_ci95),
                                            "runtime_s_candidate_build_mean": float(rt_cand_mean),
                                            "runtime_s_candidate_build_std": float(rt_cand_std),
                                            "runtime_s_candidate_build_ci95": float(rt_cand_ci95),
                                            "runtime_s_assignment_mean": float(rt_assign_mean),
                                            "runtime_s_assignment_std": float(rt_assign_std),
                                            "runtime_s_assignment_ci95": float(rt_assign_ci95),
                                            "runtime_s_fallback_mean": float(rt_fb_mean),
                                            "runtime_s_fallback_std": float(rt_fb_std),
                                            "runtime_s_fallback_ci95": float(rt_fb_ci95),
                                            "runtime_s_other_mean": float(rt_other_mean),
                                            "runtime_s_other_std": float(rt_other_std),
                                            "runtime_s_other_ci95": float(rt_other_ci95),
                                            "runtime_s_total_mean": float(rt_total_mean),
                                            "runtime_s_total_std": float(rt_total_std),
                                            "runtime_s_total_ci95": float(rt_total_ci95),
                                            "runtime_s_cost_mean": float(rt_cost_mean),
                                            "runtime_s_cost_std": float(rt_cost_std),
                                            "runtime_s_cost_ci95": float(rt_cost_ci95),
                                            "runtime_s_loop_mean": float(rt_loop_mean),
                                            "runtime_s_loop_std": float(rt_loop_std),
                                            "runtime_s_loop_ci95": float(rt_loop_ci95),
                                            "unmatched_mean": float(un_mean),
                                            "unmatched_std": float(un_std),
                                            "unmatched_ci95": float(un_ci95),
                                            "window_watermark_count_mean": float(wm_cnt_mean),
                                            "window_watermark_count_std": float(wm_cnt_std),
                                            "window_watermark_count_ci95": float(wm_cnt_ci95),
                                            "session_end_before_watermark_count_mean": float(sebw_cnt_mean),
                                            "session_end_before_watermark_count_std": float(sebw_cnt_std),
                                            "session_end_before_watermark_count_ci95": float(sebw_cnt_ci95),
                                            "session_end_fallback_count_mean": float(sef_cnt_mean),
                                            "session_end_fallback_count_std": float(sef_cnt_std),
                                            "session_end_fallback_count_ci95": float(sef_cnt_ci95),
                                            "timeline_end_fallback_count_mean": float(tef_cnt_mean),
                                            "timeline_end_fallback_count_std": float(tef_cnt_std),
                                            "timeline_end_fallback_count_ci95": float(tef_cnt_ci95),
                                            "fallback_finalized_count_mean": float(fb_cnt_mean),
                                            "fallback_finalized_count_std": float(fb_cnt_std),
                                            "fallback_finalized_count_ci95": float(fb_cnt_ci95),
                                            "window_watermark_ratio_mean": float(wm_ratio_mean),
                                            "window_watermark_ratio_std": float(wm_ratio_std),
                                            "window_watermark_ratio_ci95": float(wm_ratio_ci95),
                                            "session_end_before_watermark_ratio_mean": float(sebw_ratio_mean),
                                            "session_end_before_watermark_ratio_std": float(sebw_ratio_std),
                                            "session_end_before_watermark_ratio_ci95": float(sebw_ratio_ci95),
                                            "session_end_fallback_ratio_mean": float(sef_ratio_mean),
                                            "session_end_fallback_ratio_std": float(sef_ratio_std),
                                            "session_end_fallback_ratio_ci95": float(sef_ratio_ci95),
                                            "timeline_end_fallback_ratio_mean": float(tef_ratio_mean),
                                            "timeline_end_fallback_ratio_std": float(tef_ratio_std),
                                            "timeline_end_fallback_ratio_ci95": float(tef_ratio_ci95),
                                            "fallback_finalized_ratio_mean": float(fb_ratio_mean),
                                            "fallback_finalized_ratio_std": float(fb_ratio_std),
                                            "fallback_finalized_ratio_ci95": float(fb_ratio_ci95),
                                            "gt_in_candidates_ratio_mean": float(gt_in_cand_ratio_mean),
                                            "gt_in_candidates_ratio_std": float(gt_in_cand_ratio_std),
                                            "gt_in_candidates_ratio_ci95": float(gt_in_cand_ratio_ci95),
                                            "avg_candidates_per_ev_mean": float(avg_candidates_mean),
                                            "avg_candidates_per_ev_std": float(avg_candidates_std),
                                            "avg_candidates_per_ev_ci95": float(avg_candidates_ci95),
                                            "no_candidate_rows_total_mean": float(no_candidate_rows_total_mean),
                                            "no_candidate_rows_total_std": float(no_candidate_rows_total_std),
                                            "no_candidate_rows_total_ci95": float(no_candidate_rows_total_ci95),
                                            "fallback_to_all_rows_total_mean": float(fallback_to_all_rows_total_mean),
                                            "fallback_to_all_rows_total_std": float(fallback_to_all_rows_total_std),
                                            "fallback_to_all_rows_total_ci95": float(fallback_to_all_rows_total_ci95),
                                            "gt_recovered_by_fallback_count_mean": float(gt_recovered_by_fallback_count_mean),
                                            "gt_recovered_by_fallback_count_std": float(gt_recovered_by_fallback_count_std),
                                            "gt_recovered_by_fallback_count_ci95": float(gt_recovered_by_fallback_count_ci95),
                                            "no_candidate_rows_ratio_mean": float(no_candidate_rows_ratio_mean),
                                            "no_candidate_rows_ratio_std": float(no_candidate_rows_ratio_std),
                                            "no_candidate_rows_ratio_ci95": float(no_candidate_rows_ratio_ci95),
                                            "fallback_to_all_rows_ratio_mean": float(fallback_to_all_rows_ratio_mean),
                                            "fallback_to_all_rows_ratio_std": float(fallback_to_all_rows_ratio_std),
                                            "fallback_to_all_rows_ratio_ci95": float(fallback_to_all_rows_ratio_ci95),
                                            "gt_recovered_by_fallback_ratio_mean": float(gt_recovered_by_fallback_ratio_mean),
                                            "gt_recovered_by_fallback_ratio_std": float(gt_recovered_by_fallback_ratio_std),
                                            "gt_recovered_by_fallback_ratio_ci95": float(gt_recovered_by_fallback_ratio_ci95),
                                            "candidate_query_lead_max_s_mean": float(candidate_query_lead_max_mean),
                                            "candidate_query_lead_max_s_std": float(candidate_query_lead_max_std),
                                            "candidate_query_lead_max_s_ci95": float(candidate_query_lead_max_ci95),
                                            "candidate_future_query_violation_count_mean": float(candidate_future_query_violation_count_mean),
                                            "candidate_future_query_violation_count_std": float(candidate_future_query_violation_count_std),
                                            "candidate_future_query_violation_count_ci95": float(candidate_future_query_violation_count_ci95),
                                        }
                                    )
    return out


def _plot_metric_lines(
    summary_rows: list[dict[str, Any]],
    ev_counts: list[int],
    tau_values: list[int],
    pattern_modes: list[str],
    algo_specs: list[AlgoSpec],
    metric_prefix: str,
    y_label: str,
    title: str,
    out_path: Path,
    error_mode: str,
    scale: float = 1.0,
):
    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    color_map = {
        "time_only_baseline": "#808080",
        "single_only": "#1F77B4",
        "nomura_original_interval_hungarian": "#2CA02C",
        "bayesian_windowed": "#D62728",
    }
    marker_map = {
        "time_only_baseline": "o",
        "single_only": "s",
        "nomura_original_interval_hungarian": "^",
        "bayesian_windowed": "D",
    }
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    linestyle_map = {"current": "-", "legacy": "--"}
    used_labels: set[str] = set()

    for ti, tau_s in enumerate(tau_values):
        for mode in pattern_modes:
            for spec in algo_specs:
                rows = [
                    r
                    for r in summary_rows
                    if r["algorithm_id"] == spec.id
                    and r.get("pattern_mode") == mode
                    and int(r.get("tau_s", 60)) == int(tau_s)
                ]
                rows = sorted(rows, key=lambda r: int(r["ev_count"]))
                if len(rows) == 0:
                    continue

                x = np.asarray([int(r["ev_count"]) for r in rows], dtype=float)
                y = np.asarray([float(r[f"{metric_prefix}_mean"]) * scale for r in rows], dtype=float)
                e = np.asarray([float(r[f"{metric_prefix}_{error_mode}"]) * scale for r in rows], dtype=float)

                label_parts = [spec.label]
                if len(pattern_modes) > 1:
                    label_parts.append(PATTERN_MODE_LABELS.get(mode, mode))
                label = " | ".join(label_parts)
                draw_label = label if label not in used_labels else "_nolegend_"
                used_labels.add(label)

                ax.errorbar(
                    x,
                    y,
                    yerr=e,
                    fmt=marker_map.get(spec.id, marker_cycle[ti % len(marker_cycle)]),
                    capsize=8,
                    linewidth=2.4 if spec.is_current else 2.0,
                    markersize=10.5,
                    linestyle=linestyle_map.get(mode, "-"),
                    color=color_map.get(spec.id, None),
                    label=draw_label,
                )

    y_axis_label = str(y_label)
    if metric_prefix == "accuracy":
        y_axis_label = "Accuracy (%)"
    elif metric_prefix == "p90_latency":
        y_axis_label = "p90 decision latency (s)"
    elif metric_prefix == "avg_cost":
        y_axis_label = "Average cost"
    elif metric_prefix == "assignment_runtime":
        y_axis_label = "Runtime (s)"

    ax.set_xlabel("EV sessions", fontsize=21.0)
    ax.set_ylabel(y_axis_label, fontsize=21.0)
    xt = np.asarray(sorted(set(int(v) for v in ev_counts)), dtype=float)
    ax.set_xticks(xt)
    ax.set_xticklabels([str(int(v)) for v in xt])
    if xt.size > 0:
        ax.set_xlim(float(np.min(xt)) - 5.0, float(np.max(xt)) + 5.0)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", labelsize=14.0)
    ax.legend(loc="best", fontsize=13.0, framealpha=0.93)
    fig.tight_layout(pad=1.30)
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, columns: list[str] | None = None):
    if isinstance(rows, list) and len(rows) > 0:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=list(columns or []))

    if isinstance(columns, list) and len(columns) > 0:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df.reindex(columns=list(columns))
    df.to_csv(path, index=False, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                continue
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = str(line).strip()
                if text == "":
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return []
    return out


def _iter_jsonl(path: Path, *, max_rows: int | None = None):
    if not path.exists():
        return
    seen = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = str(line).strip()
                if text == "":
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                yield obj
                seen += 1
                if max_rows is not None and seen >= int(max_rows):
                    break
    except Exception:
        return


def _write_parquet_optional(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _build_presentation_graph_rows(summary_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    accuracy_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    scatter_rows: list[dict[str, Any]] = []

    def _base(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "tau_s": int(r.get("tau_s", 0) or 0),
            "pattern_mode": str(r.get("pattern_mode", "")),
            "pattern_assignment_mode": str(r.get("pattern_assignment_mode", "")),
            "algorithm_id": str(r.get("algorithm_id", "")),
            "algorithm_label": str(r.get("algorithm_label", "")),
            "ev_count": int(r.get("ev_count", 0) or 0),
            "runs": int(r.get("runs", 0) or 0),
        }

    for row in summary_rows:
        if not isinstance(row, dict):
            continue
        b = _base(row)
        accuracy_rows.append(
            {
                **b,
                "accuracy_mean": float(row.get("accuracy_mean", float("nan"))),
                "accuracy_ci95": float(row.get("accuracy_ci95", float("nan"))),
            }
        )
        latency_rows.append(
            {
                **b,
                "p90_latency_mean_s": float(row.get("p90_latency_mean", float("nan"))),
                "p90_latency_ci95_s": float(row.get("p90_latency_ci95", float("nan"))),
            }
        )
        runtime_rows.append(
            {
                **b,
                "runtime_mean_s": float(row.get("assignment_runtime_mean", float("nan"))),
                "runtime_ci95_s": float(row.get("assignment_runtime_ci95", float("nan"))),
            }
        )
        scatter_rows.append(
            {
                **b,
                "accuracy_mean": float(row.get("accuracy_mean", float("nan"))),
                "runtime_mean_s": float(row.get("assignment_runtime_mean", float("nan"))),
            }
        )

    sort_keys = ["tau_s", "pattern_mode", "algorithm_id", "ev_count"]
    accuracy_rows.sort(key=lambda r: tuple(r.get(k) for k in sort_keys))
    latency_rows.sort(key=lambda r: tuple(r.get(k) for k in sort_keys))
    runtime_rows.sort(key=lambda r: tuple(r.get(k) for k in sort_keys))
    scatter_rows.sort(key=lambda r: tuple(r.get(k) for k in sort_keys))

    return {
        "accuracy_vs_ev_ci95": accuracy_rows,
        "p90_latency_vs_ev_ci95": latency_rows,
        "runtime_vs_ev_ci95": runtime_rows,
        "accuracy_runtime_scatter_mean": scatter_rows,
    }


def run_algorithm_compare_report(
    ev_counts: list[int] | None = None,
    repeats: int = DEFAULT_REPEATS,
    error_mode: str = DEFAULT_ERROR_MODE,
    base_seed: int | None = None,
    pattern_mode: str = DEFAULT_PATTERN_MODE,
    pattern_assignment_mode: str = DEFAULT_PATTERN_ASSIGNMENT_MODE,
    pattern_unique_count: int = DEFAULT_PATTERN_UNIQUE_COUNT,
    command_step_count: int = DEFAULT_COMMAND_STEP_COUNT,
    tau_values: list[int] | None = None,
    algorithm_ids: list[str] | None = None,
    ingestion_enabled: bool = True,
    ev_ingest_delay_mean_s: float = float(algo.EV_INGEST_DELAY_MEAN_S),
    ev_ingest_delay_jitter_s: float = float(algo.EV_INGEST_DELAY_JITTER_S),
    ev_ingest_delay_max_s: float = float(algo.EV_INGEST_DELAY_MAX_S),
    ev_ingest_loss_prob: float = float(algo.EV_INGEST_LOSS_PROB),
    charger_sample_values: list[int] | None = None,
    ev_sample_values: list[int] | None = None,
    matcher_delay_max_values_s: list[float] | None = None,
    candidate_margin_values_s: list[int] | None = None,
    trace_level: str = "full",
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ev_counts = _validate_ev_counts(ev_counts)
    repeats = _validate_repeats(repeats)
    error_mode = _validate_error_mode(error_mode)
    pattern_mode = _validate_pattern_mode(pattern_mode)
    pattern_assignment_mode = _validate_pattern_assignment_mode(pattern_assignment_mode)
    pattern_unique_count = _validate_pattern_unique_count(pattern_unique_count)
    command_step_count = _validate_command_step_count(command_step_count)
    trace_mode = str(trace_level).strip().lower()
    if trace_mode not in {"full", "summary"}:
        raise ValueError("trace_level must be 'full' or 'summary'")
    tau_values = _validate_tau_values(tau_values)
    selected_algo_specs = _validate_algorithm_ids(algorithm_ids)

    ingestion_cfg = _validate_ingestion_inputs(
        ingestion_enabled=ingestion_enabled,
        ev_ingest_delay_mean_s=ev_ingest_delay_mean_s,
        ev_ingest_delay_jitter_s=ev_ingest_delay_jitter_s,
        ev_ingest_delay_max_s=ev_ingest_delay_max_s,
        ev_ingest_loss_prob=ev_ingest_loss_prob,
    )

    charger_sample_values = _validate_sample_values(
        charger_sample_values,
        field="charger_sample_values",
        default=list(DEFAULT_CHARGER_SAMPLE_VALUES),
        min_value=1,
        max_value=300,
    )
    ev_sample_values = _validate_sample_values(
        ev_sample_values,
        field="ev_sample_values",
        default=list(DEFAULT_EV_SAMPLE_VALUES),
        min_value=1,
        max_value=3600,
    )
    matcher_delay_max_values_s = _validate_float_values(
        matcher_delay_max_values_s,
        field="matcher_delay_max_values_s",
        default=[float(ingestion_cfg["ev_ingest_delay_max_s"])],
        min_value=0.0,
        max_value=3600.0,
    )
    candidate_margin_values_s = _validate_candidate_margin_values(candidate_margin_values_s)

    for tau_s in tau_values:
        for charger_s in charger_sample_values:
            if int(tau_s) % int(charger_s) != 0:
                raise ValueError(f"charger_sample_values must divide tau_values ({charger_s}s does not divide {tau_s}s)")

    pattern_modes = _resolve_pattern_modes(pattern_mode)
    window_values_s = [int(t * command_step_count) for t in tau_values]

    scenario_units = (
        len(tau_values)
        * len(pattern_modes)
        * len(charger_sample_values)
        * len(ev_sample_values)
        * len(matcher_delay_max_values_s)
        * len(candidate_margin_values_s)
        * len(ev_counts)
        * int(repeats)
    )
    total_work_units = int(scenario_units * len(selected_algo_specs))
    completed_units = 0

    def _emit_progress(stage: str, detail: str) -> None:
        if progress_cb is None:
            return
        pct = float(completed_units / total_work_units * 100.0) if total_work_units > 0 else 0.0
        progress_cb(
            {
                "stage": str(stage),
                "detail": str(detail),
                "completed_units": int(completed_units),
                "total_units": int(total_work_units),
                "progress_pct": float(max(0.0, min(100.0, pct))),
            }
        )

    if base_seed is None:
        base_seed = int(time.time_ns() & 0xFFFFFFFF)
    else:
        base_seed = int(base_seed)

    run_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = COMPARE_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_base = GENERATED_OUTPUT_DIR.resolve()

    experiment_cfg = ExperimentConfig(
        run_type="algorithm_compare",
        run_started=str(run_started),
        base_seed=int(base_seed),
        repeats=int(repeats),
        ev_counts=[int(x) for x in ev_counts],
        evse_count=int(FIXED_EVSE_COUNT),
        sim_hours=12.0,
        session_min_minutes=10.0,
        session_max_minutes=60.0,
        pattern_mode=str(pattern_mode),
        pattern_modes=[str(x) for x in pattern_modes],
        pattern_assignment_mode=str(pattern_assignment_mode),
        pattern_unique_count=int(pattern_unique_count),
        command_step_count=int(command_step_count),
        tau_values=[int(x) for x in tau_values],
        window_values_s=[int(x) for x in window_values_s],
        charger_sample_values_s=[int(x) for x in charger_sample_values],
        ev_sample_values_s=[int(x) for x in ev_sample_values],
        matcher_delay_max_values_s=[float(x) for x in matcher_delay_max_values_s],
        candidate_margin_values_s=[int(x) for x in candidate_margin_values_s],
        trace_level=str(trace_mode),
        algorithm_ids=[str(s.id) for s in selected_algo_specs],
        ingestion_enabled=bool(ingestion_cfg["ingestion_enabled"]),
        ev_ingest_delay_mean_s=float(ingestion_cfg["ev_ingest_delay_mean_s"]),
        ev_ingest_delay_jitter_s=float(ingestion_cfg["ev_ingest_delay_jitter_s"]),
        ev_ingest_delay_max_s=float(ingestion_cfg["ev_ingest_delay_max_s"]),
        ev_ingest_loss_prob=float(ingestion_cfg["ev_ingest_loss_prob"]),
    )
    config_dict = asdict(experiment_cfg)
    config_hash = _sha256_text(_stable_json_dumps(config_dict))
    config_path = out_dir / "config.json"
    resolved_config_path = out_dir / "resolved_runtime_config.json"
    physical_manifest_path = out_dir / "physical_model_manifest.json"
    git_rev_path = out_dir / "git_revision.txt"
    git_revision = _safe_git_revision(Path(__file__).parent)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe({"config_hash": config_hash, **config_dict}), f, ensure_ascii=False, indent=2)
    resolved_runtime_config = {
        "config_hash": str(config_hash),
        "run_id": str(tag),
        "run_started": str(run_started),
        "runtime_matching": {
            "evaluation_interval_s": int(getattr(algo, "EVALUATION_INTERVAL_S", 5)),
            "min_shared_points": int(getattr(algo, "MIN_SHARED_POINTS", 2)),
            "switch_penalty": float(getattr(algo, "SWITCH_PENALTY", 3.0)),
            "candidate_time_margin_s": int(getattr(algo, "CANDIDATE_TIME_MARGIN_S", 120)),
            "window_watermark_mode": "window_watermark",
        },
        "ingestion_effective": {
            "ev_delay_mean_s": float(ingestion_cfg["ev_ingest_delay_mean_s"]),
            "ev_delay_jitter_s": float(ingestion_cfg["ev_ingest_delay_jitter_s"]),
            "ev_delay_max_s": float(ingestion_cfg["ev_ingest_delay_max_s"]),
            "ev_loss_prob": float(ingestion_cfg["ev_ingest_loss_prob"]),
            "evse_delay_mean_s": 0.0,
            "evse_delay_jitter_s": 0.0,
            "evse_delay_max_s": 0.0,
            "evse_loss_prob": 0.0,
        },
        "algorithm_weights": {
            "refined_dtw_weight": float(getattr(algo, "REFINED_DTW_WEIGHT", 1.0)),
            "refined_step_weight": float(getattr(algo, "REFINED_STEP_WEIGHT", 0.40)),
            "refined_mean_weight": float(getattr(algo, "REFINED_MEAN_WEIGHT", 0.0)),
            "time_prior_alpha": float(getattr(algo, "TIME_PRIOR_ALPHA", 1.0)),
            "current_like_beta": float(getattr(algo, "CURRENT_LIKE_BETA", 1.0)),
            "posterior_eps": float(getattr(algo, "POSTERIOR_EPS", 1e-9)),
            "time_prior_mix": float(getattr(algo, "TIME_PRIOR_MIX", 0.35)),
            "posterior_prev_power": float(getattr(algo, "POSTERIOR_PREV_POWER", 0.60)),
            "current_cost_weight": float(getattr(algo, "CURRENT_COST_WEIGHT", 0.15)),
        },
        "sensor_model": {
            "ev_sensor_extra_delay_max_s": int(getattr(algo, "EV_SENSOR_EXTRA_DELAY_MAX_S", 0)),
            "ev_sensor_gain_std": float(getattr(algo, "EV_SENSOR_GAIN_STD", 0.0)),
            "ev_sensor_bias_std_a": float(getattr(algo, "EV_SENSOR_BIAS_STD_A", 0.0)),
            "ev_sensor_noise_std_a": float(getattr(algo, "EV_SENSOR_NOISE_STD_A", 0.0)),
            "ev_start_est_jitter_s": int(getattr(algo, "EV_START_EST_JITTER_S", 0)),
        },
    }
    with resolved_config_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(resolved_runtime_config), f, ensure_ascii=False, indent=2)
    physical_manifest = {
        "manifest_type": "physical_model_manifest",
        "model_name": "EV-Link evidence-informed baseline",
        "fitted_on": "",
        "coefficient_source": "evidence-informed baseline (non site-calibrated)",
        "device_model_note": "Device/charger-specific coefficients may vary by hardware and firmware.",
        "validity_range": {
            "sim_horizon_h": 12,
            "session_duration_min_minutes": 10,
            "session_duration_max_minutes": 60,
            "ev_sample_s": [int(x) for x in ev_sample_values],
            "evse_sample_s": [int(x) for x in charger_sample_values],
        },
        "coefficients": {
            "refined_dtw_weight": float(getattr(algo, "REFINED_DTW_WEIGHT", 1.0)),
            "refined_step_weight": float(getattr(algo, "REFINED_STEP_WEIGHT", 0.40)),
            "refined_mean_weight": float(getattr(algo, "REFINED_MEAN_WEIGHT", 0.0)),
            "ev_sensor_gain_std": float(getattr(algo, "EV_SENSOR_GAIN_STD", 0.0)),
            "ev_sensor_bias_std_a": float(getattr(algo, "EV_SENSOR_BIAS_STD_A", 0.0)),
            "ev_sensor_noise_std_a": float(getattr(algo, "EV_SENSOR_NOISE_STD_A", 0.0)),
        },
        "comments": (
            "EVSE side is console-known reference (zero uplink delay/loss in this baseline). "
            "Ingestion defaults are evidence-informed and should be recalibrated for site deployment."
        ),
    }
    with physical_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(physical_manifest), f, ensure_ascii=False, indent=2)
    git_rev_path.write_text(str(git_revision) + "\n", encoding="utf-8")

    raw_rows: list[dict[str, Any]] = []
    event_trace_rows: list[dict[str, Any]] = []
    cost_trace_rows: list[dict[str, Any]] = []
    posterior_trace_rows: list[dict[str, Any]] = []
    posterior_checkpoint_trace_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    params = SimulationParameters(
        ev_count=1,
        evse_count=FIXED_EVSE_COUNT,
        sim_hours=12.0,
        session_min_minutes=10.0,
        session_max_minutes=60.0,
        charger_sample=int(charger_sample_values[0]),
        ev_sample=int(ev_sample_values[0]),
        ingestion_enabled=bool(ingestion_cfg["ingestion_enabled"]),
        ev_ingest_delay_mean_s=float(ingestion_cfg["ev_ingest_delay_mean_s"]),
        ev_ingest_delay_jitter_s=float(ingestion_cfg["ev_ingest_delay_jitter_s"]),
        ev_ingest_delay_max_s=float(ingestion_cfg["ev_ingest_delay_max_s"]),
        matcher_delay_max_s=float(matcher_delay_max_values_s[0]),
        ev_ingest_loss_prob=float(ingestion_cfg["ev_ingest_loss_prob"]),
    )

    old_seed_env = os.getenv("EVLINK_SEED")
    replay_ingestion_mismatch_count = 0
    scenario_counter = 0

    try:
        _emit_progress("prepare", "Initializing compare run")
        with simulation_lock:
            with _silence_algo_logs():
                for tau_s in tau_values:
                    for mode in pattern_modes:
                        use_dynamic_pattern_ctx = pattern_assignment_mode == "dynamic_session"
                        pattern_ctx = (
                            _override_pattern_generator(mode, pattern_unique_count, command_step_count)
                            if use_dynamic_pattern_ctx
                            else nullcontext()
                        )
                        with pattern_ctx:
                            for charger_sample_s in charger_sample_values:
                                for ev_sample_s in ev_sample_values:
                                    for matcher_delay_max_s in matcher_delay_max_values_s:
                                        for candidate_margin_s in candidate_margin_values_s:
                                            for ev_count in ev_counts:
                                                for rep in range(repeats):
                                                    seed = int(base_seed + scenario_counter)
                                                    scenario_counter += 1
                                                    scenario_id = int(scenario_counter)
                                                    os.environ["EVLINK_SEED"] = str(seed)

                                                    params.ev_count = int(ev_count)
                                                    params.charger_sample = int(charger_sample_s)
                                                    params.ev_sample = int(ev_sample_s)
                                                    params.matcher_delay_max_s = float(matcher_delay_max_s)
                                                    params.validate()

                                                    with override_algo_config(params):
                                                        with override_ingestion_config(params):
                                                            with _override_candidate_margin(int(candidate_margin_s)):
                                                                with _override_mcct_timing(int(tau_s), command_step_count):
                                                                    algo.init_seeds()
                                                                    slot_fixed_patterns = None
                                                                    if pattern_assignment_mode == "legacy_slot_fixed":
                                                                        slot_fixed_patterns = _legacy_interval_generate_command_patterns(
                                                                            num_patterns=int(params.evse_count),
                                                                            seed=int(seed) ^ 0x1D872B41,
                                                                            command_step_count=int(command_step_count),
                                                                        )

                                                                    ev_meta, sessions_by_slot, ev_series, _patterns, dataset_diag = algo.build_dataset_with_random_arrivals(
                                                                        base_initial_ts=algo.BASE_INITIAL,
                                                                        n_evs=params.ev_count,
                                                                        n_evses=params.evse_count,
                                                                        tau=algo.TAU,
                                                                        window=algo.WINDOW,
                                                                        arrival_span_max=params.arrival_span_max,
                                                                        slot_fixed_patterns=slot_fixed_patterns,
                                                                    )
                                                                    scenario_hash = _scenario_signature_hash(ev_meta, sessions_by_slot)
                                                                    if isinstance(dataset_diag, dict):
                                                                        blocked_items = dataset_diag.get("blocked_sessions", [])
                                                                        if isinstance(blocked_items, list):
                                                                            for b in blocked_items:
                                                                                if not isinstance(b, dict):
                                                                                    continue
                                                                                blocked_rows.append(
                                                                                    {
                                                                                        "run_id": str(tag),
                                                                                        "config_hash": str(config_hash),
                                                                                        "scenario_id": int(scenario_id),
                                                                                        "scenario_seed": int(seed),
                                                                                        "repeat_idx": int(rep + 1),
                                                                                        "tau_s": int(tau_s),
                                                                                        "charger_sample_s": int(charger_sample_s),
                                                                                        "ev_sample_s": int(ev_sample_s),
                                                                                        "matcher_delay_max_s": float(matcher_delay_max_s),
                                                                                        "candidate_margin_s": int(candidate_margin_s),
                                                                                        "ev_count": int(ev_count),
                                                                                        "evse_count": int(FIXED_EVSE_COUNT),
                                                                                        "pattern_mode": str(mode),
                                                                                        "pattern_assignment_mode": str(pattern_assignment_mode),
                                                                                        "request_idx": int(b.get("request_idx", -1)),
                                                                                        "arrival_ts": str(b.get("arrival_ts", "")),
                                                                                        "active_sessions_at_arrival": int(b.get("active_sessions_at_arrival", 0)),
                                                                                        "requested_duration_s": int(b.get("requested_duration_s", 0)),
                                                                                        "blocked_reason_code": str(b.get("blocked_reason_code", "")),
                                                                                        "capacity_shortage": bool(b.get("capacity_shortage", False)),
                                                                                        "overlap_ratio": float(b.get("overlap_ratio", float("nan"))),
                                                                                        "long_session_dominance_ratio": float(b.get("long_session_dominance_ratio", float("nan"))),
                                                                                    }
                                                                                )

                                                                    # Fair compare rule: keep EV cloud ingestion realization fixed
                                                                    # across all algorithms within the same scenario/repeat.
                                                                    ingest_seed = int(seed) ^ 0x5EEDC0DE
                                                                    ingestion_sig_ref = ""

                                                                    for spec in selected_algo_specs:
                                                                        algo_seed = _derive_algo_seed(seed, spec.id)
                                                                        random.seed(int(algo_seed))
                                                                        np.random.seed(int(algo_seed))

                                                                        links, runtime_s, runtime_diag = _run_one_algorithm(
                                                                            spec.id,
                                                                            ev_meta,
                                                                            sessions_by_slot=sessions_by_slot,
                                                                            ev_series=ev_series,
                                                                            window=algo.WINDOW,
                                                                            ingest_seed=ingest_seed,
                                                                            trace_level=trace_mode,
                                                                        )
                                                                        m = _evaluate_links(ev_meta, links, runtime_s, requested_evs=int(ev_count))
                                                                        loop_diag = runtime_diag.get("runtime_loop") if isinstance(runtime_diag, dict) else {}
                                                                        if not isinstance(loop_diag, dict):
                                                                            loop_diag = {}
                                                                        ingestion_sig = ""
                                                                        if isinstance(runtime_diag, dict):
                                                                            ingestion_sig = str(runtime_diag.get("ingestion_signature_sha256", "") or "")
                                                                        ingest_match_ref = True
                                                                        if ingestion_sig_ref == "":
                                                                            ingestion_sig_ref = ingestion_sig
                                                                        elif ingestion_sig != "" and ingestion_sig_ref != "":
                                                                            ingest_match_ref = bool(ingestion_sig == ingestion_sig_ref)
                                                                            if not ingest_match_ref:
                                                                                replay_ingestion_mismatch_count += 1

                                                                        rt_candidate_build = float(loop_diag.get("stage_candidate_build_s", float("nan")))
                                                                        rt_assignment = float(loop_diag.get("stage_assignment_s", float("nan")))
                                                                        rt_fallback = float(loop_diag.get("stage_fallback_s", float("nan")))
                                                                        rt_other = float(loop_diag.get("stage_other_s", float("nan")))
                                                                        rt_total = float(loop_diag.get("total_wall_s", float("nan")))

                                                                        m.update(
                                                                            {
                                                                                "gt_in_candidates_ratio": float(loop_diag.get("gt_in_candidates_ratio", float("nan"))),
                                                                                "avg_candidates_per_ev": float(loop_diag.get("avg_candidates_per_ev", float("nan"))),
                                                                                "no_candidate_rows_total": float(loop_diag.get("no_candidate_rows_total", float("nan"))),
                                                                                "fallback_to_all_rows_total": float(loop_diag.get("fallback_to_all_rows_total", float("nan"))),
                                                                                "gt_recovered_by_fallback_count": float(loop_diag.get("gt_recovered_by_fallback_count", float("nan"))),
                                                                                "no_candidate_rows_ratio": float(loop_diag.get("no_candidate_rows_ratio", float("nan"))),
                                                                                "fallback_to_all_rows_ratio": float(loop_diag.get("fallback_to_all_rows_ratio", float("nan"))),
                                                                                "gt_recovered_by_fallback_ratio": float(loop_diag.get("gt_recovered_by_fallback_ratio", float("nan"))),
                                                                                "candidate_query_lead_max_s": float(loop_diag.get("candidate_query_lead_max_s", float("nan"))),
                                                                                "candidate_future_query_violation_count": float(
                                                                                    loop_diag.get("candidate_future_query_violation_count", float("nan"))
                                                                                ),
                                                                                "runtime_s_candidate_build": rt_candidate_build,
                                                                                "runtime_s_assignment": rt_assignment,
                                                                                "runtime_s_fallback": rt_fallback,
                                                                                "runtime_s_other": rt_other,
                                                                                # IEEE result CSV aliases.
                                                                                "runtime_s_total": rt_total,
                                                                                "runtime_s_cost": rt_candidate_build,
                                                                                "runtime_s_loop": rt_other,
                                                                            }
                                                                        )
                                                                        trace_common = {
                                                                            "run_id": str(tag),
                                                                            "config_hash": str(config_hash),
                                                                            "scenario_id": int(scenario_id),
                                                                            "scenario_seed": int(seed),
                                                                            "algo_seed": int(algo_seed),
                                                                            "repeat_idx": int(rep + 1),
                                                                            "ev_count": int(ev_count),
                                                                            "evse_count": int(FIXED_EVSE_COUNT),
                                                                            "tau_s": int(tau_s),
                                                                            "charger_sample_s": int(charger_sample_s),
                                                                            "ev_sample_s": int(ev_sample_s),
                                                                            "matcher_delay_max_s": float(matcher_delay_max_s),
                                                                            "candidate_margin_s": int(candidate_margin_s),
                                                                            "pattern_mode": str(mode),
                                                                            "pattern_assignment_mode": str(pattern_assignment_mode),
                                                                            "algorithm_id": str(spec.id),
                                                                            "algorithm_label": str(spec.label),
                                                                        }
                                                                        if isinstance(runtime_diag, dict):
                                                                            event_rows_one = runtime_diag.get("event_trace", [])
                                                                            if isinstance(event_rows_one, list):
                                                                                for tr in event_rows_one:
                                                                                    if not isinstance(tr, dict):
                                                                                        continue
                                                                                    event_trace_rows.append({**trace_common, **dict(tr)})
                                                                            cost_rows_one = runtime_diag.get("cost_trace", [])
                                                                            if isinstance(cost_rows_one, list):
                                                                                for tr in cost_rows_one:
                                                                                    if not isinstance(tr, dict):
                                                                                        continue
                                                                                    cost_trace_rows.append({**trace_common, **dict(tr)})
                                                                            posterior_rows_one = runtime_diag.get("posterior_trace", [])
                                                                            if isinstance(posterior_rows_one, list):
                                                                                for tr in posterior_rows_one:
                                                                                    if not isinstance(tr, dict):
                                                                                        continue
                                                                                    posterior_trace_rows.append({**trace_common, **dict(tr)})
                                                                            checkpoint_rows_one = runtime_diag.get("posterior_checkpoint_trace", [])
                                                                            if isinstance(checkpoint_rows_one, list):
                                                                                for tr in checkpoint_rows_one:
                                                                                    if not isinstance(tr, dict):
                                                                                        continue
                                                                                    posterior_checkpoint_trace_rows.append({**trace_common, **dict(tr)})

                                                                        raw_rows.append(
                                                                            {
                                                                                "run_id": tag,
                                                                                "config_hash": str(config_hash),
                                                                                "scenario_id": int(scenario_id),
                                                                                "scenario_seed": int(seed),
                                                                                "algo_seed": int(algo_seed),
                                                                                "seed": int(seed),
                                                                                "repeat_idx": int(rep + 1),
                                                                                "ev_count": int(ev_count),
                                                                                "evse_count": FIXED_EVSE_COUNT,
                                                                                "tau_s": int(tau_s),
                                                                                "charger_sample_s": int(charger_sample_s),
                                                                                "ev_sample_s": int(ev_sample_s),
                                                                                "matcher_delay_max_s": float(matcher_delay_max_s),
                                                                                "candidate_margin_s": int(candidate_margin_s),
                                                                                "pattern_mode": mode,
                                                                                "pattern_mode_label": PATTERN_MODE_LABELS.get(mode, mode),
                                                                                "pattern_assignment_mode": pattern_assignment_mode,
                                                                                "pattern_assignment_mode_label": PATTERN_ASSIGNMENT_MODE_LABELS.get(pattern_assignment_mode, pattern_assignment_mode),
                                                                                "pattern_unique_count": int(pattern_unique_count),
                                                                                "command_step_count": int(command_step_count),
                                                                                "window_s": int(algo.WINDOW),
                                                                                "scenario_hash": str(scenario_hash),
                                                                                "ingest_seed": int(ingest_seed),
                                                                                "ingestion_signature_sha256": str(ingestion_sig),
                                                                                "ingestion_signature_match_ref": bool(ingest_match_ref),
                                                                                "algorithm_id": spec.id,
                                                                                "algorithm_label": spec.label,
                                                                                "is_current": bool(spec.is_current),
                                                                                **m,
                                                                            }
                                                                        )
                                                                        completed_units += 1
                                                                        _emit_progress(
                                                                            "run",
                                                                            (
                                                                                f"tau={int(tau_s)}s mode={mode} ev={int(ev_count)} rep={int(rep+1)} "
                                                                                f"chg={int(charger_sample_s)}s evs={int(ev_sample_s)}s "
                                                                                f"dmax={float(matcher_delay_max_s):.3f}s Δ={int(candidate_margin_s)}s algo={spec.id}"
                                                                            ),
                                                                        )
    finally:
        if old_seed_env is None:
            os.environ.pop("EVLINK_SEED", None)
        else:
            os.environ["EVLINK_SEED"] = old_seed_env

    summary_rows = _aggregate_rows(
        raw_rows,
        ev_counts,
        tau_values,
        pattern_modes,
        pattern_unique_count,
        command_step_count,
        selected_algo_specs,
        pattern_assignment_mode,
        charger_sample_values,
        ev_sample_values,
        matcher_delay_max_values_s,
        candidate_margin_values_s,
        config_hash,
    )

    # Main-line plots keep the baseline sweep point to avoid overlaid duplicates.
    main_charger_sample_s = int(charger_sample_values[0])
    main_ev_sample_s = int(ev_sample_values[0])
    main_matcher_delay_max_s = float(matcher_delay_max_values_s[0])
    main_candidate_margin_s = int(candidate_margin_values_s[0])
    summary_rows_main = [
        r
        for r in summary_rows
        if int(r.get("charger_sample_s", main_charger_sample_s)) == int(main_charger_sample_s)
        and int(r.get("ev_sample_s", main_ev_sample_s)) == int(main_ev_sample_s)
        and abs(float(r.get("matcher_delay_max_s", main_matcher_delay_max_s)) - float(main_matcher_delay_max_s)) <= 1e-9
        and int(r.get("candidate_margin_s", main_candidate_margin_s)) == int(main_candidate_margin_s)
    ]
    if len(summary_rows_main) == 0:
        summary_rows_main = list(summary_rows)

    ingestion_effective = params.effective_ingestion_dict()

    p_acc = out_dir / "accuracy_vs_ev.png"
    p_lat = out_dir / "p90_latency_vs_ev.png"
    p_cost = out_dir / "avg_cost_vs_ev.png"
    p_runtime = out_dir / "runtime_vs_ev.png"
    p_summary_csv = out_dir / "summary.csv"
    p_raw_csv = out_dir / "raw_runs.csv"
    p_raw_results_csv = out_dir / "raw_results.csv"
    p_event_trace_jsonl = out_dir / "event_trace.jsonl"
    p_event_trace_parquet = out_dir / "event_trace.parquet"
    p_cost_trace_jsonl = out_dir / "cost_trace.jsonl"
    p_cost_trace_parquet = out_dir / "cost_trace.parquet"
    p_posterior_trace_jsonl = out_dir / "posterior_trace_A3.jsonl"
    p_posterior_trace_parquet = out_dir / "posterior_trace_A3.parquet"
    p_posterior_checkpoint_jsonl = out_dir / "posterior_checkpoint_trace_A3.jsonl"
    p_blocked_csv = out_dir / "blocked_sessions.csv"
    p_pres_acc = out_dir / "presentation_accuracy_vs_ev_ci95.csv"
    p_pres_lat = out_dir / "presentation_p90_latency_vs_ev_ci95.csv"
    p_pres_runtime = out_dir / "presentation_runtime_vs_ev_ci95.csv"
    p_pres_scatter = out_dir / "presentation_accuracy_runtime_scatter_mean.csv"

    _plot_metric_lines(
        summary_rows_main,
        ev_counts,
        tau_values,
        pattern_modes,
        selected_algo_specs,
        metric_prefix="accuracy",
        y_label="Accuracy (%)",
        title=f"Algorithm Accuracy vs EV Count (EVSE={FIXED_EVSE_COUNT}, repeats={repeats})",
        out_path=p_acc,
        error_mode=error_mode,
        scale=100.0,
    )
    _plot_metric_lines(
        summary_rows_main,
        ev_counts,
        tau_values,
        pattern_modes,
        selected_algo_specs,
        metric_prefix="p90_latency",
        y_label="P90 Latency (s)",
        title=f"Algorithm P90 Latency vs EV Count (EVSE={FIXED_EVSE_COUNT}, repeats={repeats})",
        out_path=p_lat,
        error_mode=error_mode,
        scale=1.0,
    )
    _plot_metric_lines(
        summary_rows_main,
        ev_counts,
        tau_values,
        pattern_modes,
        selected_algo_specs,
        metric_prefix="avg_cost",
        y_label="Average Matching Cost",
        title=f"Algorithm Average Cost vs EV Count (EVSE={FIXED_EVSE_COUNT}, repeats={repeats})",
        out_path=p_cost,
        error_mode=error_mode,
        scale=1.0,
    )
    _plot_metric_lines(
        summary_rows_main,
        ev_counts,
        tau_values,
        pattern_modes,
        selected_algo_specs,
        metric_prefix="assignment_runtime",
        y_label="Assignment Runtime (s)",
        title=f"Algorithm Runtime vs EV Count (EVSE={FIXED_EVSE_COUNT}, repeats={repeats})",
        out_path=p_runtime,
        error_mode=error_mode,
        scale=1.0,
    )

    _emit_progress("finalize", "Writing plots and reports")
    _write_csv(p_summary_csv, summary_rows)
    _write_csv(p_raw_csv, raw_rows)
    _write_csv(p_raw_results_csv, raw_rows)
    _write_jsonl(p_event_trace_jsonl, event_trace_rows)
    _write_jsonl(p_cost_trace_jsonl, cost_trace_rows)
    _write_jsonl(p_posterior_trace_jsonl, posterior_trace_rows)
    _write_jsonl(p_posterior_checkpoint_jsonl, posterior_checkpoint_trace_rows)
    _write_csv(
        p_blocked_csv,
        blocked_rows,
        columns=[
            "run_id",
            "config_hash",
            "scenario_id",
            "scenario_seed",
            "repeat_idx",
            "tau_s",
            "charger_sample_s",
            "ev_sample_s",
            "matcher_delay_max_s",
            "candidate_margin_s",
            "ev_count",
            "evse_count",
            "pattern_mode",
            "pattern_assignment_mode",
            "request_idx",
            "arrival_ts",
            "active_sessions_at_arrival",
            "requested_duration_s",
            "blocked_reason_code",
            "capacity_shortage",
            "overlap_ratio",
            "long_session_dominance_ratio",
        ],
    )
    wrote_event_parquet = _write_parquet_optional(p_event_trace_parquet, event_trace_rows)
    wrote_cost_parquet = _write_parquet_optional(p_cost_trace_parquet, cost_trace_rows)
    wrote_posterior_parquet = _write_parquet_optional(p_posterior_trace_parquet, posterior_trace_rows)
    presentation_rows = _build_presentation_graph_rows(summary_rows_main)
    _write_csv(p_pres_acc, presentation_rows["accuracy_vs_ev_ci95"])
    _write_csv(p_pres_lat, presentation_rows["p90_latency_vs_ev_ci95"])
    _write_csv(p_pres_runtime, presentation_rows["runtime_vs_ev_ci95"])
    _write_csv(p_pres_scatter, presentation_rows["accuracy_runtime_scatter_mean"])

    plot_paths = {
        "accuracy": _to_rel_plot(p_acc, plots_base),
        "p90_latency": _to_rel_plot(p_lat, plots_base),
        "avg_cost": _to_rel_plot(p_cost, plots_base),
        "runtime": _to_rel_plot(p_runtime, plots_base),
        "summary_csv": _to_rel_plot(p_summary_csv, plots_base),
        "raw_csv": _to_rel_plot(p_raw_csv, plots_base),
        "raw_results_csv": _to_rel_plot(p_raw_results_csv, plots_base),
        "event_trace_jsonl": _to_rel_plot(p_event_trace_jsonl, plots_base),
        "cost_trace_jsonl": _to_rel_plot(p_cost_trace_jsonl, plots_base),
        "posterior_trace_a3_jsonl": _to_rel_plot(p_posterior_trace_jsonl, plots_base),
        "blocked_sessions_csv": _to_rel_plot(p_blocked_csv, plots_base),
        "presentation_accuracy_csv": _to_rel_plot(p_pres_acc, plots_base),
        "presentation_p90_latency_csv": _to_rel_plot(p_pres_lat, plots_base),
        "presentation_runtime_csv": _to_rel_plot(p_pres_runtime, plots_base),
        "presentation_acc_runtime_scatter_csv": _to_rel_plot(p_pres_scatter, plots_base),
        "config_json": _to_rel_plot(config_path, plots_base),
        "resolved_runtime_config_json": _to_rel_plot(resolved_config_path, plots_base),
        "physical_model_manifest_json": _to_rel_plot(physical_manifest_path, plots_base),
        "git_revision_txt": _to_rel_plot(git_rev_path, plots_base),
    }
    if bool(wrote_event_parquet):
        plot_paths["event_trace_parquet"] = _to_rel_plot(p_event_trace_parquet, plots_base)
    if bool(wrote_cost_parquet):
        plot_paths["cost_trace_parquet"] = _to_rel_plot(p_cost_trace_parquet, plots_base)
    if bool(wrote_posterior_parquet):
        plot_paths["posterior_trace_a3_parquet"] = _to_rel_plot(p_posterior_trace_parquet, plots_base)

    result = {
        "run_id": tag,
        "run_started": run_started,
        "base_seed": int(base_seed),
        "config_hash": str(config_hash),
        "git_revision": str(git_revision),
        "resolved_runtime_config_json": _to_rel_plot(resolved_config_path, plots_base),
        "physical_model_manifest_json": _to_rel_plot(physical_manifest_path, plots_base),
        "repeats": int(repeats),
        "error_mode": str(error_mode),
        "pattern_mode": pattern_mode,
        "pattern_modes": pattern_modes,
        "pattern_assignment_mode": pattern_assignment_mode,
        "pattern_assignment_mode_label": PATTERN_ASSIGNMENT_MODE_LABELS.get(pattern_assignment_mode, pattern_assignment_mode),
        "pattern_unique_count": int(pattern_unique_count),
        "command_step_count": int(command_step_count),
        "tau_values": [int(x) for x in tau_values],
        "window_values_s": [int(x) for x in window_values_s],
        "tau_s": int(tau_values[0]) if len(tau_values) == 1 else None,
        "window_s": int(window_values_s[0]) if len(window_values_s) == 1 else None,
        "charger_sample_values_s": [int(x) for x in charger_sample_values],
        "ev_sample_values_s": [int(x) for x in ev_sample_values],
        "matcher_delay_max_values_s": [float(x) for x in matcher_delay_max_values_s],
        "candidate_margin_values_s": [int(x) for x in candidate_margin_values_s],
        "algorithm_ids": [s.id for s in selected_algo_specs],
        "ev_counts": [int(x) for x in ev_counts],
        "evse_fixed": int(FIXED_EVSE_COUNT),
        "algorithms": [
            {
                "id": s.id,
                "label": s.label,
                "description": s.description,
                "is_current": bool(s.is_current),
            }
            for s in selected_algo_specs
        ],
        "fixed_conditions": {
            "sim_hours": 12.0,
            "session_min_minutes": 10.0,
            "session_max_minutes": 60.0,
            "ev_sample_s": int(main_ev_sample_s),
            "charger_sample_s": int(main_charger_sample_s),
            "ev_sample_values_s": [int(x) for x in ev_sample_values],
            "charger_sample_values_s": [int(x) for x in charger_sample_values],
            "matcher_delay_max_values_s": [float(x) for x in matcher_delay_max_values_s],
            "candidate_margin_values_s": [int(x) for x in candidate_margin_values_s],
            "tau_values": [int(x) for x in tau_values],
            "window_values_s": [int(x) for x in window_values_s],
            "command_step_count": int(command_step_count),
            "pattern_modes": {m: PATTERN_MODE_LABELS.get(m, m) for m in pattern_modes},
            "pattern_unique_count": int(pattern_unique_count),
            "pattern_assignment_mode": pattern_assignment_mode,
            "pattern_assignment_mode_label": PATTERN_ASSIGNMENT_MODE_LABELS.get(pattern_assignment_mode, pattern_assignment_mode),
        },
        "online_operational_conditions": {
            "evaluation_interval_s": int(algo.EVALUATION_INTERVAL_S),
            "min_shared_points": int(algo.MIN_SHARED_POINTS),
            "switch_penalty": float(algo.SWITCH_PENALTY),
            "candidate_time_margin_s": int(main_candidate_margin_s),
            "ev_sensor_extra_delay_max_s": int(getattr(algo, "EV_SENSOR_EXTRA_DELAY_MAX_S", 0)),
            "ev_sensor_gain_std": float(getattr(algo, "EV_SENSOR_GAIN_STD", 0.0)),
            "ev_sensor_bias_std_a": float(getattr(algo, "EV_SENSOR_BIAS_STD_A", 0.0)),
            "ev_sensor_noise_std_a": float(getattr(algo, "EV_SENSOR_NOISE_STD_A", 0.0)),
            "ev_start_est_jitter_s": int(getattr(algo, "EV_START_EST_JITTER_S", 0)),
            "refined_cost_weights": {
                "dtw": float(getattr(algo, "REFINED_DTW_WEIGHT", 1.0)),
                "step_signature": float(getattr(algo, "REFINED_STEP_WEIGHT", 0.40)),
                "mean_gap": float(getattr(algo, "REFINED_MEAN_WEIGHT", 0.0)),
            },
            "bayesian_params": {
                "time_prior_alpha": float(getattr(algo, "TIME_PRIOR_ALPHA", 1.0)),
                "current_like_beta": float(getattr(algo, "CURRENT_LIKE_BETA", 1.0)),
                "posterior_eps": float(getattr(algo, "POSTERIOR_EPS", 1e-9)),
                "time_prior_mix": float(getattr(algo, "TIME_PRIOR_MIX", 0.35)),
                "posterior_prev_power": float(getattr(algo, "POSTERIOR_PREV_POWER", 0.60)),
                "current_cost_weight": float(getattr(algo, "CURRENT_COST_WEIGHT", 0.15)),
            },
            "assignment_policy": "window-fixed (arrival+window) global 1:1 on matured windows, finalization at window watermark",
            "candidate_scope": "evse_slots_active_within_effective_end_margin_clamped_to_decision_tick",
            "decision_anchor": "ev_observed_start_estimate",
            "compare_input_policy": "generate_once_replay_many_with_fixed_ingestion_seed_per_scenario",
            "runtime_loop": {
                "mode": "window_watermark",
            },
            "ingestion_model": {
                "ev": {
                    "delay_mean_s": float(ingestion_effective["ev_ingest_delay_mean_s"]),
                    "delay_jitter_s": float(ingestion_effective["ev_ingest_delay_jitter_s"]),
                    "delay_max_s": float(ingestion_effective["ev_ingest_delay_max_s"]),
                    "watermark_delay_max_s": float(main_matcher_delay_max_s),
                    "loss_prob": float(ingestion_effective["ev_ingest_loss_prob"]),
                },
                "evse": {
                    "source": "console_known",
                    "delay_mean_s": 0.0,
                    "delay_jitter_s": 0.0,
                    "delay_max_s": 0.0,
                    "loss_prob": 0.0,
                },
            },
            "ingestion_reference": ingestion_reference_payload(),
        },
        "replay_checks": {
            "scenario_count": int(scenario_counter),
            "ingestion_signature_mismatch_count": int(replay_ingestion_mismatch_count),
        },
        "pattern_generation_methods": {
            "current": "Optimized generator (maximin/proxy-distance based) used in current research code.",
            "legacy": "Legacy random unique method: [0 + (steps-1) unique values sampled from 6..30].",
            "legacy_interval": "Correlation-DTW_EV_interval style legacy generator, then fixed sequential assignment to EVSE slots.",
        },
        "note": (
            "Algorithm comparison in dynamic parking environment. "
            "Important: all selected algorithms run under the same online operational constraints "
            "(EV cloud ingestion delay/loss and window-watermark decision flow). "
            "For each scenario/repeat, one dataset and one ingestion realization are generated and replayed to every algorithm. "
            "Decision windows are anchored to EV observed-start estimates (non-oracle, jittered), not direct ground-truth plug-in timestamps. "
            "Differences are limited to each algorithm's assignment/cost strategy. "
            "EV cloud ingestion defaults follow an evidence-informed baseline (not site-calibrated). "
            "EVSE side is treated as console-known reference (no EVSE uplink delay/loss model)."
        ),
        "custom_title": "",
        "display_title": tag,
        "total_runs": int(len(raw_rows)),
        "trace_counts": {
            "event_trace_rows": int(len(event_trace_rows)),
            "cost_trace_rows": int(len(cost_trace_rows)),
            "posterior_trace_rows": int(len(posterior_trace_rows)),
            "blocked_sessions_rows": int(len(blocked_rows)),
        },
        "trace_level": str(trace_mode),
        "summary_rows": summary_rows,
        "plot_paths": plot_paths,
    }

    report_path = out_dir / "report.json"
    result["report_json"] = _to_rel_plot(report_path, plots_base)
    result["report_page"] = f"/algo-compare/report/{tag}"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(result), f, ensure_ascii=False, indent=2)

    if AUTO_PRUNE_COMPARE_REPORTS:
        try:
            prune_algorithm_compare_reports(keep_per_mode=AUTO_PRUNE_KEEP_PER_MODE)
        except Exception:
            pass

    completed_units = int(total_work_units)
    _emit_progress("done", "Completed")

    return result


def run_ieee_access_defense_bundle(
    *,
    parent_run_id: str,
    error_mode: str = DEFAULT_ERROR_MODE,
    base_seed: int,
    repeats: int,
    ev_counts: list[int],
    command_step_count: int,
    tau_values: list[int],
    algorithm_ids: list[str],
    ingestion_enabled: bool,
    ev_ingest_delay_mean_s: float,
    ev_ingest_delay_jitter_s: float,
    ev_ingest_delay_max_s: float,
    ev_ingest_loss_prob: float,
    charger_sample_values: list[int],
    ev_sample_values: list[int],
    matcher_delay_max_values_s: list[float],
    candidate_margin_values_s: list[int],
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    rid = _validate_run_id(parent_run_id)
    parent_dir = (COMPARE_DIR / rid).resolve()
    if not parent_dir.exists():
        raise FileNotFoundError(f"parent run not found: {rid}")
    plots_base = GENERATED_OUTPUT_DIR.resolve()

    base_algos = [str(a) for a in algorithm_ids if str(a) in ALGO_SPEC_BY_ID]
    if len(base_algos) == 0:
        base_algos = list(DEFAULT_ALGORITHM_IDS)
    base_algos = list(dict.fromkeys(base_algos))
    tau_main = int(tau_values[0]) if len(tau_values) > 0 else int(DEFAULT_TAU_VALUES[0])
    charger_main = int(charger_sample_values[0]) if len(charger_sample_values) > 0 else int(DEFAULT_CHARGER_SAMPLE_VALUES[0])
    evs_main = int(ev_sample_values[0]) if len(ev_sample_values) > 0 else int(DEFAULT_EV_SAMPLE_VALUES[0])
    dmax_main = float(matcher_delay_max_values_s[0]) if len(matcher_delay_max_values_s) > 0 else float(DEFAULT_MATCHER_DELAY_MAX_VALUES_S[0])
    margin_main = int(candidate_margin_values_s[0]) if len(candidate_margin_values_s) > 0 else int(DEFAULT_CANDIDATE_MARGIN_VALUES_S[0])
    repeats_small = int(max(1, min(int(repeats), 2)))
    ev_counts_main = sorted(set(int(x) for x in ev_counts))
    if len(ev_counts_main) == 0:
        ev_counts_main = list(DEFAULT_EV_COUNTS)

    def _emit(stage: str, detail: str, idx: int, total: int) -> None:
        if progress_cb is None:
            return
        pct = float(idx / max(1, total) * 100.0)
        progress_cb(
            {
                "stage": str(stage),
                "detail": str(detail),
                "completed_units": int(idx),
                "total_units": int(total),
                "progress_pct": float(max(0.0, min(100.0, pct))),
            }
        )

    def _annotate_bundle_child(child_run_id: str, role: str) -> None:
        cid = str(child_run_id).strip()
        if cid == "":
            return
        child_report = (COMPARE_DIR / cid / "report.json").resolve()
        if not child_report.exists():
            return
        try:
            with child_report.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return
            obj["bundle_parent_run_id"] = str(rid)
            obj["bundle_role"] = str(role)
            obj["bundle_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with child_report.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _load_df(run_id: str, filename: str) -> pd.DataFrame:
        path = (COMPARE_DIR / str(run_id) / str(filename)).resolve()
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _save_df(df: pd.DataFrame, path: Path) -> None:
        if df is None or df.empty:
            pd.DataFrame([]).to_csv(path, index=False, encoding="utf-8")
            return
        df.to_csv(path, index=False, encoding="utf-8")

    def _plot_metric_vs_ev(df: pd.DataFrame, *, metric_col: str, y_label: str, out_path: Path, scale: float = 1.0) -> None:
        fig, ax = plt.subplots(figsize=(10.0, 6.8))
        if df.empty or metric_col not in df.columns:
            _plot_no_data_panel(ax, x_label="EV Count", y_label=y_label, message="No data")
        else:
            drew = 0
            for aid, g in df.groupby("algorithm_id", dropna=False):
                gg = g.sort_values("ev_count")
                x = pd.to_numeric(gg["ev_count"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg[metric_col], errors="coerce").to_numpy(dtype=float) * float(scale)
                m = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(m)) <= 0:
                    continue
                ax.plot(
                    x[m],
                    y[m],
                    marker="o",
                    linewidth=3.4,
                    markersize=10.4,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    label=_paper_algo_short_label(str(aid)),
                )
                drew += 1
            if drew > 0:
                ax.set_xlabel("EV Count")
                ax.set_ylabel(str(y_label))
                ax.grid(alpha=0.30)
                _paper_ordered_legend(ax, loc="best", fontsize=max(10.0, _PAPER_LEGEND_SIZE * 0.88))
            else:
                _plot_no_data_panel(ax, x_label="EV Count", y_label=y_label, message="No finite data")
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    def _plot_highload_panel(df: pd.DataFrame, out_path: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
        metrics = [
            ("accuracy_requested_mean", "AccReq (%)", 100.0),
            ("blocked_ratio_mean", "Blocked Ratio (%)", 100.0),
            ("p90_latency_mean", "p90 Latency (s)", 1.0),
        ]
        for ax, (col, label, scale) in zip(axes, metrics):
            if df.empty or col not in df.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                continue
            for aid, g in df.groupby("algorithm_id", dropna=False):
                gg = g.sort_values("ev_count")
                x = pd.to_numeric(gg["ev_count"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg[col], errors="coerce").to_numpy(dtype=float) * float(scale)
                ax.plot(x, y, marker="o", linewidth=2.0, label=_paper_algo_short_label(str(aid)))
            ax.set_xlabel("EV Count")
            ax.set_ylabel(label)
            ax.grid(alpha=0.22)
        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    def _plot_seed_boxplot(df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(11.0, 6.8))
        if df.empty or "accuracy_requested" not in df.columns or "algorithm_id" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
        else:
            aids = sorted(set(str(x) for x in df["algorithm_id"].dropna().astype(str).tolist()))
            vals = []
            labels = []
            for aid in aids:
                arr = pd.to_numeric(df.loc[df["algorithm_id"].astype(str) == aid, "accuracy_requested"], errors="coerce").to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size <= 0:
                    continue
                vals.append(arr * 100.0)
                labels.append(_paper_algo_short_label(aid))
            if len(vals) == 0:
                ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
                ax.axis("off")
            else:
                ax.boxplot(vals, labels=labels, showfliers=False)
                ax.set_ylabel("AccReq (%)")
                ax.set_xlabel("Algorithm")
                ax.grid(axis="y", alpha=0.22)
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    def _plot_delay_misspec(df: pd.DataFrame, out_path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
        specs = [
            ("accuracy_requested_mean", "AccReq (%)", 100.0),
            ("fallback_finalized_ratio_mean", "Fallback Finalized (%)", 100.0),
        ]
        for ax, (col, yl, scale) in zip(axes, specs):
            if df.empty or col not in df.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.axis("off")
                continue
            for aid, g in df.groupby("algorithm_id", dropna=False):
                gg = g.sort_values("matcher_delay_max_s")
                x = pd.to_numeric(gg["matcher_delay_max_s"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg[col], errors="coerce").to_numpy(dtype=float) * float(scale)
                ax.plot(x, y, marker="o", linewidth=2.0, label=_paper_algo_short_label(str(aid)))
            ax.set_xlabel("Configured d_max (s)")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.22)
        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    total_stages = 4
    stage_idx = 0

    # 1) Ablation bundle
    stage_idx += 1
    _emit("defense:ablation", "Running ablation bundle", stage_idx - 1, total_stages)
    ablation_algos = [
        "single_only",
        "single_only_step_off",
        "single_only_dtw_off",
        "bayesian_windowed",
        "bayesian_windowed_memory_off",
        "bayesian_windowed_timeprior_off",
    ]
    ablation_result = run_algorithm_compare_report(
        ev_counts=list(ev_counts_main),
        repeats=int(repeats_small),
        error_mode=str(error_mode),
        base_seed=int(base_seed) ^ 0x0AB1A710,
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        command_step_count=int(command_step_count),
        tau_values=[int(tau_main)],
        algorithm_ids=ablation_algos,
        ingestion_enabled=bool(ingestion_enabled),
        ev_ingest_delay_mean_s=float(ev_ingest_delay_mean_s),
        ev_ingest_delay_jitter_s=float(ev_ingest_delay_jitter_s),
        ev_ingest_delay_max_s=float(ev_ingest_delay_max_s),
        ev_ingest_loss_prob=float(ev_ingest_loss_prob),
        charger_sample_values=[int(charger_main)],
        ev_sample_values=[int(evs_main)],
        matcher_delay_max_values_s=[float(dmax_main)],
        candidate_margin_values_s=[int(margin_main)],
        trace_level="summary",
    )
    ablation_run_id = str(ablation_result.get("run_id", "")).strip()
    _annotate_bundle_child(ablation_run_id, "ablation")
    ablation_raw_df = _load_df(ablation_run_id, "raw_runs.csv")
    ablation_summary_df = _load_df(ablation_run_id, "summary.csv")
    p_ablation_raw = parent_dir / "ablation_raw.csv"
    p_ablation_summary = parent_dir / "ablation_summary.csv"
    p_ablation_acc = parent_dir / "fig_ablation_accuracy.png"
    p_ablation_lat = parent_dir / "fig_ablation_latency.png"
    p_ablation_delta = parent_dir / "table_ablation_delta.csv"
    _save_df(ablation_raw_df, p_ablation_raw)
    _save_df(ablation_summary_df, p_ablation_summary)
    _plot_metric_vs_ev(ablation_summary_df, metric_col="accuracy_mean", y_label="Accuracy (%)", out_path=p_ablation_acc, scale=100.0)
    _plot_metric_vs_ev(ablation_summary_df, metric_col="p90_latency_mean", y_label="p90 Latency (s)", out_path=p_ablation_lat, scale=1.0)
    delta_rows: list[dict[str, Any]] = []
    if not ablation_summary_df.empty:
        fams = [
            ("A1", "single_only", ["single_only_step_off", "single_only_dtw_off"]),
            ("A3", "bayesian_windowed", ["bayesian_windowed_memory_off", "bayesian_windowed_timeprior_off"]),
        ]
        for fam, base_aid, variants in fams:
            for ev in sorted(set(pd.to_numeric(ablation_summary_df["ev_count"], errors="coerce").dropna().astype(int).tolist())):
                base_row = ablation_summary_df[
                    (ablation_summary_df["algorithm_id"].astype(str) == str(base_aid))
                    & (pd.to_numeric(ablation_summary_df["ev_count"], errors="coerce") == int(ev))
                ]
                if base_row.empty:
                    continue
                base_acc = float(pd.to_numeric(base_row["accuracy_mean"], errors="coerce").iloc[0])
                base_lat = float(pd.to_numeric(base_row["p90_latency_mean"], errors="coerce").iloc[0])
                for var_aid in variants:
                    row = ablation_summary_df[
                        (ablation_summary_df["algorithm_id"].astype(str) == str(var_aid))
                        & (pd.to_numeric(ablation_summary_df["ev_count"], errors="coerce") == int(ev))
                    ]
                    if row.empty:
                        continue
                    acc = float(pd.to_numeric(row["accuracy_mean"], errors="coerce").iloc[0])
                    lat = float(pd.to_numeric(row["p90_latency_mean"], errors="coerce").iloc[0])
                    delta_rows.append(
                        {
                            "family": str(fam),
                            "ev_count": int(ev),
                            "baseline_algorithm_id": str(base_aid),
                            "variant_algorithm_id": str(var_aid),
                            "baseline_accuracy_pct": float(base_acc * 100.0),
                            "variant_accuracy_pct": float(acc * 100.0),
                            "delta_accuracy_pp": float((acc - base_acc) * 100.0),
                            "baseline_p90_latency_s": float(base_lat),
                            "variant_p90_latency_s": float(lat),
                            "delta_p90_latency_s": float(lat - base_lat),
                        }
                    )
    _write_csv(
        p_ablation_delta,
        delta_rows,
        columns=[
            "family",
            "ev_count",
            "baseline_algorithm_id",
            "variant_algorithm_id",
            "baseline_accuracy_pct",
            "variant_accuracy_pct",
            "delta_accuracy_pp",
            "baseline_p90_latency_s",
            "variant_p90_latency_s",
            "delta_p90_latency_s",
        ],
    )
    _emit("defense:ablation", "Ablation bundle completed", stage_idx, total_stages)

    # 2) High-load bundle
    stage_idx += 1
    _emit("defense:highload", "Running high-load bundle", stage_idx - 1, total_stages)
    highload_result = run_algorithm_compare_report(
        ev_counts=[700, 1000, 1500, 2000],
        repeats=int(repeats_small),
        error_mode=str(error_mode),
        base_seed=int(base_seed) ^ 0xA11A0AD,
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        command_step_count=int(command_step_count),
        tau_values=[int(tau_main)],
        algorithm_ids=list(base_algos),
        ingestion_enabled=bool(ingestion_enabled),
        ev_ingest_delay_mean_s=float(ev_ingest_delay_mean_s),
        ev_ingest_delay_jitter_s=float(ev_ingest_delay_jitter_s),
        ev_ingest_delay_max_s=float(ev_ingest_delay_max_s),
        ev_ingest_loss_prob=float(ev_ingest_loss_prob),
        charger_sample_values=[int(charger_main)],
        ev_sample_values=[int(evs_main)],
        matcher_delay_max_values_s=[float(dmax_main)],
        candidate_margin_values_s=[int(margin_main)],
        trace_level="summary",
    )
    highload_run_id = str(highload_result.get("run_id", "")).strip()
    _annotate_bundle_child(highload_run_id, "highload")
    highload_raw_df = _load_df(highload_run_id, "raw_runs.csv")
    highload_summary_df = _load_df(highload_run_id, "summary.csv")
    p_highload_raw = parent_dir / "highload_raw.csv"
    p_highload_summary = parent_dir / "highload_summary.csv"
    p_highload_fig = parent_dir / "fig_highload_accreq_blocked_latency.png"
    _save_df(highload_raw_df, p_highload_raw)
    _save_df(highload_summary_df, p_highload_summary)
    _plot_highload_panel(highload_summary_df, p_highload_fig)
    _emit("defense:highload", "High-load bundle completed", stage_idx, total_stages)

    # 3) Seed robustness bundle
    stage_idx += 1
    _emit("defense:seed", "Running seed robustness bundle", stage_idx - 1, total_stages)
    seed_offsets = [0, 101, 211]
    seed_raw_rows: list[pd.DataFrame] = []
    seed_run_ids: list[str] = []
    for gi, off in enumerate(seed_offsets):
        rr = run_algorithm_compare_report(
            ev_counts=list(ev_counts_main),
            repeats=1,
            error_mode=str(error_mode),
            base_seed=int(base_seed + off),
            pattern_mode="current",
            pattern_assignment_mode="dynamic_session",
            command_step_count=int(command_step_count),
            tau_values=[int(tau_main)],
            algorithm_ids=list(base_algos),
            ingestion_enabled=bool(ingestion_enabled),
            ev_ingest_delay_mean_s=float(ev_ingest_delay_mean_s),
            ev_ingest_delay_jitter_s=float(ev_ingest_delay_jitter_s),
            ev_ingest_delay_max_s=float(ev_ingest_delay_max_s),
            ev_ingest_loss_prob=float(ev_ingest_loss_prob),
            charger_sample_values=[int(charger_main)],
            ev_sample_values=[int(evs_main)],
            matcher_delay_max_values_s=[float(dmax_main)],
            candidate_margin_values_s=[int(margin_main)],
            trace_level="summary",
        )
        sid = str(rr.get("run_id", "")).strip()
        if sid != "":
            seed_run_ids.append(str(sid))
            _annotate_bundle_child(sid, "seed_robustness")
        rdf = _load_df(sid, "raw_runs.csv")
        if not rdf.empty:
            rdf["seed_group"] = int(gi + 1)
            rdf["bundle_base_seed"] = int(base_seed + off)
            seed_raw_rows.append(rdf)
    if len(seed_raw_rows) > 0:
        seed_raw_df = pd.concat(seed_raw_rows, axis=0, ignore_index=True)
    else:
        seed_raw_df = pd.DataFrame()
    if not seed_raw_df.empty:
        grp = seed_raw_df.groupby(["seed_group", "bundle_base_seed", "algorithm_id"], dropna=False)
        seed_summary_df = grp.agg(
            accuracy_requested_mean=("accuracy_requested", "mean"),
            blocked_ratio_mean=("blocked_ratio", "mean"),
            p90_latency_mean_s=("p90_latency_s", "mean"),
            samples=("algorithm_id", "size"),
        ).reset_index()
    else:
        seed_summary_df = pd.DataFrame(
            columns=[
                "seed_group",
                "bundle_base_seed",
                "algorithm_id",
                "accuracy_requested_mean",
                "blocked_ratio_mean",
                "p90_latency_mean_s",
                "samples",
            ]
        )
    p_seed_summary = parent_dir / "seed_robustness_summary.csv"
    p_seed_fig = parent_dir / "fig_seed_robustness_boxplot.png"
    _save_df(seed_summary_df, p_seed_summary)
    _plot_seed_boxplot(seed_raw_df, p_seed_fig)
    _emit("defense:seed", "Seed robustness bundle completed", stage_idx, total_stages)

    # 4) Delay mis-spec bundle
    stage_idx += 1
    _emit("defense:delay_misspec", "Running delay mis-spec bundle", stage_idx - 1, total_stages)
    delay_actual = float(max(0.1, float(ev_ingest_delay_max_s)))
    delay_sweep = sorted(set(float(round(x, 3)) for x in [0.25 * delay_actual, 0.50 * delay_actual, delay_actual, 1.50 * delay_actual, 2.0 * delay_actual]))
    delay_result = run_algorithm_compare_report(
        ev_counts=[int(max(ev_counts_main))],
        repeats=int(repeats_small),
        error_mode=str(error_mode),
        base_seed=int(base_seed) ^ 0xD31A0001,
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        command_step_count=int(command_step_count),
        tau_values=[int(tau_main)],
        algorithm_ids=list(base_algos),
        ingestion_enabled=bool(ingestion_enabled),
        ev_ingest_delay_mean_s=float(ev_ingest_delay_mean_s),
        ev_ingest_delay_jitter_s=float(ev_ingest_delay_jitter_s),
        ev_ingest_delay_max_s=float(ev_ingest_delay_max_s),
        ev_ingest_loss_prob=float(ev_ingest_loss_prob),
        charger_sample_values=[int(charger_main)],
        ev_sample_values=[int(evs_main)],
        matcher_delay_max_values_s=[float(x) for x in delay_sweep],
        candidate_margin_values_s=[int(margin_main)],
        trace_level="summary",
    )
    delay_run_id = str(delay_result.get("run_id", "")).strip()
    _annotate_bundle_child(delay_run_id, "delay_misspec")
    delay_summary_df = _load_df(delay_run_id, "summary.csv")
    p_delay_summary = parent_dir / "delay_misspec_summary.csv"
    p_delay_fig = parent_dir / "fig_delay_misspec_tradeoff.png"
    _save_df(delay_summary_df, p_delay_summary)
    _plot_delay_misspec(delay_summary_df, p_delay_fig)
    _emit("defense:delay_misspec", "Delay mis-spec bundle completed", stage_idx, total_stages)

    manifest = {
        "parent_run_id": str(rid),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bundle_run_ids": {
            "ablation": str(ablation_run_id),
            "highload": str(highload_run_id),
            "seed_robustness": [str(x) for x in seed_run_ids],
            "delay_misspec": str(delay_run_id),
        },
        "artifacts": {
            "ablation_raw_csv": _to_rel_plot(p_ablation_raw, plots_base),
            "ablation_summary_csv": _to_rel_plot(p_ablation_summary, plots_base),
            "fig_ablation_accuracy": _to_rel_plot(p_ablation_acc, plots_base),
            "fig_ablation_latency": _to_rel_plot(p_ablation_lat, plots_base),
            "table_ablation_delta_csv": _to_rel_plot(p_ablation_delta, plots_base),
            "highload_raw_csv": _to_rel_plot(p_highload_raw, plots_base),
            "highload_summary_csv": _to_rel_plot(p_highload_summary, plots_base),
            "fig_highload_accreq_blocked_latency": _to_rel_plot(p_highload_fig, plots_base),
            "seed_robustness_summary_csv": _to_rel_plot(p_seed_summary, plots_base),
            "fig_seed_robustness_boxplot": _to_rel_plot(p_seed_fig, plots_base),
            "delay_misspec_summary_csv": _to_rel_plot(p_delay_summary, plots_base),
            "fig_delay_misspec_tradeoff": _to_rel_plot(p_delay_fig, plots_base),
        },
    }
    manifest_path = parent_dir / "reviewer_defense_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, ensure_ascii=False, indent=2)
    manifest["manifest_json"] = _to_rel_plot(manifest_path, plots_base)

    report_path = parent_dir / "report.json"
    if report_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                obj = {}
            obj["reviewer_defense_bundle"] = dict(manifest)
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return manifest


def run_paper_ablation_suite(
    *,
    ev_counts: list[int] | None = None,
    repeats: int = 3,
    error_mode: str = DEFAULT_ERROR_MODE,
    base_seed: int | None = None,
    command_step_count: int = DEFAULT_COMMAND_STEP_COUNT,
    tau_values: list[int] | None = None,
    ingestion_enabled: bool = True,
    ev_ingest_delay_mean_s: float = float(algo.EV_INGEST_DELAY_MEAN_S),
    ev_ingest_delay_jitter_s: float = float(algo.EV_INGEST_DELAY_JITTER_S),
    ev_ingest_delay_max_s: float = float(algo.EV_INGEST_DELAY_MAX_S),
    ev_ingest_loss_prob: float = float(algo.EV_INGEST_LOSS_PROB),
    charger_sample_values: list[int] | None = None,
    ev_sample_values: list[int] | None = None,
    matcher_delay_max_values_s: list[float] | None = None,
    candidate_margin_values_s: list[int] | None = None,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ev_counts_use = list(ev_counts) if isinstance(ev_counts, list) and len(ev_counts) > 0 else [100, 300, 500]
    repeats_use = int(max(1, int(repeats)))
    tau_use = list(tau_values) if isinstance(tau_values, list) and len(tau_values) > 0 else [60]
    charger_use = list(charger_sample_values) if isinstance(charger_sample_values, list) and len(charger_sample_values) > 0 else [5]
    ev_sample_use = list(ev_sample_values) if isinstance(ev_sample_values, list) and len(ev_sample_values) > 0 else [30]
    dmax_use = list(matcher_delay_max_values_s) if isinstance(matcher_delay_max_values_s, list) and len(matcher_delay_max_values_s) > 0 else [float(algo.EV_INGEST_DELAY_MAX_S)]
    margin_use = list(candidate_margin_values_s) if isinstance(candidate_margin_values_s, list) and len(candidate_margin_values_s) > 0 else [int(getattr(algo, "CANDIDATE_TIME_MARGIN_S", 120))]

    def _emit(stage: str, detail: str, pct: float):
        if progress_cb is None:
            return
        try:
            progress_cb(
                {
                    "stage": str(stage),
                    "detail": str(detail),
                    "progress_pct": float(max(0.0, min(100.0, pct))),
                }
            )
        except Exception:
            return

    _emit("ablation:start", "Starting paper ablation suite", 1.0)

    def _progress_bridge(info: dict[str, Any]) -> None:
        pct_raw = float(info.get("progress_pct", 0.0))
        pct = 5.0 + 0.78 * max(0.0, min(100.0, pct_raw))
        _emit(str(info.get("stage", "ablation:simulate")), str(info.get("detail", "")), pct)

    compare_result = run_algorithm_compare_report(
        ev_counts=list(ev_counts_use),
        repeats=int(repeats_use),
        error_mode=str(error_mode),
        base_seed=base_seed,
        pattern_mode="current",
        pattern_assignment_mode="dynamic_session",
        command_step_count=int(command_step_count),
        tau_values=list(tau_use),
        algorithm_ids=list(PAPER_ABLATION_ALGORITHM_IDS),
        ingestion_enabled=bool(ingestion_enabled),
        ev_ingest_delay_mean_s=float(ev_ingest_delay_mean_s),
        ev_ingest_delay_jitter_s=float(ev_ingest_delay_jitter_s),
        ev_ingest_delay_max_s=float(ev_ingest_delay_max_s),
        ev_ingest_loss_prob=float(ev_ingest_loss_prob),
        charger_sample_values=list(charger_use),
        ev_sample_values=list(ev_sample_use),
        matcher_delay_max_values_s=[float(x) for x in dmax_use],
        candidate_margin_values_s=[int(x) for x in margin_use],
        trace_level="summary",
        progress_cb=_progress_bridge,
    )

    rid = _validate_run_id(str(compare_result.get("run_id", "")).strip())
    run_dir = (COMPARE_DIR / rid).resolve()
    plots_base = GENERATED_OUTPUT_DIR.resolve()
    out_dir = run_dir / "diagnostic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    _emit("ablation:post", "Generating family-wise ablation figures", 85.0)
    report = load_algorithm_compare_report(rid)
    summary_df = _paper_summary_df(report)
    if summary_df.empty:
        raise ValueError("ablation suite summary is empty")

    if "algorithm_id" not in summary_df.columns:
        summary_df["algorithm_id"] = ""
    summary_df["algorithm_id"] = summary_df["algorithm_id"].astype(str)
    summary_df["ev_count"] = pd.to_numeric(summary_df.get("ev_count"), errors="coerce")
    summary_df["accuracy_mean"] = pd.to_numeric(summary_df.get("accuracy_mean"), errors="coerce")
    summary_df["p90_latency_mean"] = pd.to_numeric(summary_df.get("p90_latency_mean"), errors="coerce")

    suite_df = summary_df[summary_df["algorithm_id"].isin(set(PAPER_ABLATION_ALGORITHM_IDS))].copy()
    p_summary = run_dir / "ablation_suite_summary.csv"
    suite_df.to_csv(p_summary, index=False, encoding="utf-8")

    def _plot_family_metric(
        *,
        family: str,
        algo_ids: list[str],
        metric_col: str,
        y_label: str,
        scale: float,
        out_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        part = suite_df[suite_df["algorithm_id"].isin(set(str(x) for x in algo_ids))].copy()
        if part.empty:
            _plot_no_data_panel(ax, x_label="EV Count", y_label=y_label, message="No matching rows")
        else:
            drew = 0
            for aid in algo_ids:
                g = part[part["algorithm_id"].astype(str) == str(aid)].copy()
                if g.empty:
                    continue
                gg = (
                    g.groupby("ev_count", dropna=True)[metric_col]
                    .mean()
                    .reset_index()
                    .sort_values("ev_count")
                )
                x = pd.to_numeric(gg.get("ev_count"), errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg.get(metric_col), errors="coerce").to_numpy(dtype=float) * float(scale)
                m = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(m)) <= 0:
                    continue
                ax.plot(
                    x[m],
                    y[m],
                    marker="o",
                    linewidth=3.4,
                    markersize=10.4,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    label=_paper_algo_short_label(str(aid)),
                )
                drew += 1
            if drew > 0:
                ax.set_xlabel("EV Count")
                ax.set_ylabel(str(y_label))
                ax.grid(alpha=0.30)
                _paper_ordered_legend(ax, loc="best", fontsize=max(10.0, _PAPER_LEGEND_SIZE * 0.88))
                xs = pd.to_numeric(part.get("ev_count"), errors="coerce")
                xs = xs[np.isfinite(xs)]
                if len(xs) > 0 and int(pd.Series(xs).nunique()) <= 8:
                    ax.set_xticks(sorted(set(xs.tolist())))
            else:
                _plot_no_data_panel(ax, x_label="EV Count", y_label=y_label, message="No finite data")
        ax.set_title(f"{family} Ablation: {y_label}")
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    plot_paths: dict[str, str] = {}
    for family, ids in PAPER_ABLATION_FAMILIES.items():
        fkey = str(family).lower()
        p_acc = out_dir / f"fig_ablation_{fkey}_accuracy.png"
        p_lat = out_dir / f"fig_ablation_{fkey}_latency.png"
        _plot_family_metric(
            family=str(family),
            algo_ids=list(ids),
            metric_col="accuracy_mean",
            y_label="Accuracy (%)",
            scale=100.0,
            out_path=p_acc,
        )
        _plot_family_metric(
            family=str(family),
            algo_ids=list(ids),
            metric_col="p90_latency_mean",
            y_label="p90 Latency (s)",
            scale=1.0,
            out_path=p_lat,
        )
        plot_paths[f"fig_ablation_{fkey}_accuracy"] = _to_rel_plot(p_acc, plots_base)
        plot_paths[f"fig_ablation_{fkey}_latency"] = _to_rel_plot(p_lat, plots_base)

    manifest = {
        "manifest_type": "paper_ablation_suite",
        "run_id": str(rid),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "families": {str(k): [str(x) for x in v] for k, v in PAPER_ABLATION_FAMILIES.items()},
        "summary_csv": _to_rel_plot(p_summary, plots_base),
        "plot_paths": dict(plot_paths),
    }
    p_manifest = run_dir / "ablation_suite_manifest.json"
    with p_manifest.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, ensure_ascii=False, indent=2)
    manifest["manifest_json"] = _to_rel_plot(p_manifest, plots_base)

    report_path = run_dir / "report.json"
    if report_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                obj = {}
            obj["paper_ablation_suite"] = dict(manifest)
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _emit("ablation:done", "Paper ablation suite completed", 100.0)
    return {
        "run_id": str(rid),
        "report_page": f"/algo-compare/report/{rid}",
        "paper_figs_url": "/ieee-final-suite",
        "summary_csv": str(manifest.get("summary_csv", "")),
        "plot_paths": dict(plot_paths),
        "manifest_json": str(manifest.get("manifest_json", "")),
        "algorithms": [str(x) for x in PAPER_ABLATION_ALGORITHM_IDS],
        "families": {str(k): [str(x) for x in v] for k, v in PAPER_ABLATION_FAMILIES.items()},
        "note": "A1/A2/A3 family-wise ablation suite completed with standalone paper figures.",
    }


def regenerate_algorithm_compare_plots(
    run_id: str,
    *,
    error_mode: str | None = None,
) -> dict[str, Any]:
    rid = _validate_run_id(run_id)
    report_path = COMPARE_DIR / rid / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"report not found: {rid}")

    report = load_algorithm_compare_report(rid)
    mode = _validate_error_mode(
        str(error_mode).strip().lower()
        if error_mode is not None
        else str(report.get("error_mode", DEFAULT_ERROR_MODE)).strip().lower()
    )

    summary_rows_raw = report.get("summary_rows")
    summary_rows: list[dict[str, Any]] = []
    if isinstance(summary_rows_raw, list) and len(summary_rows_raw) > 0:
        summary_rows = [dict(r) for r in summary_rows_raw if isinstance(r, dict)]
    else:
        summary_df = _paper_summary_df(report)
        if summary_df.empty:
            raise ValueError("summary data not found for this run")
        summary_rows = [dict(r) for r in summary_df.to_dict(orient="records")]
    if len(summary_rows) == 0:
        raise ValueError("summary data not found for this run")

    for row in summary_rows:
        aid = str(row.get("algorithm_id", "")).strip()
        if aid == "":
            continue
        meta = _algo_meta_for_id(aid)
        row["algorithm_label"] = str(meta["label"])
        row["is_current"] = bool(meta["is_current"])

    ev_counts_from_rows: list[int] = []
    tau_values_from_rows: list[int] = []
    pattern_modes_from_rows: list[str] = []
    algo_ids_from_rows: list[str] = []
    for row in summary_rows:
        try:
            ev = int(float(row.get("ev_count", 0)))
            if ev > 0 and ev not in ev_counts_from_rows:
                ev_counts_from_rows.append(ev)
        except Exception:
            pass
        try:
            tau = int(float(row.get("tau_s", 0)))
            if tau > 0 and tau not in tau_values_from_rows:
                tau_values_from_rows.append(tau)
        except Exception:
            pass
        pmode = str(row.get("pattern_mode", "")).strip().lower()
        if pmode != "" and pmode != "nan" and pmode not in pattern_modes_from_rows:
            pattern_modes_from_rows.append(pmode)
        aid = str(row.get("algorithm_id", "")).strip()
        if aid != "" and aid not in algo_ids_from_rows:
            algo_ids_from_rows.append(aid)

    ev_counts = ev_counts_from_rows
    if len(ev_counts) == 0:
        raw = report.get("ev_counts", [])
        if isinstance(raw, list):
            for v in raw:
                try:
                    ev = int(v)
                except Exception:
                    continue
                if ev > 0 and ev not in ev_counts:
                    ev_counts.append(ev)
    if len(ev_counts) == 0:
        raise ValueError("ev_counts not found in this run")
    ev_counts = _validate_ev_counts(ev_counts)

    tau_values = tau_values_from_rows
    if len(tau_values) == 0:
        raw = report.get("tau_values", [])
        if isinstance(raw, list):
            for v in raw:
                try:
                    tau = int(v)
                except Exception:
                    continue
                if tau > 0 and tau not in tau_values:
                    tau_values.append(tau)
    if len(tau_values) == 0:
        tau_values = list(DEFAULT_TAU_VALUES)
    tau_values = _validate_tau_values(tau_values)

    pattern_modes = pattern_modes_from_rows
    if len(pattern_modes) == 0:
        raw_modes = report.get("pattern_modes", [])
        if isinstance(raw_modes, list):
            for m in raw_modes:
                mode_name = str(m).strip().lower()
                if mode_name in VALID_PATTERN_MODES and mode_name not in pattern_modes:
                    pattern_modes.append(mode_name)
        if len(pattern_modes) == 0:
            mode_name = str(report.get("pattern_mode", DEFAULT_PATTERN_MODE)).strip().lower()
            if mode_name in VALID_PATTERN_MODES:
                pattern_modes = [mode_name]
    if len(pattern_modes) == 0:
        pattern_modes = [DEFAULT_PATTERN_MODE]

    algo_ids: list[str] = []
    algos_raw = report.get("algorithms", [])
    if isinstance(algos_raw, list):
        for a in algos_raw:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id", "")).strip()
            if aid != "" and aid not in algo_ids:
                algo_ids.append(aid)
    if len(algo_ids) == 0:
        algo_ids = list(algo_ids_from_rows)
    if len(algo_ids) == 0:
        raise ValueError("algorithm_ids not found in this run")

    algo_specs: list[AlgoSpec] = []
    for aid in algo_ids:
        spec = ALGO_SPEC_BY_ID.get(str(aid))
        if spec is not None:
            algo_specs.append(spec)
            continue
        meta = _algo_meta_for_id(str(aid))
        algo_specs.append(
            AlgoSpec(
                id=str(meta["id"]),
                label=str(meta["label"]),
                description=str(meta.get("description", "")),
                is_current=bool(meta.get("is_current", False)),
            )
        )

    out_dir = COMPARE_DIR / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_base = GENERATED_OUTPUT_DIR.resolve()

    p_acc = out_dir / "accuracy_vs_ev.png"
    p_lat = out_dir / "p90_latency_vs_ev.png"
    p_cost = out_dir / "avg_cost_vs_ev.png"
    p_runtime = out_dir / "runtime_vs_ev.png"
    p_summary_csv = out_dir / "summary.csv"
    p_pres_acc = out_dir / "presentation_accuracy_vs_ev_ci95.csv"
    p_pres_lat = out_dir / "presentation_p90_latency_vs_ev_ci95.csv"
    p_pres_runtime = out_dir / "presentation_runtime_vs_ev_ci95.csv"
    p_pres_scatter = out_dir / "presentation_accuracy_runtime_scatter_mean.csv"

    _plot_metric_lines(
        summary_rows,
        ev_counts,
        tau_values,
        pattern_modes,
        algo_specs,
        metric_prefix="accuracy",
        y_label="Accuracy (%)",
        title=f"Algorithm Accuracy vs EV Count (EVSE={FIXED_EVSE_COUNT})",
        out_path=p_acc,
        error_mode=mode,
        scale=100.0,
    )
    _plot_metric_lines(
        summary_rows,
        ev_counts,
        tau_values,
        pattern_modes,
        algo_specs,
        metric_prefix="p90_latency",
        y_label="P90 Latency (s)",
        title=f"Algorithm P90 Latency vs EV Count (EVSE={FIXED_EVSE_COUNT})",
        out_path=p_lat,
        error_mode=mode,
        scale=1.0,
    )
    _plot_metric_lines(
        summary_rows,
        ev_counts,
        tau_values,
        pattern_modes,
        algo_specs,
        metric_prefix="avg_cost",
        y_label="Average Matching Cost",
        title=f"Algorithm Average Cost vs EV Count (EVSE={FIXED_EVSE_COUNT})",
        out_path=p_cost,
        error_mode=mode,
        scale=1.0,
    )
    _plot_metric_lines(
        summary_rows,
        ev_counts,
        tau_values,
        pattern_modes,
        algo_specs,
        metric_prefix="assignment_runtime",
        y_label="Assignment Runtime (s)",
        title=f"Algorithm Runtime vs EV Count (EVSE={FIXED_EVSE_COUNT})",
        out_path=p_runtime,
        error_mode=mode,
        scale=1.0,
    )

    _write_csv(p_summary_csv, summary_rows)
    presentation_rows = _build_presentation_graph_rows(summary_rows)
    _write_csv(p_pres_acc, presentation_rows["accuracy_vs_ev_ci95"])
    _write_csv(p_pres_lat, presentation_rows["p90_latency_vs_ev_ci95"])
    _write_csv(p_pres_runtime, presentation_rows["runtime_vs_ev_ci95"])
    _write_csv(p_pres_scatter, presentation_rows["accuracy_runtime_scatter_mean"])

    old_plot_paths = report.get("plot_paths")
    plot_paths: dict[str, Any] = dict(old_plot_paths) if isinstance(old_plot_paths, dict) else {}
    plot_paths.update(
        {
            "accuracy": _to_rel_plot(p_acc, plots_base),
            "p90_latency": _to_rel_plot(p_lat, plots_base),
            "avg_cost": _to_rel_plot(p_cost, plots_base),
            "runtime": _to_rel_plot(p_runtime, plots_base),
            "summary_csv": _to_rel_plot(p_summary_csv, plots_base),
            "presentation_accuracy_csv": _to_rel_plot(p_pres_acc, plots_base),
            "presentation_p90_latency_csv": _to_rel_plot(p_pres_lat, plots_base),
            "presentation_runtime_csv": _to_rel_plot(p_pres_runtime, plots_base),
            "presentation_acc_runtime_scatter_csv": _to_rel_plot(p_pres_scatter, plots_base),
        }
    )

    report["error_mode"] = str(mode)
    report["summary_rows"] = summary_rows
    report["plot_paths"] = plot_paths
    report["plot_regenerated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report["report_json"] = _to_rel_plot(report_path, plots_base)
    report["report_page"] = f"/algo-compare/report/{rid}"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)

    return load_algorithm_compare_report(rid)


def load_algorithm_compare_report(run_id: str) -> dict[str, Any]:
    rid = _validate_run_id(run_id)
    path = COMPARE_DIR / rid / "report.json"
    if not path.exists():
        raise FileNotFoundError(f"report not found: {rid}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("invalid report format")

    algorithms_raw = obj.get("algorithms")
    algorithms: list[dict[str, Any]] = []
    if isinstance(algorithms_raw, list) and len(algorithms_raw) > 0:
        for a in algorithms_raw:
            if not isinstance(a, dict):
                continue
            aid = str(a.get("id", "")).strip()
            if aid == "":
                continue
            meta = _algo_meta_for_id(aid)
            algorithms.append(
                {
                    "id": str(meta["id"]),
                    "label": str(meta["label"]),
                    "description": str(meta["description"]),
                    "is_current": bool(meta["is_current"]),
                }
            )
    if len(algorithms) == 0:
        algorithms = [
            {
                "id": s.id,
                "label": s.label,
                "description": s.description,
                "is_current": bool(s.is_current),
            }
            for s in _validate_algorithm_ids(None)
        ]
    obj["algorithms"] = algorithms
    obj["algorithm_ids"] = [str(a.get("id", "")).strip() for a in algorithms if isinstance(a, dict)]

    summary_rows = obj.get("summary_rows")
    if isinstance(summary_rows, list):
        for r in summary_rows:
            if not isinstance(r, dict):
                continue
            aid = str(r.get("algorithm_id", "")).strip()
            if aid == "":
                continue
            meta = _algo_meta_for_id(aid)
            r["algorithm_label"] = str(meta["label"])
            r["is_current"] = bool(meta["is_current"])
        obj["summary_rows"] = summary_rows
    pattern_assignment_mode = str(obj.get("pattern_assignment_mode", DEFAULT_PATTERN_ASSIGNMENT_MODE)).strip().lower()
    if pattern_assignment_mode not in VALID_PATTERN_ASSIGNMENT_MODES:
        pattern_assignment_mode = DEFAULT_PATTERN_ASSIGNMENT_MODE
    obj["pattern_assignment_mode"] = pattern_assignment_mode
    obj["pattern_assignment_mode_label"] = PATTERN_ASSIGNMENT_MODE_LABELS.get(pattern_assignment_mode, pattern_assignment_mode)

    custom_title = _normalize_custom_title(obj.get("custom_title", ""))
    obj["custom_title"] = custom_title
    obj["display_title"] = custom_title if custom_title else str(obj.get("run_id", rid))
    return obj


def _paper_summary_df(report: dict[str, Any]) -> pd.DataFrame:
    rows = report.get("summary_rows")
    if isinstance(rows, list) and len(rows) > 0:
        return pd.DataFrame(rows)

    rel = str(((report.get("plot_paths") or {}).get("summary_csv") or "")).strip()
    if rel == "":
        return pd.DataFrame()
    p = (GENERATED_OUTPUT_DIR / rel).resolve()
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _to_finite_float(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return float("nan")
    return float(num) if np.isfinite(num) else float("nan")


def _extract_results_knobs(report: dict[str, Any]) -> dict[str, float]:
    online = report.get("online_operational_conditions")
    if not isinstance(online, dict):
        online = {}
    ingestion = online.get("ingestion_model")
    if not isinstance(ingestion, dict):
        ingestion = {}
    ingest_ev = ingestion.get("ev")
    if not isinstance(ingest_ev, dict):
        ingest_ev = {}

    return {
        "delay_max_s": _to_finite_float(ingest_ev.get("delay_max_s")),
        "loss_prob": _to_finite_float(ingest_ev.get("loss_prob")),
        "candidate_margin_s": _to_finite_float(online.get("candidate_time_margin_s")),
        "sensor_noise_std_a": _to_finite_float(online.get("ev_sensor_noise_std_a")),
        "sensor_bias_std_a": _to_finite_float(online.get("ev_sensor_bias_std_a")),
    }


def _report_result_signature(report: dict[str, Any]) -> tuple[Any, ...]:
    def _int_list(v: Any) -> tuple[int, ...]:
        if not isinstance(v, list):
            return tuple()
        out: list[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return tuple(sorted(set(out)))

    mode = str(report.get("pattern_assignment_mode", DEFAULT_PATTERN_ASSIGNMENT_MODE)).strip().lower()
    if mode not in VALID_PATTERN_ASSIGNMENT_MODES:
        mode = DEFAULT_PATTERN_ASSIGNMENT_MODE

    pattern_mode = str(report.get("pattern_mode", DEFAULT_PATTERN_MODE)).strip().lower()
    if pattern_mode not in VALID_PATTERN_MODES:
        pattern_mode = DEFAULT_PATTERN_MODE

    algo_ids_raw = report.get("algorithm_ids")
    algo_ids: list[str] = []
    if isinstance(algo_ids_raw, list):
        for x in algo_ids_raw:
            sid = str(x).strip()
            if sid != "":
                algo_ids.append(sid)
    if len(algo_ids) == 0:
        algos = report.get("algorithms")
        if isinstance(algos, list):
            for a in algos:
                if not isinstance(a, dict):
                    continue
                sid = str(a.get("id", "")).strip()
                if sid != "":
                    algo_ids.append(sid)

    return (
        mode,
        pattern_mode,
        int(report.get("command_step_count", DEFAULT_COMMAND_STEP_COUNT)),
        int(report.get("pattern_unique_count", DEFAULT_PATTERN_UNIQUE_COUNT)),
        _int_list(report.get("tau_values", [])),
        _int_list(report.get("ev_counts", [])),
        tuple(sorted(set(algo_ids))),
    )


def _collect_results_sensitivity_rows(base_report: dict[str, Any]) -> pd.DataFrame:
    base_sig = _report_result_signature(base_report)
    rows: list[dict[str, Any]] = []
    if not COMPARE_DIR.exists():
        return pd.DataFrame()

    dirs = sorted([p for p in COMPARE_DIR.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in dirs:
        path = d / "report.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                rep = json.load(f)
        except Exception:
            continue
        if _report_result_signature(rep) != base_sig:
            continue

        sdf = _paper_summary_df(rep)
        if sdf.empty:
            continue
        knobs = _extract_results_knobs(rep)
        run_id = str(rep.get("run_id", d.name)).strip() or str(d.name)
        for _, rr in sdf.iterrows():
            rows.append(
                {
                    "run_id": run_id,
                    "algorithm_id": str(rr.get("algorithm_id", "")).strip(),
                    "ev_count": _to_finite_float(rr.get("ev_count")),
                    "accuracy_mean": _to_finite_float(rr.get("accuracy_mean")),
                    "accuracy_std": _to_finite_float(rr.get("accuracy_std")),
                    "accuracy_ci95": _to_finite_float(rr.get("accuracy_ci95")),
                    "p90_latency_mean": _to_finite_float(rr.get("p90_latency_mean")),
                    "assignment_runtime_mean": _to_finite_float(rr.get("assignment_runtime_mean")),
                    **knobs,
                }
            )
    return pd.DataFrame(rows)


def _algo_ordered_ids(ids: list[str]) -> list[str]:
    present = [str(x).strip() for x in ids if str(x).strip() != ""]
    known = [s.id for s in ALGO_SPECS if s.id in present]
    extra = sorted([x for x in present if x not in known])
    return known + extra


def _filter_vary_only(
    df: pd.DataFrame,
    *,
    vary_col: str,
    base_knobs: dict[str, float],
) -> pd.DataFrame:
    out = df.copy()
    for col, base_val in base_knobs.items():
        if str(col) == str(vary_col):
            continue
        if col not in out.columns:
            continue
        if not np.isfinite(float(base_val)):
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        tol = max(1e-9, abs(float(base_val)) * 1e-6)
        out = out[np.isfinite(vals) & (np.abs(vals - float(base_val)) <= tol)]
    return out


def _plot_no_data_panel(ax: Any, *, x_label: str, y_label: str, message: str) -> None:
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.text(
        0.5,
        0.5,
        str(message),
        ha="center",
        va="center",
        fontsize=13.0,
        color="#475569",
        transform=ax.transAxes,
    )


def _paper_box(
    ax: Any,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str = "#1f2a44",
    title_fs: float = 15.5,
    body_fs: float = 13.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        alpha=0.98,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + h - 0.03,
        str(title),
        ha="left",
        va="top",
        fontsize=float(title_fs),
        fontweight="semibold",
        color="#102239",
    )
    ax.text(
        x + 0.012,
        y + h - 0.075,
        str(body),
        ha="left",
        va="top",
        fontsize=float(body_fs),
        color="#22354f",
        linespacing=1.26,
    )


def _paper_arrow(
    ax: Any,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str = "",
    color: str = "#334155",
    style: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", linewidth=1.8, color=color, linestyle=style),
    )
    if str(label).strip() != "":
        ax.text((x0 + x1) * 0.5, (y0 + y1) * 0.5 + 0.018, str(label), ha="center", va="bottom", fontsize=12.0, color=color)


def _apply_ieee_figure_style(
    fig: Any,
    *,
    title_size: float = 12.5,
    label_size: float = 10.8,
    tick_size: float = 9.8,
    legend_size: float | None = 9.2,
) -> None:
    def _axis_legends(axis: Any) -> list[Legend]:
        objs: list[Legend] = []
        lg = axis.get_legend()
        if isinstance(lg, Legend):
            objs.append(lg)
        for artist in getattr(axis, "artists", []) or []:
            if isinstance(artist, Legend) and artist not in objs:
                objs.append(artist)
        return objs

    fig.patch.set_facecolor("white")
    for tx in list(getattr(fig, "texts", []) or []):
        try:
            tx.set_fontfamily(_IEEE_SERIF_FALLBACKS)
        except Exception:
            pass
    for ax in fig.get_axes():
        try:
            ax.set_facecolor("white")
        except Exception:
            pass
        try:
            ax.grid(True, which="major", axis="both", color="#BFBFBF", linewidth=0.5, linestyle=":")
        except Exception:
            pass
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)
        except Exception:
            pass
        for tx in list(getattr(ax, "texts", []) or []):
            try:
                tx.set_fontfamily(_IEEE_SERIF_FALLBACKS)
            except Exception:
                pass
        try:
            ax.title.set_fontfamily(_IEEE_SERIF_FALLBACKS)
            ax.title.set_fontsize(float(title_size))
        except Exception:
            pass
        try:
            ax.xaxis.label.set_fontfamily(_IEEE_SERIF_FALLBACKS)
            ax.xaxis.label.set_fontsize(float(label_size))
            ax.yaxis.label.set_fontfamily(_IEEE_SERIF_FALLBACKS)
            ax.yaxis.label.set_fontsize(float(label_size))
        except Exception:
            pass
        for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            try:
                lb.set_fontfamily(_IEEE_SERIF_FALLBACKS)
                lb.set_fontsize(float(tick_size))
            except Exception:
                pass
        for leg_obj in _axis_legends(ax):
            try:
                title = leg_obj.get_title()
                if title is not None:
                    title.set_fontfamily(_IEEE_SERIF_FALLBACKS)
                    title.set_fontweight("normal")
                    if legend_size is not None:
                        title.set_fontsize(float(legend_size))
            except Exception:
                pass
            for tx in leg_obj.get_texts():
                try:
                    tx.set_fontfamily(_IEEE_SERIF_FALLBACKS)
                    tx.set_fontweight("normal")
                    if legend_size is not None:
                        tx.set_fontsize(float(legend_size))
                except Exception:
                    pass
            try:
                fr = leg_obj.get_frame()
                fr.set_alpha(1.0)
                fr.set_edgecolor("#808080")
                fr.set_linewidth(0.5)
            except Exception:
                pass
            _handles = list(getattr(leg_obj, "legend_handles", []) or [])
            if len(_handles) == 0:
                _handles = list(getattr(leg_obj, "legendHandles", []) or [])
            for h in _handles:
                try:
                    if hasattr(h, "set_markersize"):
                        h.set_markersize(float(_PAPER_LEGEND_MARKER_SIZE))
                except Exception:
                    pass


def _apply_paper_style(fig: Any) -> None:
    _apply_ieee_figure_style(
        fig,
        title_size=_PAPER_TITLE_SIZE,
        label_size=_PAPER_LABEL_SIZE,
        tick_size=_PAPER_TICK_SIZE,
        legend_size=_PAPER_LEGEND_SIZE,
    )


def _apply_paper_style_scaled(
    fig: Any,
    *,
    font_scale: float = 1.0,
    override_legend_size: bool = True,
) -> None:
    scale = float(font_scale)
    if not np.isfinite(scale):
        scale = 1.0
    scale = float(np.clip(scale, 0.6, 2.5))
    legend_size_value: float | None = (_PAPER_LEGEND_SIZE * scale) if bool(override_legend_size) else None
    _apply_ieee_figure_style(
        fig,
        title_size=_PAPER_TITLE_SIZE * scale,
        label_size=_PAPER_LABEL_SIZE * scale,
        tick_size=_PAPER_TICK_SIZE * scale,
        legend_size=legend_size_value,
    )


def _legend_algo_rank(label: str) -> tuple[int, str]:
    text = str(label).strip()
    m = re.match(r"^(?:A)?(\d+)\)", text)
    if m is not None:
        try:
            return (int(m.group(1)), text)
        except Exception:
            pass
    for i, spec in enumerate(ALGO_SPECS):
        if text.startswith(str(spec.label)):
            return (int(i), text)
    return (10_000, text)


def _paper_ordered_legend(
    ax: Any,
    *,
    loc: str = "best",
    fontsize: float = _PAPER_LEGEND_SIZE,
    title: str | None = None,
    ncol: int | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) == 0:
        return
    dedup: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        key = str(l)
        if key == "_nolegend_":
            continue
        if key not in dedup:
            dedup[key] = h
    items = sorted([(h, l) for l, h in dedup.items()], key=lambda x: _legend_algo_rank(str(x[1])))
    hh = [x[0] for x in items]
    ll = [str(x[1]) for x in items]
    kwargs: dict[str, Any] = {"loc": str(loc), "fontsize": float(fontsize)}
    if title is not None:
        kwargs["title"] = str(title)
    if ncol is not None:
        kwargs["ncol"] = int(ncol)
    ax.legend(hh, ll, **kwargs)


def _runtime_tick_formatter() -> FuncFormatter:
    def _fmt(y: float, _pos: int) -> str:
        if not np.isfinite(y):
            return ""
        yy = float(y)
        if abs(yy) >= 100.0:
            return f"{yy:.0f}"
        if abs(yy) >= 10.0:
            return f"{yy:.1f}".rstrip("0").rstrip(".")
        if abs(yy) >= 1.0:
            return f"{yy:.2f}".rstrip("0").rstrip(".")
        return f"{yy:.3f}".rstrip("0").rstrip(".")

    return FuncFormatter(_fmt)


def _apply_runtime_axis_format(ax: Any, *, log_y: bool) -> None:
    fmt = _runtime_tick_formatter()
    if bool(log_y):
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_formatter(fmt)


def _paper_get_operational_values(report: dict[str, Any]) -> dict[str, float]:
    fixed = report.get("fixed_conditions")
    if not isinstance(fixed, dict):
        fixed = {}
    online = report.get("online_operational_conditions")
    if not isinstance(online, dict):
        online = {}
    ingest = online.get("ingestion_model")
    if not isinstance(ingest, dict):
        ingest = {}
    ingest_ev = ingest.get("ev")
    if not isinstance(ingest_ev, dict):
        ingest_ev = {}

    window_vals = fixed.get("window_values_s")
    if isinstance(window_vals, list) and len(window_vals) > 0:
        try:
            window_s = float(window_vals[0])
        except Exception:
            window_s = 360.0
    else:
        try:
            window_s = float(report.get("window_s", 360.0))
        except Exception:
            window_s = 360.0

    return {
        "evse_sample_s": float(fixed.get("charger_sample_s", 5.0) or 5.0),
        "ev_sample_s": float(fixed.get("ev_sample_s", 30.0) or 30.0),
        "window_s": float(window_s),
        "delay_mean_s": float(ingest_ev.get("delay_mean_s", 2.0) or 2.0),
        "delay_jitter_s": float(ingest_ev.get("delay_jitter_s", 1.0) or 1.0),
        "delay_max_s": float(ingest_ev.get("delay_max_s", 10.0) or 10.0),
        "loss_prob": float(ingest_ev.get("loss_prob", 0.005) or 0.0),
        "candidate_margin_s": float(online.get("candidate_time_margin_s", 120.0) or 120.0),
        "topk": float(CURRENT_GREEDY_TOPK),
    }


def _generate_paper_diagram_figs(
    report: dict[str, Any],
    *,
    out_dir: Path,
    plots_base: Path,
) -> dict[str, str]:
    fig_w = _PAPER_FIG_W
    fig_h = _PAPER_FIG_H
    vals = _paper_get_operational_values(report)
    evse_sample = int(round(vals["evse_sample_s"]))
    ev_sample = int(round(vals["ev_sample_s"]))
    window_s = int(round(vals["window_s"]))
    delay_mean = float(vals["delay_mean_s"])
    delay_jitter = float(vals["delay_jitter_s"])
    delay_max = float(vals["delay_max_s"])
    loss_prob = float(vals["loss_prob"])
    candidate_margin = int(round(vals["candidate_margin_s"]))
    topk = int(round(vals["topk"]))
    step_count_raw = report.get("command_step_count", 6)
    try:
        step_count = max(1.0, float(step_count_raw))
    except Exception:
        step_count = 6.0

    # Fig.1: End-to-end architecture diagram
    p1 = out_dir / "fig1_system_overview.png"
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")

    ax1.text(0.03, 0.79, "Control plane (console-known)", ha="left", va="center", fontsize=13.5, fontweight="semibold", color="#1d4ed8")
    ax1.text(0.03, 0.39, "Data plane (anonymous EV telemetry)", ha="left", va="center", fontsize=13.5, fontweight="semibold", color="#b45309")

    _paper_box(
        ax1,
        x=0.10,
        y=0.63,
        w=0.18,
        h=0.16,
        title="EVSE Pattern Control",
        body="MCCT command current\n(0A + step profile)",
        face="#dbeafe",
        edge="#2563eb",
    )
    _paper_box(
        ax1,
        x=0.34,
        y=0.63,
        w=0.18,
        h=0.16,
        title="EVSE Current Reference",
        body=f"Console-known stream\nsample={evse_sample}s",
        face="#dbeafe",
        edge="#2563eb",
    )
    _paper_box(
        ax1,
        x=0.10,
        y=0.19,
        w=0.18,
        h=0.16,
        title="EV Current Observation",
        body=f"Anonymous EV uploads\nsample={ev_sample}s, ceil_1A",
        face="#ffedd5",
        edge="#c2410c",
    )
    _paper_box(
        ax1,
        x=0.34,
        y=0.19,
        w=0.18,
        h=0.16,
        title="Cloud Ingestion",
        body=f"delay/jitter/loss\n{delay_mean:.1f}s/{delay_jitter:.1f}s/{loss_prob*100.0:.2f}%",
        face="#ffedd5",
        edge="#c2410c",
    )
    _paper_box(
        ax1,
        x=0.58,
        y=0.39,
        w=0.17,
        h=0.24,
        title="Online Identifier",
        body="candidate gating\ncost computation\nassignment",
        face="#dcfce7",
        edge="#15803d",
    )
    _paper_box(
        ax1,
        x=0.80,
        y=0.39,
        w=0.16,
        h=0.24,
        title="Match Output",
        body="EV#m <-> EVSE#n\nbilling/V2G\nresource control",
        face="#e2e8f0",
        edge="#475569",
    )

    _paper_arrow(ax1, x0=0.28, y0=0.71, x1=0.34, y1=0.71, label="known")
    _paper_arrow(ax1, x0=0.52, y0=0.71, x1=0.58, y1=0.53)
    _paper_arrow(ax1, x0=0.28, y0=0.27, x1=0.34, y1=0.27, label="anonymous")
    _paper_arrow(ax1, x0=0.52, y0=0.27, x1=0.58, y1=0.49)
    _paper_arrow(ax1, x0=0.75, y0=0.51, x1=0.80, y1=0.51, label="finalize")

    ax1.text(
        0.02,
        0.05,
        "Key point: EVSE-side reference is known to the console, while EV-side telemetry is anonymous and delayed.",
        ha="left",
        va="bottom",
        fontsize=12.2,
        color="#334155",
    )
    _apply_paper_style(fig1)
    fig1.tight_layout(pad=1.30)
    fig1.savefig(p1, dpi=220, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig1)

    # Fig.2: Asynchronous timeline with watermark
    p2 = out_dir / "fig2_async_timeline_watermark.png"
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    x_end = float(window_s + max(40, int(round(delay_max)) + 35))

    evse_t = np.arange(0.0, float(window_s) + evse_sample, float(max(1, evse_sample)))
    ev_t = np.arange(0.0, float(window_s) + ev_sample, float(max(1, ev_sample)))
    if len(ev_t) <= 1:
        ev_t = np.array([0.0, float(window_s)], dtype=float)

    phase = np.sin(np.linspace(0.0, 2.8 * np.pi, len(ev_t)))
    delays = np.clip(delay_mean + 0.65 * delay_jitter * phase + 0.35 * delay_jitter, 0.0, max(0.0, delay_max))
    ing_t = ev_t + delays

    loss_mask = np.zeros(len(ev_t), dtype=bool)
    if len(ev_t) >= 10:
        loss_mask[7] = True
    if len(ev_t) >= 18:
        loss_mask[15] = True
    ingest_kept = ing_t[~loss_mask]
    ev_lost = ev_t[loss_mask]

    ax2.hlines([3, 2, 1], xmin=0.0, xmax=x_end, colors="#cbd5e1", linestyles="--", linewidth=1.0)
    ax2.scatter(evse_t, np.full_like(evse_t, 3.0), s=15, marker="s", color="#2563eb", label=f"EVSE stream ({evse_sample}s)")
    ax2.scatter(ev_t, np.full_like(ev_t, 2.0), s=30, marker="^", color="#f59e0b", label=f"EV sampled ({ev_sample}s)")
    if len(ev_lost) > 0:
        ax2.scatter(ev_lost, np.full_like(ev_lost, 2.0), s=46, marker="x", color="#dc2626", linewidths=1.6, label="EV packet lost")
    ax2.scatter(ingest_kept, np.full_like(ingest_kept, 1.0), s=26, marker="o", color="#16a34a", label="Cloud-ingested EV")

    for t_src, t_ing in zip(ev_t, ing_t):
        ax2.plot([t_src, t_ing], [2.0, 1.0], color="#94a3b8", linewidth=0.8, alpha=0.45)

    watermark_ts = float(window_s) + float(max(0.0, delay_max))
    ax2.axvline(float(window_s), color="#111827", linestyle="--", linewidth=1.5, label="window_end")
    ax2.axvline(watermark_ts, color="#1d4ed8", linestyle="-.", linewidth=1.8, label="watermark_ready")
    ax2.axvspan(float(window_s), watermark_ts, color="#bfdbfe", alpha=0.35, label="wait for delayed EV telemetry")

    ax2.set_xlim(-5.0, x_end)
    ax2.set_ylim(0.5, 3.5)
    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(["Cloud ingest lane", "EV sample lane", "EVSE reference lane"])
    ax2.set_xlabel("Time (s)")
    ax2.grid(axis="x", alpha=0.18)
    ax2.legend(loc="upper left", ncol=2)
    ax2.text(
        0.01,
        0.01,
        f"Ingestion model: mean={delay_mean:.1f}s, jitter={delay_jitter:.1f}s, max={delay_max:.1f}s, loss={loss_prob*100.0:.2f}%",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        color="#334155",
    )
    _apply_paper_style(fig2)
    fig2.tight_layout(pad=1.30)
    fig2.savefig(p2, dpi=220, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig2)

    # Fig.3: Online event loop flowchart
    p3 = out_dir / "fig3_event_loop_flow.png"
    fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
    ax3.set_xlim(0.0, 1.0)
    ax3.set_ylim(0.0, 1.0)
    ax3.axis("off")
    _paper_box(
        ax3,
        x=0.03,
        y=0.66,
        w=0.24,
        h=0.17,
        title="Event Triggers",
        body="arrival / departure\ntick (every 5s)\nwatermark_ready",
        face="#e2e8f0",
        edge="#475569",
    )
    _paper_box(
        ax3,
        x=0.31,
        y=0.66,
        w=0.34,
        h=0.17,
        title="1) Maturity Check",
        body="is this EV eligible now?\n(watermark reached or forced fallback)",
        face="#dbeafe",
        edge="#2563eb",
    )
    _paper_box(
        ax3,
        x=0.31,
        y=0.47,
        w=0.34,
        h=0.14,
        title="2) Candidate Slot Gating",
        body=f"build active-slot candidates\nwithin +/- {candidate_margin}s near effective_end",
        face="#dbeafe",
        edge="#2563eb",
    )
    _paper_box(
        ax3,
        x=0.31,
        y=0.30,
        w=0.34,
        h=0.14,
        title="3) Cost + Assignment",
        body="time-only / coarse-refined /\nNomura / Bayesian -> assignment",
        face="#dcfce7",
        edge="#15803d",
    )
    _paper_box(
        ax3,
        x=0.31,
        y=0.13,
        w=0.34,
        h=0.14,
        title="4) Finalize",
        body="emit EV<->EVSE match\nand update state",
        face="#dcfce7",
        edge="#15803d",
    )
    _paper_box(
        ax3,
        x=0.70,
        y=0.66,
        w=0.27,
        h=0.17,
        title="No: Keep Pending",
        body="if not matured, keep in queue\nand re-check at next event",
        face="#ffedd5",
        edge="#c2410c",
    )
    _paper_box(
        ax3,
        x=0.70,
        y=0.14,
        w=0.27,
        h=0.18,
        title="Operational Output",
        body="billing identity\nV2G controllable ID\nresource accounting",
        face="#e2e8f0",
        edge="#475569",
    )

    _paper_arrow(ax3, x0=0.27, y0=0.745, x1=0.31, y1=0.745, label="event")
    _paper_arrow(ax3, x0=0.65, y0=0.745, x1=0.70, y1=0.745, label="No")
    _paper_arrow(ax3, x0=0.48, y0=0.66, x1=0.48, y1=0.61, label="Yes")
    _paper_arrow(ax3, x0=0.48, y0=0.47, x1=0.48, y1=0.44)
    _paper_arrow(ax3, x0=0.48, y0=0.30, x1=0.48, y1=0.27)
    _paper_arrow(ax3, x0=0.65, y0=0.22, x1=0.70, y1=0.23, label="final match")

    ax3.text(0.04, 0.04, "mature check -> candidate build -> cost/assignment -> finalize", ha="left", va="bottom", fontsize=12.6, color="#334155")
    _apply_paper_style(fig3)
    fig3.tight_layout(pad=1.30)
    fig3.savefig(p3, dpi=220, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig3)

    # Diagnostic signal-realism overlay (command / EVSE / sensed EV).
    p4 = out_dir / "diagnostic_signal_realism_overlay.png"
    fig4, ax4 = plt.subplots(figsize=(5.33, 3.0))
    t = np.arange(0.0, float(window_s) + 1.0, 1.0, dtype=float)
    step_len = max(5, int(window_s // max(1.0, float(step_count))))
    levels = np.array([0, 6, 12, 18, 24, 30], dtype=float)
    cmd = np.zeros_like(t, dtype=float)
    for i, tt in enumerate(t):
        lv = int(min(len(levels) - 1, max(0, int(tt // float(step_len)))))
        cmd[i] = float(levels[lv])
    ramped = np.zeros_like(cmd)
    alpha = 0.18
    for i in range(len(cmd)):
        prev = ramped[i - 1] if i > 0 else 0.0
        ramped[i] = float(prev + alpha * (cmd[i] - prev))
    evse_cur = np.clip(ramped + 0.3, 0.0, 32.0)
    jitter_phase = np.sin(np.linspace(0.0, 5.2 * np.pi, len(t)))
    ev_sensed = np.clip(evse_cur * 0.97 - 0.4 + 0.55 * jitter_phase, 0.0, 32.0)
    if len(ev_sensed) > 16:
        ev_sensed[12] = np.nan
        ev_sensed[13] = np.nan
    ax4.step(t, cmd, where="post", linewidth=2.2, color="#1d4ed8", label="Command current")
    ax4.plot(t, evse_cur, linewidth=2.1, color="#059669", label="EVSE current (ramp + offset)")
    ax4.plot(t, ev_sensed, linewidth=1.9, color="#b45309", label="Sensed EV current (bias + noise + jitter)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Current (A)")
    ax4.set_ylim(-0.5, 33.0)
    ax4.grid(color="#BFBFBF", linewidth=0.5, linestyle=":", alpha=0.9)
    ax4.legend(loc="upper left")
    ax4.text(
        0.01,
        0.02,
        "Realism factors: ramp-lag, offset, bias, noise, ingest jitter/loss",
        transform=ax4.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.2,
        color="#334155",
    )
    _apply_paper_style(fig4)
    fig4.tight_layout(pad=0.25)
    fig4.savefig(p4, dpi=300, bbox_inches="tight", pad_inches=0.05)
    try:
        fig4.savefig(p4.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    except Exception:
        pass
    plt.close(fig4)

    # Fig.5: Candidate margin illustration (delta sweep)
    p5 = out_dir / "fig5_candidate_margin_illustration.png"
    fig5, ax5 = plt.subplots(figsize=(fig_w, fig_h))
    center = float(window_s)
    intervals = [
        ("S1", center - 220.0, center - 140.0, "#94a3b8"),
        ("S2", center - 110.0, center - 15.0, "#60a5fa"),
        ("S3 (GT)", center - 35.0, center + 45.0, "#22c55e"),
        ("S4", center + 65.0, center + 150.0, "#f97316"),
        ("S5", center + 170.0, center + 255.0, "#94a3b8"),
    ]
    yvals = np.linspace(1.0, 5.0, len(intervals))
    for (label, st, et, color), yy in zip(intervals, yvals):
        ax5.hlines(yy, st, et, linewidth=8.0, color=color, alpha=0.90)
        ax5.text(st - 18.0, yy, label, ha="right", va="center", fontsize=12.2, color="#334155")
    for delta, col in ((60, "#93c5fd"), (120, "#60a5fa"), (180, "#2563eb")):
        ax5.axvspan(center - float(delta), center + float(delta), color=col, alpha=0.12, label=f"Δ={int(delta)}s")
    ax5.axvline(center, color="#111827", linestyle="--", linewidth=1.6, label="effective_end")
    ax5.set_xlabel("Time (s)")
    ax5.set_yticks([])
    ax5.set_ylim(0.2, 5.8)
    ax5.grid(axis="x", alpha=0.20)
    handles, labels = ax5.get_legend_handles_labels()
    keep_h: list[Any] = []
    keep_l: list[str] = []
    seen_l: set[str] = set()
    for h, l in zip(handles, labels):
        if l in seen_l:
            continue
        seen_l.add(l)
        keep_h.append(h)
        keep_l.append(l)
    if len(keep_h) > 0:
        ax5.legend(keep_h, keep_l, loc="upper left")
    ax5.text(
        0.01,
        0.02,
        "As Δ increases, candidate set widens and GT-in-candidates probability rises.",
        transform=ax5.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        color="#334155",
    )
    _apply_paper_style(fig5)
    fig5.tight_layout(pad=1.30)
    fig5.savefig(p5, dpi=220, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig5)

    return {
        "fig1_system_overview": _to_rel_plot(p1, plots_base),
        "fig2_async_timeline_watermark": _to_rel_plot(p2, plots_base),
        "fig3_event_loop_flow": _to_rel_plot(p3, plots_base),
        "diagnostic_signal_realism_overlay": _to_rel_plot(p4, plots_base),
        "fig5_candidate_margin_illustration": _to_rel_plot(p5, plots_base),
    }


def _generate_trace_example_figs(
    run_id: str,
    *,
    out_dir: Path,
    plots_base: Path,
) -> dict[str, str]:
    rid = _validate_run_id(run_id)
    run_dir = (COMPARE_DIR / rid).resolve()
    event_path = run_dir / "event_trace.jsonl"
    posterior_path = run_dir / "posterior_trace_A3.jsonl"
    cost_path = run_dir / "cost_trace.jsonl"

    def _to_int(value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    def _to_float(value: Any) -> float | None:
        try:
            num = float(value)
        except Exception:
            return None
        if not np.isfinite(num):
            return None
        return float(num)

    def _is_success_event(row: dict[str, Any]) -> bool:
        gt = _to_int(row.get("gt_slot"))
        pred = _to_int(row.get("pred_slot"))
        if gt is None or pred is None:
            return False
        fb = bool(row.get("fallback_triggered", False))
        return bool((gt == pred) and (not fb))

    def _pick_event_examples() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        first_row: dict[str, Any] | None = None
        success_row: dict[str, Any] | None = None
        failure_row: dict[str, Any] | None = None

        # Stream event trace to avoid loading multi-GB JSONL into memory.
        for r in _iter_jsonl(event_path, max_rows=1_500_000):
            row = dict(r)
            if first_row is None:
                first_row = row
            ok = _is_success_event(row)
            if ok and success_row is None:
                success_row = row
            if (not ok) and failure_row is None:
                failure_row = row
            if success_row is not None and failure_row is not None:
                break

        if success_row is None:
            success_row = first_row
        if failure_row is None:
            failure_row = first_row
        return success_row, failure_row

    def _plot_event_case(path: Path, *, title: str, row: dict[str, Any] | None, color: str) -> None:
        font_cfg = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }
        with plt.rc_context(font_cfg):
            fig, ax = plt.subplots(figsize=(5.33, 3.0))
        if not isinstance(row, dict):
            ax.text(0.5, 0.5, "No event trace data", ha="center", va="center")
            ax.axis("off")
            fig.tight_layout(pad=0.30)
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
            try:
                fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
            except Exception:
                pass
            plt.close(fig)
            return
        time_cols = [
            ("arrival", row.get("arrival_ts")),
            ("decision_end", row.get("decision_end_ts")),
            ("effective_end", row.get("effective_end_ts")),
            ("watermark_ready", row.get("watermark_ready_ts")),
            ("decision", row.get("decision_ts")),
            ("session_end", row.get("session_end_ts")),
        ]
        ts_vals: list[pd.Timestamp] = []
        labels: list[str] = []
        for lb, vv in time_cols:
            try:
                ts = pd.Timestamp(vv)
            except Exception:
                continue
            if pd.isna(ts):
                continue
            ts_vals.append(pd.Timestamp(ts))
            labels.append(str(lb))
        if len(ts_vals) == 0:
            ax.text(0.5, 0.5, "No valid timestamps", ha="center", va="center")
            ax.axis("off")
            fig.tight_layout(pad=0.30)
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
            try:
                fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
            except Exception:
                pass
            plt.close(fig)
            return

        t0 = min(ts_vals)
        xs = [float((ts - t0).total_seconds()) for ts in ts_vals]
        xmin = float(np.min(xs))
        xmax = float(np.max(xs))
        span_all = max(1.0, float(xmax - xmin))

        core_keys = ("arrival", "decision", "watermark", "effective_end")
        core_xs = [float(x) for x, lb in zip(xs, labels) if any(k in str(lb) for k in core_keys)]
        if len(core_xs) > 0:
            em_min = float(np.min(core_xs))
            em_max = float(np.max(core_xs))
            em_span = max(60.0, float(em_max - em_min))
            x_left = float(em_min - 0.15 * em_span)
            x_right = float(em_max + 0.40 * em_span)
        else:
            x_left = float(xmin - 0.10 * span_all)
            x_right = float(xmax + 0.10 * span_all)

        ax.hlines(0.5, x_left, x_right, color="#94a3b8", linewidth=1.4)
        ax.scatter(xs, np.full(len(xs), 0.5), s=24, color=color, zorder=3)

        clusters: dict[int, dict[str, Any]] = {}
        for x, lb in sorted(zip(xs, labels), key=lambda z: float(z[0])):
            key = int(round(float(x) / 10.0))
            ent = clusters.get(key)
            if ent is None:
                ent = {"x": float(x), "labels": []}
                clusters[key] = ent
            ent["labels"].append(str(lb))

        for occ, key in enumerate(sorted(clusters.keys())):
            ent = clusters.get(int(key), {})
            x = float(ent.get("x", 0.0))
            labs = [str(t) for t in ent.get("labels", []) if str(t).strip() != ""]
            label_text = "\n".join(labs[:3]) if len(labs) > 0 else ""
            dx = 10 if (occ % 2 == 0) else -10
            dy = 16 + 10 * (occ % 2)
            ax.annotate(
                str(label_text),
                xy=(float(x), 0.5),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#1f2937",
                arrowprops=dict(arrowstyle="-", color="#64748b", linewidth=0.7, alpha=0.85),
            )

        decision_x = None
        for x, lb in zip(xs, labels):
            if str(lb) == "decision":
                decision_x = float(x)
                break
        if decision_x is not None:
            is_success = "success" in str(title).strip().lower()
            v_col = "#16a34a" if is_success else "#dc2626"
            v_txt = "Finalized" if is_success else "Mis-matched"
            ax.axvline(decision_x, color=v_col, linestyle=":", linewidth=1.0, alpha=0.9)
            ax.text(decision_x, 0.66, v_txt, ha="center", va="bottom", fontsize=8.0, color=v_col)

        session_end = None
        for x, lb in zip(xs, labels):
            if str(lb) == "session_end":
                session_end = float(x)
                break
        core_span = max(1.0, float(x_right - x_left))
        if session_end is not None and session_end > (x_right + 3.0 * core_span):
            ax.text(
                x_right - 0.02 * core_span,
                0.58,
                f"-> session_end (t={session_end:.0f}s)",
                ha="right",
                va="center",
                fontsize=8.0,
                color="#555555",
            )

        cands = int(row.get("candidate_count", 0))
        lat = float(row.get("decision_latency_s", float("nan")))
        reason = str(row.get("finalize_reason", row.get("decision_reason", "")))
        ax.text(
            0.01,
            0.03,
            f"n_cand={cands}, t_dec={lat:.0f}s, reason={reason}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            color="#555555",
        )
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_ylim(0.38, 0.70)
        ax.set_xlim(x_left, x_right)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="x", linestyle=":", linewidth=0.5, color="#BFBFBF", alpha=0.9)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _apply_paper_style(fig)
        fig.tight_layout(pad=0.30)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
        try:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
        except Exception:
            pass
        plt.close(fig)

    def _posterior_df_from_trace() -> pd.DataFrame:
        picked_key: tuple[int | None, int | None, int] | None = None
        rows: list[dict[str, Any]] = []
        for r in _iter_jsonl(posterior_path, max_rows=2_000_000):
            if str(r.get("algorithm_id", "")).strip() != "bayesian_windowed":
                continue
            ev_idx = _to_int(r.get("ev_idx"))
            stage_id = _to_float(r.get("stage_id"))
            top1 = _to_float(r.get("top1_prob"))
            top2 = _to_float(r.get("top2_prob"))
            if ev_idx is None or stage_id is None or top1 is None or top2 is None:
                continue

            key = (_to_int(r.get("scenario_id")), _to_int(r.get("repeat_idx")), int(ev_idx))
            if picked_key is None:
                picked_key = key
            if key != picked_key:
                if len(rows) >= 2:
                    break
                continue

            rows.append(
                {
                    "ev_idx": int(ev_idx),
                    "decision_ts": r.get("decision_ts"),
                    "stage_id": float(stage_id),
                    "top1_prob": float(top1),
                    "top2_prob": float(top2),
                }
            )
        return pd.DataFrame(rows)

    def _posterior_df_from_cost() -> pd.DataFrame:
        selected_sr: tuple[int, int] | None = None
        per_ev: dict[int, list[float]] = defaultdict(list)
        for r in _iter_jsonl(cost_path, max_rows=3_000_000):
            if str(r.get("algorithm_id", "")).strip() != "bayesian_windowed":
                continue
            post = _to_float(r.get("posterior"))
            ev_idx = _to_int(r.get("ev_idx"))
            scenario_id = _to_int(r.get("scenario_id"))
            repeat_idx = _to_int(r.get("repeat_idx"))
            if post is None or ev_idx is None or scenario_id is None or repeat_idx is None:
                continue
            sr = (int(scenario_id), int(repeat_idx))
            if selected_sr is None:
                selected_sr = sr
            if sr != selected_sr:
                if len(per_ev) >= 40:
                    break
                continue
            per_ev[int(ev_idx)].append(float(post))
            if len(per_ev) >= 120 and sum(len(v) for v in per_ev.values()) >= 20000:
                break

        rows: list[dict[str, Any]] = []
        order = 0
        for ev_idx in sorted(per_ev.keys()):
            vals = [float(v) for v in per_ev.get(int(ev_idx), []) if np.isfinite(float(v))]
            if len(vals) <= 0:
                continue
            vals = sorted(vals, reverse=True)
            order += 1
            rows.append(
                {
                    "ev_idx": int(ev_idx),
                    "decision_order": int(order),
                    "stage_id": float(order),
                    "top1_prob": float(vals[0]),
                    "top2_prob": float(vals[1] if len(vals) >= 2 else np.nan),
                }
            )
        return pd.DataFrame(rows)

    p6 = out_dir / "fig6_example_success_case.png"
    p7 = out_dir / "fig7_example_failure_case.png"
    success_row, failure_row = _pick_event_examples()
    _plot_event_case(p6, title="Success Case", row=success_row, color="#16a34a")
    _plot_event_case(p7, title="Failure Case", row=failure_row, color="#dc2626")

    p8 = out_dir / "fig8_posterior_evolution_A3.png"
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
        }
    ):
        fig8, ax8 = plt.subplots(figsize=(5.33, 3.0))
    post_df = _posterior_df_from_trace()
    posterior_source = "posterior_trace"
    if post_df.empty:
        post_df = _posterior_df_from_cost()
        posterior_source = "cost_trace_fallback"

    if post_df.empty:
        ax8.text(0.5, 0.5, "No posterior data (trace/cost)", ha="center", va="center")
        ax8.axis("off")
    else:
        if str(posterior_source) == "posterior_trace":
            post_df["decision_ts"] = pd.to_datetime(post_df.get("decision_ts"), errors="coerce")
            post_df["stage_id"] = pd.to_numeric(post_df.get("stage_id"), errors="coerce")
            post_df["top1_prob"] = pd.to_numeric(post_df.get("top1_prob"), errors="coerce")
            post_df["top2_prob"] = pd.to_numeric(post_df.get("top2_prob"), errors="coerce")
            post_df["ev_idx"] = pd.to_numeric(post_df.get("ev_idx"), errors="coerce")
            post_df = post_df.dropna(subset=["ev_idx", "stage_id", "top1_prob", "top2_prob"]).copy()
            if post_df.empty:
                ax8.text(0.5, 0.5, "No finite posterior points", ha="center", va="center")
                ax8.axis("off")
            else:
                post_df = post_df.sort_values(["ev_idx", "decision_ts", "stage_id"])
                candidate_ev = (
                    post_df.groupby("ev_idx", dropna=True)["stage_id"]
                    .nunique()
                    .reset_index(name="n_stage")
                    .sort_values(["n_stage", "ev_idx"], ascending=[False, True])
                )
                if candidate_ev.empty:
                    ax8.text(0.5, 0.5, "No posterior trajectory", ha="center", va="center")
                    ax8.axis("off")
                else:
                    row_sel = candidate_ev.loc[candidate_ev["n_stage"] >= 4]
                    if row_sel.empty:
                        row_sel = candidate_ev
                    chosen_ev = int(float(row_sel.iloc[0]["ev_idx"]))
                    part = post_df[post_df["ev_idx"].astype(int) == chosen_ev].copy()
                    part = part.sort_values(["decision_ts", "stage_id"])
                    if part.empty:
                        ax8.text(0.5, 0.5, "No finite posterior points", ha="center", va="center")
                        ax8.axis("off")
                    else:
                        uniq_stages = sorted({int(float(s)) for s in part["stage_id"].tolist() if np.isfinite(float(s))})
                        if len(uniq_stages) >= 2:
                            last_stage = int(uniq_stages[-1])
                            part = part[pd.to_numeric(part.get("stage_id"), errors="coerce").astype("Int64") != last_stage].copy()
                        x = np.arange(1, len(part) + 1, dtype=float)
                        y1 = part["top1_prob"].to_numpy(dtype=float)
                        y2 = part["top2_prob"].to_numpy(dtype=float)
                        ax8.plot(x, y1, marker="o", linewidth=1.5, markersize=5.0, color="#1f77b4", label="Top-1 posterior")
                        ax8.plot(x, y2, marker="^", linewidth=1.2, markersize=4.0, linestyle="--", color="#ff7f0e", label="Top-2 posterior")
                        ax8.set_xlabel("Decision tick")
                        ax8.set_ylabel("Posterior probability")
                        ax8.set_ylim(0.0, 1.05)
                        ax8.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                        ax8.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                        ax8.grid(alpha=0.9, linestyle=":", linewidth=0.5, color="#BFBFBF")
                        ax8.spines["top"].set_visible(False)
                        ax8.spines["right"].set_visible(False)
                        ax8.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc", fancybox=False)
        elif post_df.empty:
            ax8.text(0.5, 0.5, "No A3 posterior rows", ha="center", va="center")
            ax8.axis("off")
        else:
            post_df["stage_id"] = pd.to_numeric(post_df.get("stage_id"), errors="coerce")
            post_df["top1_prob"] = pd.to_numeric(post_df.get("top1_prob"), errors="coerce")
            post_df["top2_prob"] = pd.to_numeric(post_df.get("top2_prob"), errors="coerce")
            post_df = post_df.dropna(subset=["stage_id", "top1_prob"]).copy()
            if post_df.empty:
                ax8.text(0.5, 0.5, "No finite posterior points", ha="center", va="center")
                ax8.axis("off")
            else:
                part = post_df.sort_values(["stage_id"]).copy()
                x = np.arange(1, len(part) + 1, dtype=float)
                y1 = part["top1_prob"].to_numpy(dtype=float)
                y2 = part["top2_prob"].to_numpy(dtype=float)
                ax8.plot(x, y1, marker="o", linewidth=1.5, markersize=5.0, color="#1f77b4", label="Top-1 posterior")
                ax8.plot(x, y2, marker="^", linewidth=1.2, markersize=4.0, linestyle="--", color="#ff7f0e", label="Top-2 posterior")
                ax8.set_xlabel("Decision tick")
                ax8.set_ylabel("Posterior probability")
                ax8.set_ylim(0.0, 1.05)
                ax8.set_yticks([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
                ax8.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                ax8.grid(alpha=0.9, linestyle=":", linewidth=0.5, color="#BFBFBF")
                ax8.spines["top"].set_visible(False)
                ax8.spines["right"].set_visible(False)
                ax8.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc", fancybox=False)
    _apply_paper_style(fig8)
    fig8.tight_layout(pad=0.30)
    fig8.savefig(p8, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    try:
        fig8.savefig(p8.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    except Exception:
        pass
    plt.close(fig8)

    return {
        "fig6_example_success_case": _to_rel_plot(p6, plots_base),
        "fig7_example_failure_case": _to_rel_plot(p7, plots_base),
        "fig8_posterior_evolution_A3": _to_rel_plot(p8, plots_base),
    }


def generate_results_section_figures_from_report(
    run_id: str,
    *,
    error_mode: str = DEFAULT_ERROR_MODE,
    paper_font_scale: float = 1.0,
    runtime_log_scale: bool = False,
) -> dict[str, Any]:
    rid = _validate_run_id(run_id)
    mode = str(error_mode).strip().lower()
    if mode not in VALID_ERROR_MODES:
        raise ValueError(f"error_mode must be one of {sorted(VALID_ERROR_MODES)}")

    fs = float(paper_font_scale)
    if not np.isfinite(fs):
        raise ValueError("paper_font_scale must be finite")
    if fs < 0.6 or fs > 2.5:
        raise ValueError("paper_font_scale must be in [0.6, 2.5]")
    legend_fs = float(_PAPER_LEGEND_SIZE * fs)
    case_a_size = (5.33, 3.00)   # 1.8:1 single panel
    case_b_size = (7.33, 3.07)   # 2.4:1 two-panel
    save_dpi = 300

    report = load_algorithm_compare_report(rid)
    summary_df = _paper_summary_df(report)
    if summary_df.empty:
        raise ValueError("summary data not found for this run")

    if "algorithm_id" not in summary_df.columns:
        summary_df["algorithm_id"] = ""
    summary_df["algorithm_id"] = summary_df["algorithm_id"].astype(str)

    base_numeric_cols = [
        "ev_count",
        "accuracy_mean",
        "accuracy_requested_mean",
        "p90_latency_mean",
        "assignment_runtime_mean",
        "blocked_ratio_mean",
        "avg_cost_mean",
        "unmatched_mean",
        "window_watermark_ratio_mean",
        "session_end_before_watermark_ratio_mean",
        "session_end_fallback_ratio_mean",
        "timeline_end_fallback_ratio_mean",
        "fallback_finalized_ratio_mean",
        "gt_in_candidates_ratio_mean",
        "runtime_s_candidate_build_mean",
        "runtime_s_assignment_mean",
        "runtime_s_fallback_mean",
        "runtime_s_other_mean",
        "runtime_s_total_mean",
        "runtime_s_cost_mean",
        "runtime_s_loop_mean",
        "charger_sample_s",
        "ev_sample_s",
        "matcher_delay_max_s",
        "candidate_margin_s",
        "tau_s",
    ]
    for col in base_numeric_cols:
        if col not in summary_df.columns:
            summary_df[col] = np.nan
        summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    # Runtime compatibility/fallback chain.
    if "runtime_s_total_mean" in summary_df.columns:
        summary_df["runtime_s_total_mean"] = summary_df["runtime_s_total_mean"].where(
            np.isfinite(summary_df["runtime_s_total_mean"]),
            summary_df["assignment_runtime_mean"],
        )
    else:
        summary_df["runtime_s_total_mean"] = pd.to_numeric(summary_df["assignment_runtime_mean"], errors="coerce")

    if "runtime_s_cost_mean" in summary_df.columns:
        summary_df["runtime_s_cost_mean"] = summary_df["runtime_s_cost_mean"].where(
            np.isfinite(summary_df["runtime_s_cost_mean"]),
            pd.to_numeric(summary_df.get("runtime_s_candidate_build_mean"), errors="coerce"),
        )
    else:
        summary_df["runtime_s_cost_mean"] = pd.to_numeric(summary_df.get("runtime_s_candidate_build_mean"), errors="coerce")

    if "runtime_s_loop_mean" in summary_df.columns:
        summary_df["runtime_s_loop_mean"] = summary_df["runtime_s_loop_mean"].where(
            np.isfinite(summary_df["runtime_s_loop_mean"]),
            pd.to_numeric(summary_df.get("runtime_s_other_mean"), errors="coerce"),
        )
    else:
        summary_df["runtime_s_loop_mean"] = pd.to_numeric(summary_df.get("runtime_s_other_mean"), errors="coerce")

    for col in ["pattern_mode", "pattern_assignment_mode"]:
        if col not in summary_df.columns:
            summary_df[col] = ""
        summary_df[col] = summary_df[col].astype(str)

    ev_counts = [int(x) for x in sorted(set(int(v) for v in summary_df["ev_count"].dropna().tolist()))]
    if len(ev_counts) == 0:
        raise ValueError("ev_count data not found")
    target_ev = int(max(ev_counts))

    out_dir = COMPARE_DIR / rid / "diagnostic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_base = GENERATED_OUTPUT_DIR.resolve()

    sens_df = _collect_results_sensitivity_rows(report)

    def _safe_algo_mean(df: pd.DataFrame, aid: str, col: str) -> float:
        part = df[df["algorithm_id"].astype(str) == str(aid)]
        arr = pd.to_numeric(part.get(col), errors="coerce")
        vv = arr.to_numpy(dtype=float) if isinstance(arr, pd.Series) else np.asarray([], dtype=float)
        vv = vv[np.isfinite(vv)]
        if len(vv) == 0:
            return float("nan")
        return float(np.mean(vv))

    def _first_list_num(key: str, fallback_col: str, default: float) -> float:
        raw = report.get(key)
        if isinstance(raw, list) and len(raw) > 0:
            try:
                v = float(raw[0])
                if np.isfinite(v):
                    return float(v)
            except Exception:
                pass
        vals = pd.to_numeric(summary_df.get(fallback_col), errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            return float(vals.iloc[0])
        return float(default)

    base_cfg = {
        "charger_sample_s": _first_list_num("charger_sample_values_s", "charger_sample_s", 5.0),
        "ev_sample_s": _first_list_num("ev_sample_values_s", "ev_sample_s", 30.0),
        "matcher_delay_max_s": _first_list_num("matcher_delay_max_values_s", "matcher_delay_max_s", 10.0),
        "candidate_margin_s": _first_list_num("candidate_margin_values_s", "candidate_margin_s", 120.0),
    }

    tau_values = report.get("tau_values")
    base_tau = float("nan")
    if isinstance(tau_values, list) and len(tau_values) > 0:
        try:
            base_tau = float(tau_values[0])
        except Exception:
            base_tau = float("nan")
    elif "tau_s" in summary_df.columns:
        tv = pd.to_numeric(summary_df["tau_s"], errors="coerce")
        tv = tv[np.isfinite(tv)]
        if len(tv) > 0:
            base_tau = float(tv.iloc[0])

    base_assign_mode = str(report.get("pattern_assignment_mode", "")).strip().lower()
    base_pattern_modes = report.get("pattern_modes")
    if isinstance(base_pattern_modes, list) and len(base_pattern_modes) > 0:
        base_pattern_mode = str(base_pattern_modes[0]).strip().lower()
    else:
        base_pattern_mode = str(report.get("pattern_mode", "")).strip().lower()

    def _filter_fixed(df: pd.DataFrame, *, vary_cols: set[str] | None = None) -> pd.DataFrame:
        out = df.copy()
        skip = set(vary_cols or set())

        if "pattern_assignment_mode" in out.columns and base_assign_mode != "":
            out = out[out["pattern_assignment_mode"].astype(str).str.lower() == base_assign_mode]
        if "pattern_mode" in out.columns and base_pattern_mode != "":
            out = out[out["pattern_mode"].astype(str).str.lower() == base_pattern_mode]
        if "tau_s" in out.columns and np.isfinite(base_tau) and "tau_s" not in skip:
            t = pd.to_numeric(out["tau_s"], errors="coerce")
            out = out[np.isfinite(t) & (np.abs(t - float(base_tau)) <= 1e-9)]

        for k, b in base_cfg.items():
            if k in skip:
                continue
            if k not in out.columns or not np.isfinite(float(b)):
                continue
            vals = pd.to_numeric(out[k], errors="coerce")
            tol = max(1e-9, abs(float(b)) * 1e-6)
            out = out[np.isfinite(vals) & (np.abs(vals - float(b)) <= tol)]

        return out

    def _plot_no_data_panel_scaled(ax: Any, *, x_label: str, y_label: str, message: str) -> None:
        _plot_no_data_panel(ax, x_label=x_label, y_label=y_label, message=message)

    def _plot_algo_lines(
        ax: Any,
        *,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        y_scale: float = 1.0,
        legend: bool = True,
        x_mult: float = 1.0,
    ) -> bool:
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            _plot_no_data_panel_scaled(ax, x_label=x_label, y_label=y_label, message="No matching runs")
            return False
        drawn = 0
        for aid, g in df.groupby("algorithm_id", dropna=False):
            part = (
                g.groupby(x_col, dropna=True)[y_col]
                .mean()
                .reset_index()
                .sort_values(x_col)
            )
            x = pd.to_numeric(part[x_col], errors="coerce").to_numpy(dtype=float) * float(x_mult)
            y = pd.to_numeric(part[y_col], errors="coerce").to_numpy(dtype=float) * float(y_scale)
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) < 2:
                continue
            ax.plot(
                x[m],
                y[m],
                marker="o",
                linewidth=1.5,
                markersize=5.0,
                color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                label=_paper_algo_short_label(str(aid)),
            )
            drawn += 1
        ax.set_xlabel(str(x_label))
        ax.set_ylabel(str(y_label))
        ax.grid(True, which="major", axis="both", color="#BFBFBF", linewidth=0.5, linestyle=":")
        if drawn > 0:
            if legend:
                _paper_ordered_legend(ax, loc="best", fontsize=legend_fs)
            xs = pd.to_numeric(df.get(x_col), errors="coerce")
            xs = xs[np.isfinite(xs)]
            if len(xs) > 0:
                xt = sorted(set((xs * float(x_mult)).tolist()))
                if len(xt) <= 8:
                    ax.set_xticks([float(v) for v in xt])
            return True
        _plot_no_data_panel_scaled(ax, x_label=x_label, y_label=y_label, message="Need >=2 x points per algorithm")
        return False

    def _save_axes_crop(fig: Any, axes: list[Any], out_path: Path) -> bool:
        keep_axes = [ax for ax in axes if ax is not None]
        if len(keep_axes) == 0:
            return False
        all_axes = list(getattr(fig, "axes", []) or [])
        vis_map: dict[Any, bool] = {}
        try:
            keep_set = set(keep_axes)
            for ax in all_axes:
                try:
                    vis_map[ax] = bool(ax.get_visible())
                    ax.set_visible(ax in keep_set)
                except Exception:
                    continue
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = []
            for ax in keep_axes:
                try:
                    bb = ax.get_tightbbox(renderer)
                except Exception:
                    bb = None
                if bb is not None:
                    bboxes.append(bb)
            if len(bboxes) <= 0:
                fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
                return True
            x0 = float(min(bb.x0 for bb in bboxes))
            y0 = float(min(bb.y0 for bb in bboxes))
            x1 = float(max(bb.x1 for bb in bboxes))
            y1 = float(max(bb.y1 for bb in bboxes))
            w = float(max(1.0, x1 - x0))
            h = float(max(1.0, y1 - y0))
            pad_left = max(28.0, 0.04 * w)
            pad_right = max(20.0, 0.03 * w)
            pad_bottom = max(26.0, 0.04 * h)
            pad_top = max(18.0, 0.03 * h)
            bb_all = mtransforms.Bbox.from_extents(
                float(x0 - pad_left),
                float(y0 - pad_bottom),
                float(x1 + pad_right),
                float(y1 + pad_top),
            )
            bb_inches = bb_all.transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(out_path, dpi=save_dpi, bbox_inches=bb_inches, pad_inches=0.05, facecolor="white", transparent=False)
            return True
        except Exception:
            return False
        finally:
            for ax in all_axes:
                if ax not in vis_map:
                    continue
                try:
                    ax.set_visible(bool(vis_map[ax]))
                except Exception:
                    continue

    def _select_summary_sweep(vary_col: str) -> pd.DataFrame:
        if vary_col not in summary_df.columns:
            return pd.DataFrame()
        part = _filter_fixed(summary_df, vary_cols={vary_col})
        ev = pd.to_numeric(part.get("ev_count"), errors="coerce")
        part = part[np.isfinite(ev) & (ev == float(target_ev))].copy()
        x = pd.to_numeric(part.get(vary_col), errors="coerce")
        part = part[np.isfinite(x)].copy()
        if int(pd.Series(x[np.isfinite(x)]).nunique()) < 2:
            return pd.DataFrame()
        return part

    def _select_sensitivity_sweep(vary_col: str, *, base_knobs: dict[str, float]) -> pd.DataFrame:
        if sens_df.empty or vary_col not in sens_df.columns:
            return pd.DataFrame()
        part = sens_df.copy()
        ev = pd.to_numeric(part.get("ev_count"), errors="coerce")
        part = part[np.isfinite(ev) & (ev == float(target_ev))].copy()
        part = _filter_vary_only(part, vary_col=vary_col, base_knobs=base_knobs)
        x = pd.to_numeric(part.get(vary_col), errors="coerce")
        part = part[np.isfinite(x)].copy()
        if int(pd.Series(x[np.isfinite(x)]).nunique()) < 2:
            return pd.DataFrame()
        return part

    split_result_paths: dict[str, str] = {}

    # Result R1: Accuracy(admitted) and AccReq with blocked ratio.
    pA = out_dir / "result_A_overall_accuracy.png"
    pA1 = out_dir / "result_A1_accuracy.png"
    pA2 = out_dir / "result_A2_accreq_blocked.png"
    figA, (axA1, axA2) = plt.subplots(1, 2, figsize=case_b_size)
    summary_for_ev = _filter_fixed(summary_df)

    _ = _plot_algo_lines(
        axA1,
        df=summary_for_ev,
        x_col="ev_count",
        y_col="accuracy_mean",
        x_label="EV Count",
        y_label="Accuracy (%)",
        y_scale=100.0,
        legend=True,
    )
    axA1.set_ylim(0.0, 100.0)

    _ = _plot_algo_lines(
        axA2,
        df=summary_for_ev,
        x_col="ev_count",
        y_col="accuracy_requested_mean",
        x_label="EV Count",
        y_label="AccReq (%)",
        y_scale=100.0,
        legend=True,
    )
    axA2.set_ylim(0.0, 100.0)

    blocked = (
        summary_for_ev.groupby("ev_count", dropna=True)["blocked_ratio_mean"]
        .mean()
        .reset_index()
        .sort_values("ev_count")
    )
    axA2b = None
    if not blocked.empty:
        bx = pd.to_numeric(blocked["ev_count"], errors="coerce").to_numpy(dtype=float)
        by = pd.to_numeric(blocked["blocked_ratio_mean"], errors="coerce").to_numpy(dtype=float) * 100.0
        bm = np.isfinite(bx) & np.isfinite(by)
        if int(np.sum(bm)) >= 1:
            axA2b = axA2.twinx()
            axA2b.plot(
                bx[bm],
                by[bm],
                color="#111827",
                linestyle="--",
                marker="D",
                linewidth=1.5,
                markersize=5.0,
                label="Blocked ratio",
                alpha=0.85,
            )
            axA2b.set_ylabel("Blocked Ratio (%)")
            ymax = float(np.nanmax(by[bm])) if int(np.sum(bm)) > 0 else 0.0
            axA2b.set_ylim(0.0, max(5.0, ymax * 1.25))
            axA2b.legend(loc="upper right", fontsize=legend_fs, framealpha=0.92)

    _apply_paper_style_scaled(figA, font_scale=fs, override_legend_size=False)
    figA.tight_layout(pad=1.30)
    if _save_axes_crop(figA, [axA1], pA1):
        split_result_paths["result_A1_accuracy"] = _to_rel_plot(pA1, plots_base)
    if _save_axes_crop(figA, [axA2, axA2b], pA2):
        split_result_paths["result_A2_accreq_blocked"] = _to_rel_plot(pA2, plots_base)
    figA.savefig(pA, dpi=save_dpi, facecolor="white", transparent=False)
    plt.close(figA)

    # Result R2: p90 latency + fallback-finalized ratio.
    pB = out_dir / "result_B_p90_latency.png"
    pB1 = out_dir / "result_B1_p90_latency_only.png"
    pB2 = out_dir / "result_B2_fallback_only.png"
    figB, (axB1, axB2) = plt.subplots(1, 2, figsize=case_b_size)

    _ = _plot_algo_lines(
        axB1,
        df=summary_for_ev,
        x_col="ev_count",
        y_col="p90_latency_mean",
        x_label="EV Count",
        y_label="p90 Latency (s)",
        y_scale=1.0,
        legend=True,
    )

    _ = _plot_algo_lines(
        axB2,
        df=summary_for_ev,
        x_col="ev_count",
        y_col="fallback_finalized_ratio_mean",
        x_label="EV Count",
        y_label="Fallback Finalized (%)",
        y_scale=100.0,
        legend=True,
    )
    axB2.set_ylim(0.0, 100.0)

    _apply_paper_style_scaled(figB, font_scale=fs, override_legend_size=False)
    figB.tight_layout(pad=1.30)
    if _save_axes_crop(figB, [axB1], pB1):
        split_result_paths["result_B1_p90_latency"] = _to_rel_plot(pB1, plots_base)
    if _save_axes_crop(figB, [axB2], pB2):
        split_result_paths["result_B2_fallback_finalized"] = _to_rel_plot(pB2, plots_base)
    figB.savefig(pB, dpi=save_dpi, facecolor="white", transparent=False)
    plt.close(figB)

    # Result R3: runtime total + runtime breakdown.
    pC = out_dir / "result_C_runtime.png"
    pC1 = out_dir / "result_C1_runtime_total.png"
    pC2 = out_dir / "result_C2_runtime_breakdown.png"
    figC, (axC1, axC2) = plt.subplots(1, 2, figsize=case_b_size)

    drew_runtime = _plot_algo_lines(
        axC1,
        df=summary_for_ev,
        x_col="ev_count",
        y_col="runtime_s_total_mean",
        x_label="EV Count",
        y_label="Runtime per 12 h Scenario (s)",
        y_scale=1.0,
        legend=True,
    )
    if drew_runtime and bool(runtime_log_scale):
        axC1.set_yscale("log")
        _apply_runtime_axis_format(axC1, log_y=True)
    elif drew_runtime:
        _apply_runtime_axis_format(axC1, log_y=False)

    partC = summary_for_ev[pd.to_numeric(summary_for_ev["ev_count"], errors="coerce") == float(target_ev)].copy()
    orderC = _algo_ordered_ids(partC["algorithm_id"].astype(str).tolist())
    if len(orderC) == 0:
        _plot_no_data_panel_scaled(
            axC2,
            x_label=f"Algorithm (EV={target_ev})",
            y_label="Runtime (s)",
            message="No high-load rows",
        )
    else:
        xx = np.arange(len(orderC), dtype=float)
        labelsC = [_paper_algo_short_label(aid) for aid in orderC]
        vals_cost = np.asarray([_safe_algo_mean(partC, aid, "runtime_s_cost_mean") for aid in orderC], dtype=float)
        vals_assign = np.asarray([_safe_algo_mean(partC, aid, "runtime_s_assignment_mean") for aid in orderC], dtype=float)
        vals_fb = np.asarray([_safe_algo_mean(partC, aid, "runtime_s_fallback_mean") for aid in orderC], dtype=float)
        vals_loop = np.asarray([_safe_algo_mean(partC, aid, "runtime_s_loop_mean") for aid in orderC], dtype=float)

        for arr in (vals_cost, vals_assign, vals_fb, vals_loop):
            arr[~np.isfinite(arr)] = 0.0

        axC2.bar(xx, vals_cost, color="#2563eb", alpha=0.90, label="Cost build")
        axC2.bar(xx, vals_assign, bottom=vals_cost, color="#16a34a", alpha=0.90, label="Assignment")
        axC2.bar(xx, vals_fb, bottom=vals_cost + vals_assign, color="#f59e0b", alpha=0.90, label="Fallback")
        axC2.bar(xx, vals_loop, bottom=vals_cost + vals_assign + vals_fb, color="#7c3aed", alpha=0.90, label="Loop/other")
        axC2.set_xticks(xx)
        axC2.set_xticklabels(labelsC, rotation=0)
        axC2.set_xlabel(f"Algorithm (EV={target_ev})")
        axC2.set_ylabel("Runtime Breakdown (s)")
        axC2.grid(axis="y", color="#BFBFBF", linewidth=0.5, linestyle=":")
        axC2.legend(loc="best", fontsize=legend_fs, framealpha=0.92)

    _apply_paper_style_scaled(figC, font_scale=fs, override_legend_size=False)
    figC.tight_layout(pad=1.30)
    if _save_axes_crop(figC, [axC1], pC1):
        split_result_paths["result_C1_runtime_total"] = _to_rel_plot(pC1, plots_base)
    if _save_axes_crop(figC, [axC2], pC2):
        split_result_paths["result_C2_runtime_breakdown"] = _to_rel_plot(pC2, plots_base)
    figC.savefig(pC, dpi=save_dpi, facecolor="white", transparent=False)
    plt.close(figC)

    # Result R4: d_max sweep (accuracy + latency).
    pD = out_dir / "result_D_dmax_sensitivity.png"
    pD1 = out_dir / "result_D1_dmax_accuracy.png"
    pD2 = out_dir / "result_D2_dmax_latency.png"
    figD, (axD1, axD2) = plt.subplots(1, 2, figsize=case_b_size)

    dmax_df = _select_summary_sweep("matcher_delay_max_s")
    if dmax_df.empty:
        dmax_df = _select_sensitivity_sweep(
            "delay_max_s",
            base_knobs={
                "delay_max_s": _to_finite_float(_extract_results_knobs(report).get("delay_max_s")),
                "loss_prob": _to_finite_float(_extract_results_knobs(report).get("loss_prob")),
                "candidate_margin_s": _to_finite_float(_extract_results_knobs(report).get("candidate_margin_s")),
                "sensor_noise_std_a": _to_finite_float(_extract_results_knobs(report).get("sensor_noise_std_a")),
                "sensor_bias_std_a": _to_finite_float(_extract_results_knobs(report).get("sensor_bias_std_a")),
            },
        )
        dmax_x = "delay_max_s"
    else:
        dmax_x = "matcher_delay_max_s"

    _ = _plot_algo_lines(
        axD1,
        df=dmax_df,
        x_col=dmax_x,
        y_col="accuracy_mean",
        x_label="Watermark d_max (s)",
        y_label=f"Accuracy at EV={target_ev} (%)",
        y_scale=100.0,
        legend=True,
    )
    axD1.set_ylim(0.0, 100.0)

    _ = _plot_algo_lines(
        axD2,
        df=dmax_df,
        x_col=dmax_x,
        y_col="p90_latency_mean",
        x_label="Watermark d_max (s)",
        y_label=f"p90 Latency at EV={target_ev} (s)",
        y_scale=1.0,
        legend=True,
    )

    _apply_paper_style_scaled(figD, font_scale=fs, override_legend_size=False)
    figD.tight_layout(pad=1.30)
    if _save_axes_crop(figD, [axD1], pD1):
        split_result_paths["result_D1_dmax_accuracy"] = _to_rel_plot(pD1, plots_base)
    if _save_axes_crop(figD, [axD2], pD2):
        split_result_paths["result_D2_dmax_latency"] = _to_rel_plot(pD2, plots_base)
    figD.savefig(pD, dpi=save_dpi, facecolor="white", transparent=False)
    plt.close(figD)

    # Result R5: Delta sweep (accuracy-runtime tradeoff + GT-in-candidates).
    pE = out_dir / "result_E_candidate_margin_sensitivity.png"
    pE1 = out_dir / "result_E1_candidate_margin_accuracy.png"
    pE2 = out_dir / "result_E2_candidate_margin_runtime_gt.png"
    figE, (axE1, axE2) = plt.subplots(1, 2, figsize=case_b_size)

    delta_df = _select_summary_sweep("candidate_margin_s")
    _ = _plot_algo_lines(
        axE1,
        df=delta_df,
        x_col="candidate_margin_s",
        y_col="accuracy_mean",
        x_label="Candidate Margin Δ (s)",
        y_label=f"Accuracy at EV={target_ev} (%)",
        y_scale=100.0,
        legend=True,
    )
    axE1.set_ylim(0.0, 100.0)

    axE2b = None
    if delta_df.empty:
        _plot_no_data_panel_scaled(
            axE2,
            x_label="Candidate Margin Δ (s)",
            y_label="Runtime / GT-in-candidates",
            message="Need >=2 Δ points in run",
        )
    else:
        drew = 0
        axE2b = axE2.twinx()
        for aid, g in delta_df.groupby("algorithm_id", dropna=False):
            part = (
                g.groupby("candidate_margin_s", dropna=True)[["runtime_s_total_mean", "gt_in_candidates_ratio_mean"]]
                .mean()
                .reset_index()
                .sort_values("candidate_margin_s")
            )
            x = pd.to_numeric(part["candidate_margin_s"], errors="coerce").to_numpy(dtype=float)
            yr = pd.to_numeric(part["runtime_s_total_mean"], errors="coerce").to_numpy(dtype=float)
            yg = pd.to_numeric(part["gt_in_candidates_ratio_mean"], errors="coerce").to_numpy(dtype=float) * 100.0
            m_r = np.isfinite(x) & np.isfinite(yr)
            m_g = np.isfinite(x) & np.isfinite(yg)
            if int(np.sum(m_r)) >= 2:
                axE2.plot(
                    x[m_r],
                    yr[m_r],
                    marker="o",
                    linewidth=1.5,
                    markersize=5.0,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    label=f"{_paper_algo_short_label(str(aid))} runtime",
                )
                drew += 1
            if int(np.sum(m_g)) >= 2:
                axE2b.plot(
                    x[m_g],
                    yg[m_g],
                    marker="s",
                    linestyle="--",
                    linewidth=2.4,
                    markersize=8.6,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    alpha=0.85,
                    label=f"{_paper_algo_short_label(str(aid))} GT-in-cand",
                )

        if drew <= 0:
            _plot_no_data_panel_scaled(
                axE2,
                x_label="Candidate Margin Δ (s)",
                y_label="Runtime / GT-in-candidates",
                message="Need >=2 Δ points per algorithm",
            )
        else:
            axE2.set_xlabel("Candidate Margin Δ (s)")
            axE2.set_ylabel("Runtime at target EV (s)")
            axE2.grid(True, which="major", axis="both", color="#BFBFBF", linewidth=0.5, linestyle=":")
            axE2b.set_ylabel("GT-in-candidates (%)")
            axE2b.set_ylim(0.0, 100.0)
            h1, l1 = axE2.get_legend_handles_labels()
            h2, l2 = axE2b.get_legend_handles_labels()
            if len(h1) + len(h2) > 0:
                axE2.legend(h1 + h2, l1 + l2, loc="best", fontsize=max(7.0, legend_fs * 0.85), framealpha=0.92)

    _apply_paper_style_scaled(figE, font_scale=fs, override_legend_size=False)
    figE.tight_layout(pad=1.30)
    if _save_axes_crop(figE, [axE1], pE1):
        split_result_paths["result_E1_candidate_margin_accuracy"] = _to_rel_plot(pE1, plots_base)
    if _save_axes_crop(figE, [axE2, axE2b], pE2):
        split_result_paths["result_E2_candidate_margin_runtime_gt"] = _to_rel_plot(pE2, plots_base)
    figE.savefig(pE, dpi=save_dpi, facecolor="white", transparent=False)
    plt.close(figE)

    # Result R6: Sampling sweep + impairment sweep.
    pF = out_dir / "result_F_sampling_impairment_sensitivity.png"
    pF1 = out_dir / "result_F1_sampling_sensitivity.png"
    pF2 = out_dir / "result_F2_impairment_sensitivity.png"
    figF, (axF1, axF2) = plt.subplots(1, 2, figsize=case_b_size)

    sample_df = _select_summary_sweep("ev_sample_s")
    sample_x = "ev_sample_s"
    sample_label = "EV Sample Interval (s)"
    if sample_df.empty:
        sample_df = _select_summary_sweep("charger_sample_s")
        sample_x = "charger_sample_s"
        sample_label = "EVSE Sample Interval (s)"

    _ = _plot_algo_lines(
        axF1,
        df=sample_df,
        x_col=sample_x,
        y_col="accuracy_mean",
        x_label=sample_label,
        y_label=f"Accuracy at EV={target_ev} (%)",
        y_scale=100.0,
        legend=True,
    )
    axF1.set_ylim(0.0, 100.0)

    imp_source = pd.DataFrame()
    imp_x_col = ""
    imp_x_label = ""
    imp_x_mult = 1.0
    imp_note = ""
    knobs = _extract_results_knobs(report)
    sens_base_knobs = {
        "delay_max_s": _to_finite_float(knobs.get("delay_max_s")),
        "loss_prob": _to_finite_float(knobs.get("loss_prob")),
        "candidate_margin_s": _to_finite_float(knobs.get("candidate_margin_s")),
        "sensor_noise_std_a": _to_finite_float(knobs.get("sensor_noise_std_a")),
        "sensor_bias_std_a": _to_finite_float(knobs.get("sensor_bias_std_a")),
    }

    for col, lbl, mult in [
        ("loss_prob", "Ingestion Loss Probability (%)", 100.0),
        ("sensor_noise_std_a", "Sensor Noise Std (A)", 1.0),
        ("sensor_bias_std_a", "Sensor Bias Std (A)", 1.0),
    ]:
        tmp = _select_sensitivity_sweep(col, base_knobs=sens_base_knobs)
        if not tmp.empty:
            imp_source = tmp
            imp_x_col = str(col)
            imp_x_label = str(lbl)
            imp_x_mult = float(mult)
            break

    if imp_source.empty:
        delay_misspec_csv = (COMPARE_DIR / rid / "delay_misspec_summary.csv").resolve()
        if delay_misspec_csv.exists():
            try:
                dm_df = pd.read_csv(delay_misspec_csv)
            except Exception:
                dm_df = pd.DataFrame()
            if not dm_df.empty:
                if "algorithm_id" not in dm_df.columns:
                    dm_df["algorithm_id"] = ""
                dm_df["algorithm_id"] = dm_df["algorithm_id"].astype(str)
                for col in ["ev_count", "matcher_delay_max_s", "accuracy_mean"]:
                    if col not in dm_df.columns:
                        dm_df[col] = np.nan
                    dm_df[col] = pd.to_numeric(dm_df[col], errors="coerce")
                ev = pd.to_numeric(dm_df.get("ev_count"), errors="coerce")
                part_dm = dm_df[np.isfinite(ev) & (ev == float(target_ev))].copy()
                x_dm = pd.to_numeric(part_dm.get("matcher_delay_max_s"), errors="coerce")
                part_dm = part_dm[np.isfinite(x_dm)].copy()
                if int(pd.Series(x_dm[np.isfinite(x_dm)]).nunique()) >= 2:
                    imp_source = part_dm
                    imp_x_col = "matcher_delay_max_s"
                    imp_x_label = "Delay Mis-spec d_max (s)"
                    imp_x_mult = 1.0
                    imp_note = "Impairment panel sourced from delay_misspec_summary.csv"

    _ = _plot_algo_lines(
        axF2,
        df=imp_source,
        x_col=imp_x_col if imp_x_col else "loss_prob",
        y_col="accuracy_mean",
        x_label=imp_x_label if imp_x_label else "Impairment knob",
        y_label=f"Accuracy at EV={target_ev} (%)",
        y_scale=100.0,
        legend=True,
        x_mult=imp_x_mult,
    )
    axF2.set_ylim(0.0, 100.0)

    _apply_paper_style_scaled(figF, font_scale=fs, override_legend_size=False)
    figF.tight_layout(pad=1.30)
    # Export split panels with fixed case-A canvas to keep IEEE single-panel ratio.
    figF1, axF1s = plt.subplots(1, 1, figsize=case_a_size)
    _ = _plot_algo_lines(
        axF1s,
        df=sample_df,
        x_col=sample_x,
        y_col="accuracy_mean",
        x_label=sample_label,
        y_label=f"Accuracy at EV={target_ev} (%)",
        y_scale=100.0,
        legend=True,
    )
    axF1s.set_ylim(0.0, 100.0)
    _apply_paper_style_scaled(figF1, font_scale=fs, override_legend_size=False)
    figF1.tight_layout(pad=0.30)
    figF1.savefig(pF1, dpi=save_dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    plt.close(figF1)
    split_result_paths["result_F1_sampling_sensitivity"] = _to_rel_plot(pF1, plots_base)

    figF2s, axF2s = plt.subplots(1, 1, figsize=case_a_size)
    _ = _plot_algo_lines(
        axF2s,
        df=imp_source,
        x_col=imp_x_col if imp_x_col else "loss_prob",
        y_col="accuracy_mean",
        x_label=imp_x_label if imp_x_label else "Impairment knob",
        y_label=f"Accuracy at EV={target_ev} (%)",
        y_scale=100.0,
        legend=True,
        x_mult=imp_x_mult,
    )
    axF2s.set_ylim(0.0, 100.0)
    _apply_paper_style_scaled(figF2s, font_scale=fs, override_legend_size=False)
    figF2s.tight_layout(pad=0.30)
    figF2s.savefig(pF2, dpi=save_dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white", transparent=False)
    plt.close(figF2s)
    split_result_paths["result_F2_impairment_sensitivity"] = _to_rel_plot(pF2, plots_base)

    figF.savefig(pF, dpi=save_dpi, facecolor="white", transparent=False)
    plt.close(figF)

    # Candidate-quality distribution panel (for reviewer defense / explainability).
    pG = out_dir / "result_G_candidate_quality_distribution.png"
    pG_csv = out_dir / "candidate_quality_summary.csv"
    figG, axsG = plt.subplots(2, 2, figsize=(15.4, 11.0))
    axG1, axG2, axG3, axG4 = axsG[0, 0], axsG[0, 1], axsG[1, 0], axsG[1, 1]
    event_trace_df = pd.DataFrame(_read_jsonl((COMPARE_DIR / rid / "event_trace.jsonl").resolve()))
    candidate_quality_rows: list[dict[str, Any]] = []
    if event_trace_df.empty or "algorithm_id" not in event_trace_df.columns:
        for ax in [axG1, axG2, axG3, axG4]:
            _plot_no_data_panel_scaled(ax, x_label="", y_label="", message="event_trace.jsonl not found")
    else:
        event_trace_df["algorithm_id"] = event_trace_df["algorithm_id"].astype(str)
        event_trace_df["ev_count"] = pd.to_numeric(event_trace_df.get("ev_count"), errors="coerce")
        event_trace_df["candidate_count"] = pd.to_numeric(event_trace_df.get("candidate_count"), errors="coerce")
        if "gt_in_candidates" not in event_trace_df.columns:
            event_trace_df["gt_in_candidates"] = False
        else:
            event_trace_df["gt_in_candidates"] = event_trace_df["gt_in_candidates"].astype(bool)
        if "fallback_to_all_triggered" not in event_trace_df.columns:
            event_trace_df["fallback_to_all_triggered"] = False
        else:
            event_trace_df["fallback_to_all_triggered"] = event_trace_df["fallback_to_all_triggered"].astype(bool)

        # G1: candidate_count CDF by algorithm.
        drew_g1 = 0
        for aid, g in event_trace_df.groupby("algorithm_id", dropna=False):
            arr = pd.to_numeric(g.get("candidate_count"), errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                continue
            arr = np.sort(arr)
            y = np.arange(1, arr.size + 1, dtype=float) / float(arr.size)
            axG1.plot(arr, y * 100.0, linewidth=2.2, label=_paper_algo_short_label(str(aid)))
            drew_g1 += 1
        if drew_g1 > 0:
            axG1.set_xlabel("Candidate Count")
            axG1.set_ylabel("CDF (%)")
            axG1.grid(alpha=0.25)
            _paper_ordered_legend(axG1, loc="lower right", fontsize=max(8.5, legend_fs * 0.85))
        else:
            _plot_no_data_panel_scaled(axG1, x_label="Candidate Count", y_label="CDF (%)", message="No candidate_count data")

        # G2: GT-in-candidates by EV count.
        drew_g2 = 0
        for aid, g in event_trace_df.groupby("algorithm_id", dropna=False):
            part = (
                g.groupby("ev_count", dropna=True)["gt_in_candidates"]
                .mean()
                .reset_index()
                .sort_values("ev_count")
            )
            x = pd.to_numeric(part.get("ev_count"), errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(part.get("gt_in_candidates"), errors="coerce").to_numpy(dtype=float) * 100.0
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) <= 0:
                continue
            axG2.plot(x[m], y[m], marker="o", linewidth=2.2, label=_paper_algo_short_label(str(aid)))
            drew_g2 += 1
        if drew_g2 > 0:
            axG2.set_xlabel("EV Count")
            axG2.set_ylabel("GT-in-candidates (%)")
            axG2.set_ylim(0.0, 100.0)
            axG2.grid(alpha=0.25)
            _paper_ordered_legend(axG2, loc="best", fontsize=max(8.5, legend_fs * 0.85))
        else:
            _plot_no_data_panel_scaled(axG2, x_label="EV Count", y_label="GT-in-candidates (%)", message="No GT-in-candidates data")

        # G3: no-candidate->all fallback ratio by EV count.
        drew_g3 = 0
        for aid, g in event_trace_df.groupby("algorithm_id", dropna=False):
            part = (
                g.groupby("ev_count", dropna=True)["fallback_to_all_triggered"]
                .mean()
                .reset_index()
                .sort_values("ev_count")
            )
            x = pd.to_numeric(part.get("ev_count"), errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(part.get("fallback_to_all_triggered"), errors="coerce").to_numpy(dtype=float) * 100.0
            m = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(m)) <= 0:
                continue
            axG3.plot(x[m], y[m], marker="o", linewidth=2.2, label=_paper_algo_short_label(str(aid)))
            drew_g3 += 1
        if drew_g3 > 0:
            axG3.set_xlabel("EV Count")
            axG3.set_ylabel("No-candidate -> all fallback (%)")
            axG3.grid(alpha=0.25)
            _paper_ordered_legend(axG3, loc="best", fontsize=max(8.5, legend_fs * 0.85))
        else:
            _plot_no_data_panel_scaled(axG3, x_label="EV Count", y_label="Fallback (%)", message="No fallback data")

        # G4: candidate_count boxplot by algorithm.
        aids = []
        vals = []
        for aid, g in event_trace_df.groupby("algorithm_id", dropna=False):
            arr = pd.to_numeric(g.get("candidate_count"), errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                continue
            aids.append(_paper_algo_short_label(str(aid)))
            vals.append(arr)
        if len(vals) > 0:
            axG4.boxplot(vals, labels=aids, showfliers=False)
            axG4.set_xlabel("Algorithm")
            axG4.set_ylabel("Candidate Count")
            axG4.grid(axis="y", alpha=0.25)
        else:
            _plot_no_data_panel_scaled(axG4, x_label="Algorithm", y_label="Candidate Count", message="No candidate_count data")

        def _p90_or_nan(x: Any) -> float:
            arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size <= 0:
                return float("nan")
            return float(np.quantile(arr, 0.90))

        cq = (
            event_trace_df.groupby(["algorithm_id", "ev_count"], dropna=False)
            .agg(
                candidate_count_mean=("candidate_count", "mean"),
                candidate_count_p90=("candidate_count", _p90_or_nan),
                gt_in_candidates_ratio=("gt_in_candidates", "mean"),
                fallback_to_all_ratio=("fallback_to_all_triggered", "mean"),
                rows=("algorithm_id", "size"),
            )
            .reset_index()
        )
        candidate_quality_rows = [dict(r) for r in cq.to_dict(orient="records")]

    _apply_paper_style_scaled(figG, font_scale=fs, override_legend_size=False)
    figG.tight_layout(pad=1.30)
    figG.savefig(pG, dpi=220, bbox_inches="tight", pad_inches=0.22)
    plt.close(figG)
    _write_csv(
        pG_csv,
        candidate_quality_rows,
        columns=[
            "algorithm_id",
            "ev_count",
            "candidate_count_mean",
            "candidate_count_p90",
            "gt_in_candidates_ratio",
            "fallback_to_all_ratio",
            "rows",
        ],
    )

    fig_paths = {
        "result_A_overall_accuracy": _to_rel_plot(pA, plots_base),
        "result_B_p90_latency": _to_rel_plot(pB, plots_base),
        "result_C_runtime": _to_rel_plot(pC, plots_base),
        "result_D_finalization_path": _to_rel_plot(pD, plots_base),
        "result_E_cost_unmatched": _to_rel_plot(pE, plots_base),
        "result_F_high_load_summary": _to_rel_plot(pF, plots_base),
        # IEEE manuscript naming (caption-aligned aliases).
        "result_D_dmax_sensitivity": _to_rel_plot(pD, plots_base),
        "result_E_candidate_margin_sensitivity": _to_rel_plot(pE, plots_base),
        "result_F_sampling_impairment_sensitivity": _to_rel_plot(pF, plots_base),
        "result_G_candidate_quality_distribution": _to_rel_plot(pG, plots_base),
        "candidate_quality_summary_csv": _to_rel_plot(pG_csv, plots_base),
        # Additional manifest aliases used by auxiliary plotting readers.
        "result_D_telemetry_sensitivity": _to_rel_plot(pD, plots_base),
        "result_E_candidate_margin": _to_rel_plot(pE, plots_base),
        "result_F_summary": _to_rel_plot(pF, plots_base),
    }
    fig_paths.update(split_result_paths)

    note_text = (
        "R1~R6 regenerated from compare report with sweep-aware results: "
        "d_max sensitivity, candidate-margin tradeoff (with GT-in-candidates), "
        "runtime breakdown, and sampling/impairment panels. "
        "Each two-panel result is also exported as standalone single-panel PNG files."
    )

    return {
        "run_id": rid,
        "target_ev_count": int(target_ev),
        "error_mode": str(mode),
        "paper_font_scale": float(fs),
        "runtime_log_scale": bool(runtime_log_scale),
        "plot_paths": fig_paths,
        "note": note_text,
    }


def _refresh_reviewer_defense_figures_from_csv(run_id: str) -> dict[str, str]:
    rid = _validate_run_id(run_id)
    run_dir = (COMPARE_DIR / rid).resolve()
    plots_base = GENERATED_OUTPUT_DIR.resolve()
    if not run_dir.exists():
        return {}

    def _read_csv(name: str) -> pd.DataFrame:
        path = run_dir / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _plot_metric_vs_ev(df: pd.DataFrame, *, metric_col: str, y_label: str, out_path: Path, scale: float = 1.0) -> None:
        fig, ax = plt.subplots(figsize=(10.0, 6.8))
        if df.empty or metric_col not in df.columns:
            _plot_no_data_panel(ax, x_label="EV Count", y_label=y_label, message="No data")
        else:
            drew = 0
            for aid, g in df.groupby("algorithm_id", dropna=False):
                gg = g.sort_values("ev_count")
                x = pd.to_numeric(gg.get("ev_count"), errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg.get(metric_col), errors="coerce").to_numpy(dtype=float) * float(scale)
                m = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(m)) <= 0:
                    continue
                ax.plot(
                    x[m],
                    y[m],
                    marker="o",
                    linewidth=3.4,
                    markersize=10.4,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    label=_paper_algo_short_label(str(aid)),
                )
                drew += 1
            if drew > 0:
                ax.set_xlabel("EV Count")
                ax.set_ylabel(str(y_label))
                ax.grid(alpha=0.30)
                _paper_ordered_legend(ax, loc="best", fontsize=max(10.0, _PAPER_LEGEND_SIZE * 0.88))
            else:
                _plot_no_data_panel(ax, x_label="EV Count", y_label=y_label, message="No finite data")
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    def _plot_highload_panel(df: pd.DataFrame, out_path: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.6))
        metrics = [
            ("accuracy_requested_mean", "AccReq (%)", 100.0),
            ("blocked_ratio_mean", "Blocked Ratio (%)", 100.0),
            ("p90_latency_mean", "p90 Latency (s)", 1.0),
        ]
        for ax, (col, label, scale) in zip(axes, metrics):
            if df.empty or col not in df.columns:
                _plot_no_data_panel(ax, x_label="EV Count", y_label=label, message="No data")
                continue
            drew = 0
            for aid, g in df.groupby("algorithm_id", dropna=False):
                gg = g.sort_values("ev_count")
                x = pd.to_numeric(gg.get("ev_count"), errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg.get(col), errors="coerce").to_numpy(dtype=float) * float(scale)
                m = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(m)) <= 0:
                    continue
                ax.plot(
                    x[m],
                    y[m],
                    marker="o",
                    linewidth=2.2,
                    markersize=8.6,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    label=_paper_algo_short_label(str(aid)),
                )
                drew += 1
            if drew > 0:
                ax.set_xlabel("EV Count")
                ax.set_ylabel(label)
                ax.grid(alpha=0.24)
            else:
                _plot_no_data_panel(ax, x_label="EV Count", y_label=label, message="No finite data")
        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    def _plot_seed_boxplot(df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(11.0, 6.8))
        if df.empty or "algorithm_id" not in df.columns:
            _plot_no_data_panel(ax, x_label="Algorithm", y_label="AccReq (%)", message="No data")
        else:
            value_col = "accuracy_requested"
            if value_col not in df.columns and "accuracy_requested_mean" in df.columns:
                value_col = "accuracy_requested_mean"
            if value_col not in df.columns:
                _plot_no_data_panel(ax, x_label="Algorithm", y_label="AccReq (%)", message="No accuracy columns")
            else:
                vals = []
                labels = []
                for aid, g in df.groupby("algorithm_id", dropna=False):
                    arr = pd.to_numeric(g.get(value_col), errors="coerce").to_numpy(dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size <= 0:
                        continue
                    vals.append(arr * 100.0)
                    labels.append(_paper_algo_short_label(str(aid)))
                if len(vals) <= 0:
                    _plot_no_data_panel(ax, x_label="Algorithm", y_label="AccReq (%)", message="No finite values")
                else:
                    ax.boxplot(vals, labels=labels, showfliers=False)
                    ax.set_ylabel("AccReq (%)")
                    ax.set_xlabel("Algorithm")
                    ax.grid(axis="y", alpha=0.24)
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    def _plot_delay_misspec(df: pd.DataFrame, out_path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
        specs = [
            ("accuracy_requested_mean", "AccReq (%)", 100.0),
            ("fallback_finalized_ratio_mean", "Fallback Finalized (%)", 100.0),
        ]
        for ax, (col, yl, scale) in zip(axes, specs):
            if df.empty or col not in df.columns:
                _plot_no_data_panel(ax, x_label="Configured d_max (s)", y_label=yl, message="No data")
                continue
            drew = 0
            for aid, g in df.groupby("algorithm_id", dropna=False):
                gg = g.sort_values("matcher_delay_max_s")
                x = pd.to_numeric(gg.get("matcher_delay_max_s"), errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(gg.get(col), errors="coerce").to_numpy(dtype=float) * float(scale)
                m = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(m)) <= 0:
                    continue
                ax.plot(
                    x[m],
                    y[m],
                    marker="o",
                    linewidth=2.2,
                    markersize=8.6,
                    color=_PAPER_ALGO_COLORS.get(str(aid), "#64748b"),
                    label=_paper_algo_short_label(str(aid)),
                )
                drew += 1
            if drew > 0:
                ax.set_xlabel("Configured d_max (s)")
                ax.set_ylabel(yl)
                ax.grid(alpha=0.24)
            else:
                _plot_no_data_panel(ax, x_label="Configured d_max (s)", y_label=yl, message="No finite data")
        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 0:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
        _apply_paper_style_scaled(fig, font_scale=1.0, override_legend_size=False)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=1.30)
        fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)

    p_ablation_summary = run_dir / "ablation_summary.csv"
    p_highload_summary = run_dir / "highload_summary.csv"
    p_seed_summary = run_dir / "seed_robustness_summary.csv"
    p_delay_summary = run_dir / "delay_misspec_summary.csv"

    out_paths: dict[str, tuple[Path, str]] = {
        "fig_ablation_accuracy": (run_dir / "fig_ablation_accuracy.png", "ablation"),
        "fig_ablation_latency": (run_dir / "fig_ablation_latency.png", "ablation"),
        "fig_highload_accreq_blocked_latency": (run_dir / "fig_highload_accreq_blocked_latency.png", "highload"),
        "fig_seed_robustness_boxplot": (run_dir / "fig_seed_robustness_boxplot.png", "seed"),
        "fig_delay_misspec_tradeoff": (run_dir / "fig_delay_misspec_tradeoff.png", "delay"),
    }

    ablation_df = _read_csv(p_ablation_summary.name) if p_ablation_summary.exists() else pd.DataFrame()
    highload_df = _read_csv(p_highload_summary.name) if p_highload_summary.exists() else pd.DataFrame()
    seed_df = _read_csv(p_seed_summary.name) if p_seed_summary.exists() else pd.DataFrame()
    delay_df = _read_csv(p_delay_summary.name) if p_delay_summary.exists() else pd.DataFrame()

    if p_ablation_summary.exists():
        _plot_metric_vs_ev(ablation_df, metric_col="accuracy_mean", y_label="Accuracy (%)", out_path=out_paths["fig_ablation_accuracy"][0], scale=100.0)
        _plot_metric_vs_ev(ablation_df, metric_col="p90_latency_mean", y_label="p90 Latency (s)", out_path=out_paths["fig_ablation_latency"][0], scale=1.0)
    if p_highload_summary.exists():
        _plot_highload_panel(highload_df, out_paths["fig_highload_accreq_blocked_latency"][0])
    if p_seed_summary.exists():
        _plot_seed_boxplot(seed_df, out_paths["fig_seed_robustness_boxplot"][0])
    if p_delay_summary.exists():
        _plot_delay_misspec(delay_df, out_paths["fig_delay_misspec_tradeoff"][0])

    rel: dict[str, str] = {}
    for key, (path, _) in out_paths.items():
        if path.exists():
            rel[key] = _to_rel_plot(path, plots_base)
    return rel
def generate_paper_figures_from_report(
    run_id: str,
    *,
    x_axis: str = "ev_count",
    error_mode: str = DEFAULT_ERROR_MODE,
    runtime_log_scale: bool = False,
    paper_font_scale: float = 1.0,
    selected_results: list[str] | None = None,
) -> dict[str, Any]:
    rid = _validate_run_id(run_id)
    mode = str(error_mode).strip().lower()
    if mode not in VALID_ERROR_MODES:
        raise ValueError(f"error_mode must be one of {sorted(VALID_ERROR_MODES)}")

    x_mode = str(x_axis).strip().lower()
    if x_mode != "ev_count":
        raise ValueError("x_axis is currently fixed to 'ev_count'")

    fs = float(paper_font_scale)
    if not np.isfinite(fs):
        raise ValueError("paper_font_scale must be finite")
    if fs < 0.6 or fs > 2.5:
        raise ValueError("paper_font_scale must be in [0.6, 2.5]")

    results_figures = generate_results_section_figures_from_report(
        rid,
        error_mode=mode,
        paper_font_scale=fs,
        runtime_log_scale=bool(runtime_log_scale),
    )
    results_paths = dict(results_figures.get("plot_paths", {}))

    plots_base = GENERATED_OUTPUT_DIR.resolve()
    diagnostic_plot_dir = COMPARE_DIR / rid / "diagnostic_plots"
    diagnostic_plot_dir.mkdir(parents=True, exist_ok=True)

    base_report = load_algorithm_compare_report(rid)
    diagram_paths = _generate_paper_diagram_figs(
        base_report,
        out_dir=diagnostic_plot_dir,
        plots_base=plots_base,
    )
    trace_example_paths = _generate_trace_example_figs(
        rid,
        out_dir=diagnostic_plot_dir,
        plots_base=plots_base,
    )
    diagram_paths.update(trace_example_paths)
    defense_refreshed_paths = _refresh_reviewer_defense_figures_from_csv(rid)
    diagram_captions = {
        "fig1_system_overview": "End-to-end EV-Link system overview with console-known EVSE reference and anonymous EV ingest stream.",
        "fig2_async_timeline_watermark": "Asynchronous sampling, ingestion delay/loss, and watermark-ready timing structure.",
        "fig3_event_loop_flow": "Online event-loop decision flow: maturity check, candidate gating, assignment, and finalize.",
        "diagnostic_signal_realism_overlay": "Command/EVSE/sensed-EV current overlay with ramp, offset, bias, noise, and ingest jitter.",
        "fig5_candidate_margin_illustration": "Candidate-margin sweep intuition: wider Delta increases candidate coverage and GT inclusion likelihood.",
        "fig6_example_success_case": "Event-trace based successful matching example with timeline annotations.",
        "fig7_example_failure_case": "Event-trace based failure/fallback example for reviewer-facing error analysis.",
        "fig8_posterior_evolution_A3": "A3 posterior evolution at the intermediate update stage (final saturated stage removed) with confidence threshold.",
    }

    paper_manifest_rel = ""
    paper_manifest_obj = {
        "manifest_type": "ieee_access_paper_figures",
        "run_id": str(rid),
        "x_axis": str(x_mode),
        "error_mode": str(mode),
        "runtime_log_scale": bool(runtime_log_scale),
        "paper_font_scale": float(fs),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results_figures": dict(results_paths),
        "diagram_figures": dict(diagram_paths),
        "diagram_captions": dict(diagram_captions),
    }
    try:
        paper_manifest_path = diagnostic_plot_dir / "paper_manifest.json"
        with paper_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(paper_manifest_obj), f, ensure_ascii=False, indent=2)
        paper_manifest_rel = _to_rel_plot(paper_manifest_path, plots_base)
    except Exception:
        paper_manifest_rel = ""

    full_fig_paths = {
        "figR1_overall_accuracy": str(results_paths.get("result_A_overall_accuracy", "")),
        "figR2_p90_latency": str(results_paths.get("result_B_p90_latency", "")),
        "figR3_runtime": str(results_paths.get("result_C_runtime", "")),
        "figR4_finalization_path": str(results_paths.get("result_D_finalization_path", "")),
        "figR5_cost_unmatched": str(results_paths.get("result_E_cost_unmatched", "")),
        "figR6_high_load_summary": str(results_paths.get("result_F_high_load_summary", "")),
    }

    # Backward-compatibility aliases.
    full_fig_paths.update(
        {
            "fig1a_accuracy": full_fig_paths["figR1_overall_accuracy"],
            "fig1b_accreq": full_fig_paths["figR2_p90_latency"],
            "fig2_p90_latency": full_fig_paths["figR2_p90_latency"],
            "fig3_runtime": full_fig_paths["figR3_runtime"],
            "fig4_acc_runtime_tradeoff": full_fig_paths["figR4_finalization_path"],
            "fig5_stability": full_fig_paths["figR6_high_load_summary"],
        }
    )
    full_fig_paths = {k: v for k, v in full_fig_paths.items() if str(v).strip() != ""}

    alias_to_result: dict[str, str] = {
        "r1": "result_A_overall_accuracy",
        "r2": "result_B_p90_latency",
        "r3": "result_C_runtime",
        "r4": "result_D_finalization_path",
        "r5": "result_E_cost_unmatched",
        "r6": "result_F_high_load_summary",
    }
    alias_to_paper: dict[str, str] = {
        "r1": "figR1_overall_accuracy",
        "r2": "figR2_p90_latency",
        "r3": "figR3_runtime",
        "r4": "figR4_finalization_path",
        "r5": "figR5_cost_unmatched",
        "r6": "figR6_high_load_summary",
    }
    alias_text_to_id: dict[str, str] = {
        "r1": "r1",
        "result_a_overall_accuracy": "r1",
        "figr1_overall_accuracy": "r1",
        "r2": "r2",
        "result_b_p90_latency": "r2",
        "figr2_p90_latency": "r2",
        "r3": "r3",
        "result_c_runtime": "r3",
        "figr3_runtime": "r3",
        "r4": "r4",
        "result_d_finalization_path": "r4",
        "result_d_dmax_sensitivity": "r4",
        "figr4_finalization_path": "r4",
        "r5": "r5",
        "result_e_cost_unmatched": "r5",
        "result_e_candidate_margin_sensitivity": "r5",
        "figr5_cost_unmatched": "r5",
        "r6": "r6",
        "result_f_high_load_summary": "r6",
        "result_f_sampling_impairment_sensitivity": "r6",
        "figr6_high_load_summary": "r6",
    }

    selected_aliases: list[str] = []
    if isinstance(selected_results, list) and len(selected_results) > 0:
        for raw in selected_results:
            key = str(raw).strip().lower()
            if key == "":
                continue
            alias = alias_text_to_id.get(key, "")
            if alias == "":
                continue
            if alias not in selected_aliases:
                selected_aliases.append(alias)
    if len(selected_aliases) == 0:
        selected_aliases = ["r1", "r2", "r3", "r4", "r5", "r6"]

    selected_set = set(selected_aliases)
    fig_paths: dict[str, str] = {}
    for alias in selected_aliases:
        paper_key = alias_to_paper.get(alias, "")
        if paper_key != "":
            val = str(full_fig_paths.get(paper_key, ""))
            if val != "":
                fig_paths[paper_key] = val
    if "r1" in selected_set and "figR1_overall_accuracy" in fig_paths:
        fig_paths["fig1a_accuracy"] = fig_paths["figR1_overall_accuracy"]
    if "r2" in selected_set and "figR2_p90_latency" in fig_paths:
        fig_paths["fig1b_accreq"] = fig_paths["figR2_p90_latency"]
        fig_paths["fig2_p90_latency"] = fig_paths["figR2_p90_latency"]
    if "r3" in selected_set and "figR3_runtime" in fig_paths:
        fig_paths["fig3_runtime"] = fig_paths["figR3_runtime"]
    if "r4" in selected_set and "figR4_finalization_path" in fig_paths:
        fig_paths["fig4_acc_runtime_tradeoff"] = fig_paths["figR4_finalization_path"]
    if "r6" in selected_set and "figR6_high_load_summary" in fig_paths:
        fig_paths["fig5_stability"] = fig_paths["figR6_high_load_summary"]

    results_paths_selected: dict[str, str] = {}
    for alias in selected_aliases:
        result_key = alias_to_result.get(alias, "")
        if result_key != "":
            vv = str(results_paths.get(result_key, ""))
            if vv != "":
                results_paths_selected[result_key] = vv

    report_path = COMPARE_DIR / rid / "report.json"
    if report_path.exists():
        try:
            with report_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                obj = {}
            obj["paper_figures"] = {
                "figure_set": "ieee_access_results_r1_to_r6",
                "x_axis": str(x_mode),
                "error_mode": str(mode),
                "runtime_log_scale": bool(runtime_log_scale),
                "paper_font_scale": float(fs),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target_ev_count": int(results_figures.get("target_ev_count", 0) or 0),
                "selected_results": selected_aliases,
                "plot_paths": full_fig_paths,
                "defense_paths_refreshed": dict(defense_refreshed_paths),
            }
            obj["results_figures"] = {
                "figure_set": "ieee_access_results_r1_to_r6",
                "error_mode": str(mode),
                "paper_font_scale": float(fs),
                "runtime_log_scale": bool(runtime_log_scale),
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "target_ev_count": int(results_figures.get("target_ev_count", 0) or 0),
                "selected_results": selected_aliases,
                "plot_paths": dict(results_paths),
                "note": str(results_figures.get("note", "")),
            }
            obj["paper_diagrams"] = {
                "figure_set": "ieee_access_explainability_diagrams",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "plot_paths": dict(diagram_paths),
                "captions": dict(diagram_captions),
            }
            if str(paper_manifest_rel).strip() != "":
                obj["paper_manifest"] = {
                    "path": str(paper_manifest_rel),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return {
        "run_id": rid,
        "x_axis": str(x_mode),
        "error_mode": str(mode),
        "runtime_log_scale": bool(runtime_log_scale),
        "paper_font_scale": float(fs),
        "target_ev_count": int(results_figures.get("target_ev_count", 0) or 0),
        "selected_results": selected_aliases,
        "plot_paths": fig_paths,
        "results_plot_paths": results_paths_selected,
        "diagram_plot_paths": dict(diagram_paths),
        "defense_plot_paths_refreshed": dict(defense_refreshed_paths),
        "diagram_captions": dict(diagram_captions),
        "paper_manifest_json": str(paper_manifest_rel),
        "results_note": str(results_figures.get("note", "")),
        "note": (
            "IEEE Access result figure bundle (R1~R6) and explainability diagrams generated from existing report data. "
            "No re-simulation was executed."
        ),
    }

def _normalize_custom_title(title: Any) -> str:
    if title is None:
        return ""
    text = str(title).strip()
    if len(text) > 120:
        text = text[:120]
    return text


def update_algorithm_compare_title(run_id: str, title: Any) -> dict[str, Any]:
    rid = _validate_run_id(run_id)
    path = COMPARE_DIR / rid / "report.json"
    if not path.exists():
        raise FileNotFoundError(f"report not found: {rid}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("invalid report format")

    custom_title = _normalize_custom_title(title)
    obj["custom_title"] = custom_title

    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
    return {
        "run_id": rid,
        "custom_title": custom_title,
        "display_title": custom_title if custom_title else rid,
    }


def prune_algorithm_compare_reports(keep_per_mode: int = AUTO_PRUNE_KEEP_PER_MODE) -> dict[str, int]:
    keep_n = max(0, int(keep_per_mode))
    if not COMPARE_DIR.exists():
        return {"removed_reports": 0, "removed_orphans": 0}

    removed_reports = 0
    removed_orphans = 0
    per_mode_dirs: dict[str, list[Path]] = {
        m: [] for m in VALID_PATTERN_ASSIGNMENT_MODES
    }

    dirs = sorted(
        [p for p in COMPARE_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in dirs:
        rp = d / "report.json"
        if not rp.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed_orphans += 1
            continue
        try:
            with rp.open("r", encoding="utf-8") as f:
                r = json.load(f)
            mode = str(r.get("pattern_assignment_mode", DEFAULT_PATTERN_ASSIGNMENT_MODE)).strip().lower()
            if mode not in VALID_PATTERN_ASSIGNMENT_MODES:
                mode = DEFAULT_PATTERN_ASSIGNMENT_MODE
            per_mode_dirs.setdefault(mode, []).append(d)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            removed_orphans += 1

    if keep_n == 0:
        for mode_dirs in per_mode_dirs.values():
            for d in mode_dirs:
                shutil.rmtree(d, ignore_errors=True)
                removed_reports += 1
        return {"removed_reports": int(removed_reports), "removed_orphans": int(removed_orphans)}

    for mode_dirs in per_mode_dirs.values():
        for old_dir in mode_dirs[keep_n:]:
            shutil.rmtree(old_dir, ignore_errors=True)
            removed_reports += 1

    return {"removed_reports": int(removed_reports), "removed_orphans": int(removed_orphans)}


def list_algorithm_compare_reports(
    limit: int = 100,
    pattern_assignment_mode: str | None = None,
    include_bundle_children: bool = False,
) -> list[dict[str, Any]]:
    n = max(1, int(limit))
    mode_filter: str | None = None
    if pattern_assignment_mode is not None:
        mode_filter = str(pattern_assignment_mode).strip().lower()
        if mode_filter not in VALID_PATTERN_ASSIGNMENT_MODES:
            raise ValueError(f"pattern_assignment_mode must be one of {sorted(VALID_PATTERN_ASSIGNMENT_MODES)}")
    if not COMPARE_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    dirs = sorted([p for p in COMPARE_DIR.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in dirs:
        path = d / "report.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                r = json.load(f)
            assignment_mode = str(r.get("pattern_assignment_mode", DEFAULT_PATTERN_ASSIGNMENT_MODE)).strip().lower()
            if assignment_mode not in VALID_PATTERN_ASSIGNMENT_MODES:
                assignment_mode = DEFAULT_PATTERN_ASSIGNMENT_MODE
            if mode_filter is not None and assignment_mode != mode_filter:
                continue
            if not bool(include_bundle_children):
                parent_id = str(r.get("bundle_parent_run_id", "")).strip()
                if parent_id != "":
                    continue
            algo_list_raw = r.get("algorithms", [])
            algo_list: list[dict[str, Any]] = []
            if isinstance(algo_list_raw, list) and len(algo_list_raw) > 0:
                for a in algo_list_raw:
                    if not isinstance(a, dict):
                        continue
                    aid = str(a.get("id", "")).strip()
                    if aid == "":
                        continue
                    meta = _algo_meta_for_id(aid)
                    algo_list.append(
                        {
                            "id": str(meta["id"]),
                            "label": str(meta["label"]),
                            "description": str(meta["description"]),
                            "is_current": bool(meta["is_current"]),
                        }
                    )
            if len(algo_list) == 0:
                algo_list = [
                    {
                        "id": s.id,
                        "label": s.label,
                        "description": s.description,
                        "is_current": bool(s.is_current),
                    }
                    for s in _validate_algorithm_ids(None)
                ]
            out.append(
                {
                    "run_id": str(r.get("run_id", d.name)),
                    "run_started": str(r.get("run_started", "")),
                    "repeats": int(r.get("repeats", 0)),
                    "pattern_mode": str(r.get("pattern_mode", DEFAULT_PATTERN_MODE)),
                    "pattern_assignment_mode": assignment_mode,
                    "pattern_assignment_mode_label": PATTERN_ASSIGNMENT_MODE_LABELS.get(assignment_mode, assignment_mode),
                    "pattern_unique_count": int(r.get("pattern_unique_count", DEFAULT_PATTERN_UNIQUE_COUNT)),
                    "command_step_count": int(r.get("command_step_count", DEFAULT_COMMAND_STEP_COUNT)),
                    "tau_values": [int(x) for x in r.get("tau_values", DEFAULT_TAU_VALUES)],
                    "window_values_s": [int(x) for x in r.get("window_values_s", [60 * DEFAULT_COMMAND_STEP_COUNT])],
                    "tau_s": r.get("tau_s"),
                    "window_s": r.get("window_s"),
                    "ev_counts": [int(x) for x in r.get("ev_counts", [])],
                    "total_runs": int(r.get("total_runs", 0)),
                    "algorithms": algo_list,
                    "algorithm_ids": [str(a["id"]) for a in algo_list],
                    "custom_title": _normalize_custom_title(r.get("custom_title", "")),
                    "display_title": _normalize_custom_title(r.get("custom_title", "")) or str(r.get("run_id", d.name)),
                    "report_path": f"/algo-compare/report/{d.name}",
                }
            )
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


def clear_algorithm_compare_reports(pattern_assignment_mode: str | None = None) -> dict[str, int]:
    deleted = 0
    deleted_compare_files = 0
    mode_filter: str | None = None

    if pattern_assignment_mode is not None:
        mode_filter = str(pattern_assignment_mode).strip().lower()
        if mode_filter not in VALID_PATTERN_ASSIGNMENT_MODES:
            raise ValueError(f"pattern_assignment_mode must be one of {sorted(VALID_PATTERN_ASSIGNMENT_MODES)}")

    if COMPARE_DIR.exists():
        for d in COMPARE_DIR.iterdir():
            if not d.is_dir():
                continue
            if mode_filter is not None:
                rp = d / "report.json"
                if not rp.exists():
                    continue
                try:
                    with rp.open("r", encoding="utf-8") as f:
                        r = json.load(f)
                    m = str(r.get("pattern_assignment_mode", DEFAULT_PATTERN_ASSIGNMENT_MODE)).strip().lower()
                    if m not in VALID_PATTERN_ASSIGNMENT_MODES:
                        m = DEFAULT_PATTERN_ASSIGNMENT_MODE
                    if m != mode_filter:
                        continue
                except Exception:
                    continue
            shutil.rmtree(d, ignore_errors=True)
            deleted += 1

    if mode_filter is None:
        # Legacy flat compare artifacts (plots/compare/*.png, *.csv)
        legacy_compare_dir = GENERATED_OUTPUT_DIR / "compare"
        if legacy_compare_dir.exists():
            for p in legacy_compare_dir.iterdir():
                if not p.is_file():
                    continue
                try:
                    p.unlink(missing_ok=True)
                    deleted_compare_files += 1
                except Exception:
                    continue

    return {
        "deleted": int(deleted),
        "deleted_compare_files": int(deleted_compare_files),
    }
