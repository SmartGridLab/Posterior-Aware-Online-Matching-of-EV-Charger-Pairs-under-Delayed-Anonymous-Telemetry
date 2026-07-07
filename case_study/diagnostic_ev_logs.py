# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

from case_study.diagnostic_terminal_logs import _log
from pair_identification.matching_core import N_STEPS, reindex_at_times, step_medians

# Helper for preserving output format (same as original)
def _fmt_int_list(lst) -> str:
    import math

    def _safe_int(x) -> int:
        if x is None:
            return 0
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(xf):
            return 0
        return int(round(xf))

    return "[" + ",".join(str(_safe_int(x)) for x in lst) + "]"

def _fmt_float1_list(lst) -> str:
    import math

    def _safe_float(x) -> float:
        if x is None:
            return 0.0
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return 0.0
        return xf if math.isfinite(xf) else 0.0

    return "[" + ",".join(f"{_safe_float(x):.1f}" for x in lst) + "]"

def write_ev_mcct_log(
    ev_meta,
    sessions_by_slot,
    ev_series,
    links,
    TAU: int,
    WINDOW: int,
    CHARGER_SAMPLE: int,
    logs_dir: str | None = None,
) -> str:
    """
    Functionized from the original main() block:
    'Write per-EV MCCT log (12h)'.
    Return: saved CSV path.
    """
    if logs_dir is None:
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"EV({len(ev_meta)})_mcct_acc_12h.csv")

    log_rows = []
    sample_steps = int(N_STEPS)
    for ei, meta in enumerate(ev_meta):
        # Build EV 5s grid
        t0_i = meta['arrival_ts']
        n5_i = WINDOW // CHARGER_SAMPLE + 1
        times_5s = pd.date_range(start=t0_i, periods=n5_i, freq=f"{CHARGER_SAMPLE}s")
        e_grid_raw = reindex_at_times(ev_series[ei], times_5s, method="pad").astype(float)

        # Prediction / ground truth
        pred_slot = int(links[ei]['slot'])
        gt_slot = int(meta['evse_idx_gt'])

        # EVSE 5s grid (based on predicted slot)
        s_series = evlink_slot_series_for_window(pred_slot, times_5s[0], times_5s[-1], sessions_by_slot)
        s_grid_raw = reindex_at_times(s_series, times_5s, method="pad").astype(float)

        # Step arrays
        ev_sig_comp = step_medians(e_grid_raw, tau=TAU, grid=CHARGER_SAMPLE, n_steps=sample_steps)
        evse_sig = step_medians(s_grid_raw, tau=TAU, grid=CHARGER_SAMPLE, n_steps=sample_steps)

        # Command patterns from sessions (not slot-id indexed pattern table)
        cmd_pred_sig = evlink_slot_pattern_for_window(pred_slot, times_5s[0], times_5s[-1], sessions_by_slot)
        cmd_gt_sig = evlink_slot_pattern_for_window(gt_slot, times_5s[0], times_5s[-1], sessions_by_slot)

        ev_tok = meta['token']
        evse_tok = f"{pred_slot:02d}"
        link_s = f"{ev_tok}-{gt_slot:02d}"
        cost_val = float(links[ei]['cost'])

        log_rows.append(dict(
            ev_token=ev_tok,
            evse_token=evse_tok,
            ev_step_currents_comp=_fmt_int_list(ev_sig_comp),
            evse_step_currents=_fmt_float1_list(evse_sig),
            command_step_currents="[" + ",".join(str(int(x)) for x in cmd_pred_sig) + "]",
            command_gt_step_currents="[" + ",".join(str(int(x)) for x in cmd_gt_sig) + "]",
            link=link_s,
            cost=round(cost_val, 6),
        ))

    df_log = pd.DataFrame(log_rows, columns=[
        "ev_token", "evse_token", "ev_step_currents_comp",
        "evse_step_currents", "command_step_currents", "command_gt_step_currents", "link", "cost",
    ])
    df_log.to_csv(out_path, index=False, encoding="utf-8")
    _log(f"[LOG] wrote {out_path} ({len(df_log)} rows)")
    return out_path


# Wrap helpers from pair_identification.matching_core
from pair_identification.matching_core import _slot_series_1s_for_window as evlink_slot_series_for_window
from pair_identification.matching_core import slot_pattern_for_window as evlink_slot_pattern_for_window
