# -*- coding: utf-8 -*-
import os
import pandas as pd

# Terminal log buffer (same semantics as original)
_TERM_LOGS = []

def _log(msg: str):
    """Write to console and in-memory buffer at the same time (same behavior as original)."""
    print(msg, flush=True)
    try:
        _TERM_LOGS.append({
            "ts": pd.Timestamp.now(tz=None).isoformat(timespec="seconds"),
            "message": msg
        })
    except Exception:
        pass

def write_terminal_log_csv(term_dir: str, ev_count: int) -> str:
    """Save terminal logs to CSV (functionized from the original main footer block)."""
    os.makedirs(term_dir, exist_ok=True)
    term_path = os.path.join(term_dir, f"ev({ev_count})_terminal_log.csv")
    if len(_TERM_LOGS) == 0:
        _TERM_LOGS.append({
            "ts": pd.Timestamp.now(tz=None).isoformat(timespec="seconds"),
            "message": "[INFO] (no terminal messages captured)"
        })
    df_term = pd.DataFrame(_TERM_LOGS, columns=["ts", "message"])
    df_term.to_csv(term_path, index=False, encoding="utf-8")
    if not os.path.exists(term_path):
        print(f"[LOG][WARN] Terminal log path not found after write: {term_path}", flush=True)
    _log(f"[LOG] wrote {term_path} ({len(df_term)} lines)")
    return term_path
