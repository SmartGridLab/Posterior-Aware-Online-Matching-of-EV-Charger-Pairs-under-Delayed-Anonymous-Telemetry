#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared implementation for the manuscript result figures.

Requirement (see docs/MANUSCRIPT_FIGURE_GENERATION.md):
  * Use Times New Roman for the manuscript result figures.
  * Preserve the manuscript design choices: data values, colors, markers,
    line styles, legend labels, font sizes, and panel proportions.
  * Use the canonical causal re-run data under
    case_study/scalability_analysis/accuracy/ for data-backed result figures
    and the simulated ablation table (figure6_ablation_simulated.csv) for the
    ablation figure.

Font: real "Times New Roman" is used when present (e.g. on macOS); on Linux the
metric-compatible "Liberation Serif" / "Nimbus Roman" substitute is used (visually
identical to Times New Roman). mathtext uses the serif-matching STIX set.

File-to-paper mapping in the manuscript:
  figure3_base_load_comparison.pdf   -> Figure 3
  figure4_sampling_sensitivity.pdf   -> Figure 4
  figure5_posterior_diagnostic.pdf   -> Supplementary Fig. S1
  figure6_ablation_summary.pdf       -> Figure 8
  figure7_impairment_sweeps.pdf      -> Figure 5
  figure8_heterogeneous_response.pdf -> Figure 6
  figure9_codebook_separation.pdf    -> Figure 7
"""
from __future__ import annotations
import argparse, os, shutil
from pathlib import Path
import tempfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO / "paper"
CANONICAL_RESULTS_DIR = REPO / "case_study" / "scalability_analysis" / "accuracy" / "canonical_run_20260907_111056_515561"
CANONICAL_SUMMARY_CSV = CANONICAL_RESULTS_DIR / "summary.csv"
CANONICAL_RAW_CSV = CANONICAL_RESULTS_DIR / "raw_runs.csv"
# TABLE 5 calibration/evaluation protocol: EV=300 repeats 1-5 are the
# hyperparameter-calibration slice, so every reported base-load aggregate at
# EV=300 uses the held-out evaluation repeats 6-20 (EV in {100, 500}: 1-20).
HELD_OUT_MIN_REPEAT = {100: 1, 300: 6, 500: 1}
PAPER_FIGURE_FILES = {
    3: "figure3_base_load_comparison.pdf",
    4: "figure4_sampling_sensitivity.pdf",
    5: "figure5_posterior_diagnostic.pdf",
    6: "figure6_ablation_summary.pdf",
    # figure files 7–9 (manuscript Figures 5, 6 and 7; see the module docstring)
    7: "figure7_impairment_sweeps.pdf",
    8: "figure8_heterogeneous_response.pdf",
    9: "figure9_codebook_separation.pdf",
}

# ---- font: Times New Roman (only the family changes) ----------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Liberation Serif",
                              "Nimbus Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
# Embed glyphs as TrueType subsets (fonttype 42) instead of matplotlib's default
# Type 3 fonts. IEEE PDF eXpress rejects Type 3, so the exported figure PDFs must
# carry only embedded TrueType/Type 1 subsets.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# font SIZES kept at the paper values (unchanged)
plt.rcParams.update({
    "font.size": 9.0, "axes.titlesize": 9.5, "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
})
plt.rcParams.update({
    "axes.grid": True, "grid.color": "#BFBFBF", "grid.linestyle": ":",
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5, "lines.markersize": 5.0,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---- exact paper palette / markers / labels (UNCHANGED design) -------------
STYLE = {  # algorithm_id : (label, color, marker, linestyle)
    "time_only_baseline":                 ("Time-prior",     "#7f7f7f", "s", "--"),
    "single_only":                        ("Greedy",         "#ff7f0e", "^", "-"),
    "nomura_original_interval_hungarian": ("Global",         "#2ca02c", "D", "-"),
    "bayesian_windowed":                  ("Posterior", "#1f77b4", "o", "-"),
}
ORDER = ["time_only_baseline", "single_only",
         "nomura_original_interval_hungarian", "bayesian_windowed"]

REP = dict(candidate_margin_s=120, matcher_delay_max_s=10, ev_sample_s=30, tau_s=60)

# ---- consistent sizing across the four result figures (figure files 3/4/5/6) ---
# Native figure widths are set equal to the on-page display widths so the font
# scale is 1.0 in every figure (text renders at the same size in all of them):
TEXT_W = 6.99    # \textwidth  in inches  (505.12 pt)  -> figure3/figure6 full width
COL_W  = 3.36    # \columnwidth in inches (242.67 pt)  -> figure4/figure5 single col
# Base box aspect ratio (axes height / width). The single-column pair (figure4/figure5)
# uses this; figure3 and figure6 take taller per-figure values. Tunable via the
# FIG_BOX_ASPECT environment variable.
BOX_ASPECT = float(os.environ.get("FIG_BOX_ASPECT", "0.44"))
FIGURE3_BOX_ASPECT = 0.54  # figure3: slightly taller panels than the single-column base
FIGURE6_BOX_ASPECT = 0.62  # figure6: taller than the base aspect
LEG_GAP_PTS = 2.2   # tuned so the x-label->legend frame gap renders ~6 pt (the
                    # nominal value is measured to the x-label's font-descent bbox,
                    # which sits ~4 pt below the visible glyphs)


def _box(ax, val=None):
    """Force this axes' plot box to a given (or the base) proportion."""
    ax.set_box_aspect(BOX_ASPECT if val is None else val)


def _legend_below(fig, ax, handles, labels, *, ncol, gap_pts=LEG_GAP_PTS, x=None, **kw):
    """Place a legend whose top edge sits ~gap_pts below ax's x-axis label.

    Freezes the (constrained) layout first, measures the x-label, then anchors the
    legend in figure coordinates. x=None centers it on ax; pass x=0.5 to center it
    on the whole figure (shared legend under multiple panels). bbox='tight' on save
    preserves the point-accurate gap.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    y_lbl = fig.transFigure.inverted().transform((0, ax.xaxis.label.get_window_extent(r).y0))[1]
    y = y_lbl - gap_pts / (fig.get_figheight() * 72.0)
    if x is None:
        p = ax.get_position(); x = 0.5 * (p.x0 + p.x1)
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(x, y),
                      bbox_transform=fig.transFigure, ncol=ncol, frameon=True,
                      edgecolor="#808080", **kw)


def _load_summary() -> pd.DataFrame:
    if not CANONICAL_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing canonical summary CSV: {CANONICAL_SUMMARY_CSV}")
    return pd.read_csv(CANONICAL_SUMMARY_CSV)


def _sel(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


def _load_raw() -> pd.DataFrame:
    if not CANONICAL_RAW_CSV.exists():
        raise FileNotFoundError(f"Missing canonical raw-runs CSV: {CANONICAL_RAW_CSV}")
    return pd.read_csv(CANONICAL_RAW_CSV)


def _protocol_stats(raw: pd.DataFrame, aid: str, ev: int, col: str):
    """Mean and 95% CI (1.96*sd/sqrt(n), the summary.csv convention) of one
    base-load cell over the TABLE 5 evaluation repeats."""
    m = (raw["algorithm_id"] == aid) & (raw["ev_count"] == ev)
    for k, v in REP.items():
        m &= raw[k] == v
    m &= raw["repeat_idx"] >= HELD_OUT_MIN_REPEAT[ev]
    vals = raw[m][col]
    n = int(len(vals))
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    return float(vals.mean()), 1.96 * sd / (n ** 0.5)


def _series_line(ax, xs, ys, errs, aid, label_on=True):
    lab, col, mk, ls = STYLE[aid]
    # Small markers so the 95% CI error bars/caps are not hidden underneath the
    # marker faces (the intervals are narrow at the representative operating point).
    # 2026-07-02: halved again (3.5 -> 1.8) because the CI whiskers were still
    # obscured by the marker faces in Figures 3/4.
    ax.errorbar(xs, ys, yerr=errs, label=(lab if label_on else None),
                color=col, marker=mk, linestyle=ls, markersize=1.8,
                capsize=2.5, capthick=0.8, markeredgecolor=col,
                markerfacecolor=col, elinewidth=0.8, zorder=3)


# --------------------------------------------------------------------------- figure3 (manuscript Figure 3)
def _save(fig, out: Path, stem: str, save_png: bool) -> None:
    fig.savefig(out / f"{stem}.pdf")
    if save_png:
        fig.savefig(out / f"{stem}.png")


def generate_figure3_base_load_comparison(out: Path, save_png: bool = False):
    raw = _load_raw()  # aggregated per the TABLE 5 evaluation protocol
    evs = [100, 300, 500]
    # Full-width 3-panel figure; native width == \textwidth so it is not rescaled
    # on the page. Generous height; bbox='tight' trims to content.
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 3.0), constrained_layout=True)
    panels = [("(a) Admitted-session accuracy", "accuracy", "Accuracy (%)", 100.0),
              ("(b) Requested-session accuracy", "accuracy_requested", "Accuracy (%)", 100.0),
              ("(c) Total runtime", "runtime_s_total", "Runtime (s)", 1.0)]
    for ax, (title, col, ylab, scale) in zip(axes, panels):
        for aid in ORDER:
            ys, es = [], []
            for ev in evs:
                mean, ci = _protocol_stats(raw, aid, ev, col)
                ys.append(mean * scale)
                es.append(ci * scale)
            _series_line(ax, evs, ys, es, aid)
        ax.set_title(title, loc="center")
        ax.set_xlabel("EV sessions")
        ax.set_ylabel(ylab)
        ax.set_xticks(evs)
        _box(ax, FIGURE3_BOX_ASPECT)  # a little taller than the base proportion
    handles, labels = axes[0].get_legend_handles_labels()
    # one shared legend, centered under all three panels, ~6 pt below the x-labels
    _legend_below(fig, axes[1], handles, labels, ncol=4, x=0.5)
    _save(fig, out, "figure3_base_load_comparison", save_png)
    plt.close(fig)


# --------------------------------------------------------------------------- figure4 (manuscript Figure 4)
def generate_figure4_sampling_sensitivity(out: Path, save_png: bool = False):
    s = _load_summary()
    dts = [5, 15, 30]
    # Single-column figure; native width == \columnwidth, same as figure5.
    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.7), constrained_layout=True)
    rep = {k: v for k, v in REP.items() if k != "ev_sample_s"}
    for aid in ORDER:
        ys, es = [], []
        for dt in dts:
            r = _sel(s, ev_count=500, ev_sample_s=dt, algorithm_id=aid, **rep).iloc[0]
            ys.append(float(r["accuracy_mean"]) * 100.0)
            es.append(float(r["accuracy_ci95"]) * 100.0)
        _series_line(ax, dts, ys, es, aid)
    ax.set_xlabel(r"EV sampling period $\Delta t_{\mathrm{EV}}$ (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(dts)
    _box(ax)
    # Legend below the axes, ~6 pt under the x-label (no overlap with data lines).
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, ax, h, l, ncol=2, handlelength=1.6, columnspacing=1.0)
    _save(fig, out, "figure4_sampling_sensitivity", save_png)
    plt.close(fig)


# --------------------------------------------------------------------------- figure5 (Supplementary Fig. S1)
# Posterior diagnostic curve for figure5. The six diagnostic indices are the
# renormalized top-1/top-2 posteriors recorded at three checkpoints within each of
# the two update stages of a single watermark decision for one representative
# session at the representative operating point (EV=500). The trace is the measured
# posterior of a canonical single-scenario replay (base_seed 2187631072); regenerate
# it with scripts/extract_figure5_posterior_trace.py.
FIGURE5_TRACE_CSV = (
    REPO / "case_study" / "scalability_analysis" / "accuracy" / "figure5_posterior_trace.csv"
)


def generate_figure5_posterior_diagnostic(out: Path, save_png: bool = False):
    if not FIGURE5_TRACE_CSV.exists():
        raise FileNotFoundError(
            f"Missing Figure 5 posterior trace CSV: {FIGURE5_TRACE_CSV}. "
            "Run scripts/extract_figure5_posterior_trace.py to produce it."
        )
    trace = pd.read_csv(FIGURE5_TRACE_CSV).sort_values("diagnostic_index")
    idx = [int(v) for v in trace["diagnostic_index"].tolist()]
    top1 = [float(v) for v in trace["top1_posterior"].tolist()]
    top2 = [float(v) for v in trace["top2_posterior"].tolist()]
    # Checkpoint labels come from the trace itself (stage, checkpoint): the three
    # checkpoints of each update stage are the carry-over, the time prior and the
    # waveform (current) likelihood; a dotted rule separates the two stages.
    _ckpt_word = {
        "carry_over_uniform_prior": "carry-over",
        "carry_over_prev_power": "carry-over",
        "time_prior_applied": "time prior",
        "current_likelihood_renormalized": "waveform",
    }
    labels = []
    for stage, ck in zip(trace["stage"].tolist(), trace["checkpoint"].tolist()):
        labels.append(f"S{int(stage)}\n{_ckpt_word.get(str(ck), str(ck))}")
    # Single-column figure; native width == \columnwidth, same as figure4.
    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.7), constrained_layout=True)
    ax.plot(idx, top1, color="#1f77b4", marker="o", linestyle="-",
            label="Top-1 belief")
    ax.plot(idx, top2, color="#ff7f0e", marker="^", linestyle="--",
            label="Top-2 belief")
    ax.set_xlabel("Checkpoint within one watermark decision")
    ax.set_ylabel("Normalized posterior belief")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0.0, 1.02)
    stages = [int(s) for s in trace["stage"].tolist()]
    if len(set(stages)) == 2:
        split = max(i for i, s in zip(idx, stages) if s == min(stages)) + 0.5
        ax.axvline(split, color="0.35", linestyle=":", linewidth=0.9)
        lo, hi = min(idx), max(idx)
        ax.text((lo + split) / 2.0, 0.985, "Stage 1", ha="center", va="top", fontsize=7.5)
        ax.text((split + hi) / 2.0, 0.985, "Stage 2", ha="center", va="top", fontsize=7.5)
    _box(ax, 0.58)  # taller than the single-column base aspect
    # Legend moved BELOW the axes in a single horizontal row so it no longer
    # covers the curves; ~6 pt under the x-label.
    h, l = ax.get_legend_handles_labels()
    _legend_below(fig, ax, h, l, ncol=2, handlelength=1.6, columnspacing=1.0,
                  fontsize=7.5)
    _save(fig, out, "figure5_posterior_diagnostic", save_png)
    plt.close(fig)


# --------------------------------------------------------------------------- figure6 (manuscript Figure 8)
# Ablation (3-panel). Data simulated by `reproduce_all.py --mode ablation` on the
# canonical base-load scenario seeds and tabulated into
# case_study/scalability_analysis/accuracy/figure6_ablation_simulated.csv by
# scripts/build_figure6_data.py (TABLE 5 evaluation repeats; the full-design
# rows equal the canonical raw runs aggregated under the same protocol);
# see docs/MANUSCRIPT_FIGURE_GENERATION.md.
def generate_figure6_ablation_summary(out: Path, save_png: bool = False):
    import csv as _csv
    from matplotlib import lines as _mlines
    src = REPO / "case_study" / "scalability_analysis" / "accuracy" / "figure6_ablation_simulated.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing ablation data CSV: {src}. "
            "Run scripts/build_figure6_data.py to regenerate it from the committed ablation run."
        )
    with src.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    families = [("Greedy", "(a) Greedy matcher"),
                ("Global", "(b) Global matcher"),
                ("Posterior", "(c) Posterior matcher")]
    evs = [100, 300, 500]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 4.0), constrained_layout=True)
    for ax, (fam, title) in zip(axes, families):
        seen = []  # preserve CSV order of first appearance
        for r in rows:
            if r["matcher_family"] == fam and r["variant"] not in [s[0] for s in seen]:
                seen.append((r["variant"], r["marker"], r["linestyle"],
                             r["color"], float(r["alpha"]),
                             float(r.get("dodge") or 0.0),
                             str(r.get("markerfill") or "filled")))
        handles = []
        for var, mk, ls, col, al, dodge, mfill in seen:
            m = {int(r["ev_count"]): float(r["accuracy"]) for r in rows
                 if r["matcher_family"] == fam and r["variant"] == var}
            ys = [m[ev] for ev in evs]
            full = var.endswith("(full)")
            mface = "white" if mfill == "open" else col
            msize = 2.5 if full else 2.1
            lw = 1.6 if full else 1.1
            lstyle = None if str(ls).strip().lower() == "none" else ls
            # Lines are drawn at the true x positions; only the markers of
            # coinciding curves are dodged horizontally (declared in the
            # caption), so no plotted value is displaced.
            if lstyle is not None:
                ax.plot(evs, ys, color=col, alpha=al, linestyle=lstyle,
                        marker="None", linewidth=lw)
            ax.plot([ev + dodge for ev in evs], ys, color=col, alpha=al,
                    marker=mk, linestyle="None", markersize=msize,
                    markeredgecolor=col, markerfacecolor=mface,
                    markeredgewidth=0.7)
            handles.append(_mlines.Line2D(
                [], [], color=col, alpha=al, marker=mk,
                linestyle=(lstyle or "None"), linewidth=lw, markersize=msize,
                markeredgecolor=col, markerfacecolor=mface, label=var))
        ax.set_title(title, loc="center")
        ax.set_xlabel("EV sessions")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(evs)
        ax.set_ylim(66, 101)
        _box(ax, FIGURE6_BOX_ASPECT)
        ax._fig6_handles = handles
    # per-panel legends below each panel, ~6 pt under the x-label; two columns
    # everywhere (the Greedy panel carries six series and thus three legend
    # rows — three columns would overflow into the neighbouring panel)
    for ax in axes:
        h = ax._fig6_handles
        _legend_below(fig, ax, h, [x.get_label() for x in h], ncol=2,
                      fontsize=6.8, handlelength=1.8, columnspacing=1.2,
                      labelspacing=0.42, handletextpad=0.45, borderpad=0.38,
                      markerscale=2.4)
    _save(fig, out, "figure6_ablation_summary", save_png)
    plt.close(fig)


# --------------------------------------------------------------------------- figure7 (manuscript Figure 5, impairment sweeps)
# Impairment sweeps at the representative point: accuracy of the four
# matchers vs measurement-loss probability (per EV-side measurement), mean ingestion delay (watermark d_max held at 10 s, plus
# one matched-d_max cell) and sensing-distortion multiplier. Data: figure7_impairment_sweeps.csv
# built by scripts/build_figure7_data.py from the committed impairment runs (20 paired repeats,
# same canonical EV = 500 scenario seeds in every cell).
FIGURE7_CSV = REPO / "case_study" / "scalability_analysis" / "accuracy" / "figure7_impairment_sweeps.csv"


def generate_figure7_impairment_sweeps(out: Path, save_png: bool = False):
    if not FIGURE7_CSV.exists():
        raise FileNotFoundError(
            f"Missing Figure 7 data CSV: {FIGURE7_CSV}. Run scripts/build_figure7_data.py to build it from the committed impairment runs."
        )
    df = pd.read_csv(FIGURE7_CSV)
    panels = [
        ("loss", "(a) Measurement loss", "Measurement-loss probability", "log"),
        ("delay", r"(b) Ingestion delay ($d_{\max}$ = 10 s)", "Mean ingestion delay (s)", "log"),
        ("noise", "(c) Sensing distortion", r"Sensing-distortion multiplier $m$", "log"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 3.0), constrained_layout=True)
    for ax, (axis, title, xlabel, xscale) in zip(axes, panels):
        sub = df[(df["axis"] == axis) & (~df["matched_dmax"].astype(bool))]
        xs_all = sorted(sub["x_value"].unique())
        for aid in ORDER:
            rows = sub[sub["algorithm_id"] == aid].sort_values("x_value")
            xs = rows["x_value"].to_numpy(dtype=float)
            ys = rows["accuracy_mean"].to_numpy(dtype=float) * 100.0
            es = rows["accuracy_ci95"].to_numpy(dtype=float) * 100.0
            _series_line(ax, xs, ys, es, aid)
        # matched-watermark remedy cell on the delay axis: hollow markers at the same x
        matched = df[(df["axis"] == axis) & (df["matched_dmax"].astype(bool))]
        for aid in ORDER:
            rows = matched[matched["algorithm_id"] == aid]
            if len(rows) == 0:
                continue
            lab, col, mk, _ls = STYLE[aid]
            ax.errorbar(rows["x_value"].to_numpy(dtype=float), rows["accuracy_mean"].to_numpy(dtype=float) * 100.0,
                        yerr=rows["accuracy_ci95"].to_numpy(dtype=float) * 100.0, color=col, marker=mk,
                        linestyle="None", markersize=4.0, markerfacecolor="white", markeredgecolor=col,
                        capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=4,
                        label=("matched watermark $d_{\\max}$" if aid == ORDER[-1] else None))
        ax.set_title(title, loc="center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%)")
        if xscale == "log":
            ax.set_xscale("log")
            ax.set_xticks(xs_all)
            ax.set_xticklabels([("%g" % x) for x in xs_all])
            ax.minorticks_off()
        _box(ax, FIGURE3_BOX_ASPECT)
    handles, labels = axes[1].get_legend_handles_labels()
    _legend_below(fig, axes[1], handles, labels, ncol=5, x=0.5)
    _save(fig, out, "figure7_impairment_sweeps", save_png)
    plt.close(fig)


# --------------------------------------------------------------------------- figure8 (manuscript Figure 6, response heterogeneity)
# Accuracy vs EV-session count under per-session response heterogeneity (first-step lag
# ~ U(2, 30) s, ramp ~ U(1, 10) A/s) next to the nominal fitted model (canonical run), TABLE 5
# protocol, 95 % CI. Data: figure8_heterogeneous_response.csv from scripts/build_figure8_data.py.
FIGURE8_CSV = REPO / "case_study" / "scalability_analysis" / "accuracy" / "figure8_heterogeneous_response.csv"


def generate_figure8_heterogeneous_response(out: Path, save_png: bool = False):
    from matplotlib import lines as _mlines
    if not FIGURE8_CSV.exists():
        raise FileNotFoundError(f"Missing Figure 8 data CSV: {FIGURE8_CSV}. Run scripts/build_figure8_data.py first.")
    df = pd.read_csv(FIGURE8_CSV)
    evs = [100, 300, 500]
    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.7), constrained_layout=True)
    handles = []
    for aid in ORDER:
        lab, col, mk, ls = STYLE[aid]
        for model, lstyle, face, alpha in (("nominal", "--", "white", 0.9), ("heterogeneous", "-", col, 1.0)):
            rows = df[(df["algorithm_id"] == aid) & (df["model"] == model)].sort_values("ev_count")
            if len(rows) == 0:
                continue
            ys = rows["accuracy_mean"].to_numpy(dtype=float) * 100.0
            es = rows["accuracy_ci95"].to_numpy(dtype=float) * 100.0
            ax.errorbar(rows["ev_count"].to_numpy(dtype=float), ys, yerr=es, color=col, marker=mk, linestyle=lstyle,
                        markersize=2.6, markerfacecolor=face, markeredgecolor=col, capsize=2.5, capthick=0.8,
                        elinewidth=0.8, alpha=alpha, zorder=3)
        handles.append(_mlines.Line2D([], [], color=col, marker=mk, linestyle="-", markersize=3.0, label=lab))
    handles.append(_mlines.Line2D([], [], color="#444444", linestyle="-", label="heterogeneous response"))
    # Explicit dash pattern and a longer legend handle: at the default pattern the
    # 3 pt marker covers the single dash gap, so the key rendered solid and the
    # caption's "dashed lines, hollow markers" had no matching entry.
    handles.append(_mlines.Line2D([], [], color="#444444", linestyle="--", dashes=(3.5, 1.8), marker="o", markerfacecolor="white", markersize=3.0, label="nominal fitted model"))
    ax.set_xlabel("EV sessions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(evs)
    _box(ax)
    _legend_below(fig, ax, handles, [h.get_label() for h in handles], ncol=3, handlelength=2.8, columnspacing=0.9, fontsize=7.0)
    _save(fig, out, "figure8_heterogeneous_response", save_png)
    plt.close(fig)


FIGURE9_CSV = REPO / "case_study" / "scalability_analysis" / "accuracy" / "figure9_codebook_separation.csv"


def generate_figure9_codebook_separation(out: Path, save_png: bool = False):
    """Figure file figure9 (manuscript Figure 7, codebook band sweep): accuracy vs realised codebook separation (set-point band sweep)."""
    from matplotlib import lines as _mlines
    if not FIGURE9_CSV.exists():
        raise FileNotFoundError(f"Missing Figure 9 data CSV: {FIGURE9_CSV}. Run scripts/build_figure9_data.py first.")
    df = pd.read_csv(FIGURE9_CSV)
    cells = df.drop_duplicates("cell").sort_values("realised_l1_mean_a")
    fig, ax = plt.subplots(1, 1, figsize=(COL_W, 2.8), constrained_layout=True)
    handles = []
    for aid in ("single_only", "nomura_original_interval_hungarian", "bayesian_windowed"):
        lab, col, mk, ls = STYLE[aid]
        rows = df[df["algorithm_id"] == aid].sort_values("realised_l1_mean_a")
        xs = rows["realised_l1_mean_a"].to_numpy(dtype=float)
        ys = rows["accuracy_mean"].to_numpy(dtype=float) * 100.0
        es = rows["accuracy_ci95"].to_numpy(dtype=float) * 100.0
        ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, linestyle="-", markersize=2.8, capsize=2.5, capthick=0.8, elinewidth=0.8, zorder=3)
        canon = rows[rows["cell"] == "B30"]
        if len(canon):
            ax.plot(canon["realised_l1_mean_a"].to_numpy(dtype=float), canon["accuracy_mean"].to_numpy(dtype=float) * 100.0, marker=mk, linestyle="none",
                    markersize=5.0, markerfacecolor="white", markeredgecolor=col, zorder=4)
        handles.append(_mlines.Line2D([], [], color=col, marker=mk, linestyle="-", markersize=3.0, label=lab))
    handles.append(_mlines.Line2D([], [], color="#444444", marker="o", linestyle="none", markerfacecolor="white", markersize=4.5, label="6–30 A (TABLE 6)"))
    ax.set_xlabel("Realized mean pairwise $\\ell_1$ setpoint distance (A)")
    ax.set_ylabel("Accuracy (%)")
    top = ax.secondary_xaxis("top")
    top.set_xticks(cells["realised_l1_mean_a"].to_numpy(dtype=float))
    top.set_xticklabels([f"{int(lo)}–{int(hi)} A" for lo, hi in zip(cells["setpoint_lo_a"], cells["setpoint_hi_a"])], fontsize=6.5)
    top.tick_params(length=2)
    _box(ax)
    _legend_below(fig, ax, handles, [h.get_label() for h in handles], ncol=2, handlelength=1.6, columnspacing=0.9, fontsize=7.0)
    _save(fig, out, "figure9_codebook_separation", save_png)
    plt.close(fig)


def install_paper_figure(figure_number: int, output_dir: Path, paper_dir: Path) -> Path:
    """Copy one generated PDF figure into the manuscript paper directory."""
    paper_dir.mkdir(parents=True, exist_ok=True)
    name = PAPER_FIGURE_FILES[int(figure_number)]
    src = output_dir / name
    if not src.exists():
        raise FileNotFoundError(f"Expected generated figure is missing: {src}")
    dest = paper_dir / name
    shutil.copy2(src, dest)
    return dest


def run_single_figure_cli(
    *,
    figure_number: int,
    description: str,
    generator,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir",
        default=str(Path(tempfile.gettempdir()) / f"evlink_paper_figure{figure_number}"),
        help="directory where the generated figure PDF is written",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="also write a PNG preview next to the generated PDF",
    )
    parser.add_argument(
        "--install-paper",
        action="store_true",
        help="copy the generated PDF into paper/ for manuscript compilation",
    )
    parser.add_argument("--paper-dir", default=str(PAPER_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator(output_dir, save_png=args.png)

    output_name = PAPER_FIGURE_FILES[int(figure_number)]
    print(f"[generate] wrote: {output_dir / output_name}")
    if args.install_paper:
        installed = install_paper_figure(int(figure_number), output_dir, Path(args.paper_dir))
        print(f"[generate] installed: {installed}")
    print(
        "[generate] serif font resolved to:",
        matplotlib.font_manager.FontProperties(family="serif").get_name(),
    )
    return 0
