#!/usr/bin/env python3
"""
Compare DyFO variants against EWMA and Persistence baselines.

Loads metrics from two bootstrap evaluation runs and produces a
two-panel figure suitable for the BRACIS paper:

  Top    — R² per walk-forward window (temporal stability)
  Bottom — Aggregate mean ± std for R², MAE, Spearman

Data sources (hardcoded to the canonical N=50, seed=42 runs):
  baselines (EWMA, Persistence, TGAT): bootstrap_eval_tkg_rev3_20260501_200449
  neural models (TGN, ROLAND, GAT-Static): bootstrap_eval_tkg_rev3_20260420_141237

Usage
-----
  python scripts/plot_model_comparison.py
  python scripts/plot_model_comparison.py --out_dir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Data sources
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = ROOT / "results"

# Run with persistence, ewma, tgat  (seed=42, n_tickers=50, 9 windows)
RUN_BASELINES = RESULTS_DIR / "bootstrap_eval_tkg_rev3_20260501_200449" / "bootstrap_summary_tkg_rev3.json"

# Run with tgat, tgn, roland, gat_static  (n_tickers=50, 9 windows)
RUN_NEURAL = RESULTS_DIR / "bootstrap_eval_tkg_rev3_20260420_141237" / "bootstrap_summary_tkg_rev3.json"

# ──────────────────────────────────────────────────────────────────────────────
# Visual style
# ──────────────────────────────────────────────────────────────────────────────

MODEL_STYLE: dict[str, dict] = {
    "ewma":       dict(label="EWMA",        color="#1565C0", ls="--",  lw=1.8, marker="s", ms=5,  zorder=6),
    "persistence":dict(label="Persistence", color="#757575", ls="--",  lw=1.5, marker="^", ms=5,  zorder=5),
    "tgat":       dict(label="TGAT (DyFO)", color="#C62828", ls="-",   lw=2.2, marker="o", ms=6,  zorder=8),
    "tgn":        dict(label="TGN",         color="#E65100", ls="-.",  lw=1.5, marker="D", ms=4,  zorder=4),
    "roland":     dict(label="ROLAND",      color="#6A1B9A", ls=":",   lw=1.5, marker="v", ms=4,  zorder=3),
    "gat_static": dict(label="GAT-Static",  color="#2E7D32", ls="-.",  lw=1.5, marker="P", ms=4,  zorder=3),
}

# Draw order for the summary bar chart (bottom to top)
DRAW_ORDER = ["tgn", "roland", "gat_static", "persistence", "tgat", "ewma"]

# Exact test-period labels computed from walk-forward protocol:
# train=500, val=125, test=125, step=125, bdays from 2018-01-01
# W1: 2020-05-25→2020-11-13  W2: 2020-11-16→2021-05-07  W3: 2021-05-10→2021-10-29
# W4: 2021-11-01→2022-04-22  W5: 2022-04-25→2022-10-14  W6: 2022-10-17→2023-04-07
# W7: 2023-04-10→2023-09-29  W8: 2023-10-02→2024-03-22  W9: 2024-03-25→2024-09-13
WINDOW_LABELS = [
    "May–Nov\n2020", "Nov 20–\nMay 21", "May–Oct\n2021",
    "Nov 21–\nApr 22",  "Apr–Oct\n2022",  "Oct 22–\nApr 23",
    "Apr–Sep\n2023",    "Oct 23–\nMar 24", "Mar–Sep\n2024",
]


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics(json_path: Path) -> dict[str, list[dict]]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)["metrics_by_variant"]


def merge_sources() -> dict[str, list[dict]]:
    """Merge both runs; prefer baselines run for TGAT (same seed as baselines)."""
    data: dict[str, list[dict]] = {}
    for path in (RUN_BASELINES, RUN_NEURAL):
        for variant, windows in load_metrics(path).items():
            if variant not in data:
                data[variant] = windows
    return data


def metric_array(windows: list[dict], key: str) -> np.ndarray:
    return np.array([w[key] for w in windows])


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def make_figure(data: dict[str, list[dict]], out_dir: Path) -> None:
    models = [m for m in DRAW_ORDER if m in data]
    n_windows = len(next(iter(data.values())))
    x = np.arange(1, n_windows + 1)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], hspace=0.38, wspace=0.32)
    ax_r2w  = fig.add_subplot(gs[0, :])    # full-width: R² per window
    ax_bar  = fig.add_subplot(gs[1, 0])    # mean R² bar
    ax_mae  = fig.add_subplot(gs[1, 1])    # mean MAE bar

    # ── Top: R² per window ──────────────────────────────────────────────────
    for m in models:
        st = MODEL_STYLE[m]
        r2 = metric_array(data[m], "r_squared")
        ax_r2w.plot(
            x, r2,
            color=st["color"], ls=st["ls"], lw=st["lw"],
            marker=st["marker"], markersize=st["ms"], zorder=st["zorder"],
            label=f"{st['label']}  (μ={r2.mean():.3f})",
        )

    # Annotate TGN window-8 collapse
    if "tgn" in data:
        r2_tgn = metric_array(data["tgn"], "r_squared")
        worst_w = int(np.argmin(r2_tgn))
        ax_r2w.annotate(
            f"TGN collapse\n(R²={r2_tgn[worst_w]:.3f})",
            xy=(worst_w + 1, r2_tgn[worst_w]),
            xytext=(worst_w + 1.3, r2_tgn[worst_w] - 0.12),
            fontsize=7.5, color=MODEL_STYLE["tgn"]["color"],
            arrowprops=dict(arrowstyle="->", color=MODEL_STYLE["tgn"]["color"], lw=1.0),
        )

    # Information-set callout box for EWMA
    ax_r2w.text(
        0.015, 0.96,
        "EWMA / Persistence observe $\\rho_t$ directly\n"
        "TGAT infers $\\rho_{t+1}$ from structural signals only",
        transform=ax_r2w.transAxes,
        fontsize=7.5, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#E3F2FD", ec="#1565C0", alpha=0.9),
    )

    ax_r2w.set_xticks(x)
    ax_r2w.set_xticklabels(WINDOW_LABELS[:n_windows], fontsize=8)
    ax_r2w.set_ylabel("$R^2$  (correlation forecasting)", fontsize=10)
    ax_r2w.set_title("Walk-Forward $R^2$ per Test Window — N=50 assets, 9 windows (2018–2024)", fontsize=10)
    ax_r2w.set_ylim(-0.15, 1.08)
    ax_r2w.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
    ax_r2w.grid(True, alpha=0.25, lw=0.5)
    ax_r2w.legend(
        loc="lower right", fontsize=8, framealpha=0.92,
        ncol=3, columnspacing=1.0,
    )

    # ── Bottom-left: mean R² bar ─────────────────────────────────────────────
    y_pos = np.arange(len(models))
    r2_means = [metric_array(data[m], "r_squared").mean() for m in models]
    r2_stds  = [metric_array(data[m], "r_squared").std()  for m in models]
    colors   = [MODEL_STYLE[m]["color"] for m in models]

    ax_bar.barh(y_pos, r2_means, xerr=r2_stds, color=colors, alpha=0.85,
                       height=0.55, capsize=3, error_kw=dict(lw=1.2))
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([MODEL_STYLE[m]["label"] for m in models], fontsize=9)
    ax_bar.set_xlabel("Mean $R^2$ ± std", fontsize=9)
    ax_bar.set_title("Aggregate $R^2$", fontsize=9)
    ax_bar.set_xlim(0, 1.08)
    ax_bar.axvline(0, color="black", lw=0.5)
    ax_bar.grid(True, axis="x", alpha=0.25, lw=0.5)

    for i, (mu, sd) in enumerate(zip(r2_means, r2_stds)):
        ax_bar.text(mu + sd + 0.01, i, f"{mu:.3f}", va="center", fontsize=7.5,
                    color=MODEL_STYLE[models[i]]["color"])

    # ── Bottom-right: mean MAE bar ───────────────────────────────────────────
    mae_means = [metric_array(data[m], "mae").mean() for m in models]
    mae_stds  = [metric_array(data[m], "mae").std()  for m in models]

    ax_mae.barh(y_pos, mae_means, xerr=mae_stds, color=colors, alpha=0.85,
                height=0.55, capsize=3, error_kw=dict(lw=1.2))
    ax_mae.set_yticks(y_pos)
    ax_mae.set_yticklabels([MODEL_STYLE[m]["label"] for m in models], fontsize=9)
    ax_mae.set_xlabel("Mean MAE ± std  (lower = better)", fontsize=9)
    ax_mae.set_title("Aggregate MAE", fontsize=9)
    ax_mae.grid(True, axis="x", alpha=0.25, lw=0.5)
    ax_mae.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    for i, (mu, sd) in enumerate(zip(mae_means, mae_stds)):
        ax_mae.text(mu + sd + 0.0002, i, f"{mu:.4f}", va="center", fontsize=7.5,
                    color=MODEL_STYLE[models[i]]["color"])

    # ── Shared footnote ──────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "EWMA/Persistence: observe $\\rho_t$ directly (information-set advantage).  "
        "TGAT/TGN/ROLAND/GAT-Static: infer from event stream and graph structure only.  "
        "Note: DCC-GARCH labels use full-sample (2018–2024) parameter estimates — "
        "a symmetric label-smoothness effect that benefits autoregressive baselines disproportionately.",
        ha="center", fontsize=7, style="italic", color="#555555", wrap=True,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"model_comparison.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved -> {p}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DyFO vs baseline comparison from bootstrap evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", default=str(ROOT / "figures"))
    args = parser.parse_args()

    for p in (RUN_BASELINES, RUN_NEURAL):
        if not p.exists():
            print(f"[ERROR] Result file not found: {p}")
            sys.exit(1)

    print("Loading results …")
    data = merge_sources()
    print(f"  Variants loaded: {list(data.keys())}")
    for m, windows in data.items():
        r2s = [w["r_squared"] for w in windows]
        import statistics
        print(f"  {m:15s}  R²={statistics.mean(r2s):.4f}±{statistics.stdev(r2s):.4f}  ({len(r2s)} windows)")

    make_figure(data, Path(args.out_dir))


if __name__ == "__main__":
    main()
