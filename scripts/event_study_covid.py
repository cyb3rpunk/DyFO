#!/usr/bin/env python3
"""
Event Study: COVID-19 Market Crash (March 2020)

Visualises the inherent temporal lag of autoregressive baselines (EWMA, Persistence)
versus the event-driven inference of DyFO during the COVID-19 crash, with the VIX
shown as a market-stress indicator.

Three-panel figure
------------------
  Top    — Pairwise correlation: actual (black), EWMA (blue), DyFO (red, optional),
            Persistence (gray dashed).
  Bottom — Daily returns for each asset as the event signals that DyFO observes.

Usage
-----
  # Quick demo — EWMA / Persistence only (no training required):
  python scripts/event_study_covid.py

  # Full figure including DyFO predictions:
  #   Step 1 – generate predictions (test window covers March 2020):
  python scripts/train_link_prediction.py \\
      --variant temporal_kg --start 2018-01-01 --end 2020-06-30 \\
      --n_tickers 50 --save_preds_path results/covid_dyfo_preds.csv

  #   Step 2 – plot:
  python scripts/event_study_covid.py --dyfo_preds results/covid_dyfo_preds.csv

Output
------
  figures/event_study_covid_<A>_<B>.pdf  (and .png)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────────────
# Constants (match train_link_prediction.py defaults)
# ──────────────────────────────────────────────────────────────────────────────
EWMA_ALPHA = 0.05
ROLLING_WINDOW = 63

PLOT_START = "2019-10-01"
PLOT_END = "2020-07-31"

CRASH_START = "2020-02-20"
CRASH_PEAK = "2020-03-23"

KEY_DATES = {
    "2020-02-24": "WHO warning\n(Feb 24)",
    "2020-03-09": "Black Monday\n(Mar 9)",
    "2020-03-16": "Circuit breaker\n(Mar 16)",
    "2020-03-23": "S&P bottom\n(Mar 23)",
}

# Colour palette
C_ACTUAL = "#212121"
C_PERSIST = "#9E9E9E"
C_EWMA = "#1565C0"
C_DYFO = "#C62828"
C_CRASH = "#FFCDD2"
C_VIX = "#6A1B9A"


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed — run: pip install yfinance")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"][tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return prices.dropna(how="all")


def _download_vix(start: str, end: str) -> pd.Series:
    vix = _download_prices(["^VIX"], start=start, end=end)
    if "^VIX" not in vix.columns:
        raise RuntimeError("Could not download ^VIX from Yahoo Finance.")
    return vix["^VIX"].dropna()


def _rolling_corr(prices: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    rets = prices[[a, b]].pct_change().dropna()
    return rets[a].rolling(window).corr(rets[b])


def _rolling_corr_two_series(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return df["a"].rolling(window).corr(df["b"])


def _ewma_prediction(actual: pd.Series, alpha: float) -> pd.Series:
    """One-step-ahead EWMA prediction.

    At close of day t the EWMA state absorbs ρ_t; that state is the prediction
    for day t+1.  We shift by 1 so that prediction[t] uses information up to t-1,
    matching the model's causal setup.
    """
    state = actual.ewm(alpha=alpha, adjust=False).mean()
    return state.shift(1)


def _persistence_prediction(actual: pd.Series) -> pd.Series:
    """ρ̂_{t+1} = ρ_t  (naive random-walk forecast)."""
    return actual.shift(1)


def _load_dyfo_preds(csv_path: str, a: str, b: str) -> pd.Series | None:
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"[WARN] DyFO predictions file not found: {csv_path}")
        return None
    mask = ((df["src"] == a) & (df["dst"] == b)) | ((df["src"] == b) & (df["dst"] == a))
    sub = df[mask].set_index("date").sort_index()
    if sub.empty:
        print(f"[WARN] No predictions for pair {a}–{b} in {csv_path}")
        return None
    return sub["pred"].groupby(level=0).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Lag estimation
# ──────────────────────────────────────────────────────────────────────────────

def _lag_days(
    actual: pd.Series,
    prediction: pd.Series,
    threshold_pct: float = 0.5,
) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
    """Return (lag_days, actual_cross_date, pred_cross_date)."""
    pre_mean = actual.loc[:CRASH_START].dropna().mean()
    peak = actual.loc[CRASH_START:CRASH_PEAK].max()
    if pd.isna(peak) or peak <= pre_mean:
        return 0, None, None
    thresh = pre_mean + threshold_pct * (peak - pre_mean)

    a_cross = actual.loc[CRASH_START:CRASH_PEAK].dropna()
    a_cross = a_cross[a_cross >= thresh]
    p_cross = prediction.loc[CRASH_START:CRASH_PEAK].dropna()
    p_cross = p_cross[p_cross >= thresh]

    if a_cross.empty or p_cross.empty:
        return 0, None, None
    d0, d1 = a_cross.index[0], p_cross.index[0]
    return int((d1 - d0).days), d0, d1


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def make_figure(
    actual: pd.Series,
    ewma: pd.Series,
    persist: pd.Series,
    prices: pd.DataFrame,
    vix: pd.Series,
    a: str,
    b: str,
    dyfo: pd.Series | None = None,
    model_label: str = "DyFO (event-driven)",
    out_dir: Path | None = None,
) -> None:
    def crop(s: pd.Series) -> pd.Series:
        return s.loc[PLOT_START:PLOT_END].dropna()

    act = crop(actual)
    ew = crop(ewma)
    pe = crop(persist)
    vx = crop(vix)
    rets_a = (prices[a].pct_change() * 100).loc[PLOT_START:PLOT_END].dropna()
    rets_b = (prices[b].pct_change() * 100).loc[PLOT_START:PLOT_END].dropna()
    vix_thresh = float(vx.quantile(0.8)) if len(vx) > 0 else np.nan

    lag, d_actual, d_ewma = _lag_days(actual, ewma)

    fig, (ax_c, ax_r, ax_v) = plt.subplots(
        3, 1,
        figsize=(11, 8.6),
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.0], "hspace": 0.06},
        sharex=True,
    )

    # ── crash shading ────────────────────────────────────────────────────────
    ts0, ts1 = pd.Timestamp(CRASH_START), pd.Timestamp(CRASH_PEAK)
    for ax in (ax_c, ax_r, ax_v):
        ax.axvspan(ts0, ts1, color=C_CRASH, alpha=0.45, zorder=0)

    # ── key-date verticals ───────────────────────────────────────────────────
    for date_str, lbl in KEY_DATES.items():
        dt = pd.Timestamp(date_str)
        ax_c.axvline(dt, color="#757575", lw=0.7, linestyle=":", zorder=1)
        ax_c.text(
            dt, 0.98,
            lbl,
            transform=ax_c.get_xaxis_transform(),
            fontsize=6.5, color="#757575",
            ha="center", va="top",
            rotation=90,
        )

    # ── correlation lines ────────────────────────────────────────────────────
    ax_c.plot(pe.index, pe.values, color=C_PERSIST, lw=1.2, ls="--", alpha=0.65,
              label=r"Persistence  $\hat{\rho}_{t+1}=\rho_t$", zorder=3)
    ax_c.plot(ew.index, ew.values, color=C_EWMA, lw=2.0,
              label=rf"EWMA ($\alpha={EWMA_ALPHA}$)", zorder=4)
    ax_c.plot(act.index, act.values, color=C_ACTUAL, lw=1.8,
              label=r"Actual $\rho_t$ (rolling 63-day)", zorder=5)

    if dyfo is not None:
        dy = crop(dyfo)
        ax_c.plot(dy.index, dy.values, color=C_DYFO, lw=2.0,
                  label=model_label, zorder=6)

    # ── EWMA lag annotation ──────────────────────────────────────────────────
    if lag > 0 and d_actual is not None and d_ewma is not None:
        y_arr = actual.loc[d_actual] + 0.04
        ax_c.annotate(
            "",
            xy=(d_ewma, y_arr),
            xytext=(d_actual, y_arr),
            arrowprops=dict(arrowstyle="<->", color=C_EWMA, lw=1.6),
        )
        mid = d_actual + (d_ewma - d_actual) / 2
        ax_c.text(
            mid, y_arr + 0.025,
            f"EWMA lag ≈ {lag} days",
            ha="center", va="bottom", fontsize=8, color=C_EWMA,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_EWMA, alpha=0.85),
        )

    ax_c.set_ylabel(f"Pairwise Correlation  {a}–{b}", fontsize=10)
    ax_c.set_ylim(-0.05, 1.10)
    ax_c.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_c.grid(True, alpha=0.25, lw=0.5)

    # ── returns panel ────────────────────────────────────────────────────────
    bar_colors = np.where(rets_a.values >= 0, "#2E7D32", "#B71C1C")
    ax_r.bar(rets_a.index, rets_a.values, color=bar_colors, width=0.8, alpha=0.75,
             label=f"{a} return (%)")
    ax_r.plot(rets_b.index, rets_b.values, color="#E65100", lw=0.9, alpha=0.85,
              label=f"{b} return (%)")
    ax_r.axhline(0, color="black", lw=0.5)
    ax_r.set_ylabel("Daily return (%)", fontsize=9)
    ax_r.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax_r.grid(True, alpha=0.2, lw=0.5)

    ax_v.plot(vx.index, vx.values, color=C_VIX, lw=1.8, label="VIX (market stress)")
    if np.isfinite(vix_thresh):
        ax_v.axhline(
            vix_thresh, color=C_VIX, lw=1.0, ls="--", alpha=0.8,
            label=f"VIX p80 = {vix_thresh:.1f}",
        )
        stress_mask = vx >= vix_thresh
        if stress_mask.any():
            ax_v.fill_between(
                vx.index,
                vx.values,
                vix_thresh,
                where=stress_mask.values,
                color=C_VIX,
                alpha=0.18,
            )
    ax_v.set_ylabel("VIX", fontsize=9)
    ax_v.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax_v.grid(True, alpha=0.2, lw=0.5)

    # ── x-axis ───────────────────────────────────────────────────────────────
    ax_v.xaxis.set_major_locator(mdates.MonthLocator())
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_v.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    plt.setp(ax_v.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

    # ── title ────────────────────────────────────────────────────────────────
    dyfo_note = "" if dyfo is None else "  |  DyFO reacts via event signals"
    fig.suptitle(
        f"Event Study: COVID-19 Market Crash — {a}–{b} Correlation Forecasting\n"
        rf"EWMA ($\alpha={EWMA_ALPHA}$) inherits DCC persistence lag; VIX marks the stress regime{dyfo_note}",
        fontsize=10, y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.975])

    # ── save ─────────────────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"event_study_covid_{a}_{b}"
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def make_macro_context_figure(
    spy_prices: pd.Series,
    vix: pd.Series,
    out_dir: Path | None = None,
    rolling_window: int = 21,
) -> None:
    spy = spy_prices.loc[PLOT_START:PLOT_END].dropna()
    vix_plot = vix.loc[PLOT_START:PLOT_END].dropna()
    spy_ret = spy.pct_change() * 100.0
    vix_chg = vix_plot.pct_change() * 100.0
    corr = _rolling_corr_two_series(spy_ret, vix_chg, rolling_window).dropna()
    vix_thresh = float(vix_plot.quantile(0.8)) if len(vix_plot) > 0 else np.nan

    fig, (ax_p, ax_c, ax_v) = plt.subplots(
        3, 1,
        figsize=(11, 8.6),
        gridspec_kw={"height_ratios": [1.3, 2.2, 1.1], "hspace": 0.06},
        sharex=True,
    )

    ts0, ts1 = pd.Timestamp(CRASH_START), pd.Timestamp(CRASH_PEAK)
    for ax in (ax_p, ax_c, ax_v):
        ax.axvspan(ts0, ts1, color=C_CRASH, alpha=0.45, zorder=0)

    for date_str, lbl in KEY_DATES.items():
        dt = pd.Timestamp(date_str)
        ax_c.axvline(dt, color="#757575", lw=0.7, linestyle=":", zorder=1)
        ax_c.text(
            dt, 0.98, lbl,
            transform=ax_c.get_xaxis_transform(),
            fontsize=6.5, color="#757575",
            ha="center", va="top", rotation=90,
        )

    ax_p.plot(spy.index, spy.values, color="#1B5E20", lw=2.0, label="SPY level")
    ax_p.set_ylabel("SPY", fontsize=9)
    ax_p.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_p.grid(True, alpha=0.2, lw=0.5)

    ax_c.plot(
        corr.index, corr.values, color="#37474F", lw=2.2,
        label=rf"Rolling corr: SPY return vs VIX change ({rolling_window}d)",
    )
    ax_c.axhline(0.0, color="black", lw=0.6, ls="--", alpha=0.5)
    ax_c.set_ylabel("Correlation", fontsize=10)
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_c.grid(True, alpha=0.25, lw=0.5)

    ax_v.plot(vix_plot.index, vix_plot.values, color=C_VIX, lw=1.8, label="VIX (market stress)")
    if np.isfinite(vix_thresh):
        ax_v.axhline(
            vix_thresh, color=C_VIX, lw=1.0, ls="--", alpha=0.8,
            label=f"VIX p80 = {vix_thresh:.1f}",
        )
        stress_mask = vix_plot >= vix_thresh
        if stress_mask.any():
            ax_v.fill_between(
                vix_plot.index,
                vix_plot.values,
                vix_thresh,
                where=stress_mask.values,
                color=C_VIX,
                alpha=0.18,
            )
    ax_v.set_ylabel("VIX", fontsize=9)
    ax_v.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax_v.grid(True, alpha=0.2, lw=0.5)

    ax_v.xaxis.set_major_locator(mdates.MonthLocator())
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_v.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    plt.setp(ax_v.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

    fig.suptitle(
        "Event Study: COVID-19 Market Crash — SPY and VIX Stress Context\n"
        rf"Rolling correlation between SPY returns and VIX changes ({rolling_window}-day window)",
        fontsize=10, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    if out_dir is None:
        out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "event_study_covid_spy_vix"
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved -> {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Event study: COVID-19 crash — correlation forecasting lag",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["pair", "macro_context"], default="pair")
    parser.add_argument("--ticker_a", default="AAPL")
    parser.add_argument("--ticker_b", default="MSFT")
    parser.add_argument(
        "--dyfo_preds", default=None,
        help="CSV produced by train_link_prediction.py --save_preds_path. "
             "If omitted the figure shows only EWMA and Persistence.",
    )
    parser.add_argument(
        "--model_label", default="DyFO (event-driven)",
        help="Legend label for the model predictions line (e.g. 'TGAT', 'TGN').",
    )
    parser.add_argument("--rolling_window", type=int, default=ROLLING_WINDOW)
    parser.add_argument("--vix_start", default="2019-01-01")
    parser.add_argument("--vix_end", default="2020-12-31")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.mode == "macro_context":
        print("Downloading SPY prices (2019-01-01 -> 2020-12-31) ...")
        spy_prices = _download_prices(["SPY"], start="2019-01-01", end="2020-12-31")["SPY"]
        print(f"  {len(spy_prices)} SPY points.")
        print(f"Downloading VIX (^VIX)  ({args.vix_start} -> {args.vix_end}) ...")
        vix = _download_vix(start=args.vix_start, end=args.vix_end)
        print(f"  {len(vix)} VIX points.")
        make_macro_context_figure(
            spy_prices=spy_prices,
            vix=vix,
            out_dir=out_dir,
            rolling_window=min(args.rolling_window, 21),
        )
        return

    a, b = args.ticker_a, args.ticker_b

    print(f"Downloading prices for [{a}, {b}]  (2019-01-01 → 2020-12-31) …")
    prices = _download_prices([a, b], start="2019-01-01", end="2020-12-31")
    print(f"  {len(prices)} trading days.")

    print(f"Computing {args.rolling_window}-day rolling correlation …")
    print(f"Downloading VIX (^VIX)  ({args.vix_start} -> {args.vix_end}) ...")
    vix = _download_vix(start=args.vix_start, end=args.vix_end)
    print(f"  {len(vix)} VIX points.")

    actual = _rolling_corr(prices, a, b, args.rolling_window)

    ewma = _ewma_prediction(actual, EWMA_ALPHA)
    persist = _persistence_prediction(actual)

    dyfo = None
    if args.dyfo_preds:
        print(f"Loading DyFO predictions from {args.dyfo_preds} …")
        dyfo = _load_dyfo_preds(args.dyfo_preds, a, b)
        if dyfo is not None:
            print(f"  {len(dyfo)} prediction points loaded.")
    else:
        print(
            "No --dyfo_preds provided.  Showing EWMA + Persistence only.\n"
            "To add TGAT predictions, first train TGAT and export the test predictions:\n"
            "  python scripts/train_link_prediction.py \\\n"
            "      --variant tgat --start 2018-01-01 --end 2020-06-30 \\\n"
            "      --n_tickers 50 --save_preds_path results/covid_tgat_preds.csv\n"
            "  python scripts/event_study_covid.py --dyfo_preds results/covid_tgat_preds.csv --model_label TGAT"
        )
    make_figure(actual, ewma, persist, prices, vix, a, b,
                dyfo=dyfo, model_label=args.model_label, out_dir=out_dir)


if __name__ == "__main__":
    main()
