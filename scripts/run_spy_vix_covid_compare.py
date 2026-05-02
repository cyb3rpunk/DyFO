#!/usr/bin/env python3
"""
Compare EWMA vs Persistence vs TGAT for SPY–^VIX correlation prediction
during the COVID-19 crash (March 2020).

SPY–VIX is a particularly challenging pair: the correlation is persistently
negative (≈ −0.7 to −0.5 in normal regimes) and dives further during crises,
testing whether TGAT can track the regime shift faster than autoregressive
baselines.

Pipeline
--------
1. Download SPY + ^VIX prices.
2. Compute 63-day rolling correlation (actual), EWMA and Persistence one-step
   ahead forecasts.
3. Optionally train TGAT on a small universe that includes both SPY and ^VIX,
   then save per-pair test-period predictions.
4. Generate a 3-panel comparison figure (correlation forecasts | SPY returns vs
   VIX changes | VIX level).

Usage
-----
  # Baselines only (no training, instant):
  python scripts/run_spy_vix_covid_compare.py --skip_tgat

  # Full comparison — trains TGAT from scratch:
  python scripts/run_spy_vix_covid_compare.py --epochs 10

  # Reuse previously saved TGAT predictions:
  python scripts/run_spy_vix_covid_compare.py --tgat_preds results/spy_vix_tgat_preds.csv

Output
------
  figures/spy_vix_covid_compare.pdf  (and .png)
  results/spy_vix_tgat_preds.csv     (if TGAT is trained)
"""

from __future__ import annotations

import argparse
import datetime
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

from scripts.event_study_covid import (
    _download_prices,
    _ewma_prediction,
    _persistence_prediction,
    _load_dyfo_preds,
    _lag_days,
    EWMA_ALPHA,
    ROLLING_WINDOW,
    PLOT_START,
    PLOT_END,
    CRASH_START,
    CRASH_PEAK,
    KEY_DATES,
    C_ACTUAL,
    C_PERSIST,
    C_EWMA,
    C_DYFO,
    C_CRASH,
    C_VIX,
)

# ──────────────────────────────────────────────────────────────────────────────
# Universe & split constants
# ──────────────────────────────────────────────────────────────────────────────

# Full 50-stock universe + ^VIX as node 51.
# VIX needs 50 stock neighbours generating real events (price updates, earnings,
# corporate actions) so the TGAT attention mechanism can propagate temporal
# information into the VIX embedding.  With only 10 nodes the VIX embedding
# stays near-static (no earnings, no sector peer events) and the decoder
# collapses to predicting the training-period mean.
from dyfo.core.ticker_registry import TICKERS_50
# SPY is the benchmark (not in TICKERS_50), add it explicitly as node 0.
UNIVERSE = ["SPY"] + TICKERS_50 + ["^VIX"]

TICKER_A = "SPY"
TICKER_B = "^VIX"

# Extend training back to 2015 so TGAT sees the Aug-2015 China shock and the
# Q4-2018 selloff — periods where SPY-VIX correlation varied noticeably.
# Without these regime-change examples the model memorises the stable mean.
DATA_START = "2015-01-01"
DATA_END   = "2020-12-31"
TRAIN_END  = "2019-09-30"
VAL_END    = "2019-12-31"
TEST_END   = "2020-06-30"

DEFAULT_PREDS_PATH = str(ROOT / "results" / "spy_vix_tgat_preds.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_corr_pair(prices: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    """Rolling correlation between daily returns of assets *a* and *b*."""
    rets = prices[[a, b]].pct_change().dropna()
    return rets[a].rolling(window).corr(rets[b])


# ──────────────────────────────────────────────────────────────────────────────
# TGAT training
# ──────────────────────────────────────────────────────────────────────────────

def _int_day_to_iso(day: int) -> str:
    epoch = datetime.date(2000, 1, 1)
    return (epoch + datetime.timedelta(days=int(day))).isoformat()


def _slice_dates(sorted_dates: list, start: str, end: str) -> list:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return [d for d in sorted_dates
            if start_ts <= pd.Timestamp(_int_day_to_iso(d)) <= end_ts]


def train_tgat_and_save_preds(epochs: int, seed: int, save_preds_path: str) -> None:
    """Train TGAT on UNIVERSE (pre-COVID) and export test-period predictions."""
    from dyfo.config import DataConfig, DyFOConfig
    from dyfo.logging_utils import setup_logging
    from scripts.run_bootstrap_eval_v5 import (
        TGN_LR, TGN_PATIENCE, TGN_USE_COSINE, load_or_prepare_data,
    )
    from scripts.train_link_prediction import train_link_prediction

    logger = setup_logging("dyfo.spy_vix_covid", log_to_file=False)
    config = DyFOConfig(model_variant="tgat")
    data_config = DataConfig(
        tickers=UNIVERSE,
        benchmark_ticker="SPY",
        start_date=DATA_START,
        end_date=DATA_END,
    )

    logger.info("Preparing data for universe: %s", UNIVERSE)
    data = load_or_prepare_data(
        tickers=UNIVERSE,
        start=DATA_START,
        end=DATA_END,
        benchmark="SPY",
        config=config,
        data_config=data_config,
        logger=logger,
    )

    train_dates = _slice_dates(data["sorted_dates"], DATA_START, TRAIN_END)
    val_dates = _slice_dates(
        data["sorted_dates"],
        (pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)).date().isoformat(),
        VAL_END,
    )
    test_dates = _slice_dates(
        data["sorted_dates"],
        (pd.Timestamp(VAL_END) + pd.Timedelta(days=1)).date().isoformat(),
        TEST_END,
    )

    if not train_dates or not val_dates or not test_dates:
        raise RuntimeError(
            "One of the train/val/test slices is empty. Check date boundaries.\n"
            f"  train: {len(train_dates)} | val: {len(val_dates)} | test: {len(test_dates)}"
        )

    logger.info(
        "Split | train=%s..%s (%d days) | val=%s..%s (%d days) | test=%s..%s (%d days)",
        _int_day_to_iso(train_dates[0]), _int_day_to_iso(train_dates[-1]), len(train_dates),
        _int_day_to_iso(val_dates[0]),   _int_day_to_iso(val_dates[-1]),   len(val_dates),
        _int_day_to_iso(test_dates[0]),  _int_day_to_iso(test_dates[-1]),  len(test_dates),
    )

    metrics = train_link_prediction(
        tickers=UNIVERSE,
        start=DATA_START,
        end=TEST_END,
        benchmark="SPY",
        num_epochs=epochs,
        lr=TGN_LR,
        corr_threshold=0.3,
        neg_ratio=1.0,
        early_stopping_patience=TGN_PATIENCE,
        weight_decay=1e-4,
        pos_weight=1.0,
        mode="regression",
        model_variant="tgat",
        seed=seed,
        prepared_data=data,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        use_cosine_schedule=TGN_USE_COSINE,
        save_preds_path=save_preds_path,
    )
    r2 = metrics.get("r_squared", float("nan"))
    mae = metrics.get("mae", float("nan"))
    print(f"TGAT test metrics (all pairs) — R²={r2:.4f}  MAE={mae:.4f}")
    print(f"TGAT predictions saved -> {save_preds_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

def make_spy_vix_figure(
    actual: pd.Series,
    ewma: pd.Series,
    persist: pd.Series,
    spy_prices: pd.Series,
    vix_level: pd.Series,
    tgat: pd.Series | None = None,
    out_dir: Path | None = None,
) -> None:
    """3-panel figure: correlation forecasts | SPY returns | VIX level."""

    def crop(s: pd.Series) -> pd.Series:
        return s.loc[PLOT_START:PLOT_END].dropna()

    act = crop(actual)
    ew  = crop(ewma)
    pe  = crop(persist)
    vx  = crop(vix_level)
    spy_ret = (spy_prices.pct_change() * 100).loc[PLOT_START:PLOT_END].dropna()
    vix_ret = (vix_level.pct_change() * 100).loc[PLOT_START:PLOT_END].dropna()
    vix_thresh = float(vx.quantile(0.8)) if len(vx) > 0 else np.nan

    # Lag analysis against EWMA
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
            dt, 0.98, lbl,
            transform=ax_c.get_xaxis_transform(),
            fontsize=6.5, color="#757575",
            ha="center", va="top", rotation=90,
        )

    # ── correlation lines ────────────────────────────────────────────────────
    ax_c.plot(pe.index, pe.values, color=C_PERSIST, lw=1.2, ls="--", alpha=0.65,
              label=r"Persistence  $\hat{\rho}_{t+1}=\rho_t$", zorder=3)
    ax_c.plot(ew.index, ew.values, color=C_EWMA, lw=2.0,
              label=rf"EWMA ($\alpha={EWMA_ALPHA}$)", zorder=4)
    ax_c.plot(act.index, act.values, color=C_ACTUAL, lw=1.8,
              label=r"Actual $\rho_t$ (rolling 63-day)", zorder=5)
    if tgat is not None:
        tg = crop(tgat)
        ax_c.plot(tg.index, tg.values, color=C_DYFO, lw=2.0,
                  label="TGAT (event-driven)", zorder=6)

    # ── EWMA lag annotation ──────────────────────────────────────────────────
    # For SPY-VIX the correlation drops (becomes more negative) during a crash,
    # so the lag is measured as the delay until the correlation falls below
    # a threshold — _lag_days() uses the rise in magnitude, which still applies.
    if lag > 0 and d_actual is not None and d_ewma is not None:
        y_arr = float(actual.loc[d_actual]) - 0.05
        ax_c.annotate(
            "",
            xy=(d_ewma, y_arr),
            xytext=(d_actual, y_arr),
            arrowprops=dict(arrowstyle="<->", color=C_EWMA, lw=1.6),
        )
        mid = d_actual + (d_ewma - d_actual) / 2
        ax_c.text(
            mid, y_arr - 0.03,
            f"EWMA lag ≈ {lag} days",
            ha="center", va="top", fontsize=8, color=C_EWMA,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_EWMA, alpha=0.85),
        )

    ax_c.set_ylabel(f"Pairwise Correlation  {TICKER_A}–{TICKER_B}", fontsize=10)
    # Dynamic y-limits: SPY-VIX correlation is mostly negative
    y_min = min(act.min(), ew.min(), pe.min()) - 0.08
    y_max = max(act.max(), ew.max(), pe.max()) + 0.08
    if tgat is not None and not tgat.empty:
        tg = crop(tgat)
        y_min = min(y_min, tg.min() - 0.08)
        y_max = max(y_max, tg.max() + 0.08)
    ax_c.set_ylim(np.clip(y_min, -1.05, None), np.clip(y_max, None, 1.05))
    ax_c.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax_c.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax_c.grid(True, alpha=0.25, lw=0.5)

    # ── returns panel: SPY bars, VIX change line ─────────────────────────────
    bar_colors = np.where(spy_ret.values >= 0, "#2E7D32", "#B71C1C")
    ax_r.bar(spy_ret.index, spy_ret.values, color=bar_colors, width=0.8, alpha=0.75,
             label="SPY daily return (%)")
    ax_r.plot(vix_ret.index, vix_ret.values, color=C_VIX, lw=0.9, alpha=0.8,
              label="^VIX daily change (%)")
    ax_r.axhline(0, color="black", lw=0.5)
    ax_r.set_ylabel("Daily change (%)", fontsize=9)
    ax_r.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax_r.grid(True, alpha=0.2, lw=0.5)

    # ── VIX level panel ──────────────────────────────────────────────────────
    ax_v.plot(vx.index, vx.values, color=C_VIX, lw=1.8, label="VIX level (market stress)")
    if np.isfinite(vix_thresh):
        ax_v.axhline(vix_thresh, color=C_VIX, lw=1.0, ls="--", alpha=0.8,
                     label=f"VIX p80 = {vix_thresh:.1f}")
        stress_mask = vx >= vix_thresh
        if stress_mask.any():
            ax_v.fill_between(
                vx.index, vx.values, vix_thresh,
                where=stress_mask.values, color=C_VIX, alpha=0.18,
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
    tgat_note = "" if tgat is None else "  |  TGAT reacts via event signals"
    fig.suptitle(
        f"Event Study: COVID-19 Market Crash — {TICKER_A}–{TICKER_B} Correlation Forecasting\n"
        rf"EWMA ($\alpha={EWMA_ALPHA}$) vs Persistence vs TGAT — negative regime, VIX stress indicator{tgat_note}",
        fontsize=10, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    # ── save ─────────────────────────────────────────────────────────────────
    if out_dir is None:
        out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "spy_vix_covid_compare"
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved -> {p}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare EWMA vs Persistence vs TGAT for SPY–^VIX COVID correlation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip_tgat", action="store_true",
        help="Skip TGAT training; plot only EWMA and Persistence.",
    )
    parser.add_argument(
        "--tgat_preds", default=None,
        help="Path to pre-computed TGAT predictions CSV (produced by "
             "train_link_prediction.py --save_preds_path). If provided, skips training.",
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="TGAT training epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for figures (default: figures/).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None

    # ── 1. Prices ─────────────────────────────────────────────────────────────
    print(f"Downloading prices for {TICKER_A} and {TICKER_B} (2019-01-01 -> 2020-12-31) ...")
    prices = _download_prices([TICKER_A, TICKER_B], start="2019-01-01", end="2020-12-31")
    missing = [t for t in [TICKER_A, TICKER_B] if t not in prices.columns]
    if missing:
        raise RuntimeError(f"Could not download prices for: {missing}")
    print(f"  {len(prices)} trading days.")

    # ── 2. Baselines ──────────────────────────────────────────────────────────
    print(f"Computing {ROLLING_WINDOW}-day rolling correlation {TICKER_A}-{TICKER_B} ...")
    actual  = _rolling_corr_pair(prices, TICKER_A, TICKER_B, ROLLING_WINDOW)
    ewma    = _ewma_prediction(actual, EWMA_ALPHA)
    persist = _persistence_prediction(actual)

    vix_level = prices[TICKER_B].dropna()

    # ── 3. TGAT predictions ───────────────────────────────────────────────────
    tgat_preds = None
    if not args.skip_tgat:
        preds_path = args.tgat_preds or DEFAULT_PREDS_PATH
        if args.tgat_preds and Path(args.tgat_preds).exists():
            print(f"Loading TGAT predictions from {args.tgat_preds} ...")
        else:
            if args.tgat_preds:
                print(f"[WARN] {args.tgat_preds} not found — training TGAT from scratch.")
            print(f"Training TGAT on universe={UNIVERSE}  (epochs={args.epochs}, seed={args.seed}) ...")
            print("  This may take several minutes on CPU.")
            train_tgat_and_save_preds(
                epochs=args.epochs,
                seed=args.seed,
                save_preds_path=preds_path,
            )
        tgat_preds = _load_dyfo_preds(preds_path, TICKER_A, TICKER_B)
        if tgat_preds is not None:
            print(f"  Loaded {len(tgat_preds)} TGAT prediction points for {TICKER_A}-{TICKER_B}.")
        else:
            print(
                f"[WARN] No TGAT predictions found for {TICKER_A}-{TICKER_B} in {preds_path}.\n"
                "       Check that the CSV contains 'src' and 'dst' columns with those ticker names."
            )

    # ── 4. Summary stats ──────────────────────────────────────────────────────
    test_slice = actual.loc["2020-01-01":"2020-06-30"].dropna()
    if len(test_slice) > 0:
        print("\nCorrelation stats during COVID test period (Jan-Jun 2020):")
        print(f"  Actual rolling rho: mean={test_slice.mean():.3f}  min={test_slice.min():.3f}  max={test_slice.max():.3f}")
        ew_test = ewma.loc["2020-01-01":"2020-06-30"].dropna()
        pe_test = persist.loc["2020-01-01":"2020-06-30"].dropna()
        if len(ew_test) > 0:
            mae_ew = (ew_test - test_slice.loc[ew_test.index]).abs().mean()
            mae_pe = (pe_test - test_slice.loc[pe_test.index]).abs().mean()
            print(f"  EWMA MAE:         {mae_ew:.4f}")
            print(f"  Persistence MAE:  {mae_pe:.4f}")
        if tgat_preds is not None:
            tg_test = tgat_preds.loc["2020-01-01":"2020-06-30"].dropna()
            if len(tg_test) > 0:
                common_idx = tg_test.index.intersection(test_slice.index)
                if len(common_idx) > 0:
                    mae_tg = (tg_test.loc[common_idx] - test_slice.loc[common_idx]).abs().mean()
                    print(f"  TGAT MAE:         {mae_tg:.4f}")

    # ── 5. Figure ─────────────────────────────────────────────────────────────
    make_spy_vix_figure(
        actual=actual,
        ewma=ewma,
        persist=persist,
        spy_prices=prices[TICKER_A],
        vix_level=vix_level,
        tgat=tgat_preds,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
