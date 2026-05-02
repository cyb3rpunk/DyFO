#!/usr/bin/env python3
"""Stress-event comparison: DyFO/TGAT vs EWMA and Persistence.

The default pair remains SPY-^VIX, but the script can now run a cross-asset
COVID stress battery.  The reporting is intentionally honest: EWMA is expected
to be very strong on smooth DCC/rolling-correlation R2, while TGAT/DyFO is
evaluated for event/regime adaptation, lag reduction, and stress-window error.

Examples
--------
  # Original SPY-VIX style figure, baselines only:
  python scripts/run_spy_vix_covid_compare.py --skip_tgat

  # Reuse existing TGAT predictions for SPY-^VIX:
  python scripts/run_spy_vix_covid_compare.py --tgat_preds results/spy_vix_tgat_preds.csv

  # Cross-asset stress battery, no training:
  python scripts/run_spy_vix_covid_compare.py --mode battery --skip_tgat

  # Print the official walk-forward command for the S&P 50 protocol:
  python scripts/run_spy_vix_covid_compare.py --print_walk_forward_command

Outputs
-------
  figures/stress_event_compare_<PAIR>.pdf/.png
  results/stress_event_compare/<PAIR>_predictions.csv
  results/stress_event_compare/<PAIR>_metrics.json
  results/stress_event_compare/stress_event_summary.json
  results/stress_event_compare/stress_event_report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dyfo.core.ticker_registry import TICKERS_50
from scripts.event_study_covid import (
    C_ACTUAL,
    C_CRASH,
    C_DYFO,
    C_EWMA,
    C_PERSIST,
    C_VIX,
    CRASH_PEAK,
    CRASH_START,
    EWMA_ALPHA,
    KEY_DATES,
    PLOT_END,
    PLOT_START,
    ROLLING_WINDOW,
    _download_prices,
    _ewma_prediction,
    _persistence_prediction,
)

DATA_START = "2015-01-01"
DATA_END = "2020-12-31"
TRAIN_END = "2019-09-30"
VAL_END = "2019-12-31"
TEST_END = "2020-06-30"
PRICE_START = "2019-01-01"
PRICE_END = "2020-12-31"
TEST_START = "2020-01-01"

DEFAULT_PAIR = ("SPY", "^VIX")
DEFAULT_PREDS_PATH = str(ROOT / "results" / "spy_vix_tgat_preds.csv")
DEFAULT_RESULTS_DIR = ROOT / "results" / "stress_event_compare"
DEFAULT_FIGURES_DIR = ROOT / "figures"
DEFAULT_BATTERY_TGAT_PREDS = "stress_battery_tgat_preds.csv"
STRESS_PAIRS = [
    ("SPY", "^VIX"),
    ("SPY", "BTC-USD"),
    ("SPY", "GLD"),
    ("SPY", "TLT"),
    ("QQQ", "BTC-USD"),
    ("GLD", "BTC-USD"),
    ("XLE", "SPY"),
    ("XLK", "SPY"),
]


def _safe_name(*parts: str) -> str:
    text = "_".join(parts)
    return (
        text.replace("^", "")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _as_datetime_index(obj):
    """Return a copy with a timezone-naive DatetimeIndex for robust date slicing."""
    out = obj.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def _pair_label(a: str, b: str) -> str:
    return f"{a}-{b}"


def _int_day_to_iso(day: int) -> str:
    epoch = datetime.date(2000, 1, 1)
    return (epoch + datetime.timedelta(days=int(day))).isoformat()


def _slice_dates(sorted_dates: list[int], start: str, end: str) -> list[int]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    return [
        d for d in sorted_dates
        if start_ts <= pd.Timestamp(_int_day_to_iso(d)) <= end_ts
    ]


def _rolling_corr_pair(prices: pd.DataFrame, a: str, b: str, window: int) -> pd.Series:
    rets = prices[[a, b]].pct_change(fill_method=None).dropna()
    return _as_datetime_index(rets[a].rolling(window).corr(rets[b]))


def _stress_universe(extra_tickers: Iterable[str], include_sp50: bool = True) -> list[str]:
    universe: list[str] = []
    base = ["SPY", *TICKERS_50] if include_sp50 else ["SPY"]
    for ticker in [*base, *extra_tickers]:
        if ticker not in universe:
            universe.append(ticker)
    return universe


def _extra_tickers_from_pairs(pairs: Iterable[tuple[str, str]]) -> list[str]:
    extra: list[str] = []
    for a, b in pairs:
        for ticker in (a, b):
            if ticker not in extra:
                extra.append(ticker)
    return extra


def train_tgat_for_tickers_and_save_preds(
    extra_tickers: Iterable[str],
    epochs: int,
    seed: int,
    save_preds_path: str,
    include_sp50: bool = True,
) -> None:
    """Train TGAT on the S&P 50 universe plus extra tickers and export preds."""
    from dyfo.config import DataConfig, DyFOConfig
    from dyfo.logging_utils import setup_logging
    from scripts.run_bootstrap_eval_v5 import (
        TGN_LR,
        TGN_PATIENCE,
        TGN_USE_COSINE,
        load_or_prepare_data,
    )
    from scripts.train_link_prediction import train_link_prediction

    universe = _stress_universe(extra_tickers, include_sp50=include_sp50)
    logger = setup_logging("dyfo.stress_event_compare", log_to_file=False)
    config = DyFOConfig(model_variant="tgat")
    data_config = DataConfig(
        tickers=universe,
        benchmark_ticker="SPY",
        start_date=DATA_START,
        end_date=DATA_END,
    )

    logger.info("Preparing data for universe: %s", universe)
    data = load_or_prepare_data(
        tickers=universe,
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
            f"  train={len(train_dates)} val={len(val_dates)} test={len(test_dates)}"
        )

    metrics = train_link_prediction(
        tickers=universe,
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
    print(
        "TGAT test metrics (all pairs) "
        f"R2={metrics.get('r_squared', float('nan')):.4f} "
        f"MAE={metrics.get('mae', float('nan')):.4f}"
    )
    print(f"TGAT predictions saved -> {save_preds_path}")


def train_tgat_and_save_preds(
    pair: tuple[str, str],
    epochs: int,
    seed: int,
    save_preds_path: str,
) -> None:
    """Train TGAT on the S&P 50 universe plus the requested pair and export preds."""
    train_tgat_for_tickers_and_save_preds(pair, epochs, seed, save_preds_path, include_sp50=True)


def load_tgat_preds_with_diagnostics(
    csv_path: str,
    a: str,
    b: str,
) -> tuple[pd.Series | None, pd.Series | None, dict]:
    """Load TGAT predictions and report whether they are degenerate/constant."""
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except FileNotFoundError:
        print(f"[WARN] TGAT predictions file not found: {csv_path}")
        return None, None, {"available": False, "reason": "file_not_found"}

    mask = ((df["src"] == a) & (df["dst"] == b)) | ((df["src"] == b) & (df["dst"] == a))
    sub = df[mask].copy()
    if sub.empty:
        print(f"[WARN] No predictions for pair {a}-{b} in {csv_path}")
        return None, None, {"available": False, "reason": "pair_missing"}

    pred = sub.set_index("date").sort_index()["pred"].groupby(level=0).mean()
    pred = _as_datetime_index(pred)
    true = None
    if "true" in sub.columns:
        true = sub.set_index("date").sort_index()["true"].groupby(level=0).mean()
        true = _as_datetime_index(true)
    std = float(pred.std()) if len(pred) > 1 else 0.0
    nunique = int(pred.nunique(dropna=True))
    degenerate = nunique <= 1 or std <= 1e-10
    diag = {
        "available": True,
        "rows": int(len(sub)),
        "dates": int(pred.index.nunique()),
        "pred_nunique": nunique,
        "pred_std": std,
        "degenerate_constant": bool(degenerate),
        "source_csv": str(csv_path),
    }
    if degenerate:
        print(
            f"[WARN] TGAT predictions for {a}-{b} are constant "
            f"(nunique={nunique}, std={std:.3e}) in {csv_path}"
        )
    return pred, true, diag


def _align(actual: pd.Series, pred: pd.Series) -> pd.DataFrame:
    df = pd.concat([actual.rename("actual"), pred.rename("pred")], axis=1).dropna()
    return df


def _r2(y: pd.Series, yhat: pd.Series) -> float:
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot


def _directional_accuracy(actual: pd.Series, pred: pd.Series) -> float:
    df = _align(actual.diff(), pred.diff())
    if df.empty:
        return float("nan")
    mask = (df["actual"] != 0) & (df["pred"] != 0)
    if not mask.any():
        return float("nan")
    return float((np.sign(df.loc[mask, "actual"]) == np.sign(df.loc[mask, "pred"])).mean())


def _turning_point_delay_days(actual: pd.Series, pred: pd.Series) -> float:
    df = _align(actual, pred)
    df = df.loc[CRASH_START:CRASH_PEAK]
    if len(df) < 3:
        return float("nan")
    actual_date = df["actual"].diff().abs().idxmax()
    pred_date = df["pred"].diff().abs().idxmax()
    return float((pred_date - actual_date).days)


def _lag_to_threshold(actual: pd.Series, pred: pd.Series, threshold_pct: float = 0.5) -> float:
    actual_crash = actual.loc[CRASH_START:CRASH_PEAK].dropna()
    pred_crash = pred.loc[CRASH_START:CRASH_PEAK].dropna()
    pre = actual.loc[:CRASH_START].dropna()
    if actual_crash.empty or pred_crash.empty or pre.empty:
        return float("nan")

    pre_mean = float(pre.mean())
    peak_date = (actual_crash - pre_mean).abs().idxmax()
    peak_value = float(actual_crash.loc[peak_date])
    threshold = pre_mean + threshold_pct * (peak_value - pre_mean)
    direction = 1.0 if peak_value >= pre_mean else -1.0

    actual_cross = actual_crash[direction * (actual_crash - threshold) >= 0]
    pred_cross = pred_crash[direction * (pred_crash - threshold) >= 0]
    if actual_cross.empty or pred_cross.empty:
        return float("nan")
    return float((pred_cross.index[0] - actual_cross.index[0]).days)


def _period_masks(index: pd.DatetimeIndex, vix: pd.Series | None) -> dict[str, pd.Index]:
    periods = {
        "full_test": index[(index >= TEST_START) & (index <= TEST_END)],
        "pre_crash": index[(index >= "2020-01-01") & (index < CRASH_START)],
        "covid_crash": index[(index >= CRASH_START) & (index <= CRASH_PEAK)],
        "post_crash": index[(index > CRASH_PEAK) & (index <= TEST_END)],
    }
    if vix is not None and not vix.dropna().empty:
        vx = vix.reindex(index).ffill()
        threshold = vx.loc[PLOT_START:PLOT_END].quantile(0.8)
        periods["high_vix_days"] = index[vx >= threshold]
    else:
        periods["high_vix_days"] = index[:0]
    return periods


def _metrics_for_period(actual: pd.Series, pred: pd.Series, idx: pd.Index) -> dict[str, float]:
    df = _align(actual.reindex(idx), pred.reindex(idx))
    if df.empty:
        return {
            "n": 0,
            "mae": float("nan"),
            "mse": float("nan"),
            "r_squared": float("nan"),
            "spearman": float("nan"),
            "directional_accuracy": float("nan"),
        }
    spearman = (
        float("nan")
        if df["actual"].nunique(dropna=True) <= 1 or df["pred"].nunique(dropna=True) <= 1
        else float(df["actual"].corr(df["pred"], method="spearman"))
    )
    return {
        "n": int(len(df)),
        "mae": float((df["pred"] - df["actual"]).abs().mean()),
        "mse": float(((df["pred"] - df["actual"]) ** 2).mean()),
        "r_squared": _r2(df["actual"], df["pred"]),
        "spearman": spearman,
        "directional_accuracy": _directional_accuracy(df["actual"], df["pred"]),
    }


def compute_pair_metrics(
    actual: pd.Series,
    predictions: dict[str, pd.Series | None],
    vix: pd.Series | None,
) -> dict[str, dict]:
    idx = actual.dropna().index
    periods = _period_masks(idx, vix)
    out: dict[str, dict] = {}

    for model, pred in predictions.items():
        if pred is None:
            continue
        model_metrics: dict[str, dict | float | bool] = {}
        for period_name, period_idx in periods.items():
            model_metrics[period_name] = _metrics_for_period(actual, pred, period_idx)
        model_metrics["turning_point_delay_days"] = _turning_point_delay_days(actual, pred)
        model_metrics["lag_to_threshold"] = _lag_to_threshold(actual, pred)
        model_metrics["stress_mae"] = model_metrics["covid_crash"]["mae"]
        out[model] = model_metrics

    if "tgat" in out and "ewma" in out:
        tgat_stress = out["tgat"]["covid_crash"]["mae"]
        ewma_stress = out["ewma"]["covid_crash"]["mae"]
        tgat_lag = out["tgat"]["lag_to_threshold"]
        ewma_lag = out["ewma"]["lag_to_threshold"]
        out["comparison"] = {
            "tgat_event_window_win": bool(
                np.isfinite(tgat_stress) and np.isfinite(ewma_stress) and tgat_stress <= ewma_stress
            ),
            "tgat_lag_reduction_days": (
                float(ewma_lag - tgat_lag)
                if np.isfinite(ewma_lag) and np.isfinite(tgat_lag)
                else float("nan")
            ),
        }
    return out


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_pair_predictions(
    path: Path,
    actual: pd.Series,
    ewma: pd.Series,
    persistence: pd.Series,
    tgat: pd.Series | None,
    vix: pd.Series | None,
) -> None:
    df = pd.concat(
        [
            actual.rename("rho_t_plus_1_real"),
            ewma.rename("rho_t_plus_1_ewma"),
            (
                tgat.rename("rho_t_plus_1_tgat")
                if tgat is not None
                else pd.Series(dtype=float, name="rho_t_plus_1_tgat")
            ),
            persistence.rename("rho_t_plus_1_persistence"),
            (vix.rename("vix") if vix is not None else pd.Series(dtype=float, name="vix")),
        ],
        axis=1,
    )
    df = _as_datetime_index(df)
    if "vix" in df.columns and df["vix"].notna().any():
        threshold = df["vix"].loc[PLOT_START:PLOT_END].quantile(0.8)
        df["high_vix_day"] = df["vix"] >= threshold
    else:
        df["high_vix_day"] = False
    df.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _forecast_preview_rows(
    actual: pd.Series,
    ewma: pd.Series,
    persistence: pd.Series,
    tgat: pd.Series | None,
    start: str = CRASH_START,
    end: str = CRASH_PEAK,
) -> list[dict]:
    df = pd.concat(
        [
            actual.rename("rho_t_plus_1_real"),
            ewma.rename("rho_t_plus_1_ewma"),
            (
                tgat.rename("rho_t_plus_1_tgat")
                if tgat is not None
                else pd.Series(dtype=float, name="rho_t_plus_1_tgat")
            ),
            persistence.rename("rho_t_plus_1_persistence"),
        ],
        axis=1,
    )
    df = _as_datetime_index(df).loc[start:end].dropna(how="all")
    rows = []
    for date, row in df.iterrows():
        rows.append(
            {
                "date": date.date().isoformat(),
                "rho_t_plus_1_real": _json_ready(row.get("rho_t_plus_1_real", float("nan"))),
                "rho_t_plus_1_ewma": _json_ready(row.get("rho_t_plus_1_ewma", float("nan"))),
                "rho_t_plus_1_tgat": _json_ready(row.get("rho_t_plus_1_tgat", float("nan"))),
                "rho_t_plus_1_persistence": _json_ready(row.get("rho_t_plus_1_persistence", float("nan"))),
            }
        )
    return rows


def make_pair_figure(
    pair: tuple[str, str],
    prices: pd.DataFrame,
    actual: pd.Series,
    ewma: pd.Series,
    persistence: pd.Series,
    vix: pd.Series,
    tgat: pd.Series | None,
    out_dir: Path,
) -> None:
    a, b = pair

    def crop(s: pd.Series) -> pd.Series:
        return s.loc[PLOT_START:PLOT_END].dropna()

    act = crop(actual)
    ew = crop(ewma)
    pe = crop(persistence)
    vx = crop(vix)
    tgat_crop = crop(tgat) if tgat is not None else None

    ret_a = (prices[a].pct_change(fill_method=None) * 100.0).loc[PLOT_START:PLOT_END].dropna()
    ret_b = (prices[b].pct_change(fill_method=None) * 100.0).loc[PLOT_START:PLOT_END].dropna()
    vix_thresh = float(vx.quantile(0.8)) if len(vx) else np.nan

    fig, (ax_c, ax_r, ax_v) = plt.subplots(
        3,
        1,
        figsize=(11, 8.6),
        gridspec_kw={"height_ratios": [3.0, 1.2, 1.0], "hspace": 0.06},
        sharex=True,
    )

    for ax in (ax_c, ax_r, ax_v):
        ax.axvspan(pd.Timestamp(CRASH_START), pd.Timestamp(CRASH_PEAK), color=C_CRASH, alpha=0.45)

    for date_str, label in KEY_DATES.items():
        dt = pd.Timestamp(date_str)
        ax_c.axvline(dt, color="#757575", lw=0.7, linestyle=":")
        ax_c.text(
            dt,
            0.98,
            label,
            transform=ax_c.get_xaxis_transform(),
            fontsize=6.5,
            color="#757575",
            ha="center",
            va="top",
            rotation=90,
        )

    ax_c.plot(pe.index, pe.values, color=C_PERSIST, lw=1.2, ls="--", alpha=0.65, label="Persistence")
    ax_c.plot(ew.index, ew.values, color=C_EWMA, lw=2.0, label=f"EWMA alpha={EWMA_ALPHA}")
    ax_c.plot(act.index, act.values, color=C_ACTUAL, lw=1.8, label="Actual rolling rho")
    if tgat_crop is not None and not tgat_crop.empty:
        ax_c.plot(tgat_crop.index, tgat_crop.values, color=C_DYFO, lw=2.0, label="TGAT")

    all_lines = [act, ew, pe] + ([tgat_crop] if tgat_crop is not None and not tgat_crop.empty else [])
    y_min = min(s.min() for s in all_lines if not s.empty) - 0.08
    y_max = max(s.max() for s in all_lines if not s.empty) + 0.08
    ax_c.set_ylim(max(y_min, -1.05), min(y_max, 1.05))
    ax_c.axhline(0.0, color="black", lw=0.5, ls="--", alpha=0.4)
    ax_c.set_ylabel(f"Correlation {a}-{b}", fontsize=10)
    ax_c.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_c.grid(True, alpha=0.25, lw=0.5)

    colors = np.where(ret_a.values >= 0, "#2E7D32", "#B71C1C")
    ax_r.bar(ret_a.index, ret_a.values, color=colors, width=0.8, alpha=0.75, label=f"{a} return (%)")
    ax_r.plot(ret_b.index, ret_b.values, color="#E65100", lw=0.9, alpha=0.85, label=f"{b} return (%)")
    ax_r.axhline(0.0, color="black", lw=0.5)
    ax_r.set_ylabel("Daily return (%)", fontsize=9)
    ax_r.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    ax_r.grid(True, alpha=0.2, lw=0.5)

    ax_v.plot(vx.index, vx.values, color=C_VIX, lw=1.8, label="VIX level")
    if np.isfinite(vix_thresh):
        ax_v.axhline(vix_thresh, color=C_VIX, lw=1.0, ls="--", alpha=0.8, label=f"VIX p80={vix_thresh:.1f}")
        stress_mask = vx >= vix_thresh
        if stress_mask.any():
            ax_v.fill_between(vx.index, vx.values, vix_thresh, where=stress_mask.values, color=C_VIX, alpha=0.18)
    ax_v.set_ylabel("VIX", fontsize=9)
    ax_v.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    ax_v.grid(True, alpha=0.2, lw=0.5)

    ax_v.xaxis.set_major_locator(mdates.MonthLocator())
    ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_v.set_xlim(pd.Timestamp(PLOT_START), pd.Timestamp(PLOT_END))
    plt.setp(ax_v.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

    dyfo_note = "" if tgat is None else " | TGAT included"
    fig.suptitle(
        f"Stress Event Study: {a}-{b} Correlation Forecasting\n"
        f"EWMA is a strong smooth baseline; DyFO/TGAT is judged on stress adaptation{dyfo_note}",
        fontsize=10,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"stress_event_compare_{_safe_name(a, b)}"
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved -> {path}")
    plt.close(fig)


def run_pair(
    pair: tuple[str, str],
    args: argparse.Namespace,
    results_dir: Path,
    figures_dir: Path,
) -> dict:
    a, b = pair
    print(f"\n=== Pair {_pair_label(a, b)} ===")
    needed = list(dict.fromkeys([a, b, "^VIX"]))
    prices = _as_datetime_index(_download_prices(needed, start=PRICE_START, end=PRICE_END))
    missing = [ticker for ticker in [a, b] if ticker not in prices.columns]
    if missing:
        raise RuntimeError(f"Could not download prices for: {missing}")
    if "^VIX" in prices.columns:
        vix = _as_datetime_index(prices["^VIX"].dropna())
    else:
        vix = _as_datetime_index(_download_prices(["^VIX"], start=PRICE_START, end=PRICE_END)["^VIX"].dropna())

    actual = _rolling_corr_pair(prices, a, b, args.rolling_window)
    ewma = _ewma_prediction(actual, EWMA_ALPHA)
    persistence = _persistence_prediction(actual)

    tgat = None
    tgat_true = None
    tgat_diagnostics = {"available": False, "reason": "skipped" if args.skip_tgat else "not_loaded"}
    if not args.skip_tgat:
        preds_path = Path(args.tgat_preds) if args.tgat_preds else results_dir / f"{_safe_name(a, b)}_tgat_preds.csv"
        if preds_path.exists():
            print(f"Loading TGAT predictions from {preds_path}")
        else:
            print(f"Training TGAT for {_pair_label(a, b)} -> {preds_path}")
            train_tgat_and_save_preds(pair, args.epochs, args.seed, str(preds_path))
        tgat, tgat_true, tgat_diagnostics = load_tgat_preds_with_diagnostics(str(preds_path), a, b)
        if tgat is None:
            print(f"[WARN] No TGAT predictions found for {_pair_label(a, b)} in {preds_path}")
        elif tgat_true is not None and not tgat_true.dropna().empty:
            print(f"Using TGAT CSV 'true' column as rho(t+1) real for {_pair_label(a, b)}.")
            actual = tgat_true
            ewma = _ewma_prediction(actual, EWMA_ALPHA)
            persistence = _persistence_prediction(actual)

    predictions = {"ewma": ewma, "persistence": persistence, "tgat": tgat}
    metrics = compute_pair_metrics(actual, predictions, vix)
    if "tgat" in metrics:
        metrics["tgat"]["diagnostics"] = tgat_diagnostics

    pair_name = _safe_name(a, b)
    save_pair_predictions(
        results_dir / f"{pair_name}_predictions.csv",
        actual,
        ewma,
        persistence,
        tgat,
        vix,
    )
    with open(results_dir / f"{pair_name}_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            _json_ready(
                {
                    "pair": [a, b],
                    "metrics": metrics,
                    "forecast_preview_rows": _forecast_preview_rows(actual, ewma, persistence, tgat),
                }
            ),
            fh,
            indent=2,
        )

    make_pair_figure(pair, prices, actual, ewma, persistence, vix, tgat, figures_dir)
    _print_pair_summary(pair, metrics)
    return {
        "pair": [a, b],
        "metrics": metrics,
        "forecast_preview_rows": _forecast_preview_rows(actual, ewma, persistence, tgat),
    }


def _fmt(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "na"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(val):
        return "na"
    return f"{val:.{ndigits}f}"


def _print_pair_summary(pair: tuple[str, str], metrics: dict) -> None:
    print(f"Metrics for {_pair_label(*pair)}:")
    for model in ("ewma", "persistence", "tgat"):
        if model not in metrics:
            continue
        full = metrics[model]["full_test"]
        crash = metrics[model]["covid_crash"]
        print(
            f"  {model:11s} full R2={_fmt(full['r_squared'])} "
            f"full MAE={_fmt(full['mae'])} crash MAE={_fmt(crash['mae'])} "
            f"lag_days={_fmt(metrics[model]['lag_to_threshold'], 1)}"
        )
    if "comparison" in metrics:
        cmp = metrics["comparison"]
        print(
            "  TGAT vs EWMA: "
            f"event_win={cmp['tgat_event_window_win']} "
            f"lag_reduction_days={_fmt(cmp['tgat_lag_reduction_days'], 1)}"
        )


def aggregate_results(pair_results: list[dict], results_dir: Path) -> dict:
    comparable = [r for r in pair_results if "comparison" in r.get("metrics", {})]
    wins = [
        r["metrics"]["comparison"]["tgat_event_window_win"]
        for r in comparable
    ]
    lag_reductions = [
        r["metrics"]["comparison"]["tgat_lag_reduction_days"]
        for r in comparable
        if r["metrics"]["comparison"]["tgat_lag_reduction_days"] is not None
    ]
    finite_lags = [float(v) for v in lag_reductions if math.isfinite(float(v))]

    summary = {
        "claim": (
            "EWMA remains a strong smooth autoregressive baseline; DyFO/TGAT is "
            "evaluated for stress-event adaptation and lag reduction."
        ),
        "n_pairs": len(pair_results),
        "n_pairs_with_tgat": len(comparable),
        "tgat_event_window_wins": int(sum(bool(v) for v in wins)),
        "tgat_event_window_win_rate": float(sum(bool(v) for v in wins) / len(wins)) if wins else None,
        "mean_tgat_lag_reduction_days": float(np.mean(finite_lags)) if finite_lags else None,
        "pairs": pair_results,
        "official_walk_forward_command": official_walk_forward_command(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "stress_event_summary.json", "w", encoding="utf-8") as fh:
        json.dump(_json_ready(summary), fh, indent=2)
    write_markdown_report(summary, results_dir / "stress_event_report.md")
    print(f"\nSaved aggregate summary -> {results_dir / 'stress_event_summary.json'}")
    print(f"Saved aggregate report  -> {results_dir / 'stress_event_report.md'}")
    return summary


def write_markdown_report(summary: dict, path: Path) -> None:
    lines = [
        "# DyFO/TGAT vs EWMA Stress-Event Evidence",
        "",
        "EWMA is treated as a strong smooth autoregressive baseline. The central claim is not that TGAT dominates every R2 table, but that DyFO can add value in stress regimes, event windows, and downstream portfolio utility.",
        "",
        "## Aggregate",
        "",
        f"- Pairs evaluated: {summary['n_pairs']}",
        f"- Pairs with TGAT predictions: {summary['n_pairs_with_tgat']}",
        f"- TGAT event-window wins vs EWMA: {summary['tgat_event_window_wins']} ({_fmt(summary['tgat_event_window_win_rate'])})",
        f"- Mean TGAT lag reduction vs EWMA, days: {_fmt(summary['mean_tgat_lag_reduction_days'], 2)}",
        "",
        "## Pair Table",
        "",
        "| Pair | EWMA full R2 | EWMA crash MAE | TGAT crash MAE | TGAT lag reduction days |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["pairs"]:
        pair = "-".join(result["pair"])
        metrics = result["metrics"]
        ewma = metrics.get("ewma", {})
        tgat = metrics.get("tgat", {})
        cmp = metrics.get("comparison", {})
        lines.append(
            "| "
            f"{pair} | "
            f"{_fmt(ewma.get('full_test', {}).get('r_squared'))} | "
            f"{_fmt(ewma.get('covid_crash', {}).get('mae'))} | "
            f"{_fmt(tgat.get('covid_crash', {}).get('mae'))} | "
            f"{_fmt(cmp.get('tgat_lag_reduction_days'), 1)} |"
        )
    lines.extend(
        [
            "",
            "## TGAT Diagnostics",
            "",
            "| Pair | TGAT rows | TGAT unique predictions | TGAT std | Constant? |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for result in summary["pairs"]:
        pair = "-".join(result["pair"])
        diag = result["metrics"].get("tgat", {}).get("diagnostics", {})
        lines.append(
            "| "
            f"{pair} | "
            f"{diag.get('rows', 'na')} | "
            f"{diag.get('pred_nunique', 'na')} | "
            f"{_fmt(diag.get('pred_std'))} | "
            f"{diag.get('degenerate_constant', 'na')} |"
        )

    lines.extend(
        [
            "",
            "## Crash-Window rho(t+1) Forecasts",
            "",
            "Rows below use the target-date convention: each date is the realized `rho(t+1)` target date, and forecasts are causal one-step-ahead estimates produced from information available before that date.",
            "",
        ]
    )
    for result in summary["pairs"]:
        pair = "-".join(result["pair"])
        lines.extend(
            [
                f"### {pair}",
                "",
                "| Date | rho+1 real | EWMA rho+1 pred | TGAT rho+1 pred | Persistence rho+1 pred |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result.get("forecast_preview_rows", []):
            lines.append(
                "| "
                f"{row['date']} | "
                f"{_fmt(row.get('rho_t_plus_1_real'))} | "
                f"{_fmt(row.get('rho_t_plus_1_ewma'))} | "
                f"{_fmt(row.get('rho_t_plus_1_tgat'))} | "
                f"{_fmt(row.get('rho_t_plus_1_persistence'))} |"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Official Walk-Forward Command",
            "",
            "```powershell",
            official_walk_forward_command(),
            "```",
            "",
            "This command uses only `tgat`, `ewma`, and `persistence`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def official_walk_forward_command() -> str:
    return (
        "python scripts/run_bootstrap_eval_temporal_kg_rev3.py "
        "--variants tgat ewma persistence --n_tickers 50 --step_days 125"
    )


def maybe_run_walk_forward(args: argparse.Namespace) -> None:
    if args.print_walk_forward_command:
        print("\nOfficial S&P 50 walk-forward command:")
        print(official_walk_forward_command())
    if args.run_walk_forward_protocol:
        cmd = official_walk_forward_command().split()
        print("\nRunning official S&P 50 walk-forward protocol:")
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


def parse_pair(text: str) -> tuple[str, str]:
    if "," in text:
        left, right = text.split(",", 1)
    elif ":" in text:
        left, right = text.split(":", 1)
    else:
        raise argparse.ArgumentTypeError("Pair must be formatted as TICKER_A,TICKER_B")
    return left.strip(), right.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare EWMA vs Persistence vs TGAT in COVID stress windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["pair", "battery"], default="pair")
    parser.add_argument("--pair", type=parse_pair, default=DEFAULT_PAIR, help="Pair as TICKER_A,TICKER_B")
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=parse_pair,
        default=None,
        help="Override battery pairs, each formatted as TICKER_A,TICKER_B",
    )
    parser.add_argument("--skip_tgat", action="store_true", help="Skip TGAT training/loading.")
    parser.add_argument(
        "--tgat_preds",
        default=None,
        help=(
            "Path to pre-computed TGAT predictions. In battery mode this is mainly "
            "useful for SPY,^VIX; missing pair rows are reported but not fatal."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10, help="TGAT epochs when training is needed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rolling_window", type=int, default=ROLLING_WINDOW)
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_FIGURES_DIR), help="Output directory for figures.")
    parser.add_argument(
        "--train_battery_tgat",
        action="store_true",
        help=(
            "In battery mode, train one TGAT run for all stress-pair tickers "
            "and use that CSV for every pair."
        ),
    )
    parser.add_argument(
        "--tgat_universe",
        choices=["stress_only", "sp50_plus"],
        default="stress_only",
        help=(
            "Universe used by --train_battery_tgat. 'stress_only' is fast and "
            "covers all requested pairs; 'sp50_plus' is heavier and adds the S&P 50."
        ),
    )
    parser.add_argument("--print_walk_forward_command", action="store_true")
    parser.add_argument(
        "--run_walk_forward_protocol",
        action="store_true",
        help="Run the expensive official S&P 50 tgat/ewma/persistence protocol.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.out_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    maybe_run_walk_forward(args)

    if args.mode == "battery":
        pairs = args.pairs or STRESS_PAIRS
    else:
        pairs = [args.pair]

    if args.mode == "battery" and args.train_battery_tgat and not args.skip_tgat:
        battery_preds = results_dir / DEFAULT_BATTERY_TGAT_PREDS
        print(f"\nTraining one TGAT battery model for all stress pairs -> {battery_preds}")
        train_tgat_for_tickers_and_save_preds(
            _extra_tickers_from_pairs(pairs),
            args.epochs,
            args.seed,
            str(battery_preds),
            include_sp50=args.tgat_universe == "sp50_plus",
        )
        args.tgat_preds = str(battery_preds)

    pair_results = [run_pair(pair, args, results_dir, figures_dir) for pair in pairs]
    aggregate_results(pair_results, results_dir)


if __name__ == "__main__":
    main()
