#!/usr/bin/env python3
"""
Diversified Basket Correlation Prediction: DyFO (TGN) vs EWMA.

Compares correlation prediction quality on a multi-asset basket that spans:
  - US Equity indices (SPY, QQQ)
  - Cryptocurrency    (BTC-USD)
  - Commodities       (GLD, DBC)
  - Fixed Income      (BIL, TLT, LQD, HYG)
  - Real Estate       (VNQ)
  - Emerging Markets  (EEM)
  - Energy            (IYE)

Walk-forward protocol
---------------------
  Data  : DATA_START – DATA_END
  Train : first  TRAIN_FRAC of trading days
  Val   : next   VAL_FRAC
  Test  : last   TEST_FRAC

Metrics reported (test set only):
  MAE, RMSE, R² for ρ-prediction (regression mode)

Usage
-----
  python scripts/run_diversified_basket_comparison.py
  python scripts/run_diversified_basket_comparison.py --epochs 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.train_link_prediction import prepare_data, train_link_prediction

# ---------------------------------------------------------------------------
# Diversified basket definition
# ---------------------------------------------------------------------------

BASKET: dict[str, list[str]] = {
    "US Equity":        ["SPY", "QQQ"],
    "Crypto":           ["BTC-USD"],
    "Commodities":      ["GLD", "DBC"],
    "Fixed Income":     ["BIL", "TLT", "LQD", "HYG"],
    "Real Estate":      ["VNQ"],
    "Emerging Markets": ["EEM"],
    "Energy":           ["IYE"],
}

TICKERS: list[str] = [t for assets in BASKET.values() for t in assets]

# ---------------------------------------------------------------------------
# Walk-forward config
# ---------------------------------------------------------------------------

DATA_START  = "2020-01-01"
DATA_END    = "2024-12-31"
BENCHMARK   = "SPY"
TRAIN_FRAC  = 0.60
VAL_FRAC    = 0.20
SINGLE_WINDOW = 63
MULTI_WINDOWS = [30, 90, 252, 365]
LABEL_WINDOW = 63
# TEST_FRAC = 0.20 (remainder)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_dates(sorted_dates: list[int]) -> tuple[list[int], list[int], list[int]]:
    n = len(sorted_dates)
    tr = int(n * TRAIN_FRAC)
    va = int(n * VAL_FRAC)
    return sorted_dates[:tr], sorted_dates[tr : tr + va], sorted_dates[tr + va :]


def _fmt(metrics: dict) -> str:
    parts = []
    for k in ("test_mae", "test_rmse", "test_r2"):
        v = metrics.get(k)
        if v is None:
            v = metrics.get(k.replace("test_", ""))  # fallback key name
        parts.append(f"{k.replace('test_','').upper():6s} = {v:.4f}" if v is not None else f"{k}: N/A")
    return "  |  ".join(parts)


def _extract_test_metrics(result: dict) -> dict:
    """Pull out the canonical test metrics from the dict returned by train_link_prediction.

    train_link_prediction returns test_metrics directly with keys:
      "mae", "r_squared", "loss" (=MSE), "spearman", ...
    """
    mae   = result.get("mae")
    r2    = result.get("r_squared")
    mse   = result.get("loss")
    rmse  = float(np.sqrt(mse)) if mse is not None else None
    return {
        "test_mae":  mae,
        "test_rmse": rmse,
        "test_r2":   r2,
        "test_mse":  mse,
        "test_spearman": result.get("spearman"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    epochs: int,
    seed: int,
    use_dcc: bool,
    single_window: int,
    multi_windows: list[int],
    label_window: int,
) -> dict:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"diversified_basket_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("dyfo.diversified_basket", log_to_file=False)

    # Build a DyFOConfig that uses rolling Pearson (faster for diverse assets)
    # or DCC-GARCH if requested
    corr_method = "dcc_garch" if use_dcc else "rolling_pearson"
    base_config = DyFOConfig(
        model_variant="tgn",
        correlation_method=corr_method,
        memory_dim=32,          # REDUCED: for 12-node graph (was 64)
        embedding_dim=32,       # REDUCED: for 12-node graph (was 64)
        num_attention_heads=1,  # REDUCED: minimal attention (was 2)
        num_gat_layers=1,
    )

    data_config = DataConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start_date=DATA_START,
        end_date=DATA_END,
    )

    logger.info("=" * 65)
    logger.info("Diversified Basket | %d assets | %d pairs", len(TICKERS), len(TICKERS) * (len(TICKERS) - 1) // 2)
    logger.info("Basket composition:")
    for cls, tkrs in BASKET.items():
        logger.info("  %-20s %s", cls + ":", ", ".join(tkrs))
    logger.info("Data     : %s → %s", DATA_START, DATA_END)
    logger.info("Corr     : %s", corr_method)
    if not use_dcc:
        logger.info("A/B      : TGN-A single=%s | TGN-B multi=%s | labels=%s", [single_window], multi_windows, label_window)
    logger.info("Epochs   : %d | Seed: %d", epochs, seed)
    logger.info("=" * 65)

    # ── 1. Define A/B arms ────────────────────────────────────────────
    arms = [
        {"name": "EWMA", "variant": "ewma", "windows": [single_window], "lr": 1e-3},
        {"name": "TGN_A_SINGLE", "variant": "tgn", "windows": [single_window], "lr": 5e-4},
        {"name": "TGN_B_MULTI", "variant": "tgn", "windows": multi_windows, "lr": 5e-4},
    ]

    # ── 2. Prepare data per window-set and run arms ──────────────────
    prepared_cache: dict[tuple[int, ...], dict] = {}
    train_dates: list[int] | None = None
    val_dates: list[int] | None = None
    test_dates: list[int] | None = None

    results: dict[str, dict] = {}

    for arm in arms:
        logger.info("")
        logger.info(">> Running arm: %s", arm["name"])

        arm_windows = sorted({int(w) for w in arm["windows"] if int(w) > 1})
        if not arm_windows:
            raise ValueError(f"Invalid windows for arm {arm['name']}: {arm['windows']}")

        cfg = DyFOConfig(
            model_variant=arm["variant"],
            correlation_method=corr_method,
            memory_dim=base_config.memory_dim,
            embedding_dim=base_config.embedding_dim,
            num_attention_heads=base_config.num_attention_heads,
            num_gat_layers=base_config.num_gat_layers,
            rolling_corr_window=single_window,
            rolling_corr_windows=arm_windows,
            rolling_label_window=label_window,
        )

        cache_key = tuple(arm_windows)
        if cache_key not in prepared_cache:
            logger.info("Preparing data for windows=%s …", arm_windows)
            prepared_cache[cache_key] = prepare_data(
                tickers=TICKERS,
                start=DATA_START,
                end=DATA_END,
                benchmark=BENCHMARK,
                config=cfg,
                data_config=data_config,
                logger=logger,
            )

        prepared = prepared_cache[cache_key]
        if train_dates is None or val_dates is None or test_dates is None:
            train_dates, val_dates, test_dates = _split_dates(prepared["sorted_dates"])
            logger.info(
                "Split | train=%d  val=%d  test=%d days",
                len(train_dates), len(val_dates), len(test_dates),
            )
            if not test_dates:
                raise RuntimeError("Test set is empty — adjust DATA_START/DATA_END or fractions.")

        try:
            metrics = train_link_prediction(
                tickers=TICKERS,
                start=DATA_START,
                end=DATA_END,
                benchmark=BENCHMARK,
                num_epochs=epochs,
                lr=arm["lr"],
                corr_threshold=0.0,     # keep all pairs (sparse basket)
                neg_ratio=1.0,
                early_stopping_patience=2,      # REDUCED: aggressive early stop (was 5)
                weight_decay=1e-4,              # INCREASED: strong L2 regularization (was 1e-5)
                mode="regression",
                model_variant=arm["variant"],
                seed=seed,
                prepared_data=prepared,
                config=cfg,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("%s failed: %s", arm["name"], exc)
            results[arm["name"]] = {"test_mae": None, "test_rmse": None, "test_r2": None}
            continue

        test_m = _extract_test_metrics(metrics)
        results[arm["name"]] = test_m
        logger.info("  → %s", _fmt(test_m))

    # ── 3. Summary table ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("%-12s  %8s  %8s  %8s", "Variant", "MAE", "RMSE", "R²")
    logger.info("-" * 65)
    for arm in arms:
        v = arm["name"]
        m = results.get(v, {})
        mae  = m.get("test_mae")
        rmse = m.get("test_rmse")
        r2   = m.get("test_r2")
        mae_s  = f"{mae:.4f}"  if mae  is not None else "  N/A  "
        rmse_s = f"{rmse:.4f}" if rmse is not None else "  N/A  "
        r2_s   = f"{r2:.4f}"   if r2   is not None else "  N/A  "
        logger.info("%-12s  %8s  %8s  %8s", v, mae_s, rmse_s, r2_s)
    logger.info("=" * 65)

    # Δ vs EWMA and B vs A
    if "EWMA" in results and "TGN_B_MULTI" in results:
        for key, label in [("test_mae", "ΔMAE (B−EWMA)"), ("test_r2", "ΔR² (B−EWMA)")]:
            e = results["EWMA"].get(key)
            t = results["TGN_B_MULTI"].get(key)
            if e is not None and t is not None:
                logger.info("  %-22s %+.4f", label, t - e)
    if "TGN_A_SINGLE" in results and "TGN_B_MULTI" in results:
        for key, label in [("test_mae", "ΔMAE (B−A)"), ("test_r2", "ΔR² (B−A)")]:
            a = results["TGN_A_SINGLE"].get(key)
            b = results["TGN_B_MULTI"].get(key)
            if a is not None and b is not None:
                logger.info("  %-22s %+.4f", label, b - a)

    # ── 4. Save JSON ──────────────────────────────────────────────────
    summary = {
        "run_tag": out_dir.name,
        "basket": BASKET,
        "n_tickers": len(TICKERS),
        "data": {
            "start": DATA_START,
            "end": DATA_END,
            "corr_method": corr_method,
            "single_window": single_window,
            "multi_windows": multi_windows,
            "label_window": label_window,
        },
        "split": {
            "train_days": len(train_dates),
            "val_days":   len(val_dates),
            "test_days":  len(test_dates),
        },
        "run": {"epochs": epochs, "seed": seed},
        "arms": arms,
        "metrics": results,
    }
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Results saved → %s", out_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    def _parse_windows(raw: str) -> list[int]:
        vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
        vals = sorted({v for v in vals if v > 1})
        if not vals:
            raise argparse.ArgumentTypeError("windows must contain at least one integer > 1")
        return vals

    parser = argparse.ArgumentParser(
        description="Diversified basket A/B: EWMA vs TGN(single-window) vs TGN(multi-window)"
    )
    parser.add_argument("--epochs",  type=int, default=5,
                        help="Training epochs for DyFO (default: 5)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dcc",     action="store_true",
                        help="Use DCC-GARCH instead of rolling Pearson (slower)")
    parser.add_argument("--single-window", type=int, default=SINGLE_WINDOW,
                        help=f"Single rolling window for EWMA and TGN-A (default: {SINGLE_WINDOW})")
    parser.add_argument("--multi-windows", type=_parse_windows,
                        default=MULTI_WINDOWS,
                        help="Comma-separated rolling windows for TGN-B, e.g. 30,90,252,365")
    parser.add_argument("--label-window", type=int, default=LABEL_WINDOW,
                        help=f"Unsparsified label correlation window (default: {LABEL_WINDOW})")
    args = parser.parse_args()

    run(
        epochs=args.epochs,
        seed=args.seed,
        use_dcc=args.dcc,
        single_window=args.single_window,
        multi_windows=args.multi_windows,
        label_window=args.label_window,
    )


if __name__ == "__main__":
    main()
