"""Scaling experiment for DyFO: 30 vs 50 vs 100 assets.

This script executes the DyFO (TGAT) model on universes of 30, 50, and 100 stocks,
evaluates their predictive performance, and performs rigorous statistical comparisons
(Diebold-Mariano and Wilcoxon) on their daily aggregate errors to test if
scaling the spatial graph size degrades or improves predictive accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.train_link_prediction import prepare_data, train_link_prediction

TICKERS_30 = [
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM",
    "JPM", "GS", "MA", "BRK-B",
    "JNJ", "UNH", "LLY",
    "AMZN", "TSLA", "HD",
    "PG", "KO",
    "XOM", "CVX",
    "CAT", "BA", "RTX",
    "META", "GOOGL", "DIS",
    "LIN", "APD",
    "NEE", "DUK",
    "PLD",
]

TICKERS_50 = TICKERS_30 + [
    "V", "AXP", "BAC", "C", "WFC", 
    "PEP", "COST", "WMT", "MRK", "ABBV", 
    "PFE", "TMO", "DHR", "INTC", "AMD", 
    "CSCO", "QCOM", "TXN", "ORCL", "IBM"
]

TICKERS_100 = TICKERS_50 + [
    "MCD", "SBUX", "NKE", "PM", "MO", "T", "VZ", "CMCSA", "NFLX", "ADBE", 
    "INTU", "NOW", "GE", "MMM", "HON", "LMT", "NOC", "GD", "COP", "EOG", 
    "SLB", "HAL", "DE", "VRTX", "REGN", "BIIB", "AMGN", "GILD", "BMY", "MDT", 
    "SYK", "ISRG", "ZTS", "BDX", "UNP", "CSX", "NSC", "BLK", "SPGI", "CME", 
    "SCHW", "GS", "MS", "TGT", "LOW", "BKNG", "TJX", "ADP", "INTU", "CB"
]
TICKERS_100 = list(dict.fromkeys(TICKERS_100)) # remove duplicates if any

def cohens_d(diff: np.ndarray) -> float:
    std = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 1e-10 else 0.0

def rank_biserial_from_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    nonzero_mask = d != 0
    d_nz = d[nonzero_mask]
    if len(d_nz) == 0:
        return 0.0

    ranks = rankdata(np.abs(d_nz))
    w_plus = float(np.sum(ranks[d_nz > 0]))
    w_minus = float(np.sum(ranks[d_nz < 0]))
    total = w_plus + w_minus
    if total == 0:
        return 0.0
    return (w_minus - w_plus) / total

def newey_west_variance(d: np.ndarray, max_lags: Optional[int] = None) -> float:
    t = len(d)
    if max_lags is None:
        max_lags = int(np.floor(t ** (1.0 / 3.0)))

    d_demeaned = d - np.mean(d)
    gamma = np.zeros(max_lags + 1)
    for lag in range(max_lags + 1):
        gamma[lag] = np.dot(d_demeaned[: t - lag], d_demeaned[lag:]) / t

    var_hac = gamma[0]
    for lag in range(1, max_lags + 1):
        weight = 1.0 - lag / (max_lags + 1)
        var_hac += 2.0 * weight * gamma[lag]

    return max(var_hac, 1e-20)

def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    loss: str = "mae",
    alternative: str = "two-sided",
) -> dict:
    if loss == "mse":
        loss_a = errors_a ** 2
        loss_b = errors_b ** 2
    elif loss == "mae":
        loss_a = np.abs(errors_a)
        loss_b = np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = loss_a - loss_b
    t = len(d)
    d_bar = float(np.mean(d))
    var_hac = newey_west_variance(d)
    se = np.sqrt(var_hac / t)
    dm_stat = d_bar / se if se > 1e-10 else 0.0

    if alternative == "less":
        p_val = float(norm.cdf(dm_stat))
    elif alternative == "greater":
        p_val = float(1 - norm.cdf(dm_stat))
    else:
        p_val = float(2 * norm.cdf(-abs(dm_stat)))

    return {
        "dm_statistic": dm_stat,
        "p_value": p_val,
        "mean_loss_diff": d_bar,
        "effect_size_d": cohens_d(d),
        "n_days": t,
    }

def load_or_prepare_data(tickers, start, end, benchmark, config, data_config, logger):
    cache_key = hashlib.md5(f"{sorted(tickers)}_{len(tickers)}{start}{end}{benchmark}".encode()).hexdigest()[:10]
    cache_path = RESULTS_DIR / f"prepared_data_{len(tickers)}_{cache_key}.pkl"

    if cache_path.exists():
        logger.info(f"Loading cached prepared_data for {len(tickers)} tickers: {cache_path}")
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
    else:
        logger.info(f"Downloading/preparing data for {len(tickers)} tickers...")
        data = prepare_data(tickers, start, end, benchmark, config, data_config, logger)
        with open(cache_path, "wb") as fh:
            pickle.dump(data, fh)
    return data

def build_windows(sorted_dates: List[int], train_size: int, val_size: int, test_size: int, step_size: int) -> List[Tuple[List[int], List[int], List[int]]]:
    windows = []
    cursor = 0
    total_days = len(sorted_dates)
    while cursor + train_size + val_size + test_size <= total_days:
        train = sorted_dates[cursor : cursor + train_size]
        val = sorted_dates[cursor + train_size : cursor + train_size + val_size]
        test = sorted_dates[cursor + train_size + val_size : cursor + train_size + val_size + test_size]
        windows.append((train, val, test))
        cursor += step_size
    return windows

def extract_daily_errors(preds: np.ndarray, targets: np.ndarray, logger: object, label: str, n_pairs_per_day: List[int]) -> Optional[dict]:
    n_total = len(preds)
    if n_total != len(targets):
        return None
    if n_total != sum(n_pairs_per_day):
        return None

    daily_mae = []
    daily_mse = []
    cursor = 0
    for n_pairs in n_pairs_per_day:
        if n_pairs == 0:
            daily_mae.append(0.0)
            daily_mse.append(0.0)
            continue
        day_preds = preds[cursor : cursor + n_pairs]
        day_targets = targets[cursor : cursor + n_pairs]
        err = np.abs(day_preds - day_targets)
        daily_mae.append(err.mean())
        daily_mse.append((err ** 2).mean())
        cursor += n_pairs

    return {
        "daily_mae": np.array(daily_mae),
        "daily_mse": np.array(daily_mse),
        "n_days": len(n_pairs_per_day),
    }

def run_scaling_experiment(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    epochs: int = 15,
    train_days: int = 500,
    val_days: int = 125,
    test_days: int = 125,
    step_days: int = 125,
):
    logger = setup_logging("dyfo.scaling_experiment", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Scaling Experiment: TGAT(30) vs TGAT(50) vs TGAT(100)")
    logger.info("=" * 60)

    config = DyFOConfig()
    benchmark = "SPY"

    configs = {
        "TGAT_30": TICKERS_30,
        "TGAT_50": TICKERS_50[:50],  # Ensure exact amounts
        "TGAT_100": TICKERS_100[:100]
    }

    # We collect pooled daily metrics across windows to run valid statistical tests.
    pooled_daily_mae = {name: [] for name in configs}
    scalar_metrics = {name: [] for name in configs}

    # Pre-load data to get intersection dates (since calendars might slightly differ if stocks are missing, we intersect)
    datasets = {}
    for name, tickers in configs.items():
        dc = DataConfig(tickers=tickers, benchmark_ticker=benchmark, start_date=start, end_date=end)
        datasets[name] = load_or_prepare_data(tickers, start, end, benchmark, config, dc, logger)
    
    # Use dates from the 100-asset dataset as the gold standard, or intersect them
    intersect_dates = set(datasets["TGAT_30"]["sorted_dates"])
    for name in configs:
        intersect_dates &= set(datasets[name]["sorted_dates"])
    
    sorted_dates = sorted(list(intersect_dates))
    # --- Use a single Train/Val/Test (60/20/20) split to save compute time ---
    total_days = len(sorted_dates)
    train_end = int(total_days * 0.6)
    val_end = int(total_days * 0.8)
    
    train_dates = sorted_dates[:train_end]
    val_dates = sorted_dates[train_end:val_end]
    test_dates = sorted_dates[val_end:]
    
    logger.info(f"Using a single chronological split: Train ({len(train_dates)} days) -> Val ({len(val_dates)}) -> Test ({len(test_dates)})")

    for name, tickers in configs.items():
        logger.info(f"--- Training {name} ---")
        metrics = train_link_prediction(
            tickers=tickers,
            start=start,
            end=end,
            benchmark=benchmark,
            num_epochs=epochs,
            lr=1e-3,
            mode="regression",
            model_variant="tgat",  # TGAT architecture
            seed=42, # fixed seed for reproducibility without multiple windows
            prepared_data=datasets[name],
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
        )
        
        # Store scalar metrics (R2, Spearman, etc)
        scalar = {k: float(v) for k, v in metrics.items() if not k.startswith("_")}
        scalar_metrics[name].append(scalar)

        # Extract daily errors for this window
        n_pairs_per_day = metrics.get("_n_pairs_per_day", [])
        preds = metrics["_all_preds"].detach().cpu().numpy()
        targets = metrics["_all_targets"].detach().cpu().numpy()
        
        day_level = extract_daily_errors(preds, targets, logger, name, n_pairs_per_day)
        if day_level:
            pooled_daily_mae[name].extend(day_level["daily_mae"].tolist())

    logger.info("=" * 60)
    logger.info("AGGREGATED BENCHMARK METRICS (Averaged across windows)")
    logger.info("=" * 60)
    for name in configs:
        r2 = np.mean([m.get("r_squared", np.nan) for m in scalar_metrics[name]])
        spearman = np.mean([m.get("spearman", np.nan) for m in scalar_metrics[name]])
        mae = np.mean([m.get("mae", np.nan) for m in scalar_metrics[name]])
        f1 = np.mean([m.get("cls_f1", np.nan) for m in scalar_metrics[name]])
        logger.info(f"{name:10} | R2: {r2:.3f} | Spearman: {spearman:.3f} | MAE: {mae:.3f} | F1: {f1:.3f}")

    logger.info("=" * 60)
    logger.info("STATISTICAL COMPARISONS (Diebold-Mariano & Wilcoxon on Series MAE)")
    logger.info("=" * 60)
    
    comparisons = [
        ("TGAT_50", "TGAT_30"),
        ("TGAT_100", "TGAT_30"),
        ("TGAT_100", "TGAT_50"),
    ]

    stat_results = {}
    for a, b in comparisons:
        mae_a = np.array(pooled_daily_mae[a])
        mae_b = np.array(pooled_daily_mae[b])
        
        # Trim to identical length if dates somehow mismatched
        min_len = min(len(mae_a), len(mae_b))
        mae_a = mae_a[:min_len]
        mae_b = mae_b[:min_len]
        
        # Wilcoxon 
        try:
            stat, p_val = wilcoxon(mae_a, mae_b)
        except ValueError:
            stat, p_val = np.nan, np.nan
        
        # Diebold-Mariano
        dm = diebold_mariano_test(mae_a, mae_b, loss="mae", alternative="two-sided")
        
        stat_results[f"{a}_vs_{b}"] = {
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(p_val),
            "dm_stat": float(dm["dm_statistic"]),
            "dm_p": float(dm["p_value"])
        }
        
        logger.info(f"{a} vs {b}:")
        logger.info(f"  Wilcoxon p-value: {p_val:.4e}")
        logger.info(f"  Diebold-Mariano p-value: {dm['p_value']:.4e} (DM Stat: {dm['dm_statistic']:.3f})")

    out_dir = RESULTS_DIR / f"scaling_experiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scaling_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "metrics": scalar_metrics,
            "statistics": stat_results
        }, fh, indent=2)
    logger.info(f"Saved results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_scaling_experiment(epochs=args.epochs)
