"""Walk-forward runner for DyFO link prediction.

Executes multiple sliding-window experiments to validate the H4 hypothesis.
Each window has its own training, validation, and testing dates.
Reports metrics for TGN and baselines across all windows.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Make sure project root is on path when running from repo root
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

REGRESSION_METRICS = ["r_squared", "spearman", "mae", "loss", "cls_f1"]


def run_walk_forward(
    tickers: List[str],
    start: str,
    end: str,
    benchmark: str,
    train_size: int = 500,
    val_size: int = 125,
    test_size: int = 125,
    step_size: int = 125,
    mode: str = "regression",
    model_variants: List[str] = ["tgn", "roland", "gat_static"],
    seed: int = 42,
    num_epochs: int = 5,
) -> Dict:
    """Run rolling walk-forward evaluation across multiple windows."""

    root_logger = setup_logging("dyfo.walk_forward", log_to_file=False)
    root_logger.info("=" * 60)
    root_logger.info("Walk-Forward Experiment")
    root_logger.info("Variants: %s", model_variants)
    root_logger.info("Mode:     %s", mode)
    root_logger.info("=" * 60)

    config = DyFOConfig()
    data_config = DataConfig(
        tickers=tickers, benchmark_ticker=benchmark,
        start_date=start, end_date=end,
    )

    # 1. Prepare global data
    root_logger.info("Preparing global data for range %s to %s...", start, end)
    data = prepare_data(tickers, start, end, benchmark, config, data_config, root_logger)
    sorted_dates = data["sorted_dates"]
    total_days = len(sorted_dates)
    root_logger.info("Total trading days available: %d", total_days)

    # 2. Define windows
    windows: List[Tuple[List[int], List[int], List[int]]] = []
    curr = 0
    while curr + train_size + val_size + test_size <= total_days:
        t_start = curr
        t_end = curr + train_size
        v_start = t_end
        v_end = t_end + val_size
        te_start = v_end
        te_end = v_end + test_size
        
        windows.append((
            sorted_dates[t_start:t_end],
            sorted_dates[v_start:v_end],
            sorted_dates[te_start:te_end]
        ))
        curr += step_size

    root_logger.info("Generated %d walk-forward windows.", len(windows))
    if not windows:
        root_logger.error("Not enough data for even one window! Check sizes/dates.")
        return {}

    # 3. Execution loop
    results_by_variant = {v: [] for v in model_variants}

    for w_idx, (w_train, w_val, w_test) in enumerate(windows):
        root_logger.info("\n" + "#" * 40)
        root_logger.info("WINDOW %d / %d", w_idx + 1, len(windows))
        root_logger.info("Train: %d to %d", w_train[0], w_train[-1])
        root_logger.info("Test:  %d to %d", w_test[0], w_test[-1])
        root_logger.info("#" * 40)

        for variant in model_variants:
            root_logger.info("--- Training Variant: %s ---", variant.upper())
            
            # Call training for this window
            metrics = train_link_prediction(
                tickers=tickers,
                start=start,
                end=end,
                benchmark=benchmark,
                num_epochs=num_epochs,
                lr=2e-4,
                mode=mode,
                model_variant=variant,
                seed=seed,
                prepared_data=data,
                train_dates=w_train,
                val_dates=w_val,
                test_dates=w_test,
            )
            
            # Collect scalar metrics
            scalar_metrics = {
                k: float(v) for k, v in metrics.items()
                if not k.startswith("_")
            }
            results_by_variant[variant].append(scalar_metrics)

    # 4. Aggregation and H4 validation
    summary = {
        "mode": mode,
        "windows_count": len(windows),
        "variants": model_variants,
        "results": results_by_variant,
    }

    root_logger.info("\n" + "=" * 60)
    root_logger.info("FINAL WALK-FORWARD SUMMARY")
    root_logger.info("=" * 60)

    # Key metric for H4: sharpe_proxy (primary economic utility)
    benchmark_metric = "sharpe_proxy" if mode == "regression" else "auc"
    
    for variant in model_variants:
        metrics_list = results_by_variant[variant]
        vals = [m.get(benchmark_metric, 0.0) for m in metrics_list]
        avg = np.mean(vals)
        std = np.std(vals)
        root_logger.info("%-10s | %s: %.4f ± %.4f", variant.upper(), benchmark_metric, avg, std)

    # H4 Check: TGN vs ROLAND
    if "tgn" in model_variants and "roland" in model_variants and mode == "regression":
        tgn_vals = [m.get(benchmark_metric, 0.0) for m in results_by_variant["tgn"]]
        roland_vals = [m.get(benchmark_metric, 0.0) for m in results_by_variant["roland"]]
        
        wins = sum(1 for t, r in zip(tgn_vals, roland_vals) if t >= r)
        win_rate = wins / len(windows)
        
        root_logger.info("-" * 30)
        root_logger.info("H4 Validation (TGN >= ROLAND)")
        root_logger.info("Win rate: %.1f%% (%d/%d windows)", win_rate * 100, wins, len(windows))
        if win_rate >= 0.70:
            root_logger.info(">>> HYPOTHESIS H4 SUPPORTED! ✅")
        else:
            root_logger.info(">>> HYPOTHESIS H4 NOT SUPPORTED. ❌")

    # Save to file
    out_dir = RESULTS_DIR / f"walk_forward_{mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    root_logger.info("\nResults saved to %s", summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="DyFO Walk-Forward Experiment Runner")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--train_days", type=int, default=500, help="Size of training window")
    parser.add_argument("--val_days", type=int, default=125, help="Size of validation window")
    parser.add_argument("--test_days", type=int, default=125, help="Size of test window")
    parser.add_argument("--step_days", type=int, default=125, help="Step between windows")
    parser.add_argument("--mode", choices=["regression", "classification"], default="regression")
    parser.add_argument("--variants", nargs="+", default=["tgn", "roland", "gat_static"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_walk_forward(
        tickers=TICKERS_30,
        start=args.start,
        end=args.end,
        benchmark="SPY",
        train_size=args.train_days,
        val_size=args.val_days,
        test_size=args.test_days,
        step_size=args.step_days,
        mode=args.mode,
        model_variants=args.variants,
        num_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
