"""Walk-forward evaluation runner for DyFO link prediction.

Trains TGN, ROLAND, and GAT-Static on rolling windows and validates H4
(TGN Sharpe >= ROLAND in >= 70% of windows) using the window win-rate criterion.

For statistical significance via block bootstrap, use run_bootstrap_eval.py instead.

Usage example:
    python scripts/run_walk_forward.py --epochs 30 --train_days 500 --step_days 125
    python scripts/run_walk_forward.py --mode regression --epochs 2 --train_days 200 \\
        --step_days 100 --start 2023-01-01 --end 2024-12-31 --num_windows 6 --workers 6
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


def _train_window(
    w_idx: int,
    w_train: List[int],
    w_val: List[int],
    w_test: List[int],
    tickers: List[str],
    start: str,
    end: str,
    benchmark: str,
    model_variants: List[str],
    num_epochs: int,
    mode: str,
    seed: int,
    data: dict,
) -> Tuple[int, Dict[str, Dict]]:
    """Train all variants for a single window. Designed for thread-pool execution."""
    logger = setup_logging(f"dyfo.walk_forward.w{w_idx}", log_to_file=False)
    logger.info("Window %d | train=%d val=%d test=%d days",
                w_idx + 1, len(w_train), len(w_val), len(w_test))

    window_results: Dict[str, Dict] = {}
    for variant in model_variants:
        logger.info("Window %d | variant=%s", w_idx + 1, variant.upper())
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
        window_results[variant] = {k: float(v) for k, v in metrics.items() if not k.startswith("_")}

    return w_idx, window_results


def run_walk_forward(
    tickers: List[str],
    start: str,
    end: str,
    benchmark: str,
    train_size: int = 500,
    val_size: int = 125,
    test_size: int = 125,
    step_size: int = 125,
    num_windows: Optional[int] = None,
    mode: str = "regression",
    model_variants: List[str] = ["tgn", "roland", "gat_static"],
    seed: int = 42,
    num_epochs: int = 30,
    workers: int = 1,
) -> Dict:
    """Run rolling walk-forward evaluation across multiple windows.

    Parameters
    ----------
    tickers : list of str
        Asset universe.
    start, end : str
        Date range (YYYY-MM-DD) for the full data download.
    benchmark : str
        Benchmark ticker (e.g. "SPY") for data alignment.
    train_size, val_size, test_size : int
        Number of trading days per split within each window.
    step_size : int
        Number of days to advance between consecutive windows.
    num_windows : int, optional
        Maximum number of windows to run. None = run all available.
    mode : str
        "regression" (predict continuous rho) or "classification".
    model_variants : list of str
        Variants to compare: "tgn", "roland", "gat_static".
    seed : int
        Random seed (same across variants for fair comparison).
    num_epochs : int
        Max epochs per training run (early stopping with patience=5 applies).
    workers : int
        Number of parallel threads for window execution. Each thread trains
        all variants for one window. Use 1 for sequential (safe on GPU).
        Use N > 1 only on CPU or when each window fits in memory independently.

    Returns
    -------
    dict
        Summary with per-window results and H4 validation.
    """
    logger = setup_logging("dyfo.walk_forward", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Walk-Forward Experiment")
    logger.info("Variants  : %s", model_variants)
    logger.info("Mode      : %s", mode)
    logger.info("Window    : train=%d  val=%d  test=%d  step=%d days",
                train_size, val_size, test_size, step_size)
    logger.info("Workers   : %d", workers)
    logger.info("=" * 60)

    config = DyFOConfig()
    data_config = DataConfig(
        tickers=tickers, benchmark_ticker=benchmark,
        start_date=start, end_date=end,
    )

    # ------------------------------------------------------------------ #
    # 1. Download and prepare data once (shared across all windows/variants)
    # ------------------------------------------------------------------ #
    logger.info("Preparing global data for range %s → %s...", start, end)
    data = prepare_data(tickers, start, end, benchmark, config, data_config, logger)
    sorted_dates = data["sorted_dates"]
    total_days = len(sorted_dates)
    logger.info("Total trading days available: %d", total_days)

    # ------------------------------------------------------------------ #
    # 2. Build window index list
    # ------------------------------------------------------------------ #
    windows: List[Tuple[List[int], List[int], List[int]]] = []
    curr = 0
    while curr + train_size + val_size + test_size <= total_days:
        t_slice  = sorted_dates[curr : curr + train_size]
        v_slice  = sorted_dates[curr + train_size : curr + train_size + val_size]
        te_slice = sorted_dates[curr + train_size + val_size : curr + train_size + val_size + test_size]
        windows.append((t_slice, v_slice, te_slice))
        curr += step_size

    if not windows:
        logger.error(
            "Not enough data for even one window. "
            "Reduce train/val/test sizes or widen the date range."
        )
        return {}

    if num_windows is not None:
        windows = windows[:num_windows]

    logger.info("Running %d walk-forward window(s)%s.",
                len(windows),
                f" (capped at --num_windows {num_windows})" if num_windows else "")

    # ------------------------------------------------------------------ #
    # 3. Training loop — sequential or parallel
    # ------------------------------------------------------------------ #
    # Pre-allocate ordered results list
    results_per_window: List[Optional[Dict[str, Dict]]] = [None] * len(windows)

    if workers <= 1:
        for w_idx, (w_train, w_val, w_test) in enumerate(windows):
            logger.info("")
            logger.info("#" * 50)
            logger.info("WINDOW %d / %d", w_idx + 1, len(windows))
            logger.info("  Train : %d days", len(w_train))
            logger.info("  Val   : %d days", len(w_val))
            logger.info("  Test  : %d days", len(w_test))
            logger.info("#" * 50)
            _, window_results = _train_window(
                w_idx, w_train, w_val, w_test,
                tickers, start, end, benchmark,
                model_variants, num_epochs, mode, seed, data,
            )
            results_per_window[w_idx] = window_results
    else:
        effective_workers = min(workers, len(windows))
        logger.info("Launching %d parallel workers for %d windows...",
                    effective_workers, len(windows))
        futures = {}
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            for w_idx, (w_train, w_val, w_test) in enumerate(windows):
                future = pool.submit(
                    _train_window,
                    w_idx, w_train, w_val, w_test,
                    tickers, start, end, benchmark,
                    model_variants, num_epochs, mode, seed, data,
                )
                futures[future] = w_idx

            for future in as_completed(futures):
                w_idx, window_results = future.result()
                results_per_window[w_idx] = window_results
                logger.info("Window %d completed.", w_idx + 1)

    # Re-index as variant → list of per-window dicts
    results_by_variant: Dict[str, List[Dict]] = {v: [] for v in model_variants}
    for window_results in results_per_window:
        if window_results is None:
            continue
        for variant in model_variants:
            results_by_variant[variant].append(window_results.get(variant, {}))

    # ------------------------------------------------------------------ #
    # 4. Aggregate and report
    # ------------------------------------------------------------------ #
    logger.info("")
    logger.info("=" * 60)
    logger.info("WALK-FORWARD SUMMARY  (%d windows)", len(windows))
    logger.info("=" * 60)

    benchmark_metric = "sharpe_proxy" if mode == "regression" else "auc"

    for variant in model_variants:
        vals = [m.get(benchmark_metric, 0.0) for m in results_by_variant[variant]]
        logger.info(
            "%-12s | %s: %.4f ± %.4f  (min=%.4f  max=%.4f)",
            variant.upper(), benchmark_metric,
            np.mean(vals), np.std(vals), np.min(vals), np.max(vals),
        )

    # H4: TGN win-rate >= ROLAND
    if "tgn" in model_variants and "roland" in model_variants and mode == "regression":
        tgn_vals    = [m.get(benchmark_metric, 0.0) for m in results_by_variant["tgn"]]
        roland_vals = [m.get(benchmark_metric, 0.0) for m in results_by_variant["roland"]]
        wins = sum(1 for t, r in zip(tgn_vals, roland_vals) if t >= r)
        win_rate = wins / len(windows)

        logger.info("-" * 40)
        logger.info("H4: TGN >= ROLAND per window")
        logger.info("Win rate: %.1f%%  (%d / %d windows)", win_rate * 100, wins, len(windows))
        if win_rate >= 0.70:
            logger.info(">>> HYPOTHESIS H4 SUPPORTED! (win-rate >= 70%%) ✅")
        else:
            logger.info(">>> HYPOTHESIS H4 NOT SUPPORTED. (win-rate < 70%%) ❌")

    # Supplementary correlation metrics
    if mode == "regression":
        logger.info("-" * 40)
        for metric in ("r_squared", "spearman", "mae", "cls_f1"):
            logger.info("  %-12s", metric)
            for variant in model_variants:
                vals = [m.get(metric, 0.0) for m in results_by_variant[variant]]
                logger.info("    %-12s %.4f ± %.4f", variant.upper(), np.mean(vals), np.std(vals))

    # ------------------------------------------------------------------ #
    # 5. Save results
    # ------------------------------------------------------------------ #
    summary = {
        "mode": mode,
        "windows_count": len(windows),
        "window_config": {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "step_size": step_size,
        },
        "variants": model_variants,
        "results": results_by_variant,
    }

    out_dir = RESULTS_DIR / f"walk_forward_{mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nResults saved to %s", summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="DyFO Walk-Forward Experiment Runner")
    parser.add_argument("--start",       default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",         default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--train_days",  type=int, default=500, help="Training window in trading days")
    parser.add_argument("--val_days",    type=int, default=125, help="Validation window in trading days")
    parser.add_argument("--test_days",   type=int, default=125, help="Test window in trading days")
    parser.add_argument("--step_days",   type=int, default=125, help="Step between windows in trading days")
    parser.add_argument("--num_windows", type=int, default=None, help="Max windows to run (default: all available)")
    parser.add_argument("--mode",        choices=["regression", "classification"], default="regression")
    parser.add_argument("--variants",    nargs="+", default=["tgn", "roland", "gat_static"])
    parser.add_argument("--epochs",      type=int, default=30, help="Max epochs per training run")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--workers",     type=int, default=1,
                        help="Parallel threads for window execution (1=sequential, N>1=parallel). "
                             "Use N>1 only on CPU.")
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
        num_windows=args.num_windows,
        mode=args.mode,
        model_variants=args.variants,
        num_epochs=args.epochs,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
