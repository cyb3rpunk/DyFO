"""Script to evaluate H4 via Out-of-Sample Block Bootstrap.

This script replaces the expensive multi-window walk-forward validation.
It trains TGN, ROLAND, and GAT_STATIC once on a single historical split
(e.g., Train: 60%, Val: 20%, Test: 20%).
After training, it extracts the out-of-sample portfolio returns
(_realized_returns) from the Test period and performs Block Bootstrapping
to generate empirical p-values for H4 (TGN > ROLAND).
"""

import sys
from pathlib import Path
import json

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

def block_bootstrap_sharpe(returns: np.ndarray, block_size: int = 5, n_iterations: int = 10000, seed: int = 42):
    """Computes distribution of Sharpe proxy using block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    sharpes = []

    for _ in range(n_iterations):
        # Sample block starting indices with replacement
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled_returns = []
        for start_idx in start_indices:
            sampled_returns.extend(returns[start_idx:start_idx + block_size])

        sampled_returns = np.array(sampled_returns[:n])  # Trim to exactly n

        std = np.std(sampled_returns)
        if std > 1e-8:
            sharpe = (np.mean(sampled_returns) / std) * np.sqrt(252)
        else:
            sharpe = 0.0
        sharpes.append(sharpe)

    return np.array(sharpes)

def run_bootstrap_eval(
    start="2020-01-01",
    end="2024-12-31",
    model_variants=["tgn", "roland", "gat_static"],
    epochs=10
):
    logger = setup_logging("dyfo.bootstrap_eval", log_to_file=False)
    logger.info("Starting Bootstrap Evaluation for H4")

    config = DyFOConfig()
    data_config = DataConfig(tickers=TICKERS_30, benchmark_ticker="SPY", start_date=start, end_date=end)

    logger.info("Preparing data...")
    data = prepare_data(TICKERS_30, start, end, "SPY", config, data_config, logger)

    results = {}
    returns_dict = {}

    for variant in model_variants:
        logger.info(f"Training variant {variant}...")
        test_metrics = train_link_prediction(
            tickers=TICKERS_30,
            start=start,
            end=end,
            benchmark="SPY",
            num_epochs=epochs,
            lr=2e-4,
            mode="regression",
            model_variant=variant,
            seed=42,
            prepared_data=data,
        )
        results[variant] = {k: v for k, v in test_metrics.items() if not k.startswith("_")}

        # Save realized returns
        # _realized_returns contains list of daily GMV returns computed in run_split for "test"
        ret = test_metrics.get("_realized_returns", [])
        returns_dict[variant] = np.array(ret)

    logger.info("Training complete. Starting Block Bootstrap...")

    # Bootstrap
    n_iters = 10000
    block_size = 5
    bootstrap_sharpes = {}

    for variant, rets in returns_dict.items():
        if len(rets) == 0:
            logger.error(f"No returns found for {variant}. Skipping bootstrap.")
            continue

        sharpes = block_bootstrap_sharpe(rets, block_size=block_size, n_iterations=n_iters, seed=42)
        bootstrap_sharpes[variant] = sharpes

        ci_lower = np.percentile(sharpes, 2.5)
        ci_upper = np.percentile(sharpes, 97.5)
        avg_sharpe = np.mean(sharpes)
        logger.info(f"{variant.upper()} - Sharpe: {results[variant].get('sharpe_proxy', 0):.4f} | Bootstrap Mean: {avg_sharpe:.4f} | 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    p_value = None
    if "tgn" in bootstrap_sharpes and "roland" in bootstrap_sharpes:
        tgn_s = bootstrap_sharpes["tgn"]
        roland_s = bootstrap_sharpes["roland"]
        # H0: TGN <= ROLAND
        # H1: TGN > ROLAND
        # p-value is the probability that TGN <= ROLAND
        p_value = np.mean(tgn_s <= roland_s)

        logger.info("=" * 50)
        logger.info("HYPOTHESIS H4 VALIDATION")
        logger.info("=" * 50)
        logger.info(f"P(TGN <= ROLAND) [p-value] = {p_value:.4f}")
        if p_value < 0.05:
            logger.info(">>> HYPOTHESIS H4 SUPPORTED! (p < 0.05) ✅")
        else:
            logger.info(">>> HYPOTHESIS H4 NOT SIGNIFICANTLY SUPPORTED. ❌")

    # Also check gat_static just for info
    if "tgn" in bootstrap_sharpes and "gat_static" in bootstrap_sharpes:
        tgn_s = bootstrap_sharpes["tgn"]
        p_val_gat = np.mean(tgn_s <= bootstrap_sharpes["gat_static"])
        logger.info(f"P(TGN <= GAT_STATIC) = {p_val_gat:.4f}")

    # Output to File
    out_dir = RESULTS_DIR / f"bootstrap_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bootstrap_summary.json", "w", encoding="utf-8") as f:
        json_res = {
            "p_value_tgn_vs_roland": float(p_value) if p_value is not None else None,
            "metrics": results,
            "bootstrap_mean_sharpes": {k: float(np.mean(v)) for k, v in bootstrap_sharpes.items()},
            "bootstrap_ci_2.5": {k: float(np.percentile(v, 2.5)) for k, v in bootstrap_sharpes.items()},
            "bootstrap_ci_97.5": {k: float(np.percentile(v, 97.5)) for k, v in bootstrap_sharpes.items()},
        }
        json.dump(json_res, f, indent=2)

    logger.info(f"Summary saved to {out_dir / 'bootstrap_summary.json'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_bootstrap_eval(epochs=args.epochs)
