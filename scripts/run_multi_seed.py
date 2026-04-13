"""Multi-seed runner for DyFO link-prediction pre-training.

Prepares data once, then trains the model with N different seeds.
Reports mean ± std across seeds for all key metrics.

Usage:
    python scripts/run_multi_seed.py                    # 5 seeds, regression mode
    python scripts/run_multi_seed.py --seeds 42 43 44   # custom seeds
    python scripts/run_multi_seed.py --mode classification
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

REGRESSION_METRICS = ["r_squared", "spearman", "mae", "loss",
                       "cls_precision", "cls_recall", "cls_f1"]
CLASSIFICATION_METRICS = ["auc", "f1", "accuracy", "precision", "recall", "loss"]


def run_multi_seed(
    seeds: list[int],
    tickers: list[str],
    start: str,
    end: str,
    benchmark: str,
    num_epochs: int,
    lr: float,
    early_stopping_patience: int,
    weight_decay: float,
    mode: str,
) -> dict:
    """Prepare data once, then run training for each seed."""

    root_logger = setup_logging("dyfo.multi_seed", log_to_file=False)
    root_logger.info("=" * 60)
    root_logger.info("Multi-seed run: %d seeds %s", len(seeds), seeds)
    root_logger.info("=" * 60)

    config = DyFOConfig()
    data_config = DataConfig(
        tickers=tickers, benchmark_ticker=benchmark,
        start_date=start, end_date=end,
    )

    # ------------------------------------------------------------------
    # Prepare data once — API calls + DCC-GARCH, reused across seeds
    # ------------------------------------------------------------------
    root_logger.info("Preparing data (will be reused across all seeds)...")
    data = prepare_data(
        tickers, start, end, benchmark, config, data_config, root_logger
    )
    root_logger.info("Data ready. Starting seed loop.")

    # ------------------------------------------------------------------
    # Seed loop
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []
    run_tags: list[str] = []

    for seed in seeds:
        root_logger.info("-" * 40)
        root_logger.info("Seed %d / %s", seed, seeds)
        root_logger.info("-" * 40)

        metrics = train_link_prediction(
            tickers=tickers,
            start=start,
            end=end,
            benchmark=benchmark,
            num_epochs=num_epochs,
            lr=lr,
            corr_threshold=0.3,
            neg_ratio=1.0,
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            pos_weight=1.0,
            mode=mode,
            seed=seed,
            prepared_data=data,
        )

        # Strip tensor-valued keys (e.g. _all_preds) before collecting
        scalar_metrics = {
            k: float(v) for k, v in metrics.items()
            if not k.startswith("_")
        }
        all_metrics.append(scalar_metrics)
        root_logger.info("Seed %d done: %s", seed, scalar_metrics)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    key_metrics = REGRESSION_METRICS if mode == "regression" else CLASSIFICATION_METRICS
    summary: dict = {"seeds": seeds, "n": len(seeds), "mode": mode, "per_seed": all_metrics}

    root_logger.info("=" * 60)
    root_logger.info("SUMMARY (N=%d seeds)", len(seeds))
    root_logger.info("=" * 60)

    agg: dict = {}
    for key in key_metrics:
        values = [m[key] for m in all_metrics if key in m]
        if not values:
            continue
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1) if len(values) > 1 else 0.0)
        agg[key] = {"mean": mean, "std": std, "values": values}
        root_logger.info("  %-20s  %.4f ± %.4f  %s", key, mean, std,
                         [f"{v:.4f}" for v in values])

    summary["aggregated"] = agg

    # ------------------------------------------------------------------
    # Persist summary
    # ------------------------------------------------------------------
    out_dir = RESULTS_DIR / f"multi_seed_{mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "multi_seed_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    root_logger.info("Summary saved to %s", summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="DyFO multi-seed training runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mode", choices=["regression", "classification"], default="regression")
    args = parser.parse_args()

    run_multi_seed(
        seeds=args.seeds,
        tickers=TICKERS_30,
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        num_epochs=args.num_epochs,
        lr=args.lr,
        early_stopping_patience=args.patience,
        weight_decay=args.weight_decay,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
