#!/usr/bin/env python3
"""Distinct-period COVID smoketest for TGAT vs EWMA vs Persistence.

This runner is intentionally small and causal:
- train: pre-COVID only
- validation: pre-COVID only
- test: COVID period only

TGAT is trained with the same ``train_link_prediction`` path used by DyFO,
while EWMA and Persistence are evaluated through the exact same split API.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.ticker_registry import get_tickers
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_v5 import (
    BASELINE_LR,
    BASELINE_PATIENCE,
    BASELINE_USE_COSINE,
    TGN_LR,
    TGN_PATIENCE,
    TGN_USE_COSINE,
    load_or_prepare_data,
)
from scripts.train_link_prediction import train_link_prediction


EPOCH = date(2000, 1, 1)


def int_day_to_iso(day: int) -> str:
    return (EPOCH + timedelta(days=int(day))).isoformat()


def slice_dates(sorted_dates: List[int], start: str, end: str) -> List[int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected: List[int] = []
    for day in sorted_dates:
        ts = pd.Timestamp(int_day_to_iso(day))
        if start_ts <= ts <= end_ts:
            selected.append(day)
    return selected


def hyperparams_for_variant(variant: str) -> tuple[float, bool, int]:
    if variant == "tgat":
        return TGN_LR, TGN_USE_COSINE, TGN_PATIENCE
    return BASELINE_LR, BASELINE_USE_COSINE, BASELINE_PATIENCE


def run_smoketest(
    n_tickers: int,
    epochs: int,
    seed: int,
    data_start: str,
    data_end: str,
    train_end: str,
    val_end: str,
    test_end: str,
) -> dict:
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"smoketest_covid_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging("dyfo.smoketest_covid_compare", log_to_file=False)
    logger.info("COVID smoketest | distinct train/val/test periods")

    tickers = get_tickers(n_tickers)
    config = DyFOConfig(model_variant="tgat")
    data_config = DataConfig(
        tickers=tickers,
        benchmark_ticker="SPY",
        start_date=data_start,
        end_date=data_end,
    )

    data = load_or_prepare_data(
        tickers=tickers,
        start=data_start,
        end=data_end,
        benchmark="SPY",
        config=config,
        data_config=data_config,
        logger=logger,
    )

    train_dates = slice_dates(data["sorted_dates"], data_start, train_end)
    val_dates = slice_dates(data["sorted_dates"], (pd.Timestamp(train_end) + pd.Timedelta(days=1)).date().isoformat(), val_end)
    test_dates = slice_dates(data["sorted_dates"], (pd.Timestamp(val_end) + pd.Timedelta(days=1)).date().isoformat(), test_end)

    if not train_dates or not val_dates or not test_dates:
        raise RuntimeError("One of the train/val/test slices is empty. Check the chosen date boundaries.")

    logger.info(
        "Split | train=%s..%s (%d) | val=%s..%s (%d) | test=%s..%s (%d)",
        int_day_to_iso(train_dates[0]), int_day_to_iso(train_dates[-1]), len(train_dates),
        int_day_to_iso(val_dates[0]), int_day_to_iso(val_dates[-1]), len(val_dates),
        int_day_to_iso(test_dates[0]), int_day_to_iso(test_dates[-1]), len(test_dates),
    )

    summary = {
        "run_tag": out_dir.name,
        "purpose": "COVID distinct-period smoketest",
        "run_config": {
            "variants": ["tgat", "ewma", "persistence"],
            "n_tickers": n_tickers,
            "epochs": epochs,
            "seed": seed,
            "data_start": data_start,
            "data_end": data_end,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
        },
        "split": {
            "train": {"start": int_day_to_iso(train_dates[0]), "end": int_day_to_iso(train_dates[-1]), "n_days": len(train_dates)},
            "val": {"start": int_day_to_iso(val_dates[0]), "end": int_day_to_iso(val_dates[-1]), "n_days": len(val_dates)},
            "test": {"start": int_day_to_iso(test_dates[0]), "end": int_day_to_iso(test_dates[-1]), "n_days": len(test_dates)},
        },
        "metrics_by_variant": {},
    }

    for variant in ["tgat", "ewma", "persistence"]:
        lr, use_cosine, patience = hyperparams_for_variant(variant)
        logger.info("Running %s", variant.upper())
        metrics = train_link_prediction(
            tickers=tickers,
            start=data_start,
            end=test_end,
            benchmark="SPY",
            num_epochs=epochs,
            lr=lr,
            corr_threshold=0.3,
            neg_ratio=1.0,
            early_stopping_patience=patience,
            weight_decay=1e-4,
            pos_weight=1.0,
            mode="regression",
            model_variant=variant,
            seed=seed,
            prepared_data=data,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
            use_cosine_schedule=use_cosine,
        )
        summary["metrics_by_variant"][variant] = {
            "r_squared": float(metrics.get("r_squared", float("nan"))),
            "mae": float(metrics.get("mae", float("nan"))),
            "mse": float(metrics.get("mse", metrics.get("loss", float("nan")))),
            "spearman": float(metrics.get("spearman", float("nan"))),
            "loss": float(metrics.get("loss", float("nan"))),
            "cls_f1": float(metrics.get("cls_f1", float("nan"))),
            "best_epoch": int(metrics.get("_best_epoch", 1)),
            "best_val_r_squared": float(metrics.get("_best_val_r_squared", float("nan"))),
        }

    ranking = sorted(
        (
            {"variant": variant, **metrics}
            for variant, metrics in summary["metrics_by_variant"].items()
        ),
        key=lambda row: row["r_squared"],
        reverse=True,
    )
    summary["ranking_by_r_squared"] = ranking

    out_path = out_dir / "smoketest_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved summary -> {out_path}")
    print(
        "Test period:",
        f"{summary['split']['test']['start']} -> {summary['split']['test']['end']}",
        f"({summary['split']['test']['n_days']} trading days)",
    )
    for row in ranking:
        print(
            f"{row['variant']:12s} "
            f"R2={row['r_squared']:.4f}  "
            f"MAE={row['mae']:.4f}  "
            f"Spearman={row['spearman']:.4f}"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distinct-period COVID smoketest: TGAT vs EWMA vs Persistence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_tickers", type=int, choices=[30, 50, 100], default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_start", default="2018-01-01")
    parser.add_argument("--data_end", default="2024-12-31")
    parser.add_argument("--train_end", default="2019-09-30")
    parser.add_argument("--val_end", default="2019-12-31")
    parser.add_argument("--test_end", default="2020-06-30")
    args = parser.parse_args()

    run_smoketest(
        n_tickers=args.n_tickers,
        epochs=args.epochs,
        seed=args.seed,
        data_start=args.data_start,
        data_end=args.data_end,
        train_end=args.train_end,
        val_end=args.val_end,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
