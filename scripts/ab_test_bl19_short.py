"""Short A/B test for BL-19 (gradient clipping + LR scheduler).

Runs two short training experiments with identical settings:
- A (baseline): gradient clipping enabled, scheduler disabled
- B (BL-19): gradient clipping enabled, ReduceLROnPlateau enabled

Outputs a compact JSON summary with objective metric deltas.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from train_link_prediction import train_link_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Smaller universe to keep A/B runtime practical.
TICKERS_10 = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
    "JPM", "XOM", "JNJ", "PG", "MA",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _list_link_pred_runs() -> set[str]:
    if not RESULTS_DIR.exists():
        return set()
    return {
        d.name
        for d in RESULTS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("link_pred_")
    }


def _find_new_run(before: set[str], after: set[str]) -> Path:
    new_runs = sorted(after - before)
    if not new_runs:
        raise RuntimeError("Could not detect new run directory in results/.")
    return RESULTS_DIR / new_runs[-1]


def _load_results(run_dir: Path) -> Dict:
    with open(run_dir / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(results: Dict, mode: str) -> Dict[str, float]:
    metrics = results.get("metrics", {})
    if mode == "regression":
        return {
            "best_val_score": float(metrics.get("best_val_r_squared", 0.0)),
            "test_r_squared": float(metrics.get("test_r_squared", 0.0)),
            "test_mae": float(metrics.get("test_mae", 0.0)),
            "test_spearman": float(metrics.get("test_spearman", 0.0)),
            "test_cls_f1": float(metrics.get("test_cls_f1", 0.0)),
            "final_lr": float(metrics.get("final_lr", 0.0)),
            "best_epoch": int(metrics.get("best_epoch", 0)),
        }

    return {
        "best_val_score": float(metrics.get("best_val_auc", 0.0)),
        "test_auc": float(metrics.get("test_auc", 0.0)),
        "test_f1": float(metrics.get("test_f1", 0.0)),
        "test_precision": float(metrics.get("test_precision", 0.0)),
        "test_recall": float(metrics.get("test_recall", 0.0)),
        "final_lr": float(metrics.get("final_lr", 0.0)),
        "best_epoch": int(metrics.get("best_epoch", 0)),
    }


def _compute_delta(a: Dict, b: Dict) -> Dict:
    delta = {}
    for k, a_val in a.items():
        b_val = b.get(k)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            delta[k] = b_val - a_val
    return delta


def run_ab_test(
    tickers: List[str],
    start: str,
    end: str,
    benchmark: str,
    mode: str,
    num_epochs: int,
    seed: int,
) -> Path:
    common_kwargs = {
        "tickers": tickers,
        "start": start,
        "end": end,
        "benchmark": benchmark,
        "num_epochs": num_epochs,
        "lr": 1e-3,
        "corr_threshold": 0.3,
        "neg_ratio": 1.0,
        "early_stopping_patience": 3,
        "weight_decay": 1e-4,
        "pos_weight": 1.0,
        "grad_clip_enabled": True,
        "grad_clip_max_norm": 1.0,
        "mode": mode,
    }

    # Run A: baseline (without scheduler)
    _set_seed(seed)
    before_a = _list_link_pred_runs()
    train_link_prediction(
        scheduler_enabled=False,
        scheduler_factor=0.5,
        scheduler_patience=2,
        scheduler_threshold=1e-4,
        scheduler_min_lr=1e-6,
        scheduler_cooldown=0,
        **common_kwargs,
    )
    after_a = _list_link_pred_runs()
    run_a_dir = _find_new_run(before_a, after_a)
    res_a = _load_results(run_a_dir)

    # Run B: BL-19 (with scheduler)
    _set_seed(seed)
    before_b = _list_link_pred_runs()
    train_link_prediction(
        scheduler_enabled=True,
        scheduler_factor=0.5,
        scheduler_patience=2,
        scheduler_threshold=1e-4,
        scheduler_min_lr=1e-6,
        scheduler_cooldown=0,
        **common_kwargs,
    )
    after_b = _list_link_pred_runs()
    run_b_dir = _find_new_run(before_b, after_b)
    res_b = _load_results(run_b_dir)

    metrics_a = _extract_metrics(res_a, mode)
    metrics_b = _extract_metrics(res_b, mode)

    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "BL-19 short A/B",
            "mode": mode,
            "epochs": num_epochs,
            "seed": seed,
            "start": start,
            "end": end,
            "benchmark": benchmark,
            "num_tickers": len(tickers),
            "tickers": tickers,
        },
        "run_a_baseline": {
            "name": "A_baseline_no_scheduler",
            "run_dir": str(run_a_dir),
            "metrics": metrics_a,
        },
        "run_b_bl19": {
            "name": "B_bl19_with_scheduler",
            "run_dir": str(run_b_dir),
            "metrics": metrics_b,
        },
        "delta_b_minus_a": _compute_delta(metrics_a, metrics_b),
    }

    out_path = RESULTS_DIR / f"ab_bl19_short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("BL-19 short A/B completed")
    print(f"A run: {run_a_dir.name}")
    print(f"B run: {run_b_dir.name}")
    print(f"Summary: {out_path}")
    print("Delta (B - A):")
    for k, v in summary["delta_b_minus_a"].items():
        if isinstance(v, float):
            print(f"  {k:>18}: {v:+.6f}")
        else:
            print(f"  {k:>18}: {v}")
    print("=" * 72)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short A/B test for BL-19")
    parser.add_argument("--epochs", type=int, default=3, choices=[2, 3], help="Number of epochs for each run")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker")
    parser.add_argument("--mode", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_ab_test(
        tickers=TICKERS_10,
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        mode=args.mode,
        num_epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
