"""Full A/B test for BL-20 (30 assets, multi-seed) to re-evaluate BL-19.

Runs paired experiments for each seed with identical settings:
- A (baseline): gradient clipping enabled, scheduler disabled
- B (BL-19): gradient clipping enabled, ReduceLROnPlateau enabled

Outputs:
- per-seed run metrics and deltas
- aggregate mean/std deltas across seeds
- recommendation on whether to promote scheduler as default
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

TICKERS_30 = [
    # Information Technology (5)
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM",
    # Financials (4)
    "JPM", "GS", "MA", "BRK-B",
    # Health Care (3)
    "JNJ", "UNH", "LLY",
    # Consumer Discretionary (3)
    "AMZN", "TSLA", "HD",
    # Consumer Staples (2)
    "PG", "KO",
    # Energy (2)
    "XOM", "CVX",
    # Industrials (3)
    "CAT", "BA", "RTX",
    # Communication Services (3)
    "META", "GOOGL", "DIS",
    # Materials (2)
    "LIN", "APD",
    # Utilities (2)
    "NEE", "DUK",
    # Real Estate (1)
    "PLD",
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


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}


def run_full_ab_test(
    tickers: List[str],
    start: str,
    end: str,
    benchmark: str,
    mode: str,
    num_epochs: int,
    seeds: List[int],
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
        "early_stopping_patience": 5,
        "weight_decay": 1e-4,
        "pos_weight": 1.0,
        "grad_clip_enabled": True,
        "grad_clip_max_norm": 1.0,
        "mode": mode,
    }

    per_seed = []

    for seed in seeds:
        # A run: baseline (without scheduler)
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

        # B run: BL-19 (with scheduler)
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

        per_seed.append(
            {
                "seed": seed,
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
        )

    metric_names = list(per_seed[0]["delta_b_minus_a"].keys()) if per_seed else []
    agg_delta = {}
    for m in metric_names:
        agg_delta[m] = _mean_std([s["delta_b_minus_a"][m] for s in per_seed])

    scheduler_wins_r2 = sum(1 for s in per_seed if s["delta_b_minus_a"].get("test_r_squared", 0.0) > 0)
    scheduler_wins_spearman = sum(1 for s in per_seed if s["delta_b_minus_a"].get("test_spearman", 0.0) > 0)
    scheduler_wins_mae = sum(1 for s in per_seed if s["delta_b_minus_a"].get("test_mae", 0.0) < 0)

    # Promotion rule: require mean improvement AND majority wins across seeds.
    # This avoids promoting a configuration on a single large outlier seed.
    total_seeds = len(seeds)
    majority = total_seeds // 2 + 1
    recommend_scheduler_default = (
        agg_delta.get("test_r_squared", {}).get("mean", 0.0) > 0
        and agg_delta.get("test_spearman", {}).get("mean", 0.0) > 0
        and agg_delta.get("test_mae", {}).get("mean", 0.0) < 0
        and scheduler_wins_r2 >= majority
        and scheduler_wins_spearman >= majority
        and scheduler_wins_mae >= majority
    )

    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": "BL-20 full A/B",
            "purpose": "Re-evaluate BL-19 scheduler decision",
            "mode": mode,
            "epochs": num_epochs,
            "seeds": seeds,
            "start": start,
            "end": end,
            "benchmark": benchmark,
            "num_tickers": len(tickers),
            "tickers": tickers,
        },
        "per_seed": per_seed,
        "aggregate_delta_b_minus_a": agg_delta,
        "win_count": {
            "scheduler_wins_test_r_squared": scheduler_wins_r2,
            "scheduler_wins_test_spearman": scheduler_wins_spearman,
            "scheduler_wins_test_mae": scheduler_wins_mae,
            "total_seeds": len(seeds),
        },
        "decision": {
            "recommend_scheduler_default": recommend_scheduler_default,
            "rule": "Promote only if mean delta improves R2/Spearman and reduces MAE, plus majority wins per metric across seeds",
        },
    }

    out_path = RESULTS_DIR / f"ab_bl19_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("BL-20 full A/B completed")
    print(f"Seeds: {seeds}")
    print(f"Summary: {out_path}")
    print("Aggregate delta (B - A):")
    for metric, stats in agg_delta.items():
        print(f"  {metric:>18}: mean={stats['mean']:+.6f}, std={stats['std']:.6f}")
    print("Win count (scheduler):")
    print(f"  R2:       {scheduler_wins_r2}/{len(seeds)}")
    print(f"  Spearman: {scheduler_wins_spearman}/{len(seeds)}")
    print(f"  MAE:      {scheduler_wins_mae}/{len(seeds)}")
    print(f"Recommend scheduler default: {recommend_scheduler_default}")
    print("=" * 72)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full BL-20 A/B test for BL-19")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run (recommended: 10-15)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--benchmark", type=str, default="SPY", help="Benchmark ticker")
    parser.add_argument("--mode", type=str, default="regression", choices=["regression", "classification"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 777],
        help="Seeds for paired A/B runs",
    )
    args = parser.parse_args()

    run_full_ab_test(
        tickers=TICKERS_30,
        start=args.start,
        end=args.end,
        benchmark=args.benchmark,
        mode=args.mode,
        num_epochs=args.epochs,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
