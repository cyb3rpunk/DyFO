"""Bootstrap eval for BL-18 Temporal KG ablation."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_ra_htgn import train_variant_for_window_ra_htgn
from scripts.run_bootstrap_eval_v5 import (
    BASELINE_LR,
    BASELINE_PATIENCE,
    BASELINE_USE_COSINE,
    TGN_LR,
    TGN_PATIENCE,
    TGN_USE_COSINE,
    TICKERS_30,
    _cvar,
    _interpret_effect_r,
    _sharpe,
    build_windows,
    diebold_mariano_test,
    extract_daily_errors,
    holm_bonferroni,
    load_or_prepare_data,
    paired_block_bootstrap_multi,
    run_window_wilcoxon,
)
from scripts.train_link_prediction import train_link_prediction


DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_DAYS = 500
DEFAULT_VAL_DAYS = 125
DEFAULT_TEST_DAYS = 125
DEFAULT_STEP_DAYS = 63
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BLOCK_SIZE = 5
DEFAULT_MAX_WINDOWS = 12

TKG_VARIANTS = ["temporal_kg", "ra_htgn", "tgn", "roland", "gat_static"]
TKG_COMPARISON_PAIRS = list(combinations(TKG_VARIANTS, 2))
N_PAIRS = 435


def train_variant_for_window_temporal_kg(
    variant: str,
    data: dict,
    start: str,
    end: str,
    benchmark: str,
    epochs: int,
    train_dates: List[int],
    val_dates: List[int],
    test_dates: List[int],
) -> dict:
    if variant != "temporal_kg":
        return train_variant_for_window_ra_htgn(
            variant=variant,
            data=data,
            start=start,
            end=end,
            benchmark=benchmark,
            epochs=epochs,
            train_dates=train_dates,
            val_dates=val_dates,
            test_dates=test_dates,
        )

    return train_link_prediction(
        tickers=TICKERS_30,
        start=start,
        end=end,
        benchmark=benchmark,
        num_epochs=epochs,
        mode="regression",
        model_variant=variant,
        seed=42,
        prepared_data=data,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        weight_decay=1e-4,
        lr=BASELINE_LR,
        use_cosine_schedule=BASELINE_USE_COSINE,
        early_stopping_patience=BASELINE_PATIENCE,
    )


def run_bootstrap_eval_temporal_kg(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    epochs: int = DEFAULT_EPOCHS,
    train_days: int = DEFAULT_TRAIN_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_windows: Optional[int] = DEFAULT_MAX_WINDOWS,
):
    logger = setup_logging("dyfo.bootstrap_eval_temporal_kg", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Bootstrap Eval Temporal KG - BL-18 walk-forward validation")
    logger.info("=" * 60)

    config = DyFOConfig(model_variant="temporal_kg")
    data_config = DataConfig(
        tickers=TICKERS_30,
        benchmark_ticker="SPY",
        start_date=start,
        end_date=end,
    )
    data = load_or_prepare_data(TICKERS_30, start, end, "SPY", config, data_config, logger)
    windows = build_windows(
        data["sorted_dates"],
        train_size=train_days,
        val_size=val_days,
        test_size=test_days,
        step_size=step_days,
        max_windows=max_windows,
    )
    if not windows:
        raise RuntimeError("No walk-forward windows could be constructed with the requested settings.")
    logger.info(
        "Walk-forward windows: %d | train=%d val=%d test=%d step=%d | date span=%s..%s",
        len(windows), train_days, val_days, test_days, step_days, start, end,
    )

    scalar_results = {variant: [] for variant in TKG_VARIANTS}
    realized_returns = {variant: [] for variant in TKG_VARIANTS}
    daily_losses = {variant: [] for variant in TKG_VARIANTS}
    temporal_kg_artifacts = []
    window_reports = []
    confirmatory_p_values: Dict[str, Optional[float]] = {}
    exploratory_p_values: Dict[str, Optional[float]] = {}

    for window_idx, (train_dates, val_dates, test_dates) in enumerate(windows, start=1):
        logger.info("-" * 60)
        logger.info(
            "Window %d/%d | train=%d val=%d test=%d",
            window_idx, len(windows), len(train_dates), len(val_dates), len(test_dates),
        )

        window_raw = {}
        for variant in TKG_VARIANTS:
            logger.info("Training %s on window %d", variant.upper(), window_idx)
            metrics = train_variant_for_window_temporal_kg(
                variant=variant,
                data=data,
                start=start,
                end=end,
                benchmark="SPY",
                epochs=epochs,
                train_dates=train_dates,
                val_dates=val_dates,
                test_dates=test_dates,
            )
            window_raw[variant] = metrics
            scalar_results[variant].append({
                key: float(val)
                for key, val in metrics.items()
                if not key.startswith("_")
            })
            realized_returns[variant].append(np.array(metrics.get("_realized_returns", []), dtype=float))

            if variant == "temporal_kg":
                temporal_kg_artifacts.append(metrics.get("_temporal_kg_artifacts", {}))

            if "_all_preds" in metrics and "_all_targets" in metrics:
                preds = metrics["_all_preds"].detach().cpu().numpy()
                targets = metrics["_all_targets"].detach().cpu().numpy()
                day_level = extract_daily_errors(
                    preds,
                    targets,
                    logger,
                    label=f"{variant}/window_{window_idx}",
                    n_pairs=N_PAIRS,
                )
                daily_losses[variant].append(day_level)
            else:
                daily_losses[variant].append(None)

        report = {
            "window_index": window_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "metrics": {variant: scalar_results[variant][-1] for variant in TKG_VARIANTS},
            "comparisons": {},
            "temporal_kg_explanations": temporal_kg_artifacts[-1].get("top_explanations", []) if temporal_kg_artifacts else [],
            "temporal_kg_relation_scores": temporal_kg_artifacts[-1].get("last_relation_scores", {}) if temporal_kg_artifacts else {},
        }

        for left, right in TKG_COMPARISON_PAIRS:
            rets_left = realized_returns[left][-1]
            rets_right = realized_returns[right][-1]
            if len(rets_left) == 0 or len(rets_right) == 0 or len(rets_left) != len(rets_right):
                continue

            paired = paired_block_bootstrap_multi(
                rets_left,
                rets_right,
                block_size=block_size,
                n_iterations=n_bootstrap,
                seed=42 + window_idx,
            )
            diff_b = paired["sharpe_diffs"]
            cvar_diff_b = paired["cvar_diffs"]
            d_obs = _sharpe(rets_left) - _sharpe(rets_right)
            cvar_obs = _cvar(rets_left) - _cvar(rets_right)

            report["comparisons"][f"{left}_vs_{right}"] = {
                "sharpe_obs_diff": float(d_obs),
                "sharpe_bootstrap_ci_2.5": float(np.percentile(diff_b, 2.5)),
                "sharpe_bootstrap_ci_97.5": float(np.percentile(diff_b, 97.5)),
                "sharpe_p_direct": float(np.mean(diff_b <= 0.0)),
                "sharpe_effect_size_d": float(np.mean(diff_b) / np.std(diff_b, ddof=1)) if len(diff_b) > 1 and np.std(diff_b, ddof=1) > 1e-10 else 0.0,
                "cvar_obs_diff": float(cvar_obs),
                "cvar_bootstrap_ci_2.5": float(np.percentile(cvar_diff_b, 2.5)),
                "cvar_bootstrap_ci_97.5": float(np.percentile(cvar_diff_b, 97.5)),
                "cvar_p_direct": float(np.mean(cvar_diff_b >= 0.0)),
            }
        window_reports.append(report)

    tkg_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["temporal_kg"]], dtype=float)
    tgn_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["tgn"]], dtype=float)
    ra_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["ra_htgn"]], dtype=float)

    valid_primary_mask = np.isfinite(tkg_window_sharpes) & np.isfinite(ra_window_sharpes)
    primary_wins = int(np.sum(tkg_window_sharpes[valid_primary_mask] >= ra_window_sharpes[valid_primary_mask]))
    primary_n = int(np.sum(valid_primary_mask))
    primary_win_rate = float(primary_wins / primary_n) if primary_n > 0 else float("nan")

    if primary_n == 0:
        raise RuntimeError("No valid windows available for the primary Temporal KG vs RA-HTGN comparison.")

    logger.info(
        "Temporal KG >= RA-HTGN in %d/%d windows (%.1f%%)",
        primary_wins,
        primary_n,
        primary_win_rate * 100 if primary_n else float("nan"),
    )

    binom_res = binomtest(primary_wins, primary_n, p=0.50, alternative="greater")
    binom_ci = binom_res.proportion_ci(confidence_level=0.95)
    confirmatory_p_values["temporal_kg_win_rate_gt_0.50_vs_ra_htgn"] = float(binom_res.pvalue)

    primary_wilcoxon = run_window_wilcoxon(
        tkg_window_sharpes[valid_primary_mask],
        ra_window_sharpes[valid_primary_mask],
        alternative="greater",
    )
    if primary_wilcoxon is not None:
        confirmatory_p_values["temporal_kg_window_wilcoxon_sharpe_vs_ra_htgn"] = primary_wilcoxon["p_value"]
        logger.info(
            "Window-level Wilcoxon Sharpe: p=%.4e | r=%.3f (%s)",
            primary_wilcoxon["p_value"],
            primary_wilcoxon["effect_size_r"],
            _interpret_effect_r(primary_wilcoxon["effect_size_r"]),
        )

    if primary_win_rate > 0.50 and binom_res.pvalue < 0.05:
        primary_conclusion = "BL18 INTERPRETABLE ARM COMPETITIVE"
        primary_reason = (
            "Temporal KG outperformed RA-HTGN in a majority of windows and the "
            "one-sided exact binomial test was significant at alpha=0.05."
        )
    elif primary_win_rate <= 0.50:
        primary_conclusion = "BL18 INTERPRETABLE ARM TRAILS RA-HTGN"
        primary_reason = "Temporal KG did not beat RA-HTGN in a majority of walk-forward windows."
    else:
        primary_conclusion = "BL18 INTERPRETABLE ARM INCONCLUSIVE"
        primary_reason = (
            "Temporal KG showed a majority win-rate against RA-HTGN, but the one-sided exact "
            "binomial test did not reject the null at alpha=0.05."
        )

    pairwise_window_tests = {}
    for left, right in TKG_COMPARISON_PAIRS:
        left_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[left]], dtype=float)
        right_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[right]], dtype=float)
        valid_mask = np.isfinite(left_sharpes) & np.isfinite(right_sharpes)
        res = run_window_wilcoxon(left_sharpes[valid_mask], right_sharpes[valid_mask], alternative="greater")
        if res is not None:
            pairwise_window_tests[f"{left}_vs_{right}"] = res
            exploratory_p_values[f"window_wilcoxon_{left}_vs_{right}"] = res["p_value"]

    pooled_dm_results = {}
    if step_days >= test_days:
        for left, right in TKG_COMPARISON_PAIRS:
            left_daily = [entry for entry in daily_losses[left] if entry is not None]
            right_daily = [entry for entry in daily_losses[right] if entry is not None]
            paired_daily = list(zip(left_daily, right_daily))
            if not paired_daily:
                continue

            left_mae = np.concatenate([x["daily_mae"] for x, _ in paired_daily])
            right_mae = np.concatenate([y["daily_mae"] for _, y in paired_daily])
            left_mse = np.concatenate([x["daily_mse"] for x, _ in paired_daily])
            right_mse = np.concatenate([y["daily_mse"] for _, y in paired_daily])

            dm_mae = diebold_mariano_test(left_mae, right_mae, loss="mae", alternative="less")
            dm_mse = diebold_mariano_test(left_mse, right_mse, loss="mae", alternative="less")
            pooled_dm_results[f"{left}_vs_{right}_mae"] = dm_mae
            pooled_dm_results[f"{left}_vs_{right}_mse"] = dm_mse
            exploratory_p_values[f"dm_mae_{left}_vs_{right}"] = dm_mae["p_value"]
            exploratory_p_values[f"dm_mse_{left}_vs_{right}"] = dm_mse["p_value"]
    else:
        logger.warning("step_days < test_days, so test windows overlap. Skipping pooled daily DM tests.")

    summary = {
        "version": "temporal_kg_v1_bl18",
        "spec_alignment": {
            "source_protocol": "run_bootstrap_eval_v5.py",
            "minimum_bl18_comparisons": [
                "temporal_kg vs tgn",
                "temporal_kg vs ra_htgn",
            ],
            "all_vs_all_pairwise_tests": [f"{left} vs {right}" for left, right in TKG_COMPARISON_PAIRS],
            "interpretability_output": "top temporal facts and relation-level plausibility scores per window",
        },
        "window_config": {
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "step_days": step_days,
            "n_windows": len(windows),
        },
        "variants": TKG_VARIANTS,
        "comparison_pairs": TKG_COMPARISON_PAIRS,
        "metrics_by_variant": scalar_results,
        "window_reports": window_reports,
        "temporal_kg_interpretability": temporal_kg_artifacts,
        "primary_temporal_kg_vs_ra_htgn": {
            "wins": primary_wins,
            "n_windows": primary_n,
            "win_rate": primary_win_rate,
            "threshold": 0.50,
            "conclusion": primary_conclusion,
            "conclusion_reason": primary_reason,
            "binomial_test": {
                "p_value": float(binom_res.pvalue),
                "ci_95_low": float(binom_ci.low),
                "ci_95_high": float(binom_ci.high),
            },
            "window_wilcoxon_sharpe": primary_wilcoxon,
            "holm_bonferroni_confirmatory": holm_bonferroni(confirmatory_p_values),
        },
        "pairwise_window_tests": pairwise_window_tests,
        "pooled_predictive_tests": pooled_dm_results,
        "holm_bonferroni_exploratory": holm_bonferroni(exploratory_p_values),
        "descriptive_summary": {
            "mean_window_sharpe": {
                variant: float(np.nanmean(np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[variant]], dtype=float)))
                for variant in TKG_VARIANTS
            },
            "std_window_sharpe": {
                variant: (
                    float(np.nanstd(np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[variant]], dtype=float), ddof=1))
                    if len(scalar_results[variant]) > 1
                    else 0.0
                )
                for variant in TKG_VARIANTS
            },
            "mean_window_metrics": {
                variant: {
                    metric: float(np.mean([entry.get(metric, np.nan) for entry in scalar_results[variant]]))
                    for metric in ["r_squared", "spearman", "mae", "loss", "cls_f1", "sharpe_proxy"]
                }
                for variant in TKG_VARIANTS
            },
        },
        "config": {
            "epochs": epochs,
            "bootstrap_n_iterations_per_window": n_bootstrap,
            "bootstrap_block_size": block_size,
            "tgn_lr": TGN_LR,
            "baseline_lr": BASELINE_LR,
            "tgn_patience": TGN_PATIENCE,
            "baseline_patience": BASELINE_PATIENCE,
            "tgn_use_cosine": TGN_USE_COSINE,
            "baseline_use_cosine": BASELINE_USE_COSINE,
        },
    }

    out_dir = RESULTS_DIR / f"bootstrap_eval_temporal_kg_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_summary_temporal_kg.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved summary to %s", out_path)
    logger.info(
        "BL-18 result: %s | win_rate=%.1f%% | binomial p=%.4e",
        primary_conclusion,
        primary_win_rate * 100,
        binom_res.pvalue,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="DyFO bootstrap eval for BL-18 Temporal KG")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train_days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--val_days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--test_days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step_days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max_windows", type=int, default=DEFAULT_MAX_WINDOWS)
    args = parser.parse_args()

    run_bootstrap_eval_temporal_kg(
        start=args.start,
        end=args.end,
        epochs=args.epochs,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        n_bootstrap=args.n_bootstrap,
        block_size=args.block_size,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
