"""Bootstrap eval for BL-17 relation-aware HTGN.

This runner mirrors the walk-forward protocol from ``run_bootstrap_eval_v5.py``
while adding the new ``ra_htgn`` variant as the primary candidate model.
The frozen v5 runner remains untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.logging_utils import RESULTS_DIR, setup_logging
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
    rank_biserial_from_wilcoxon,
    run_window_wilcoxon,
)
from scripts.train_link_prediction import train_link_prediction

RA_VARIANTS = ["ra_htgn", "tgn", "roland", "gat_static"]
RA_COMPARISON_PAIRS = [
    ("ra_htgn", "tgn"),
    ("ra_htgn", "roland"),
    ("ra_htgn", "gat_static"),
]
N_PAIRS = 435


def train_variant_for_window_ra_htgn(
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
    common_kwargs = {
        "tickers": TICKERS_30,
        "start": start,
        "end": end,
        "benchmark": benchmark,
        "num_epochs": epochs,
        "mode": "regression",
        "model_variant": variant,
        "seed": 42,
        "prepared_data": data,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "weight_decay": 1e-4,
    }

    if variant in {"tgn", "ra_htgn"}:
        return train_link_prediction(
            lr=TGN_LR,
            use_cosine_schedule=TGN_USE_COSINE,
            early_stopping_patience=TGN_PATIENCE,
            **common_kwargs,
        )

    return train_link_prediction(
        lr=BASELINE_LR,
        use_cosine_schedule=BASELINE_USE_COSINE,
        early_stopping_patience=BASELINE_PATIENCE,
        **common_kwargs,
    )


def run_bootstrap_eval_ra_htgn(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    epochs: int = 30,
    train_days: int = 500,
    val_days: int = 125,
    test_days: int = 125,
    step_days: int = 125,
    n_bootstrap: int = 500,
    block_size: int = 5,
    max_windows: Optional[int] = None,
):
    logger = setup_logging("dyfo.bootstrap_eval_ra_htgn", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Bootstrap Eval RA-HTGN - BL-17 walk-forward validation")
    logger.info("=" * 60)

    config = DyFOConfig(model_variant="ra_htgn")
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
        "Walk-forward windows: %d | train=%d val=%d test=%d step=%d",
        len(windows), train_days, val_days, test_days, step_days,
    )

    scalar_results = {variant: [] for variant in RA_VARIANTS}
    realized_returns = {variant: [] for variant in RA_VARIANTS}
    daily_losses = {variant: [] for variant in RA_VARIANTS}
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
        for variant in RA_VARIANTS:
            logger.info("Training %s on window %d", variant.upper(), window_idx)
            metrics = train_variant_for_window_ra_htgn(
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

        ra_sharpe = float(window_raw["ra_htgn"].get("sharpe_proxy", np.nan))
        tgn_sharpe = float(window_raw["tgn"].get("sharpe_proxy", np.nan))
        roland_sharpe = float(window_raw["roland"].get("sharpe_proxy", np.nan))
        gat_sharpe = float(window_raw["gat_static"].get("sharpe_proxy", np.nan))

        report = {
            "window_index": window_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "metrics": {
                variant: scalar_results[variant][-1] for variant in RA_VARIANTS
            },
            "comparisons": {},
        }

        for left, right in RA_COMPARISON_PAIRS:
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

        report["ra_vs_tgn_window_win"] = bool(np.isfinite(ra_sharpe) and np.isfinite(tgn_sharpe) and ra_sharpe >= tgn_sharpe)
        report["ra_vs_tgn_margin"] = float(ra_sharpe - tgn_sharpe)
        report["ra_vs_roland_margin"] = float(ra_sharpe - roland_sharpe)
        report["ra_vs_gat_margin"] = float(ra_sharpe - gat_sharpe)
        window_reports.append(report)

    ra_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["ra_htgn"]], dtype=float)
    tgn_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["tgn"]], dtype=float)
    roland_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["roland"]], dtype=float)
    gat_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["gat_static"]], dtype=float)

    valid_primary_mask = np.isfinite(ra_window_sharpes) & np.isfinite(tgn_window_sharpes)
    primary_wins = int(np.sum(ra_window_sharpes[valid_primary_mask] >= tgn_window_sharpes[valid_primary_mask]))
    primary_n = int(np.sum(valid_primary_mask))
    primary_win_rate = float(primary_wins / primary_n) if primary_n > 0 else float("nan")

    logger.info("=" * 60)
    logger.info("PRIMARY BL-17 EVALUATION")
    logger.info(
        "RA-HTGN >= TGN in %d/%d windows (%.1f%%)",
        primary_wins,
        primary_n,
        primary_win_rate * 100 if primary_n else float("nan"),
    )

    if primary_n == 0:
        raise RuntimeError("No valid windows available for the primary RA-HTGN vs TGN comparison.")

    binom_res = binomtest(primary_wins, primary_n, p=0.70, alternative="greater")
    binom_ci = binom_res.proportion_ci(confidence_level=0.95)
    confirmatory_p_values["ra_htgn_win_rate_gt_0.70_vs_tgn"] = float(binom_res.pvalue)

    small_sample_warning = None
    if primary_n < 5:
        small_sample_warning = (
            f"Only {primary_n} walk-forward windows were available. "
            "Treat this as a smoke test or preliminary read, not strong confirmatory evidence."
        )
        logger.warning(small_sample_warning)

    logger.info(
        "Exact binomial test vs 70%% threshold: p=%.4e | 95%% CI=[%.3f, %.3f]",
        binom_res.pvalue, binom_ci.low, binom_ci.high,
    )

    primary_wilcoxon = run_window_wilcoxon(
        ra_window_sharpes[valid_primary_mask],
        tgn_window_sharpes[valid_primary_mask],
        alternative="greater",
    )
    if primary_wilcoxon is not None:
        confirmatory_p_values["ra_htgn_window_wilcoxon_sharpe_vs_tgn"] = primary_wilcoxon["p_value"]
        logger.info(
            "Window-level Wilcoxon Sharpe: p=%.4e | r=%.3f (%s)",
            primary_wilcoxon["p_value"],
            primary_wilcoxon["effect_size_r"],
            _interpret_effect_r(primary_wilcoxon["effect_size_r"]),
        )
    else:
        logger.info("Window-level Wilcoxon Sharpe: skipped (requires at least 2 windows)")

    if primary_win_rate >= 0.70 and binom_res.pvalue < 0.05:
        primary_conclusion = "BL17 PRIMARY SUPPORTED"
        primary_reason = (
            "RA-HTGN met or exceeded the 70% window win-rate threshold against TGN and the "
            "one-sided exact binomial test was significant at alpha=0.05."
        )
    elif primary_win_rate < 0.70:
        primary_conclusion = "BL17 PRIMARY NOT SUPPORTED"
        primary_reason = (
            "Observed RA-HTGN win-rate against TGN was below the 70% threshold."
        )
    else:
        primary_conclusion = "BL17 PRIMARY INCONCLUSIVE"
        primary_reason = (
            "Observed RA-HTGN win-rate reached the 70% threshold, but the one-sided exact "
            "binomial test did not reject the null at alpha=0.05."
        )

    logger.info("BL-17 primary conclusion: %s", primary_conclusion)
    logger.info("Reason: %s", primary_reason)

    confirmatory_holm = holm_bonferroni(confirmatory_p_values)

    pairwise_window_tests = {}
    for left, right in RA_COMPARISON_PAIRS:
        left_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[left]], dtype=float)
        right_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[right]], dtype=float)
        valid_mask = np.isfinite(left_sharpes) & np.isfinite(right_sharpes)
        res = run_window_wilcoxon(left_sharpes[valid_mask], right_sharpes[valid_mask], alternative="greater")
        if res is not None:
            pairwise_window_tests[f"{left}_vs_{right}"] = res
            exploratory_p_values[f"window_wilcoxon_{left}_vs_{right}"] = res["p_value"]

    pooled_dm_results = {}
    if step_days >= test_days:
        for variant in ["tgn", "roland", "gat_static"]:
            ra_daily = [entry for entry in daily_losses["ra_htgn"] if entry is not None]
            var_daily = [entry for entry in daily_losses[variant] if entry is not None]
            paired_daily = list(zip(ra_daily, var_daily))
            if not paired_daily:
                continue

            ra_mae = np.concatenate([x["daily_mae"] for x, _ in paired_daily])
            var_mae = np.concatenate([y["daily_mae"] for _, y in paired_daily])
            ra_mse = np.concatenate([x["daily_mse"] for x, _ in paired_daily])
            var_mse = np.concatenate([y["daily_mse"] for _, y in paired_daily])

            dm_mae = diebold_mariano_test(ra_mae, var_mae, loss="mae", alternative="less")
            dm_mse = diebold_mariano_test(ra_mse, var_mse, loss="mae", alternative="less")
            pooled_dm_results[f"ra_htgn_vs_{variant}_mae"] = dm_mae
            pooled_dm_results[f"ra_htgn_vs_{variant}_mse"] = dm_mse
            exploratory_p_values[f"dm_mae_ra_htgn_vs_{variant}"] = dm_mae["p_value"]
            exploratory_p_values[f"dm_mse_ra_htgn_vs_{variant}"] = dm_mse["p_value"]
    else:
        logger.warning("step_days < test_days, so test windows overlap. Skipping pooled daily DM tests.")

    exploratory_holm = holm_bonferroni(exploratory_p_values)

    summary = {
        "version": "ra_htgn_v1_bl17",
        "spec_alignment": {
            "source_protocol": "run_bootstrap_eval_v5.py",
            "primary_hypothesis": "RA-HTGN Sharpe >= TGN in >= 70% of walk-forward windows",
            "primary_test": "exact binomial test on RA-HTGN-vs-TGN window win-rate against p0=0.70",
            "secondary_confirmatory_test": "window-level Wilcoxon on RA-HTGN vs TGN Sharpe across windows",
            "minimum_bl17_comparisons": [
                "ra_htgn vs tgn",
                "ra_htgn vs roland",
                "ra_htgn vs gat_static",
            ],
            "predictive_tests_note": "Daily DM is only pooled when test windows do not overlap.",
        },
        "window_config": {
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "step_days": step_days,
            "n_windows": len(windows),
        },
        "variants": RA_VARIANTS,
        "comparison_pairs": RA_COMPARISON_PAIRS,
        "metrics_by_variant": scalar_results,
        "window_reports": window_reports,
        "primary_ra_htgn_vs_tgn": {
            "wins": primary_wins,
            "n_windows": primary_n,
            "win_rate": primary_win_rate,
            "threshold": 0.70,
            "conclusion": primary_conclusion,
            "conclusion_reason": primary_reason,
            "small_sample_warning": small_sample_warning,
            "binomial_test": {
                "p_value": float(binom_res.pvalue),
                "ci_95_low": float(binom_ci.low),
                "ci_95_high": float(binom_ci.high),
            },
            "window_wilcoxon_sharpe": primary_wilcoxon,
            "holm_bonferroni_confirmatory": confirmatory_holm,
        },
        "pairwise_window_tests": pairwise_window_tests,
        "pooled_predictive_tests": pooled_dm_results,
        "holm_bonferroni_exploratory": exploratory_holm,
        "descriptive_summary": {
            "mean_window_sharpe": {
                "ra_htgn": float(np.nanmean(ra_window_sharpes)),
                "tgn": float(np.nanmean(tgn_window_sharpes)),
                "roland": float(np.nanmean(roland_window_sharpes)),
                "gat_static": float(np.nanmean(gat_window_sharpes)),
            },
            "std_window_sharpe": {
                "ra_htgn": float(np.nanstd(ra_window_sharpes, ddof=1)) if len(ra_window_sharpes) > 1 else 0.0,
                "tgn": float(np.nanstd(tgn_window_sharpes, ddof=1)) if len(tgn_window_sharpes) > 1 else 0.0,
                "roland": float(np.nanstd(roland_window_sharpes, ddof=1)) if len(roland_window_sharpes) > 1 else 0.0,
                "gat_static": float(np.nanstd(gat_window_sharpes, ddof=1)) if len(gat_window_sharpes) > 1 else 0.0,
            },
            "mean_window_metrics": {
                variant: {
                    metric: float(np.mean([entry.get(metric, np.nan) for entry in scalar_results[variant]]))
                    for metric in ["r_squared", "spearman", "mae", "loss", "cls_f1", "sharpe_proxy"]
                }
                for variant in RA_VARIANTS
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
        },
    }

    out_dir = RESULTS_DIR / f"bootstrap_eval_ra_htgn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_summary_ra_htgn.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 60)
    logger.info("Saved summary to %s", out_path)
    logger.info(
        "BL-17 primary result: %s | win_rate=%.1f%% | binomial p=%.4e",
        primary_conclusion,
        primary_win_rate * 100,
        binom_res.pvalue,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="DyFO bootstrap eval for BL-17 RA-HTGN")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_days", type=int, default=500)
    parser.add_argument("--val_days", type=int, default=125)
    parser.add_argument("--test_days", type=int, default=125)
    parser.add_argument("--step_days", type=int, default=125)
    parser.add_argument("--n_bootstrap", type=int, default=500)
    parser.add_argument("--block_size", type=int, default=5)
    parser.add_argument("--max_windows", type=int, default=None)
    args = parser.parse_args()

    run_bootstrap_eval_ra_htgn(
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
