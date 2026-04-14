"""Bootstrap eval v5 - aligned with the spec's walk-forward H4 protocol.

Main changes vs v4:

  1. TRUE WALK-FORWARD: trains every variant on multiple rolling windows.
  2. H4 AT THE RIGHT LEVEL: primary confirmatory test is the TGN-vs-ROLAND
     Sharpe win-rate across windows, tested against the spec threshold of 70%.
  3. EXACT BINOMIAL TEST: one-sided test for H0: p_win <= 0.70 vs H1: p_win > 0.70.
  4. WINDOW-LEVEL BOOTSTRAP: paired block bootstrap is run inside each test window.
  5. NO PSEUDOREPLICATION: predictive tests aggregate to daily or window level.
  6. CLEAN MULTIPLE TESTING: Holm-Bonferroni is applied to the confirmatory family.

The output is meant to answer the spec directly:
  "O Sharpe condicional do TGN >= ROLAND em >=70% das janelas walk-forward."
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
from scipy.stats import binomtest, norm, rankdata, wilcoxon

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

N_PAIRS = 435

TGN_LR = 1e-3
TGN_USE_COSINE = False
TGN_PATIENCE = 5

BASELINE_LR = 1e-3
BASELINE_USE_COSINE = False
BASELINE_PATIENCE = 5


def _sharpe(arr: np.ndarray) -> float:
    std = np.std(arr, ddof=1)
    return (float(np.mean(arr)) / std) * np.sqrt(252) if std > 1e-8 else 0.0


def _cvar(arr: np.ndarray, alpha: float = 0.05) -> float:
    cutoff = np.percentile(arr, alpha * 100)
    tail = arr[arr <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else float(np.min(arr))


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


def _interpret_effect_r(r: float) -> str:
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    if ar < 0.3:
        return "small"
    if ar < 0.5:
        return "medium"
    return "large"


def _interpret_cohens_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


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
    alternative: str = "less",
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
        "max_lags_hac": int(np.floor(t ** (1.0 / 3.0))),
    }


def holm_bonferroni(p_values: Dict[str, Optional[float]]) -> dict:
    valid = {k: v for k, v in p_values.items() if v is not None}
    if not valid:
        return {}

    sorted_tests = sorted(valid.items(), key=lambda x: x[1])
    m = len(sorted_tests)
    corrected = {}
    max_so_far = 0.0
    for rank, (name, p_val) in enumerate(sorted_tests):
        adj_p = min(1.0, p_val * (m - rank))
        adj_p = max(adj_p, max_so_far)
        max_so_far = adj_p
        corrected[name] = {
            "original_p": p_val,
            "corrected_p": adj_p,
            "significant_at_0.05": adj_p < 0.05,
        }
    return corrected


def block_bootstrap_metrics(
    returns: np.ndarray,
    block_size: int = 5,
    n_iterations: int = 500,
    seed: int = 42,
    cvar_alpha: float = 0.05,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    sharpes = []
    cvars = []
    for _ in range(n_iterations):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = []
        for start_idx in start_indices:
            sampled.extend(returns[start_idx : start_idx + block_size])
        sampled = np.array(sampled[:n])
        sharpes.append(_sharpe(sampled))
        cvars.append(_cvar(sampled, cvar_alpha))

    return {"sharpes": np.array(sharpes), "cvars": np.array(cvars)}


def paired_block_bootstrap_multi(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    block_size: int = 5,
    n_iterations: int = 500,
    seed: int = 42,
    cvar_alpha: float = 0.05,
) -> dict:
    if len(returns_a) != len(returns_b):
        raise ValueError(
            f"Series have different lengths: {len(returns_a)} vs {len(returns_b)}."
        )

    rng = np.random.default_rng(seed)
    n = len(returns_a)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    sharpe_diffs = []
    cvar_diffs = []
    for _ in range(n_iterations):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled_a = []
        sampled_b = []
        for start_idx in start_indices:
            sampled_a.extend(returns_a[start_idx : start_idx + block_size])
            sampled_b.extend(returns_b[start_idx : start_idx + block_size])

        sa = np.array(sampled_a[:n])
        sb = np.array(sampled_b[:n])
        sharpe_diffs.append(_sharpe(sa) - _sharpe(sb))
        cvar_diffs.append(_cvar(sa, cvar_alpha) - _cvar(sb, cvar_alpha))

    return {
        "sharpe_diffs": np.array(sharpe_diffs),
        "cvar_diffs": np.array(cvar_diffs),
    }


def load_or_prepare_data(tickers, start, end, benchmark, config, data_config, logger):
    cache_key = hashlib.md5(
        f"{sorted(tickers)}{start}{end}{benchmark}".encode()
    ).hexdigest()[:10]
    cache_path = RESULTS_DIR / f"prepared_data_cache_{cache_key}.pkl"

    if cache_path.exists():
        logger.info("Loading cached prepared_data: %s", cache_path)
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
    else:
        logger.info("Prepared-data cache miss. Downloading data...")
        data = prepare_data(tickers, start, end, benchmark, config, data_config, logger)
        with open(cache_path, "wb") as fh:
            pickle.dump(data, fh)
        logger.info("Prepared data cached at %s", cache_path)

    return data


def build_windows(
    sorted_dates: List[int],
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    max_windows: Optional[int] = None,
) -> List[Tuple[List[int], List[int], List[int]]]:
    windows = []
    cursor = 0
    total_days = len(sorted_dates)
    while cursor + train_size + val_size + test_size <= total_days:
        train_dates = sorted_dates[cursor : cursor + train_size]
        val_dates = sorted_dates[cursor + train_size : cursor + train_size + val_size]
        test_dates = sorted_dates[
            cursor + train_size + val_size : cursor + train_size + val_size + test_size
        ]
        windows.append((train_dates, val_dates, test_dates))
        cursor += step_size

    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def train_variant_for_window(
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

    if variant == "tgn":
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


def extract_daily_errors(
    preds: np.ndarray,
    targets: np.ndarray,
    logger,
    label: str,
    n_pairs: int = N_PAIRS,
) -> Optional[dict]:
    n_total = len(preds)
    if n_total != len(targets):
        logger.warning("%s: preds/targets with different lengths (%d vs %d)", label, n_total, len(targets))
        return None

    if n_total % n_pairs != 0:
        logger.warning("%s: n_total=%d is not a multiple of n_pairs=%d", label, n_total, n_pairs)
        return None

    n_days = n_total // n_pairs
    if n_days == 0:
        logger.warning("%s: zero daily observations after reshape", label)
        return None

    err = np.abs(preds - targets).reshape(n_days, n_pairs)
    return {
        "daily_mae": err.mean(axis=1),
        "daily_mse": (err ** 2).mean(axis=1),
        "n_days": int(n_days),
    }


def run_window_wilcoxon(
    series_a: np.ndarray,
    series_b: np.ndarray,
    alternative: str = "greater",
) -> Optional[dict]:
    if len(series_a) != len(series_b) or len(series_a) < 2:
        return None

    try:
        stat, p_val = wilcoxon(series_a, series_b, alternative=alternative)
    except ValueError:
        return None

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "effect_size_r": float(rank_biserial_from_wilcoxon(series_a, series_b)),
        "n": int(len(series_a)),
    }


def run_bootstrap_eval_v5(
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
    logger = setup_logging("dyfo.bootstrap_eval_v5", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Bootstrap Eval v5 - Spec-aligned walk-forward validation")
    logger.info("=" * 60)

    config = DyFOConfig()
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

    variants = ["tgn", "roland", "gat_static"]
    scalar_results = {variant: [] for variant in variants}
    realized_returns = {variant: [] for variant in variants}
    daily_losses = {variant: [] for variant in variants}
    window_reports = []
    confirmatory_p_values = {}
    exploratory_p_values = {}

    for window_idx, (train_dates, val_dates, test_dates) in enumerate(windows, start=1):
        logger.info("-" * 60)
        logger.info(
            "Window %d/%d | train=%d val=%d test=%d",
            window_idx, len(windows), len(train_dates), len(val_dates), len(test_dates),
        )

        window_raw = {}
        for variant in variants:
            logger.info("Training %s on window %d", variant.upper(), window_idx)
            metrics = train_variant_for_window(
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
                )
                daily_losses[variant].append(day_level)
            else:
                daily_losses[variant].append(None)

        tgn_sharpe = float(window_raw["tgn"].get("sharpe_proxy", np.nan))
        roland_sharpe = float(window_raw["roland"].get("sharpe_proxy", np.nan))
        gat_sharpe = float(window_raw["gat_static"].get("sharpe_proxy", np.nan))

        report = {
            "window_index": window_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "metrics": {
                variant: scalar_results[variant][-1] for variant in variants
            },
            "comparisons": {},
        }

        for left, right in [("tgn", "roland"), ("tgn", "gat_static")]:
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
                "sharpe_effect_size_d": float(cohens_d(diff_b)),
                "cvar_obs_diff": float(cvar_obs),
                "cvar_bootstrap_ci_2.5": float(np.percentile(cvar_diff_b, 2.5)),
                "cvar_bootstrap_ci_97.5": float(np.percentile(cvar_diff_b, 97.5)),
                "cvar_p_direct": float(np.mean(cvar_diff_b >= 0.0)),
            }

        report["h4_window_win"] = bool(np.isfinite(tgn_sharpe) and np.isfinite(roland_sharpe) and tgn_sharpe >= roland_sharpe)
        report["h4_window_margin"] = float(tgn_sharpe - roland_sharpe)
        report["gat_margin"] = float(tgn_sharpe - gat_sharpe)
        window_reports.append(report)

    tgn_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["tgn"]], dtype=float)
    roland_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["roland"]], dtype=float)
    gat_window_sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results["gat_static"]], dtype=float)

    valid_h4_mask = np.isfinite(tgn_window_sharpes) & np.isfinite(roland_window_sharpes)
    h4_wins = int(np.sum(tgn_window_sharpes[valid_h4_mask] >= roland_window_sharpes[valid_h4_mask]))
    h4_n = int(np.sum(valid_h4_mask))
    h4_win_rate = float(h4_wins / h4_n) if h4_n > 0 else float("nan")

    logger.info("=" * 60)
    logger.info("PRIMARY H4 EVALUATION")
    logger.info("TGN >= ROLAND in %d/%d windows (%.1f%%)", h4_wins, h4_n, h4_win_rate * 100 if h4_n else float("nan"))

    if h4_n == 0:
        raise RuntimeError("No valid windows available for the primary H4 comparison.")

    binom_res = binomtest(h4_wins, h4_n, p=0.70, alternative="greater")
    binom_ci = binom_res.proportion_ci(confidence_level=0.95)
    confirmatory_p_values["h4_win_rate_gt_0.70"] = float(binom_res.pvalue)

    small_sample_warning = None
    if h4_n < 5:
        small_sample_warning = (
            f"Only {h4_n} walk-forward windows were available. "
            "Treat this as a smoke test or preliminary read, not strong confirmatory evidence."
        )
        logger.warning(small_sample_warning)

    logger.info(
        "Exact binomial test vs 70%% threshold: p=%.4e | 95%% CI=[%.3f, %.3f]",
        binom_res.pvalue, binom_ci.low, binom_ci.high,
    )

    h4_window_wilcoxon = run_window_wilcoxon(
        tgn_window_sharpes[valid_h4_mask],
        roland_window_sharpes[valid_h4_mask],
        alternative="greater",
    )
    if h4_window_wilcoxon is not None:
        confirmatory_p_values["h4_window_wilcoxon_sharpe"] = h4_window_wilcoxon["p_value"]
        logger.info(
            "Window-level Wilcoxon Sharpe: p=%.4e | r=%.3f (%s)",
            h4_window_wilcoxon["p_value"],
            h4_window_wilcoxon["effect_size_r"],
            _interpret_effect_r(h4_window_wilcoxon["effect_size_r"]),
        )
    else:
        logger.info("Window-level Wilcoxon Sharpe: skipped (requires at least 2 windows)")

    if h4_win_rate >= 0.70 and binom_res.pvalue < 0.05:
        h4_conclusion = "H4 SUPPORTED"
        h4_conclusion_reason = (
            "TGN met or exceeded the 70% window win-rate threshold and the one-sided exact binomial test "
            "was significant at alpha=0.05."
        )
    elif h4_win_rate < 0.70:
        h4_conclusion = "H4 NOT SUPPORTED"
        h4_conclusion_reason = (
            "Observed TGN win-rate was below the 70% threshold required by the spec."
        )
    else:
        h4_conclusion = "H4 INCONCLUSIVE"
        h4_conclusion_reason = (
            "Observed TGN win-rate reached the 70% threshold, but the one-sided exact binomial test "
            "did not reject the null at alpha=0.05."
        )

    logger.info("H4 conclusion: %s", h4_conclusion)
    logger.info("Reason: %s", h4_conclusion_reason)

    confirmatory_holm = holm_bonferroni(confirmatory_p_values)

    pooled_dm_results = {}
    if step_days >= test_days:
        for variant in ["roland", "gat_static"]:
            tgn_daily = [entry for entry in daily_losses["tgn"] if entry is not None]
            var_daily = [entry for entry in daily_losses[variant] if entry is not None]
            paired_daily = list(zip(tgn_daily, var_daily))
            if not paired_daily:
                continue

            tgn_mae = np.concatenate([x["daily_mae"] for x, _ in paired_daily])
            var_mae = np.concatenate([y["daily_mae"] for _, y in paired_daily])
            tgn_mse = np.concatenate([x["daily_mse"] for x, _ in paired_daily])
            var_mse = np.concatenate([y["daily_mse"] for _, y in paired_daily])

            dm_mae = diebold_mariano_test(tgn_mae, var_mae, loss="mae", alternative="less")
            dm_mse = diebold_mariano_test(tgn_mse, var_mse, loss="mae", alternative="less")
            pooled_dm_results[f"tgn_vs_{variant}_mae"] = dm_mae
            pooled_dm_results[f"tgn_vs_{variant}_mse"] = dm_mse
            exploratory_p_values[f"dm_mae_tgn_vs_{variant}"] = dm_mae["p_value"]
            exploratory_p_values[f"dm_mse_tgn_vs_{variant}"] = dm_mse["p_value"]
    else:
        logger.warning("step_days < test_days, so test windows overlap. Skipping pooled daily DM tests.")

    exploratory_holm = holm_bonferroni(exploratory_p_values)

    summary = {
        "version": "v5_spec_aligned",
        "spec_alignment": {
            "primary_hypothesis": "TGN Sharpe >= ROLAND in >= 70% of walk-forward windows",
            "primary_test": "exact binomial test on window win-rate against p0=0.70",
            "secondary_confirmatory_test": "window-level Wilcoxon on Sharpe across windows",
            "predictive_tests_note": "Daily DM is only pooled when test windows do not overlap.",
        },
        "window_config": {
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "step_days": step_days,
            "n_windows": len(windows),
        },
        "metrics_by_variant": scalar_results,
        "window_reports": window_reports,
        "primary_h4": {
            "wins": h4_wins,
            "n_windows": h4_n,
            "win_rate": h4_win_rate,
            "threshold": 0.70,
            "conclusion": h4_conclusion,
            "conclusion_reason": h4_conclusion_reason,
            "small_sample_warning": small_sample_warning,
            "binomial_test": {
                "p_value": float(binom_res.pvalue),
                "ci_95_low": float(binom_ci.low),
                "ci_95_high": float(binom_ci.high),
            },
            "window_wilcoxon_sharpe": h4_window_wilcoxon,
            "holm_bonferroni_confirmatory": confirmatory_holm,
        },
        "pooled_predictive_tests": pooled_dm_results,
        "holm_bonferroni_exploratory": exploratory_holm,
        "descriptive_summary": {
            "mean_window_sharpe": {
                "tgn": float(np.nanmean(tgn_window_sharpes)),
                "roland": float(np.nanmean(roland_window_sharpes)),
                "gat_static": float(np.nanmean(gat_window_sharpes)),
            },
            "std_window_sharpe": {
                "tgn": float(np.nanstd(tgn_window_sharpes, ddof=1)) if len(tgn_window_sharpes) > 1 else 0.0,
                "roland": float(np.nanstd(roland_window_sharpes, ddof=1)) if len(roland_window_sharpes) > 1 else 0.0,
                "gat_static": float(np.nanstd(gat_window_sharpes, ddof=1)) if len(gat_window_sharpes) > 1 else 0.0,
            },
            "mean_window_metrics": {
                variant: {
                    metric: float(np.mean([entry.get(metric, np.nan) for entry in scalar_results[variant]]))
                    for metric in ["r_squared", "spearman", "mae", "loss", "cls_f1", "sharpe_proxy"]
                }
                for variant in variants
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

    out_dir = RESULTS_DIR / f"bootstrap_eval_v5_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_summary_v5.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 60)
    logger.info("Saved summary to %s", out_path)
    logger.info(
        "H4 result: %s | win_rate=%.1f%% | binomial p=%.4e",
        h4_conclusion,
        h4_win_rate * 100,
        binom_res.pvalue,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="DyFO bootstrap eval v5 - spec-aligned walk-forward validation")
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

    run_bootstrap_eval_v5(
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
