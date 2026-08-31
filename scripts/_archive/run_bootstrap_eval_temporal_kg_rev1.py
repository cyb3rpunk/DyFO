"""Bootstrap eval for BL-18 Temporal KG — Revision 1.

Differences from run_bootstrap_eval_temporal_kg.py (rev 0):

  1. NON-OVERLAPPING WINDOWS: step_days defaults to test_days (125), the same
     protocol used in run_bootstrap_eval_v5.py.  When step_days < test_days,
     the TGN memory carries leakage from one window's test period into the
     next window's training set, artificially degrading TGN (which has a live
     memory buffer) while barely affecting stateless baselines (GAT, ROLAND).

  2. TGN GETS ITS ORIGINAL HYPERPARAM BUDGET: LR=TGN_LR=1e-3 and
     TGN_PATIENCE=5 — identical to run_bootstrap_eval_v5.py.  The v0 runner
     accidentally routed the TGN through BASELINE_LR inside
     train_variant_for_window_temporal_kg.  This is fixed here by delegating
     to the corrected local function.

  3. EXPLICIT MEMORY RESET GUARD: after every window, each TGN/RA-HTGN model
     is discarded and re-instantiated — this is the existing behaviour because
     train_link_prediction rebuilds the model from scratch per call.  Rev 1
     adds a comment and a log line so this contract is visible.

  4. ALL OTHER STATISTICAL MACHINERY is identical to rev 0 / BL-17 runner:
     paired block bootstrap, Diebold-Mariano, Wilcoxon, Holm-Bonferroni.

Protocol reference: the spec's H4 says
  "TGN Sharpe >= ROLAND in >=70% of walk-forward windows".
  In BL-18 the primary test is now
  "Temporal KG Sharpe >= RA-HTGN in >=50% of walk-forward windows".
  TGN is restored as a *proper* midpoint baseline between the interpretable
  arm (Temporal KG) and the raw baselines (ROLAND, GAT Static).
"""

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


# ---------------------------------------------------------------------------
# Defaults — non-overlapping by design (step == test)
# ---------------------------------------------------------------------------

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_DAYS = 500
DEFAULT_VAL_DAYS = 125
DEFAULT_TEST_DAYS = 125
DEFAULT_STEP_DAYS = 125        # ← REV 1 KEY CHANGE: step == test (no overlap)
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BLOCK_SIZE = 5
DEFAULT_MAX_WINDOWS = None     # use all available windows

TKG_VARIANTS = ["temporal_kg", "ra_htgn", "tgn", "roland", "gat_static"]
TKG_COMPARISON_PAIRS = list(combinations(TKG_VARIANTS, 2))
N_PAIRS = 435


# ---------------------------------------------------------------------------
# Per-window training dispatcher — REV 1 fix
# ---------------------------------------------------------------------------

def train_variant_for_window_rev1(
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
    """Train a single variant for one walk-forward window.

    REV 1 correction: every variant — including ``"tgn"`` — is dispatched
    through its correct hyperparameter budget.  In rev 0, the TGN was routed
    through ``train_variant_for_window_ra_htgn`` which applied TGN_LR only
    for ``tgn`` and ``ra_htgn``, but the temporal_kg script's outer function
    then inadvertently used BASELINE_LR for tgn in certain code paths.
    Here the mapping is explicit and unambiguous.
    """
    common_kwargs = dict(
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
    )

    # TGN and RA-HTGN: temporal memory models — get the TGN hyperparam budget
    if variant in {"tgn", "ra_htgn", "temporal_kg"}:
        return train_link_prediction(
            lr=TGN_LR,
            use_cosine_schedule=TGN_USE_COSINE,
            early_stopping_patience=TGN_PATIENCE,
            **common_kwargs,
        )

    # Stateless baselines: ROLAND, GAT Static
    return train_link_prediction(
        lr=BASELINE_LR,
        use_cosine_schedule=BASELINE_USE_COSINE,
        early_stopping_patience=BASELINE_PATIENCE,
        **common_kwargs,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_bootstrap_eval_temporal_kg_rev1(
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
    logger = setup_logging("dyfo.bootstrap_eval_temporal_kg_rev1", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Bootstrap Eval Temporal KG Rev1 — BL-18 walk-forward (non-overlapping)")
    logger.info("=" * 60)
    logger.info(
        "step_days=%d | test_days=%d | overlap=%s",
        step_days, test_days,
        "YES (contamination risk)" if step_days < test_days else "NO",
    )
    if step_days < test_days:
        logger.warning(
            "step_days (%d) < test_days (%d): test windows OVERLAP — "
            "TGN memory leakage is possible.  Set step_days >= test_days "
            "for a fair comparison.",
            step_days, test_days,
        )

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
        raise RuntimeError(
            "No walk-forward windows could be constructed with the requested settings."
        )
    logger.info(
        "Walk-forward windows: %d | train=%d val=%d test=%d step=%d | date span=%s..%s",
        len(windows), train_days, val_days, test_days, step_days, start, end,
    )

    scalar_results: Dict[str, List[dict]] = {v: [] for v in TKG_VARIANTS}
    realized_returns: Dict[str, List[np.ndarray]] = {v: [] for v in TKG_VARIANTS}
    daily_losses: Dict[str, List] = {v: [] for v in TKG_VARIANTS}
    temporal_kg_artifacts: List[dict] = []
    window_reports: List[dict] = []
    confirmatory_p_values: Dict[str, Optional[float]] = {}
    exploratory_p_values: Dict[str, Optional[float]] = {}

    for window_idx, (train_dates, val_dates, test_dates) in enumerate(windows, start=1):
        logger.info("-" * 60)
        logger.info(
            "Window %d/%d | train=%d val=%d test=%d",
            window_idx, len(windows), len(train_dates), len(val_dates), len(test_dates),
        )
        logger.info(
            "  NOTE: each variant is re-instantiated from scratch — "
            "no memory cross-contamination between windows."
        )

        for variant in TKG_VARIANTS:
            logger.info("Training %s on window %d", variant.upper(), window_idx)
            metrics = train_variant_for_window_rev1(
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
            scalar_results[variant].append(
                {k: float(v) for k, v in metrics.items() if not k.startswith("_")}
            )
            realized_returns[variant].append(
                np.array(metrics.get("_realized_returns", []), dtype=float)
            )

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

        # ------------------------------------------------------------------
        # Per-window comparisons (block bootstrap)
        # ------------------------------------------------------------------
        report: dict = {
            "window_index": window_idx,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "metrics": {v: scalar_results[v][-1] for v in TKG_VARIANTS},
            "comparisons": {},
            "temporal_kg_explanations": (
                temporal_kg_artifacts[-1].get("top_explanations", [])
                if temporal_kg_artifacts else []
            ),
            "temporal_kg_relation_scores": (
                temporal_kg_artifacts[-1].get("last_relation_scores", {})
                if temporal_kg_artifacts else {}
            ),
        }

        for left, right in TKG_COMPARISON_PAIRS:
            rl = realized_returns[left][-1]
            rr = realized_returns[right][-1]
            if len(rl) == 0 or len(rr) == 0 or len(rl) != len(rr):
                continue

            paired = paired_block_bootstrap_multi(
                rl, rr,
                block_size=block_size,
                n_iterations=n_bootstrap,
                seed=42 + window_idx,
            )
            diff_b = paired["sharpe_diffs"]
            cvar_diff_b = paired["cvar_diffs"]
            d_obs = _sharpe(rl) - _sharpe(rr)
            cvar_obs = _cvar(rl) - _cvar(rr)

            report["comparisons"][f"{left}_vs_{right}"] = {
                "sharpe_obs_diff": float(d_obs),
                "sharpe_bootstrap_ci_2.5": float(np.percentile(diff_b, 2.5)),
                "sharpe_bootstrap_ci_97.5": float(np.percentile(diff_b, 97.5)),
                "sharpe_p_direct": float(np.mean(diff_b <= 0.0)),
                "sharpe_effect_size_d": (
                    float(np.mean(diff_b) / np.std(diff_b, ddof=1))
                    if len(diff_b) > 1 and np.std(diff_b, ddof=1) > 1e-10
                    else 0.0
                ),
                "cvar_obs_diff": float(cvar_obs),
                "cvar_bootstrap_ci_2.5": float(np.percentile(cvar_diff_b, 2.5)),
                "cvar_bootstrap_ci_97.5": float(np.percentile(cvar_diff_b, 97.5)),
                "cvar_p_direct": float(np.mean(cvar_diff_b >= 0.0)),
            }

        window_reports.append(report)

    # ------------------------------------------------------------------
    # Global confirmatory test: Temporal KG vs RA-HTGN (H4 primary arm)
    # ------------------------------------------------------------------
    tkg_sharpes = np.array(
        [m.get("sharpe_proxy", np.nan) for m in scalar_results["temporal_kg"]], dtype=float
    )
    ra_sharpes = np.array(
        [m.get("sharpe_proxy", np.nan) for m in scalar_results["ra_htgn"]], dtype=float
    )
    tgn_sharpes = np.array(
        [m.get("sharpe_proxy", np.nan) for m in scalar_results["tgn"]], dtype=float
    )

    valid_primary = np.isfinite(tkg_sharpes) & np.isfinite(ra_sharpes)
    if not valid_primary.any():
        raise RuntimeError("No valid windows for the primary Temporal KG vs RA-HTGN comparison.")

    primary_wins = int(np.sum(tkg_sharpes[valid_primary] >= ra_sharpes[valid_primary]))
    primary_n = int(np.sum(valid_primary))
    primary_win_rate = float(primary_wins / primary_n)

    logger.info("=" * 60)
    logger.info("PRIMARY BL-18 EVALUATION (rev1 — non-overlapping windows)")
    logger.info(
        "Temporal KG >= RA-HTGN in %d/%d windows (%.1f%%)",
        primary_wins, primary_n, primary_win_rate * 100,
    )

    binom_res = binomtest(primary_wins, primary_n, p=0.50, alternative="greater")
    binom_ci = binom_res.proportion_ci(confidence_level=0.95)
    confirmatory_p_values["temporal_kg_win_rate_gt_0.50_vs_ra_htgn"] = float(binom_res.pvalue)

    primary_wilcoxon = run_window_wilcoxon(
        tkg_sharpes[valid_primary],
        ra_sharpes[valid_primary],
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

    # ------------------------------------------------------------------
    # TGN vs RA-HTGN: historical H4 reference (TGN should be > baselines)
    # ------------------------------------------------------------------
    valid_tgn_ra = np.isfinite(tgn_sharpes) & np.isfinite(ra_sharpes)
    tgn_vs_ra_wins = int(np.sum(tgn_sharpes[valid_tgn_ra] >= ra_sharpes[valid_tgn_ra])) if valid_tgn_ra.any() else 0
    tgn_vs_ra_n = int(np.sum(valid_tgn_ra))
    logger.info(
        "TGN >= RA-HTGN in %d/%d windows (%.1f%%) — historical H4 reference",
        tgn_vs_ra_wins, tgn_vs_ra_n,
        100 * tgn_vs_ra_wins / tgn_vs_ra_n if tgn_vs_ra_n else float("nan"),
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

    logger.info("BL-18 (rev1) primary conclusion: %s", primary_conclusion)

    # ------------------------------------------------------------------
    # Pairwise window-level Wilcoxon (all vs all)
    # ------------------------------------------------------------------
    pairwise_window_tests: dict = {}
    for left, right in TKG_COMPARISON_PAIRS:
        left_sh = np.array(
            [m.get("sharpe_proxy", np.nan) for m in scalar_results[left]], dtype=float
        )
        right_sh = np.array(
            [m.get("sharpe_proxy", np.nan) for m in scalar_results[right]], dtype=float
        )
        valid = np.isfinite(left_sh) & np.isfinite(right_sh)
        res = run_window_wilcoxon(left_sh[valid], right_sh[valid], alternative="greater")
        if res is not None:
            pairwise_window_tests[f"{left}_vs_{right}"] = res
            exploratory_p_values[f"window_wilcoxon_{left}_vs_{right}"] = res["p_value"]

    # ------------------------------------------------------------------
    # Pooled Diebold-Mariano (only valid when windows don't overlap)
    # ------------------------------------------------------------------
    pooled_dm_results: dict = {}
    if step_days >= test_days:
        for left, right in TKG_COMPARISON_PAIRS:
            left_daily = [e for e in daily_losses[left] if e is not None]
            right_daily = [e for e in daily_losses[right] if e is not None]
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
        logger.warning(
            "step_days (%d) < test_days (%d): test windows overlap — "
            "skipping pooled DM tests to avoid pseudoreplication.",
            step_days, test_days,
        )

    # ------------------------------------------------------------------
    # Assemble summary
    # ------------------------------------------------------------------
    summary = {
        "version": "temporal_kg_rev1_bl18",
        "revision_notes": (
            "Rev 1: non-overlapping walk-forward windows (step_days=test_days=125); "
            "TGN/RA-HTGN/Temporal-KG all use TGN_LR=1e-3 and TGN_PATIENCE=5; "
            "memory is fully reset between windows by model re-instantiation."
        ),
        "spec_alignment": {
            "source_protocol": "run_bootstrap_eval_v5.py",
            "primary_hypothesis": (
                "Temporal KG Sharpe >= RA-HTGN in >= 50% of non-overlapping walk-forward windows"
            ),
            "secondary_reference": (
                "TGN >= ROLAND in >= 70% of windows (H4 from v5 — restored as midpoint baseline)"
            ),
            "all_vs_all_pairwise_tests": [
                f"{left} vs {right}" for left, right in TKG_COMPARISON_PAIRS
            ],
        },
        "window_config": {
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "step_days": step_days,
            "n_windows": len(windows),
            "windows_overlap": step_days < test_days,
        },
        "variants": TKG_VARIANTS,
        "comparison_pairs": TKG_COMPARISON_PAIRS,
        "metrics_by_variant": scalar_results,
        "window_reports": window_reports,
        "temporal_kg_interpretability": temporal_kg_artifacts,
        "tgn_recovery": {
            "tgn_vs_ra_htgn_wins": tgn_vs_ra_wins,
            "tgn_vs_ra_htgn_n_windows": tgn_vs_ra_n,
            "tgn_vs_ra_htgn_win_rate": (
                float(tgn_vs_ra_wins / tgn_vs_ra_n) if tgn_vs_ra_n else float("nan")
            ),
            "note": (
                "TGN should be >= RA-HTGN in ~70%+ windows if the TGN was truly "
                "degraded by overlapping windows in rev 0.  Lower rates suggest "
                "RA-HTGN is genuinely superior."
            ),
        },
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
                v: float(
                    np.nanmean(
                        np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[v]], dtype=float)
                    )
                )
                for v in TKG_VARIANTS
            },
            "std_window_sharpe": {
                v: (
                    float(
                        np.nanstd(
                            np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[v]], dtype=float),
                            ddof=1,
                        )
                    )
                    if len(scalar_results[v]) > 1
                    else 0.0
                )
                for v in TKG_VARIANTS
            },
            "mean_window_metrics": {
                v: {
                    metric: float(
                        np.nanmean([entry.get(metric, np.nan) for entry in scalar_results[v]])
                    )
                    for metric in ["r_squared", "spearman", "mae", "loss", "cls_f1", "sharpe_proxy"]
                }
                for v in TKG_VARIANTS
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

    out_dir = RESULTS_DIR / f"bootstrap_eval_tkg_rev1_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_summary_tkg_rev1.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved summary to %s", out_path)
    logger.info(
        "BL-18 (rev1) result: %s | win_rate=%.1f%% | binomial p=%.4e",
        primary_conclusion,
        primary_win_rate * 100,
        binom_res.pvalue,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DyFO bootstrap eval BL-18 Temporal KG — Rev 1 (non-overlapping windows)"
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train_days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--val_days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--test_days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument(
        "--step_days",
        type=int,
        default=DEFAULT_STEP_DAYS,
        help=(
            "Step between windows. Defaults to test_days (%(default)s) for non-overlapping "
            "windows. Set < test_days only for ablation — will trigger a contamination warning."
        ),
    )
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--max_windows",
        type=int,
        default=DEFAULT_MAX_WINDOWS,
        help="Cap on number of windows. Default=None (use all).",
    )
    args = parser.parse_args()

    run_bootstrap_eval_temporal_kg_rev1(
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
