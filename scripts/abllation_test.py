"""Paper-facing TGAT ablation runner for DyFO.

This script narrows the broader Rev 2 evaluation runner to the article scope:

- Primary ablation target: ``tgat``
- Article baselines only: ``tgat``, ``tgn``, ``roland``, ``gat_static``
- Edge ablation combinations (always keeping CORR): ``CORR_only``, ``CORR+SECT``, ``CORR+FACT``
- ``SUPL`` is explicitly excluded from execution and reported as future work
  because a stable free API/data source is not currently available

It delegates the heavy lifting to
``scripts/run_bootstrap_eval_temporal_kg_rev3.py`` and then writes a compact,
paper-friendly summary alongside the raw results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_temporal_kg_rev3 import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_END,
    DEFAULT_EPOCHS,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEEDS,
    DEFAULT_START,
    DEFAULT_STEP_DAYS,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    DEFAULT_VAL_DAYS,
    MULTISEED_SEEDS,
    run_bootstrap_eval_temporal_kg_rev3,
)


ARTICLE_VARIANTS = ["tgat", "tgn", "roland", "gat_static"]
ARTICLE_ABLATION_VARIANT = "tgat"
ARTICLE_DEFAULT_N_TICKERS = 50


def _validate_variants(variants: List[str]) -> None:
    invalid = [variant for variant in variants if variant not in ARTICLE_VARIANTS]
    if invalid:
        allowed = ", ".join(ARTICLE_VARIANTS)
        invalid_str = ", ".join(invalid)
        raise ValueError(
            f"Article scope only allows variants {{{allowed}}}; got unsupported: {invalid_str}."
        )


def _window_metric_series(
    metrics_by_variant: Dict[str, List[Dict[str, float]]],
    metric_name: str,
) -> Dict[str, List[float]]:
    return {
        variant: [float(window.get(metric_name, 0.0)) for window in windows]
        for variant, windows in metrics_by_variant.items()
    }


def _paper_summary(raw_summary: Dict[str, Any]) -> Dict[str, Any]:
    metrics_by_variant = raw_summary.get("metrics_by_variant", {})
    sharpe_by_variant = _window_metric_series(metrics_by_variant, "sharpe_proxy")
    mdd_by_variant = _window_metric_series(metrics_by_variant, "mdd_proxy")
    turnover_by_variant = _window_metric_series(metrics_by_variant, "turnover_proxy")
    cumret_by_variant = _window_metric_series(metrics_by_variant, "cumret_proxy")
    vol_by_variant = _window_metric_series(metrics_by_variant, "vol_proxy")
    r2_by_variant = _window_metric_series(metrics_by_variant, "r_squared")
    spearman_by_variant = _window_metric_series(metrics_by_variant, "spearman")
    mae_by_variant = _window_metric_series(metrics_by_variant, "mae")
    mse_by_variant = _window_metric_series(metrics_by_variant, "mse")
    loss_by_variant = _window_metric_series(metrics_by_variant, "loss")

    ablation = raw_summary.get("ablation", {})
    ablation_results = ablation.get("ablation_results", {})

    return {
        "paper_scope": {
            "article_variants": ARTICLE_VARIANTS,
            "primary_model": "tgat",
            "ablation_edge_types": ["CORR_only", "CORR+SECT", "CORR+FACT"],
            "excluded_edge_types": ["SUPL"],
            "future_work_note": (
                "SUPL remains documented in the DyFO spec but is excluded from this "
                "article-level ablation because we do not yet have a reliable free API "
                "for supply-chain relations."
            ),
        },
        "run_config": raw_summary.get("run_config", {}),
        "descriptive_summary": raw_summary.get("descriptive_summary", {}),
        "primary_comparison": raw_summary.get("primary_comparison", {}),
        "pairwise_window_tests": raw_summary.get("pairwise_window_tests", {}),
        "pooled_predictive_tests": raw_summary.get("pooled_predictive_tests", {}),
        "holm_bonferroni_exploratory": raw_summary.get("holm_bonferroni_exploratory", {}),
        "window_metric_series": {
            "sharpe_proxy": sharpe_by_variant,
            "mdd_proxy": mdd_by_variant,
            "turnover_proxy": turnover_by_variant,
            "cumret_proxy": cumret_by_variant,
            "vol_proxy": vol_by_variant,
            "r_squared": r2_by_variant,
            "spearman": spearman_by_variant,
            "mae": mae_by_variant,
            "mse": mse_by_variant,
            "loss": loss_by_variant,
        },
        "ablation": {
            "ablation_variant": ablation.get("ablation_variant", ARTICLE_ABLATION_VARIANT),
            "ranking_by_sharpe": ablation.get("ablation_ranking_by_sharpe", []),
            "results": {
                label: {
                    "active_edges": result.get("active_edges", []),
                    "mean_sharpe": result.get("mean_sharpe"),
                    "std_sharpe": result.get("std_sharpe"),
                    "mean_mdd": result.get("mean_mdd"),
                    "mean_turnover": result.get("mean_turnover"),
                    "mean_cumret": result.get("mean_cumret"),
                    "mean_vol": result.get("mean_vol"),
                    "mean_r_squared": result.get("mean_r_squared"),
                    "mean_spearman": result.get("mean_spearman"),
                    "mean_mae": result.get("mean_mae"),
                    "mean_mse": result.get("mean_mse"),
                    "mean_loss": result.get("mean_loss"),
                }
                for label, result in ablation_results.items()
            },
        },
    }


def _markdown_report(paper_summary: Dict[str, Any], raw_result_path: Path | None) -> str:
    run_config = paper_summary.get("run_config", {})
    descriptive = paper_summary.get("descriptive_summary", {})
    mean_sharpe = descriptive.get("mean_window_sharpe", {})
    ablation = paper_summary.get("ablation", {})
    ranking = ablation.get("ranking_by_sharpe", [])
    primary = paper_summary.get("primary_comparison", {})

    lines = [
        "# DyFO TGAT Ablation Report",
        "",
        "## Scope",
        f"- Variants in article scope: {', '.join(ARTICLE_VARIANTS)}",
        "- TGAT is the only model ablated in this script.",
        "- Ablation edge types: CORR_only, CORR+SECT, CORR+FACT.",
        "- SUPL is excluded and left as future work due to the lack of a reliable free API.",
        "",
        "## Run Configuration",
        f"- n_tickers: {run_config.get('n_tickers')}",
        f"- walk-forward windows: {run_config.get('n_windows')}",
        f"- epochs: {run_config.get('epochs')}",
        f"- bootstrap iterations: {run_config.get('n_bootstrap')}",
        f"- block size: {run_config.get('block_size')}",
        "",
        "## Descriptive Metric Summary (Mean across windows)",
    ]

    for variant in ARTICLE_VARIANTS:
        if variant in mean_sharpe:
            r2 = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("r_squared", 0.0)
            mae = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("mae", 0.0)
            mse = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("mse", 0.0)
            spearman = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("spearman", 0.0)
            loss = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("loss", 0.0)
            mdd = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("mdd_proxy", 0.0)
            turnover = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("turnover_proxy", 0.0)
            cumret = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("cumret_proxy", 0.0)
            vol = descriptive.get("mean_window_metrics", {}).get(variant, {}).get("vol_proxy", 0.0)
            
            lines.append(f"### {variant}")
            lines.append(f"- R²: {r2:.4f}")
            lines.append(f"- MAE: {mae:.4f}")
            lines.append(f"- MSE: {mse:.6f}")
            lines.append(f"- Spearman: {spearman:.4f}")
            lines.append(f"- Loss: {loss:.6f}")
            lines.append(f"- Sharpe: {mean_sharpe[variant]:.4f}")
            lines.append(f"- MDD: {mdd:.4f}")
            lines.append(f"- Turnover: {turnover:.4f}")
            lines.append(f"- CumRet: {cumret:.4f}")
            lines.append(f"- Volatility: {vol:.4f}")

    lines.extend(["", "## TGAT Ablation Ranking"])
    for position, item in enumerate(ranking, start=1):
        label, value = item
        lines.append(f"- {position}. {label}: {value:.4f}")

    if primary:
        lines.extend(
            [
                "",
                "## Primary Comparison",
                f"- left: {primary.get('left')}",
                f"- right: {primary.get('right')}",
                f"- win_rate: {primary.get('win_rate')}",
                f"- n_windows: {primary.get('n_windows')}",
            ]
        )

    if raw_result_path is not None:
        lines.extend(["", "## Raw Output", f"- Raw Rev 3 summary: `{raw_result_path}`"])

    return "\n".join(lines) + "\n"


def _find_latest_rev3_summary(after_start: pd.Timestamp) -> Path | None:
    candidates = []
    for path in RESULTS_DIR.glob("bootstrap_eval_tkg_rev3_*"):
        summary_path = path / "bootstrap_summary_tkg_rev3.json"
        if not summary_path.exists():
            continue
        modified = pd.Timestamp(summary_path.stat().st_mtime, unit="s")
        if modified >= after_start:
            candidates.append((modified, summary_path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_abllation_test(
    variants: List[str],
    n_tickers: int,
    ablation: str,
    start: str,
    end: str,
    epochs: int,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    n_bootstrap: int,
    block_size: int,
    max_windows: int | None,
    seeds: List[int] | None = None,
) -> Dict[str, Any]:
    _validate_variants(variants)

    logger = setup_logging("dyfo.paper_abllation_test", log_to_file=True, run_tag=f"paper_abllation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    logger.info("=" * 60)
    logger.info("DyFO paper ablation runner")
    logger.info("Variants restricted to article scope: %s", variants)
    logger.info("TGAT ablation over CORR_only/CORR+SECT/CORR+FACT only")
    logger.info("SUPL is intentionally excluded and reported as future work")
    logger.info("=" * 60)

    started_at = pd.Timestamp.now()
    
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"paper_abllation_tgat_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for execution log in the results directory
    _exec_log_path = out_dir / "execution.log"
    _exec_fh = logging.FileHandler(_exec_log_path, encoding="utf-8")
    _exec_fh.setLevel(logging.INFO)
    _exec_fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_exec_fh)

    summary_path = out_dir / "paper_abllation_summary.json"
    markdown_path = out_dir / "paper_abllation_report.md"

    def _write_logs(summary_data, raw_res_path=None):
        ps = _paper_summary(summary_data)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(ps, fh, indent=2)
        with open(markdown_path, "w", encoding="utf-8") as fh:
            fh.write(_markdown_report(ps, raw_res_path))

    def on_progress(partial_raw_summary: dict):
        _write_logs(partial_raw_summary)

    raw_summary = run_bootstrap_eval_temporal_kg_rev3(
        variants=variants,
        n_tickers=n_tickers,
        ablation=ablation,
        ablation_variant=ARTICLE_ABLATION_VARIANT,
        start=start,
        end=end,
        epochs=epochs,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
        step_days=step_days,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        max_windows=max_windows,
        on_progress=on_progress,
        seeds=seeds,
    )

    raw_result_path = _find_latest_rev3_summary(started_at)
    paper_summary = _paper_summary(raw_summary)
    _write_logs(raw_summary, raw_result_path)

    logger.info("Saved paper summary -> %s", summary_path)
    logger.info("Saved markdown report -> %s", markdown_path)

    return {
        "paper_summary": paper_summary,
        "paper_summary_path": str(summary_path),
        "paper_report_path": str(markdown_path),
        "execution_log_path": str(_exec_log_path),
        "raw_rev2_summary_path": str(raw_result_path) if raw_result_path is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DyFO article-scoped TGAT ablation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=ARTICLE_VARIANTS,
        help=(
            "Subset of article variants to compare alongside the TGAT ablation. "
            f"Allowed: {ARTICLE_VARIANTS}"
        ),
    )
    parser.add_argument(
        "--n_tickers",
        type=int,
        choices=[30, 50, 100],
        default=ARTICLE_DEFAULT_N_TICKERS,
        help="Universe size. The article default is 50.",
    )
    parser.add_argument(
        "--ablation",
        choices=["basic", "full", "corr_fact"],
        default="basic",
        help="Ablation breadth. 'basic' is the paper default for CORR_only/CORR+SECT/CORR+FACT baseline runs, 'corr_fact' runs only CORR+FACT.",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train_days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--val_days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--test_days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step_days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help=(
            "RNG seeds for multi-seed ablation. Default: [42]. "
            "Use --seeds 42 123 456 789 2024 for 5-seed validation."
        ),
    )
    args = parser.parse_args()

    run_abllation_test(
        variants=args.variants,
        n_tickers=args.n_tickers,
        ablation=args.ablation,
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
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
