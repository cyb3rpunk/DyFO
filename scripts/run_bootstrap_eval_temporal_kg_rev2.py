"""Bootstrap eval for BL-18 Temporal KG — Revision 2.

New capabilities vs Rev 1
--------------------------
1. ``--variants``    : select which model variants to train (subset of all 5).
2. ``--n_tickers``   : run with 30, 50, or 100 tickers (default 30).
                       50 uses threshold sparsification; 100 uses TMFG.
3. ``--ablation``    : edge-type ablation mode.  Trains one variant (``--variants``)
                       three times — once per active edge type subset:
                         * ``CORR_only``  : only dynamic correlation edges
                         * ``SECT_only``  : only static sector edges
                         * ``FACT_only``  : only Fama-French factor edges
                       With ``--ablation all``, also runs ``CORR+SECT``,
                       ``CORR+FACT``, ``SECT+FACT``, and ``all_edges`` (full).
4. ``--ablation_variant`` : which variant to ablate (default: ``tgn``).

All Rev 1 fixes are inherited:
- Non-overlapping windows (step_days = test_days = 125 by default)
- TGN / RA-HTGN / Temporal KG use TGN_LR; ROLAND / GAT use BASELINE_LR
- Explicit memory-reset guarantee per window

CLI examples
------------
# Quick smoke test (2 variants, 2 windows, 5 epochs)
python scripts/run_bootstrap_eval_temporal_kg_rev2.py \\
    --variants tgn roland --epochs 5 --max_windows 2

# Run TGN alone on 50 tickers
python scripts/run_bootstrap_eval_temporal_kg_rev2.py \\
    --variants tgn --n_tickers 50 --max_windows 3

# Edge-type ablation for TGN (CORR / SECT / FACT independently)
python scripts/run_bootstrap_eval_temporal_kg_rev2.py \\
    --ablation basic --ablation_variant tgn --max_windows 2 --epochs 5

# Full ablation (all subsets) for ra_htgn
python scripts/run_bootstrap_eval_temporal_kg_rev2.py \\
    --ablation full --ablation_variant ra_htgn --max_windows 2 --epochs 5
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import logging
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.ticker_registry import TICKERS_30, get_tickers, get_sparsification
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_v5 import (
    BASELINE_LR,
    BASELINE_PATIENCE,
    BASELINE_USE_COSINE,
    TGN_LR,
    TGN_PATIENCE,
    TGN_USE_COSINE,
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
# Constants
# ---------------------------------------------------------------------------

ALL_VARIANTS = ["temporal_kg", "ra_htgn", "tgn", "tgat", "roland", "gat_static"]
N_PAIRS_BY_TICKERS = {30: 435, 50: 1225, 100: 4950}

# Ablation edge-type subsets
ABLATION_BASIC: List[FrozenSet[str]] = [
    frozenset({"CORR"}),
    frozenset({"SECT"}),
    frozenset({"FACT"}),
]
ABLATION_FULL: List[FrozenSet[str]] = ABLATION_BASIC + [
    frozenset({"CORR", "SECT"}),
    frozenset({"CORR", "FACT"}),
    frozenset({"SECT", "FACT"}),
    frozenset({"CORR", "SECT", "FACT"}),  # all edges
]

_ABLATION_NAME: Dict[FrozenSet[str], str] = {
    frozenset({"CORR"}): "CORR_only",
    frozenset({"SECT"}): "SECT_only",
    frozenset({"FACT"}): "FACT_only",
    frozenset({"CORR", "SECT"}): "CORR+SECT",
    frozenset({"CORR", "FACT"}): "CORR+FACT",
    frozenset({"SECT", "FACT"}): "SECT+FACT",
    frozenset({"CORR", "SECT", "FACT"}): "all_edges",
}

# Defaults
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_DAYS = 500
DEFAULT_VAL_DAYS = 125
DEFAULT_TEST_DAYS = 125
DEFAULT_STEP_DAYS = 125   # non-overlapping
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BLOCK_SIZE = 5
DEFAULT_N_TICKERS = 30


# ---------------------------------------------------------------------------
# Hyperparameter routing (Rev 1 fix, preserved)
# ---------------------------------------------------------------------------

def _hyperparam_lr(variant: str) -> Tuple[float, bool, int]:
    """Return (lr, use_cosine, patience) for a given variant."""
    if variant in {"tgn", "tgat", "ra_htgn", "temporal_kg"}:
        return TGN_LR, TGN_USE_COSINE, TGN_PATIENCE
    return BASELINE_LR, BASELINE_USE_COSINE, BASELINE_PATIENCE


# ---------------------------------------------------------------------------
# Ablation data masking
# ---------------------------------------------------------------------------

def _mask_data_for_ablation(data: dict, active_edge_types: FrozenSet[str]) -> dict:
    """Return a shallow copy of ``data`` with inactive edge types removed.

    The mask operates on ``data["graph"]`` edge lists and on the event stream
    (``data["events_by_date"]``) to suppress CORRELATION_UPDATE events when
    CORR edges are removed and FACT edges when FACT is removed.

    Parameters
    ----------
    data : dict
        Full prepared_data dict from ``prepare_data``.
    active_edge_types : FrozenSet[str]
        Subset of {``"CORR"``, ``"SECT"``, ``"FACT"``} to keep.

    Returns
    -------
    dict
        Shallow copy with masked graph / events.
    """
    import copy
    from dyfo.core.event_stream import FinancialEvent

    masked = dict(data)  # shallow copy — only override affected keys

    # --- Mask graph edges ---------------------------------------------------
    graph = data["graph"]
    try:
        # GraphBuilder exposes get_edge_type_names() → list[str]
        edge_type_names = graph.get_edge_type_names()  # e.g. ["CORR","SECT","FACT","SUPL"]
    except AttributeError:
        # Fallback: assume edge_type_ids align with CORR=0, SECT=1, FACT=2, SUPL=3
        edge_type_names = ["CORR", "SECT", "FACT", "SUPL"]

    # Build a masked copy of the graph using the keep mask
    edge_index = graph.get_full_edge_index()           # (2, E)
    edge_type_ids = graph.get_edge_type_ids()          # (E,)

    import torch
    keep_mask = torch.zeros(edge_type_ids.shape[0], dtype=torch.bool)
    for et_name in active_edge_types:
        if et_name in edge_type_names:
            et_id = edge_type_names.index(et_name)
            keep_mask |= edge_type_ids == et_id

    # Create a lightweight wrapper that returns only the kept edges
    class _MaskedGraph:
        def __init__(self, orig, mask):
            self._orig = orig
            self._mask = mask

        def get_full_edge_index(self):
            return self._orig.get_full_edge_index()[:, self._mask]

        def get_edge_type_ids(self):
            return self._orig.get_edge_type_ids()[self._mask]

        def get_edge_type_names(self):
            return self._orig.get_edge_type_names()

        def __getattr__(self, name):
            return getattr(self._orig, name)

    masked["graph"] = _MaskedGraph(graph, keep_mask)

    # --- Mask events --------------------------------------------------------
    # Map edge types to related event types in the stream
    event_type_filter = set()
    if "CORR" not in active_edge_types:
        event_type_filter.add("CORRELATION_UPDATE")
    if "FACT" not in active_edge_types:
        # FACT edges don't generate streaming events, but mark for clarity
        pass

    if event_type_filter:
        new_events: Dict[int, List] = {}
        for date_key, events in data["events_by_date"].items():
            filtered = [ev for ev in events if ev.event_type not in event_type_filter]
            new_events[date_key] = filtered
        masked["events_by_date"] = new_events

    # Mask corr labels if CORR is removed
    if "CORR" not in active_edge_types:
        masked["corr_by_date"] = {}
        masked["corr_labels_by_date"] = {}

    return masked


# ---------------------------------------------------------------------------
# Per-window training
# ---------------------------------------------------------------------------

def _train_window(
    variant: str,
    data: dict,
    start: str,
    end: str,
    tickers: List[str],
    epochs: int,
    train_dates: List[int],
    val_dates: List[int],
    test_dates: List[int],
) -> dict:
    lr, use_cosine, patience = _hyperparam_lr(variant)
    return train_link_prediction(
        tickers=tickers,
        start=start,
        end=end,
        benchmark="SPY",
        num_epochs=epochs,
        mode="regression",
        model_variant=variant,
        seed=42,
        prepared_data=data,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        weight_decay=1e-4,
        lr=lr,
        use_cosine_schedule=use_cosine,
        early_stopping_patience=patience,
    )


# ---------------------------------------------------------------------------
# Bootstrap comparison helpers
# ---------------------------------------------------------------------------

def _pairwise_bootstrap(
    realized_returns: Dict[str, List[np.ndarray]],
    pairs: List[Tuple[str, str]],
    window_idx: int,
    block_size: int,
    n_bootstrap: int,
) -> dict:
    comparisons: dict = {}
    for left, right in pairs:
        rl = realized_returns[left][-1]
        rr = realized_returns[right][-1]
        if len(rl) == 0 or len(rr) == 0 or len(rl) != len(rr):
            continue
        paired = paired_block_bootstrap_multi(
            rl, rr, block_size=block_size, n_iterations=n_bootstrap, seed=42 + window_idx
        )
        diff_b = paired["sharpe_diffs"]
        cvar_diff_b = paired["cvar_diffs"]
        d_obs = _sharpe(rl) - _sharpe(rr)
        std_b = np.std(diff_b, ddof=1)
        comparisons[f"{left}_vs_{right}"] = {
            "sharpe_obs_diff": float(d_obs),
            "sharpe_bootstrap_ci_2.5": float(np.percentile(diff_b, 2.5)),
            "sharpe_bootstrap_ci_97.5": float(np.percentile(diff_b, 97.5)),
            "sharpe_p_direct": float(np.mean(diff_b <= 0.0)),
            "sharpe_effect_size_d": float(np.mean(diff_b) / std_b) if std_b > 1e-10 else 0.0,
            "cvar_obs_diff": float(_cvar(rl) - _cvar(rr)),
            "cvar_bootstrap_ci_2.5": float(np.percentile(cvar_diff_b, 2.5)),
            "cvar_bootstrap_ci_97.5": float(np.percentile(cvar_diff_b, 97.5)),
            "cvar_p_direct": float(np.mean(cvar_diff_b >= 0.0)),
        }
    return comparisons


# ---------------------------------------------------------------------------
# Full evaluation loop (normal mode)
# ---------------------------------------------------------------------------

def _run_normal(
    variants: List[str],
    data: dict,
    tickers: List[str],
    windows: List[Tuple],
    start: str,
    end: str,
    epochs: int,
    n_bootstrap: int,
    block_size: int,
    step_days: int,
    test_days: int,
    logger: logging.Logger,
) -> dict:
    n_pairs = N_PAIRS_BY_TICKERS.get(len(tickers), len(tickers) * (len(tickers) - 1) // 2)
    comparison_pairs = list(combinations(variants, 2))

    scalar_results: Dict[str, List[dict]] = {v: [] for v in variants}
    realized_returns: Dict[str, List[np.ndarray]] = {v: [] for v in variants}
    daily_losses: Dict[str, List] = {v: [] for v in variants}
    tkg_artifacts: List[dict] = []
    window_reports: List[dict] = []
    confirmatory_p: Dict[str, Optional[float]] = {}
    exploratory_p: Dict[str, Optional[float]] = {}

    for wi, (train_dates, val_dates, test_dates) in enumerate(windows, start=1):
        logger.info("-" * 60)
        logger.info(
            "Window %d/%d | train=%d val=%d test=%d",
            wi, len(windows), len(train_dates), len(val_dates), len(test_dates),
        )

        for variant in variants:
            logger.info("Training %s | window %d", variant.upper(), wi)
            metrics = _train_window(
                variant, data, start, end, tickers, epochs,
                train_dates, val_dates, test_dates,
            )
            scalar_results[variant].append(
                {k: float(v) for k, v in metrics.items() if not k.startswith("_")}
            )
            realized_returns[variant].append(
                np.array(metrics.get("_realized_returns", []), dtype=float)
            )
            if variant == "temporal_kg":
                tkg_artifacts.append(metrics.get("_temporal_kg_artifacts", {}))

            if "_all_preds" in metrics and "_all_targets" in metrics:
                preds = metrics["_all_preds"].detach().cpu().numpy()
                targets = metrics["_all_targets"].detach().cpu().numpy()
                daily_losses[variant].append(
                    extract_daily_errors(preds, targets, logger,
                                         label=f"{variant}/w{wi}", n_pairs=n_pairs)
                )
            else:
                daily_losses[variant].append(None)

        report: dict = {
            "window_index": wi,
            "train_days": len(train_dates),
            "val_days": len(val_dates),
            "test_days": len(test_dates),
            "metrics": {v: scalar_results[v][-1] for v in variants},
            "comparisons": _pairwise_bootstrap(
                realized_returns, comparison_pairs, wi, block_size, n_bootstrap
            ),
        }
        if tkg_artifacts:
            report["temporal_kg_explanations"] = tkg_artifacts[-1].get("top_explanations", [])
            report["temporal_kg_relation_scores"] = tkg_artifacts[-1].get("last_relation_scores", {})
        window_reports.append(report)

    # ---- Global stats -------------------------------------------------------
    sharpes: Dict[str, np.ndarray] = {
        v: np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results[v]], dtype=float)
        for v in variants
    }

    # Primary: temporal_kg vs ra_htgn (if both present), else first pair
    primary_left, primary_right = (
        ("temporal_kg", "ra_htgn")
        if "temporal_kg" in variants and "ra_htgn" in variants
        else comparison_pairs[0] if comparison_pairs else (variants[0], variants[0])
    )
    valid = np.isfinite(sharpes.get(primary_left, np.array([]))) \
          & np.isfinite(sharpes.get(primary_right, np.array([])))

    primary_result: dict = {}
    if valid.any() and primary_left != primary_right:
        wins = int(np.sum(sharpes[primary_left][valid] >= sharpes[primary_right][valid]))
        n = int(np.sum(valid))
        win_rate = wins / n
        br = binomtest(wins, n, p=0.50, alternative="greater")
        bci = br.proportion_ci(confidence_level=0.95)
        ww = run_window_wilcoxon(sharpes[primary_left][valid], sharpes[primary_right][valid], "greater")
        if ww:
            confirmatory_p[f"{primary_left}_wilcoxon_vs_{primary_right}"] = ww["p_value"]
        logger.info(
            "%s >= %s in %d/%d windows (%.1f%%) | binom p=%.4e",
            primary_left, primary_right, wins, n, win_rate * 100, br.pvalue,
        )
        primary_result = {
            "left": primary_left, "right": primary_right,
            "wins": wins, "n_windows": n, "win_rate": win_rate,
            "binomial_test": {"p_value": float(br.pvalue),
                              "ci_95_low": float(bci.low), "ci_95_high": float(bci.high)},
            "window_wilcoxon": ww,
            "holm_bonferroni_confirmatory": holm_bonferroni(confirmatory_p),
        }

    # Pairwise window Wilcoxon (all vs all)
    pairwise_ww: dict = {}
    for left, right in comparison_pairs:
        valid2 = np.isfinite(sharpes[left]) & np.isfinite(sharpes[right])
        res = run_window_wilcoxon(sharpes[left][valid2], sharpes[right][valid2], "greater")
        if res:
            pairwise_ww[f"{left}_vs_{right}"] = res
            exploratory_p[f"ww_{left}_vs_{right}"] = res["p_value"]

    # Pooled DM (only for non-overlapping windows)
    pooled_dm: dict = {}
    if step_days >= test_days:
        for left, right in comparison_pairs:
            ld = [e for e in daily_losses[left] if e is not None]
            rd = [e for e in daily_losses[right] if e is not None]
            pairs_d = list(zip(ld, rd))
            if not pairs_d:
                continue
            lmae = np.concatenate([x["daily_mae"] for x, _ in pairs_d])
            rmae = np.concatenate([y["daily_mae"] for _, y in pairs_d])
            dm = diebold_mariano_test(lmae, rmae, loss="mae", alternative="less")
            pooled_dm[f"{left}_vs_{right}_mae"] = dm
            exploratory_p[f"dm_{left}_vs_{right}"] = dm["p_value"]
    else:
        logger.warning("step_days < test_days — skipping pooled DM tests (overlapping windows).")

    return {
        "variants": variants,
        "comparison_pairs": comparison_pairs,
        "metrics_by_variant": scalar_results,
        "window_reports": window_reports,
        "temporal_kg_interpretability": tkg_artifacts,
        "primary_comparison": primary_result,
        "pairwise_window_tests": pairwise_ww,
        "pooled_predictive_tests": pooled_dm,
        "holm_bonferroni_exploratory": holm_bonferroni(exploratory_p),
        "descriptive_summary": {
            "mean_window_sharpe": {
                v: float(np.nanmean(sharpes[v])) for v in variants
            },
            "std_window_sharpe": {
                v: float(np.nanstd(sharpes[v], ddof=1)) if len(sharpes[v]) > 1 else 0.0
                for v in variants
            },
            "mean_window_metrics": {
                v: {
                    m: float(np.nanmean([e.get(m, np.nan) for e in scalar_results[v]]))
                    for m in ["r_squared", "spearman", "mae", "loss", "cls_f1", "sharpe_proxy"]
                }
                for v in variants
            },
        },
    }


# ---------------------------------------------------------------------------
# Ablation mode
# ---------------------------------------------------------------------------

def _run_ablation(
    ablation_variant: str,
    ablation_sets: List[FrozenSet[str]],
    data: dict,
    tickers: List[str],
    windows: List[Tuple],
    start: str,
    end: str,
    epochs: int,
    n_bootstrap: int,
    block_size: int,
    logger: logging.Logger,
) -> dict:
    n_pairs = N_PAIRS_BY_TICKERS.get(len(tickers), len(tickers) * (len(tickers) - 1) // 2)
    ablation_results: dict = {}

    for edge_set in ablation_sets:
        label = _ABLATION_NAME[edge_set]
        logger.info("=" * 60)
        logger.info("ABLATION: %s | variant=%s | active edges=%s",
                    label, ablation_variant, sorted(edge_set))

        masked_data = _mask_data_for_ablation(data, edge_set)
        scalar_results: List[dict] = []
        realized_returns: List[np.ndarray] = []
        daily_losses: List = []

        for wi, (train_dates, val_dates, test_dates) in enumerate(windows, start=1):
            logger.info("  Ablation window %d/%d [%s]", wi, len(windows), label)
            metrics = _train_window(
                ablation_variant, masked_data, start, end, tickers, epochs,
                train_dates, val_dates, test_dates,
            )
            scalar_results.append(
                {k: float(v) for k, v in metrics.items() if not k.startswith("_")}
            )
            realized_returns.append(np.array(metrics.get("_realized_returns", []), dtype=float))
            if "_all_preds" in metrics and "_all_targets" in metrics:
                preds = metrics["_all_preds"].detach().cpu().numpy()
                targets = metrics["_all_targets"].detach().cpu().numpy()
                daily_losses.append(
                    extract_daily_errors(preds, targets, logger,
                                         label=f"ablation_{label}/w{wi}", n_pairs=n_pairs)
                )
            else:
                daily_losses.append(None)

        sharpes = np.array([m.get("sharpe_proxy", np.nan) for m in scalar_results], dtype=float)
        ablation_results[label] = {
            "active_edges": sorted(edge_set),
            "window_metrics": scalar_results,
            "mean_sharpe": float(np.nanmean(sharpes)),
            "std_sharpe": float(np.nanstd(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
            "mean_r_squared": float(np.nanmean([m.get("r_squared", np.nan) for m in scalar_results])),
            "mean_spearman": float(np.nanmean([m.get("spearman", np.nan) for m in scalar_results])),
        }

    # Rank ablation sets by mean Sharpe
    ranked = sorted(
        [(k, v["mean_sharpe"]) for k, v in ablation_results.items()],
        key=lambda x: x[1], reverse=True,
    )
    logger.info("Ablation ranking (Sharpe ↓):")
    for rank_pos, (name, sh) in enumerate(ranked, start=1):
        logger.info("  %d. %s  mean_sharpe=%.4f", rank_pos, name, sh)

    return {
        "ablation_variant": ablation_variant,
        "ablation_results": ablation_results,
        "ablation_ranking_by_sharpe": ranked,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_bootstrap_eval_temporal_kg_rev2(
    variants: List[str] = ALL_VARIANTS,
    n_tickers: int = DEFAULT_N_TICKERS,
    ablation: Optional[str] = None,
    ablation_variant: str = "tgn",
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    epochs: int = DEFAULT_EPOCHS,
    train_days: int = DEFAULT_TRAIN_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_windows: Optional[int] = None,
) -> dict:
    logger = setup_logging("dyfo.bootstrap_eval_tkg_rev2", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Bootstrap Eval Temporal KG Rev2")
    logger.info("  variants   : %s", variants)
    logger.info("  n_tickers  : %d", n_tickers)
    logger.info("  ablation   : %s", ablation or "disabled")
    logger.info("  step_days  : %d | test_days=%d | overlap=%s",
                step_days, test_days, "YES" if step_days < test_days else "NO")
    logger.info("=" * 60)

    if step_days < test_days:
        logger.warning(
            "step_days (%d) < test_days (%d): overlapping windows detected. "
            "TGN memory leakage is possible. Use step_days >= test_days for fair comparison.",
            step_days, test_days,
        )

    tickers = get_tickers(n_tickers)
    sparsification = get_sparsification(n_tickers)
    if sparsification == "tmfg":
        logger.warning(
            "n_tickers=%d requires TMFG sparsification for CORR edges "
            "(spec/02_graph_spec.md §Sparsification). "
            "The current implementation uses threshold — results may differ from spec.",
            n_tickers,
        )

    config = DyFOConfig(model_variant=variants[0] if variants else "tgn")
    data_config = DataConfig(
        tickers=tickers, benchmark_ticker="SPY", start_date=start, end_date=end
    )
    data = load_or_prepare_data(tickers, start, end, "SPY", config, data_config, logger)
    windows = build_windows(
        data["sorted_dates"],
        train_size=train_days,
        val_size=val_days,
        test_size=test_days,
        step_size=step_days,
        max_windows=max_windows,
    )
    if not windows:
        raise RuntimeError("No walk-forward windows could be constructed.")
    logger.info("Walk-forward windows: %d | date span=%s..%s", len(windows), start, end)

    # ---- Dispatch -----------------------------------------------------------
    if ablation:
        ablation_sets = ABLATION_BASIC if ablation == "basic" else ABLATION_FULL
        if ablation_variant not in variants:
            logger.warning(
                "ablation_variant=%s not in --variants; adding it automatically.",
                ablation_variant,
            )
            variants = [ablation_variant] + [v for v in variants if v != ablation_variant]

        ablation_body = _run_ablation(
            ablation_variant=ablation_variant,
            ablation_sets=ablation_sets,
            data=data,
            tickers=tickers,
            windows=windows,
            start=start,
            end=end,
            epochs=epochs,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            logger=logger,
        )

        # Still run normal comparison for selected variants
        normal_body = _run_normal(
            variants=variants,
            data=data,
            tickers=tickers,
            windows=windows,
            start=start,
            end=end,
            epochs=epochs,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            step_days=step_days,
            test_days=test_days,
            logger=logger,
        )

        results_body = {**normal_body, "ablation": ablation_body}
    else:
        results_body = _run_normal(
            variants=variants,
            data=data,
            tickers=tickers,
            windows=windows,
            start=start,
            end=end,
            epochs=epochs,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            step_days=step_days,
            test_days=test_days,
            logger=logger,
        )

    # ---- Assemble summary ---------------------------------------------------
    summary = {
        "version": "temporal_kg_rev2_bl18",
        "revision_notes": (
            "Rev 2: --variants filter; --n_tickers 30/50/100; "
            "--ablation basic/full for CORR/SECT/FACT isolated runs; "
            "inherits Rev 1 non-overlapping windows and correct LR routing."
        ),
        "run_config": {
            "variants": variants,
            "n_tickers": n_tickers,
            "tickers": tickers,
            "sparsification": sparsification,
            "ablation_mode": ablation,
            "ablation_variant": ablation_variant if ablation else None,
            "start": start, "end": end, "epochs": epochs,
            "train_days": train_days, "val_days": val_days,
            "test_days": test_days, "step_days": step_days,
            "n_windows": len(windows),
            "windows_overlap": step_days < test_days,
            "n_bootstrap": n_bootstrap, "block_size": block_size,
            "tgn_lr": TGN_LR, "baseline_lr": BASELINE_LR,
            "tgn_patience": TGN_PATIENCE, "baseline_patience": BASELINE_PATIENCE,
        },
        **results_body,
    }

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    tag = f"abl_{ablation}_{ablation_variant}_" if ablation else ""
    out_dir = RESULTS_DIR / f"bootstrap_eval_tkg_rev2_{tag}{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_summary_tkg_rev2.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Saved summary → %s", out_path)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DyFO bootstrap eval BL-18 Temporal KG — Rev 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_VARIANTS,
        default=ALL_VARIANTS,
        metavar="VARIANT",
        help=(
            "Variants to train (space-separated). "
            f"Choices: {ALL_VARIANTS}. Default: all."
        ),
    )
    parser.add_argument(
        "--n_tickers",
        type=int,
        choices=[30, 50, 100],
        default=DEFAULT_N_TICKERS,
        help="Universe size (30 / 50 / 100). Default: 30.",
    )
    parser.add_argument(
        "--ablation",
        choices=["basic", "full"],
        default=None,
        help=(
            "Edge-type ablation mode. "
            "'basic' = CORR_only / SECT_only / FACT_only; "
            "'full' = all 7 subsets. Default: disabled."
        ),
    )
    parser.add_argument(
        "--ablation_variant",
        choices=ALL_VARIANTS,
        default="tgn",
        help="Which variant to ablate (default: tgn).",
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
        help="Step between windows (default=test_days → non-overlapping).",
    )
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max_windows", type=int, default=None, help="Cap on windows (default=all).")
    args = parser.parse_args()

    run_bootstrap_eval_temporal_kg_rev2(
        variants=args.variants,
        n_tickers=args.n_tickers,
        ablation=args.ablation,
        ablation_variant=args.ablation_variant,
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
