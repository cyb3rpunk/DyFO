"""Compute MDD & Turnover from saved checkpoints (NO re-training).

Locates the link_pred_*_s42 run directories that match the
bootstrap_eval_tkg_rev2_20260418_130703 results, loads each
best_model.pt checkpoint, replays the test phase to collect
daily realized returns and portfolio weights, then computes:
  - Maximum Drawdown (per window + concatenated)
  - Portfolio Turnover (per window + mean)

Usage:
  cd d:\projetos\DyFO
  python scripts/compute_mdd_turnover_full.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.ticker_registry import get_tickers
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_v5 import build_windows, load_or_prepare_data

# ---------------------------------------------------------------------------
# Constants — hard-coded from the run we want to analyze
# ---------------------------------------------------------------------------

BOOTSTRAP_SUMMARY = (
    RESULTS_DIR
    / "bootstrap_eval_tkg_rev2_20260418_130703"
    / "bootstrap_summary_tkg_rev2.json"
)

# Window → run directories (matched by sharpe_proxy values from the summary)
# These are the runs from the bootstrap eval that started 2026-04-17 ~22:58
# and finished 2026-04-18 ~13:07.
VARIANT_RUN_DIRS: Dict[str, List[str]] = {
    "tgat": [
        "link_pred_tgat_s42_20260417_225805",  # W1
        "link_pred_tgat_s42_20260418_001033",  # W2
        "link_pred_tgat_s42_20260418_013639",  # W3
        "link_pred_tgat_s42_20260418_032748",  # W4
        "link_pred_tgat_s42_20260418_051318",  # W5
        "link_pred_tgat_s42_20260418_070239",  # W6
        "link_pred_tgat_s42_20260418_085758",  # W7
        "link_pred_tgat_s42_20260418_095309",  # W8
        "link_pred_tgat_s42_20260418_113122",  # W9
    ],
    "tgn": [
        "link_pred_tgn_s42_20260417_232025",   # W1
        "link_pred_tgn_s42_20260418_003341",   # W2
        "link_pred_tgn_s42_20260418_020332",   # W3
        "link_pred_tgn_s42_20260418_040551",   # W4
        "link_pred_tgn_s42_20260418_054946",   # W5
        "link_pred_tgn_s42_20260418_073324",   # W6
        "link_pred_tgn_s42_20260418_092102",   # W7
        "link_pred_tgn_s42_20260418_102611",   # W8
        "link_pred_tgn_s42_20260418_120135",   # W9
    ],
    "roland": [
        "link_pred_roland_s42_20260418_000753",  # W1
        "link_pred_roland_s42_20260418_013512",  # W2
        "link_pred_roland_s42_20260418_032234",  # W3
        "link_pred_roland_s42_20260418_051005",  # W4
        "link_pred_roland_s42_20260418_065830",  # W5
        "link_pred_roland_s42_20260418_085515",  # W6
        "link_pred_roland_s42_20260418_095036",  # W7
        "link_pred_roland_s42_20260418_112821",  # W8
        "link_pred_roland_s42_20260418_130334",  # W9
    ],
    "gat_static": [
        "link_pred_gat_static_s42_20260418_000905",  # W1
        "link_pred_gat_static_s42_20260418_013600",  # W2
        "link_pred_gat_static_s42_20260418_032424",  # W3
        "link_pred_gat_static_s42_20260418_051126",  # W4
        "link_pred_gat_static_s42_20260418_065922",  # W5
        "link_pred_gat_static_s42_20260418_085634",  # W6
        "link_pred_gat_static_s42_20260418_095145",  # W7
        "link_pred_gat_static_s42_20260418_112944",  # W8
        "link_pred_gat_static_s42_20260418_130433",  # W9
    ],
}

N_TICKERS = 50
START = "2018-01-01"
END = "2024-12-31"
TRAIN_DAYS = 500
VAL_DAYS = 125
TEST_DAYS = 125
STEP_DAYS = 125

# Epoch reference for date conversion
import datetime
_EPOCH = datetime.date(2000, 1, 1)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """MDD from daily returns. Returns (mdd, peak_idx, trough_idx)."""
    if len(returns) == 0:
        return 0.0, 0, 0
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1.0
    trough = int(np.argmin(dd))
    peak = int(np.argmax(cum[:trough + 1])) if trough > 0 else 0
    return float(dd[trough]), peak, trough


def compute_turnover(weights_list: List[np.ndarray]) -> float:
    """Mean one-way turnover from daily weight vectors."""
    if len(weights_list) < 2:
        return 0.0
    t_vals = []
    for t in range(1, len(weights_list)):
        t_vals.append(np.sum(np.abs(weights_list[t] - weights_list[t - 1])) / 2.0)
    return float(np.mean(t_vals))


# ---------------------------------------------------------------------------
# Test-phase replay (inference only, no gradients)
# ---------------------------------------------------------------------------

def replay_test_phase(
    variant: str,
    checkpoint_path: Path,
    data: dict,
    tickers: List[str],
    train_dates: List[int],
    val_dates: List[int],
    test_dates: List[int],
    logger,
) -> dict:
    """Load saved model, replay train+val state, then run test to get daily returns & weights."""
    from scipy.optimize import minimize
    from dyfo.core.model_variants import build_encoder
    from dyfo.core.link_prediction import CorrelationRegressor, build_regression_labels

    num_nodes = len(tickers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DyFOConfig(model_variant=variant)
    encoder = build_encoder(config, num_nodes, variant=variant).to(device)

    # Variant-specific setup
    if variant == "gat_static":
        encoder.set_static_graph_from_correlations(
            data["corr_labels_by_date"], train_dates
        )
    elif variant == "roland":
        encoder.precompute_monthly_snapshots(data["corr_labels_by_date"])

    decoder = CorrelationRegressor(
        embedding_dim=config.embedding_dim, hidden_dim=64, dropout=config.dropout
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    # Remove snap buffers to prevent size mismatch errors (they get recomputed anyway)
    keys_to_remove = [k for k in ckpt["encoder_state"].keys() if "_snap_edge" in k]
    for k in keys_to_remove:
        ckpt["encoder_state"].pop(k, None)

    encoder.load_state_dict(ckpt["encoder_state"], strict=False)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)
    encoder.eval()
    decoder.eval()

    edge_index = data["graph"].get_full_edge_index().to(device)
    edge_type_ids = data["graph"].get_edge_type_ids().to(device)
    edge_timestamps = torch.zeros(edge_index.shape[1], device=device)

    # Node features helper
    nf_dates = sorted(data["node_features_by_date"].keys())

    def get_nf(date_key):
        closest = nf_dates[0]
        for d in nf_dates:
            if d <= str(date_key):
                closest = d
            else:
                break
        return data["node_features_by_date"][closest]

    # GMV optimizer
    def optimize_gmv(cov):
        n = cov.shape[0]
        w0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        res = minimize(lambda w: w.T @ cov @ w, w0, bounds=bounds,
                       constraints=cons, method="SLSQP", tol=1e-6)
        return res.x if res.success else w0

    # Price returns + vols
    price_returns = data["prices"].pct_change().fillna(0)
    price_vols = price_returns.rolling(window=21).std() * np.sqrt(252)
    price_vols = price_vols.fillna(0.15)

    # --- Replay train + val to build temporal state ---
    encoder.reset_state()
    for split_dates in [train_dates, val_dates]:
        for d_idx in range(len(split_dates)):
            today = split_dates[d_idx]
            day_events = data["events_by_date"].get(today, [])
            nf = get_nf(today).to(device)
            t = float(today) + 0.99
            with torch.no_grad():
                encoder.advance_day(day_events, nf, edge_index, edge_type_ids,
                                    edge_timestamps, t)

    # --- Test phase: collect daily returns and weights ---
    daily_returns = []
    daily_weights = []

    for d_idx in range(len(test_dates) - 1):
        today = test_dates[d_idx]
        tomorrow = test_dates[d_idx + 1]

        day_events = data["events_by_date"].get(today, [])
        nf = get_nf(today).to(device)
        t = float(today) + 0.99

        corr_tomorrow = data["corr_labels_by_date"].get(tomorrow, {})

        with torch.no_grad():
            encoder.advance_day(day_events, nf, edge_index, edge_type_ids,
                                edge_timestamps, t)

        if not corr_tomorrow:
            continue

        # Get predictions
        src, dst, targets = build_regression_labels(corr_tomorrow, num_nodes)
        src = src.to(device)
        dst = dst.to(device)

        with torch.no_grad():
            z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids,
                                            edge_timestamps, t)
            preds = decoder(z[src], z[dst])

        # Build correlation matrix from predictions
        corr_matrix = np.eye(num_nodes)
        seen = set()
        pair_list = []
        for (i, j) in corr_tomorrow.keys():
            pair = (min(i, j), max(i, j))
            if pair not in seen:
                seen.add(pair)
                pair_list.append(pair)

        p_np = preds.cpu().numpy()
        if len(p_np) == len(pair_list):
            for p_idx, (i, j) in enumerate(pair_list):
                rho = p_np[p_idx]
                corr_matrix[i, j] = rho
                corr_matrix[j, i] = rho

            # Build covariance
            date_str = str(_EPOCH + datetime.timedelta(days=tomorrow))
            vols = (price_vols.loc[date_str].values
                    if date_str in price_vols.index
                    else np.full(num_nodes, 0.15))
            cov = np.diag(vols) @ corr_matrix @ np.diag(vols)
            cov += np.eye(num_nodes) * 1e-4

            # Optimize weights
            weights = optimize_gmv(cov)
            daily_weights.append(weights.copy())

            # Realized return
            day_ret = (price_returns.loc[date_str].values
                       if date_str in price_returns.index
                       else np.zeros(num_nodes))
            realized_ret = np.dot(weights, day_ret)
            daily_returns.append(float(realized_ret))

    return {
        "daily_returns": np.array(daily_returns),
        "daily_weights": daily_weights,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger = setup_logging("dyfo.mdd_turnover", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Computing MDD & Turnover from saved checkpoints")
    logger.info("=" * 60)

    # Verify checkpoint mapping by cross-checking sharpe values
    with open(BOOTSTRAP_SUMMARY, "r") as f:
        summary = json.load(f)

    logger.info("Loading prepared data (cached)...")
    tickers = get_tickers(N_TICKERS)
    config = DyFOConfig(model_variant="tgat")
    data_config = DataConfig(
        tickers=tickers, benchmark_ticker="SPY",
        start_date=START, end_date=END,
    )
    data = load_or_prepare_data(tickers, START, END, "SPY", config, data_config, logger)

    windows = build_windows(
        data["sorted_dates"],
        train_size=TRAIN_DAYS, val_size=VAL_DAYS,
        test_size=TEST_DAYS, step_size=STEP_DAYS,
    )
    logger.info("Walk-forward windows: %d", len(windows))

    # Verify mapping: check sharpe from results.json matches summary
    for variant, dirs in VARIANT_RUN_DIRS.items():
        for wi, run_dir_name in enumerate(dirs):
            rpath = RESULTS_DIR / run_dir_name / "results.json"
            if not rpath.exists():
                logger.error("MISSING: %s", rpath)
                continue
            with open(rpath) as f:
                rj = json.load(f)
            saved_sharpe = rj["metrics"].get("test_sharpe_proxy", None)
            expected_sharpe = summary["metrics_by_variant"][variant][wi]["sharpe_proxy"]
            if saved_sharpe is not None and abs(saved_sharpe - expected_sharpe) > 1e-6:
                logger.warning(
                    "SHARPE MISMATCH %s W%d: saved=%.6f expected=%.6f",
                    variant, wi + 1, saved_sharpe, expected_sharpe,
                )
            else:
                logger.info("[OK] %s W%d sharpe verified (%.4f)", variant, wi + 1, expected_sharpe)

    # --- Replay test phase for each model ---
    all_results: Dict[str, dict] = {}

    for variant in VARIANT_RUN_DIRS:
        logger.info("=" * 60)
        logger.info("Processing %s (%d windows)", variant.upper(), len(VARIANT_RUN_DIRS[variant]))

        per_window_mdd = []
        per_window_turnover = []
        per_window_cumret = []
        per_window_vol = []
        all_daily_returns = []

        for wi, run_dir_name in enumerate(VARIANT_RUN_DIRS[variant]):
            ckpt_path = RESULTS_DIR / run_dir_name / "best_model.pt"
            if not ckpt_path.exists():
                logger.error("Checkpoint not found: %s", ckpt_path)
                per_window_mdd.append(np.nan)
                per_window_turnover.append(np.nan)
                per_window_cumret.append(np.nan)
                per_window_vol.append(np.nan)
                continue

            train_d, val_d, test_d = windows[wi]
            logger.info("  W%d: replaying test phase for %s...", wi + 1, variant.upper())

            out = replay_test_phase(
                variant, ckpt_path, data, tickers,
                train_d, val_d, test_d, logger,
            )

            rets = out["daily_returns"]
            weights = out["daily_weights"]

            if len(rets) > 0:
                mdd, _, _ = compute_max_drawdown(rets)
                cumret = float(np.prod(1.0 + rets) - 1.0)
                vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0
            else:
                mdd, cumret, vol = 0.0, 0.0, 0.0

            turnover = compute_turnover(weights)

            per_window_mdd.append(mdd)
            per_window_turnover.append(turnover)
            per_window_cumret.append(cumret)
            per_window_vol.append(vol)
            all_daily_returns.extend(rets.tolist())

            logger.info(
                "    W%d %s: MDD=%.4f | Turnover=%.4f | CumRet=%.4f | Vol=%.4f | n_days=%d",
                wi + 1, variant.upper(), mdd, turnover, cumret, vol, len(rets),
            )

        # Overall
        all_rets = np.array(all_daily_returns)
        overall_mdd, _, _ = compute_max_drawdown(all_rets) if len(all_rets) > 0 else (0.0, 0, 0)
        overall_cumret = float(np.prod(1.0 + all_rets) - 1.0) if len(all_rets) > 0 else 0.0
        overall_vol = float(np.std(all_rets) * np.sqrt(252)) if len(all_rets) > 1 else 0.0
        overall_sharpe = float(
            (np.mean(all_rets) / (np.std(all_rets) + 1e-8)) * np.sqrt(252)
        ) if len(all_rets) > 1 else 0.0

        all_results[variant] = {
            "per_window_mdd": per_window_mdd,
            "per_window_turnover": per_window_turnover,
            "per_window_cumret": per_window_cumret,
            "per_window_vol": per_window_vol,
            "mean_window_mdd": float(np.nanmean(per_window_mdd)),
            "worst_window_mdd": float(np.nanmin(per_window_mdd)),
            "mean_turnover": float(np.nanmean(per_window_turnover)),
            "overall_mdd": overall_mdd,
            "overall_cumret": overall_cumret,
            "overall_vol_ann": overall_vol,
            "overall_sharpe": overall_sharpe,
            "n_daily_returns": len(all_rets),
        }

    # --- Save ---
    output = {
        "source": str(BOOTSTRAP_SUMMARY),
        "n_tickers": N_TICKERS,
        "n_windows": len(windows),
        "per_variant": all_results,
    }

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"mdd_turnover_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mdd_turnover_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved → %s", out_path)

    # --- Print table ---
    print("\n" + "=" * 100)
    print("MDD & TURNOVER — PER WINDOW")
    print("=" * 100)
    for variant in VARIANT_RUN_DIRS:
        r = all_results[variant]
        print(f"\n{variant.upper()}")
        print(f"  {'Window':<8} {'MDD':>10} {'Turnover':>10} {'CumRet':>10} {'Vol(ann)':>10}")
        print(f"  {'-'*48}")
        for wi in range(len(r["per_window_mdd"])):
            print(f"  W{wi+1:<7} {r['per_window_mdd'][wi]:>10.4f} "
                  f"{r['per_window_turnover'][wi]:>10.4f} "
                  f"{r['per_window_cumret'][wi]:>10.4f} "
                  f"{r['per_window_vol'][wi]:>10.4f}")
        print(f"  {'MEAN':<8} {r['mean_window_mdd']:>10.4f} "
              f"{r['mean_turnover']:>10.4f}")

    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Variant':<12} {'MDD Overall':>12} {'MDD Mean':>10} {'MDD Worst':>10} "
          f"{'Turnover':>10} {'CumRet':>10} {'Vol(ann)':>10} {'Sharpe':>10}")
    print("-" * 88)
    for v in VARIANT_RUN_DIRS:
        s = all_results[v]
        print(f"{v:<12} {s['overall_mdd']:>12.4f} {s['mean_window_mdd']:>10.4f} "
              f"{s['worst_window_mdd']:>10.4f} {s['mean_turnover']:>10.4f} "
              f"{s['overall_cumret']:>10.4f} {s['overall_vol_ann']:>10.4f} "
              f"{s['overall_sharpe']:>10.4f}")


if __name__ == "__main__":
    main()
