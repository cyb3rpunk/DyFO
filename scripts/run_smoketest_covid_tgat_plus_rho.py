#!/usr/bin/env python3
"""COVID smoketest for Persistence vs TGAT vs TGAT + rho_t.

Goal
----
Measure how much graph structure adds on top of the autoregressive signal
``rho_t`` during the COVID test period.

Variants
--------
- ``persistence``: predict ``rho_{t+1} = rho_t``
- ``tgat``: standard DyFO TGAT decoder
- ``tgat_plus_rho``: same TGAT encoder, but decoder also receives ``rho_t``

Default split
-------------
- train: 2018-01-01 .. 2019-09-30
- val:   2019-10-01 .. 2019-12-31
- test:  2020-01-01 .. 2020-06-30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.link_prediction import (
    CorrelationRegressor,
    build_regression_labels,
    compute_regression_metrics,
)
from dyfo.core.model_variants import BaseGraphEncoder, build_encoder
from dyfo.core.ticker_registry import get_tickers
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_v5 import (
    TGN_LR,
    TGN_PATIENCE,
    TGN_USE_COSINE,
    load_or_prepare_data,
)
from scripts.train_link_prediction import set_seed


EPOCH = date(2000, 1, 1)


class CorrelationRegressorWithRho(nn.Module):
    """Pairwise correlation decoder with an explicit autoregressive rho_t input."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, rho_t: torch.Tensor) -> torch.Tensor:
        if rho_t.dim() == 1:
            rho_t = rho_t.unsqueeze(-1)
        h = torch.cat([z_i, z_j, rho_t], dim=-1)
        return torch.tanh(self.net(h).squeeze(-1))


def int_day_to_iso(day: int) -> str:
    return (EPOCH + timedelta(days=int(day))).isoformat()


def slice_dates(sorted_dates: List[int], start: str, end: str) -> List[int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out: List[int] = []
    for day in sorted_dates:
        ts = pd.Timestamp(int_day_to_iso(day))
        if start_ts <= ts <= end_ts:
            out.append(day)
    return out


def get_node_feature_getter(data: dict):
    nf_dates = sorted(data["node_features_by_date"].keys())

    def get_node_features(date_key: int) -> torch.Tensor:
        closest = nf_dates[0]
        for d in nf_dates:
            if d <= str(date_key):
                closest = d
            else:
                break
        return data["node_features_by_date"][closest]

    return get_node_features


def build_rho_today_feature(
    corr_today: Dict[Tuple[int, int], float],
    src: torch.Tensor,
    dst: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    vals = [
        float(corr_today.get((int(s), int(d)), 0.0))
        for s, d in zip(src.cpu().numpy(), dst.cpu().numpy())
    ]
    return torch.tensor(vals, dtype=torch.float32, device=device)


def build_scheduler(optimizer: optim.Optimizer, num_epochs: int, use_cosine: bool):
    warmup_epochs = min(2, num_epochs)
    if use_cosine and num_epochs > 4:
        def _cosine_lr(ep: int, _w: int = warmup_epochs, _n: int = num_epochs) -> float:
            if ep < _w:
                return (ep + 1) / max(1, _w)
            progress = (ep - _w) / max(1, _n - _w)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lr)
    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs),
    )


def run_persistence(data: dict, dates: List[int]) -> dict:
    all_preds = []
    all_targets = []
    for d_idx in range(len(dates) - 1):
        today = dates[d_idx]
        tomorrow = dates[d_idx + 1]
        corr_tomorrow = data["corr_labels_by_date"].get(tomorrow, {})
        if not corr_tomorrow:
            continue
        src, dst, targets = build_regression_labels(corr_tomorrow, num_nodes=0)
        if len(src) == 0:
            continue
        corr_today = data["corr_labels_by_date"].get(today, {})
        preds = torch.tensor(
            [float(corr_today.get((int(s), int(d)), 0.0)) for s, d in zip(src.numpy(), dst.numpy())],
            dtype=torch.float32,
        )
        all_preds.append(preds)
        all_targets.append(targets)

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    metrics = compute_regression_metrics(preds_cat, targets_cat)
    metrics["_all_preds"] = preds_cat
    metrics["_all_targets"] = targets_cat
    return metrics


def run_tgat_family(
    variant: str,
    data: dict,
    config: DyFOConfig,
    num_nodes: int,
    train_dates: List[int],
    val_dates: List[int],
    test_dates: List[int],
    num_epochs: int,
    lr: float,
    patience: int,
    use_cosine: bool,
    seed: int,
    logger,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder: BaseGraphEncoder = build_encoder(config, num_nodes, variant="tgat").to(device)
    if variant == "tgat_plus_rho":
        decoder = CorrelationRegressorWithRho(
            embedding_dim=config.embedding_dim,
            hidden_dim=64,
            dropout=config.dropout,
        ).to(device)
    else:
        decoder = CorrelationRegressor(
            embedding_dim=config.embedding_dim,
            hidden_dim=64,
            dropout=config.dropout,
        ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = build_scheduler(optimizer, num_epochs, use_cosine)
    loss_fn = nn.SmoothL1Loss()
    edge_index = data["graph"].get_full_edge_index().to(device)
    edge_type_ids = data["graph"].get_edge_type_ids().to(device)
    edge_timestamps = torch.zeros(edge_index.shape[1], device=device)
    get_node_features = get_node_feature_getter(data)

    def run_split(
        dates: List[int],
        split_name: str,
        train_mode: bool,
        collect_predictions: bool = False,
    ) -> dict:
        if train_mode:
            encoder.train()
            decoder.train()
        else:
            encoder.eval()
            decoder.eval()

        all_preds = []
        all_targets = []
        total_metrics: Dict[str, float] = {}
        num_batches = 0

        for d_idx in range(len(dates) - 1):
            today = dates[d_idx]
            tomorrow = dates[d_idx + 1]
            day_events = data["events_by_date"].get(today, [])
            node_feat = get_node_features(today).to(device)
            current_time = float(today) + 0.99
            corr_tomorrow = data["corr_labels_by_date"].get(tomorrow, {})

            if not corr_tomorrow:
                with torch.no_grad():
                    encoder.advance_day(
                        day_events, node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                continue

            src, dst, targets = build_regression_labels(corr_tomorrow, num_nodes)
            if len(src) == 0:
                with torch.no_grad():
                    encoder.advance_day(
                        day_events, node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                continue

            src = src.to(device)
            dst = dst.to(device)
            targets = targets.to(device)
            corr_today = data["corr_labels_by_date"].get(today, {})
            rho_t = build_rho_today_feature(corr_today, src, dst, device)

            if train_mode:
                encoder.advance_day(
                    day_events, node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                )
                z = encoder.get_node_embeddings(
                    node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                )
                if variant == "tgat_plus_rho":
                    preds = decoder(z[src], z[dst], rho_t)
                else:
                    preds = decoder(z[src], z[dst])
                loss = loss_fn(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    max_norm=0.5,
                )
                optimizer.step()
                encoder.detach_state()
            else:
                with torch.no_grad():
                    encoder.advance_day(
                        day_events, node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                    z = encoder.get_node_embeddings(
                        node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                    if variant == "tgat_plus_rho":
                        preds = decoder(z[src], z[dst], rho_t)
                    else:
                        preds = decoder(z[src], z[dst])

            metrics = compute_regression_metrics(preds.detach(), targets)
            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + float(value)
            num_batches += 1
            if collect_predictions:
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())

        averaged = {k: v / max(1, num_batches) for k, v in total_metrics.items()}
        if collect_predictions and all_preds:
            averaged["_all_preds"] = torch.cat(all_preds)
            averaged["_all_targets"] = torch.cat(all_targets)
        return averaged

    best_val_r2 = float("-inf")
    best_epoch = 1
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        encoder.reset_state()
        train_metrics = run_split(train_dates, "train", train_mode=True)
        val_metrics = run_split(val_dates, "val", train_mode=False)
        logger.info(
            "%s epoch %d/%d | train R2=%.4f MAE=%.4f | val R2=%.4f MAE=%.4f",
            variant.upper(),
            epoch,
            num_epochs,
            float(train_metrics.get("r_squared", float("nan"))),
            float(train_metrics.get("mae", float("nan"))),
            float(val_metrics.get("r_squared", float("nan"))),
            float(val_metrics.get("mae", float("nan"))),
        )
        if val_metrics["r_squared"] > best_val_r2:
            best_val_r2 = float(val_metrics["r_squared"])
            best_epoch = epoch
            best_state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("%s early stop at epoch %d", variant.upper(), epoch)
                break
        scheduler.step()

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])

    encoder.reset_state()
    _ = run_split(train_dates, "train_replay", train_mode=False)
    _ = run_split(val_dates, "val_replay", train_mode=False)
    test_metrics = run_split(test_dates, "test", train_mode=False, collect_predictions=True)
    test_metrics["_best_epoch"] = best_epoch
    test_metrics["_best_val_r_squared"] = best_val_r2
    return test_metrics


def run_experiment(
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
    out_dir = RESULTS_DIR / f"smoketest_covid_tgat_plus_rho_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("dyfo.smoketest_covid_tgat_plus_rho", log_to_file=False)

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
        raise RuntimeError("Empty split detected. Check date boundaries.")

    logger.info(
        "Split | train=%s..%s (%d) | val=%s..%s (%d) | test=%s..%s (%d)",
        int_day_to_iso(train_dates[0]), int_day_to_iso(train_dates[-1]), len(train_dates),
        int_day_to_iso(val_dates[0]), int_day_to_iso(val_dates[-1]), len(val_dates),
        int_day_to_iso(test_dates[0]), int_day_to_iso(test_dates[-1]), len(test_dates),
    )

    persistence_metrics = run_persistence(data, test_dates)
    tgat_metrics = run_tgat_family(
        variant="tgat",
        data=data,
        config=config,
        num_nodes=len(tickers),
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        num_epochs=epochs,
        lr=TGN_LR,
        patience=TGN_PATIENCE,
        use_cosine=TGN_USE_COSINE,
        seed=seed,
        logger=logger,
    )
    tgat_plus_rho_metrics = run_tgat_family(
        variant="tgat_plus_rho",
        data=data,
        config=config,
        num_nodes=len(tickers),
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        num_epochs=epochs,
        lr=TGN_LR,
        patience=TGN_PATIENCE,
        use_cosine=TGN_USE_COSINE,
        seed=seed,
        logger=logger,
    )

    summary = {
        "run_tag": out_dir.name,
        "purpose": "TGAT + rho_t vs persistence",
        "split": {
            "train": {"start": int_day_to_iso(train_dates[0]), "end": int_day_to_iso(train_dates[-1]), "n_days": len(train_dates)},
            "val": {"start": int_day_to_iso(val_dates[0]), "end": int_day_to_iso(val_dates[-1]), "n_days": len(val_dates)},
            "test": {"start": int_day_to_iso(test_dates[0]), "end": int_day_to_iso(test_dates[-1]), "n_days": len(test_dates)},
        },
        "metrics_by_variant": {
            "persistence": {k: float(v) for k, v in persistence_metrics.items() if not k.startswith("_")},
            "tgat": {k: float(v) for k, v in tgat_metrics.items() if not k.startswith("_")},
            "tgat_plus_rho": {k: float(v) for k, v in tgat_plus_rho_metrics.items() if not k.startswith("_")},
        },
    }

    p = summary["metrics_by_variant"]["persistence"]
    g = summary["metrics_by_variant"]["tgat_plus_rho"]
    summary["uplift_vs_persistence"] = {
        "delta_r_squared": g["r_squared"] - p["r_squared"],
        "delta_mae": g["mae"] - p["mae"],
        "delta_spearman": g["spearman"] - p["spearman"],
    }

    out_path = out_dir / "smoketest_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved summary -> {out_path}")
    print(f"Test period: {summary['split']['test']['start']} -> {summary['split']['test']['end']}")
    for variant in ["persistence", "tgat", "tgat_plus_rho"]:
        row = summary["metrics_by_variant"][variant]
        print(
            f"{variant:14s} "
            f"R2={row['r_squared']:.4f}  "
            f"MAE={row['mae']:.4f}  "
            f"Spearman={row['spearman']:.4f}"
        )
    print(
        "Graph uplift vs persistence:",
        f"dR2={summary['uplift_vs_persistence']['delta_r_squared']:+.4f}",
        f"dMAE={summary['uplift_vs_persistence']['delta_mae']:+.4f}",
        f"dSpearman={summary['uplift_vs_persistence']['delta_spearman']:+.4f}",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COVID smoketest: Persistence vs TGAT vs TGAT + rho_t",
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

    run_experiment(
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
