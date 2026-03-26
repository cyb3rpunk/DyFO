"""Training script — Self-supervised link prediction pre-training for DyFO TGN.

Trains the TGN encoder + link predictor to predict which pairs of assets
will have high correlation tomorrow, given today's embeddings.

Walk-forward protocol:
  - Train: first 60% of days
  - Validation: next 20%
  - Test: last 20%
  - Memory is inherited across splits (not zeroed)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.dyfo_module import DyFOModule
from dyfo.core.edge_features import (
    build_sector_edges,
    compute_dcc_garch_correlations,
    compute_rolling_correlations,
)
from dyfo.core.event_stream import EventStreamBuilder, FinancialEvent, timestamp_to_float
from dyfo.core.graph_builder import GraphBuilder
from dyfo.core.link_prediction import (
    LinkPredictor, build_link_labels, compute_metrics,
    CorrelationRegressor, build_regression_labels, compute_regression_metrics,
)
from dyfo.core.node_features import NodeFeatureBuilder
from dyfo.data.fred_adapter import detect_macro_events, download_fred_series
from dyfo.data.yfinance_adapter import (
    download_ohlcv,
    download_prices,
    get_corporate_actions,
    get_earnings_dates,
    get_ticker_info,
)
from dyfo.logging_utils import ResultLogger, setup_logging


# ---------------------------------------------------------------------------
# Data preparation (reused from test_real_data.py, refactored)
# ---------------------------------------------------------------------------

def prepare_data(
    tickers: List[str],
    start: str,
    end: str,
    benchmark: str,
    config: DyFOConfig,
    data_config: DataConfig,
    logger,
) -> dict:
    """Download and prepare all data needed for training."""

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    logger.info("Downloading prices for %d tickers...", len(tickers))
    prices = download_prices(tickers, start, end)

    logger.info("Downloading OHLCV...")
    ohlcv = download_ohlcv(tickers, start, end)
    volumes = pd.DataFrame({t: ohlcv[t]["Volume"] for t in tickers if t in ohlcv})
    volumes = volumes.reindex(prices.index)

    logger.info("Downloading benchmark (%s)...", benchmark)
    bench = download_prices([benchmark], start, end)
    bench_series = bench[benchmark] if benchmark in bench.columns else None

    logger.info("Fetching ticker info...")
    ticker_info = get_ticker_info(tickers)

    logger.info("Fetching earnings dates...")
    earnings_df = get_earnings_dates(tickers, start, end)

    logger.info("Fetching corporate actions...")
    actions_df = get_corporate_actions(tickers, start, end)

    logger.info("Downloading FRED macro series...")
    from dotenv import load_dotenv
    load_dotenv()
    fred_key = os.environ.get("FRED_API_KEY", "")
    macro_df = download_fred_series(data_config.fred_series, start, end, api_key=fred_key)
    macro_events_df = detect_macro_events(macro_df, threshold_std=1.5)

    # Node features
    nf_builder = NodeFeatureBuilder(
        tickers=tickers,
        ticker_to_idx=ticker_to_idx,
        gics_sectors=data_config.gics_sectors,
        num_regimes=config.num_regimes,
    )
    node_features_by_date = nf_builder.build_daily_features(
        prices=prices, volumes=volumes, benchmark_prices=bench_series, ticker_info=ticker_info,
    )

    # Edges
    sector_edges = build_sector_edges(ticker_info, ticker_to_idx)

    # Correlation method: DCC-GARCH (Engle 2002) or rolling Pearson
    use_dcc = config.correlation_method == "dcc_garch"
    if use_dcc:
        logger.info("Computing DCC-GARCH correlations...")
        # Compute full (unsparsified) DCC correlations once — expensive
        corr_series_all, corr_pairs_all = compute_dcc_garch_correlations(
            prices, window=config.dcc_garch_window, threshold=0.0,
        )
        # Sparsified version for CORRELATION_UPDATE events
        corr_series = corr_series_all.copy()
        for col in corr_series.columns:
            mask = corr_series[col].abs() < config.corr_sparsify_threshold
            corr_series.loc[mask, col] = np.nan
        corr_series = corr_series.dropna(axis=1, how="all")
        corr_pairs = [
            p for p in corr_pairs_all if f"{p[0]}_{p[1]}" in corr_series.columns
        ]
    else:
        logger.info("Computing rolling Pearson correlations...")
        corr_series, corr_pairs = compute_rolling_correlations(
            prices, window=config.rolling_corr_window,
            threshold=config.corr_sparsify_threshold,
        )

    # Event stream
    builder = GraphBuilder(config=config, tickers=tickers)
    graph = builder.build_initial_graph(sector_edges=sector_edges, supply_chain_edges=[], factor_edges=[])

    esb = EventStreamBuilder(ticker_to_idx)
    price_events = esb.build_price_events(prices, volumes)
    earnings_events = esb.build_earnings_events(earnings_df)
    action_events = esb.build_corp_action_events(actions_df)
    macro_events = esb.build_macro_events(macro_events_df, len(tickers))
    corr_events = esb.build_correlation_events(corr_series, corr_pairs)
    all_events = EventStreamBuilder.merge_and_sort(
        price_events, earnings_events, action_events, macro_events, corr_events,
    )

    # Group events by date key
    events_by_date: Dict[int, List[FinancialEvent]] = defaultdict(list)
    for ev in all_events:
        events_by_date[int(ev.timestamp)].append(ev)

    # Build daily correlation dicts for labels: date_key -> {(i,j): rho}
    # Use sparsified correlations for classification labels (backward compat)
    corr_by_date: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    for ev in corr_events:
        date_key = int(ev.timestamp)
        i, j = ev.source_node, ev.target_node
        rho = ev.features[0].item()
        corr_by_date[date_key][(i, j)] = rho
        corr_by_date[date_key][(j, i)] = rho

    # Unsparsified correlations for regression labels (continuous ρ prediction)
    if not use_dcc:
        # Rolling Pearson: need to compute unsparsified version separately
        corr_series_all, corr_pairs_all = compute_rolling_correlations(
            prices, window=config.rolling_corr_window, threshold=0.0,
        )
    # else: corr_series_all / corr_pairs_all already computed above
    corr_labels_by_date: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    for date in corr_series_all.index:
        date_key = int(timestamp_to_float(pd.Timestamp(date)))
        for pair_col, (tk_i, tk_j) in zip(corr_series_all.columns, corr_pairs_all):
            rho = corr_series_all.at[date, pair_col]
            if pd.isna(rho):
                continue
            idx_i = ticker_to_idx.get(tk_i)
            idx_j = ticker_to_idx.get(tk_j)
            if idx_i is not None and idx_j is not None:
                corr_labels_by_date[date_key][(idx_i, idx_j)] = rho
                corr_labels_by_date[date_key][(idx_j, idx_i)] = rho

    logger.info(
        "Data prepared: %d events, %d dates with correlations (sparsified), %d dates with all correlations (regression)",
        len(all_events), len(corr_by_date), len(corr_labels_by_date),
    )

    return {
        "prices": prices,
        "ticker_to_idx": ticker_to_idx,
        "ticker_info": ticker_info,
        "node_features_by_date": node_features_by_date,
        "node_feature_dim": nf_builder.feature_dim,
        "graph": graph,
        "events_by_date": dict(events_by_date),
        "corr_by_date": dict(corr_by_date),
        "corr_labels_by_date": dict(corr_labels_by_date),
        "sorted_dates": sorted(events_by_date.keys()),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_link_prediction(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    benchmark: str = "SPY",
    num_epochs: int = 5,
    lr: float = 1e-3,
    corr_threshold: float = 0.3,
    neg_ratio: float = 1.0,
    early_stopping_patience: int = 5,
    weight_decay: float = 1e-4,
    pos_weight: float = 0.5,
    mode: str = "regression",
):
    """Full training pipeline for link prediction pre-training."""

    run_tag = f"link_pred_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging("dyfo", run_tag=run_tag)
    results = ResultLogger(run_tag=run_tag)

    config = DyFOConfig()
    data_config = DataConfig(tickers=tickers, benchmark_ticker=benchmark, start_date=start, end_date=end)

    results.log_params({
        "task": "link_prediction",
        "mode": mode,
        "correlation_method": config.correlation_method,
        "tickers": tickers,
        "start": start,
        "end": end,
        "num_epochs": num_epochs,
        "lr": lr,
        "corr_threshold": corr_threshold,
        "neg_ratio": neg_ratio,
        "weight_decay": weight_decay,
        "pos_weight": pos_weight,
        "memory_dim": config.memory_dim,
        "embedding_dim": config.embedding_dim,
    })

    # ------------------------------------------------------------------
    # 1. Prepare data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Preparing data...")
    logger.info("=" * 60)
    data = prepare_data(tickers, start, end, benchmark, config, data_config, logger)

    sorted_dates = data["sorted_dates"]
    num_nodes = len(tickers)

    # Walk-forward split: 60/20/20
    n = len(sorted_dates)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_dates = sorted_dates[:train_end]
    val_dates = sorted_dates[train_end:val_end]
    test_dates = sorted_dates[val_end:]

    logger.info("Walk-forward split: train=%d, val=%d, test=%d days", len(train_dates), len(val_dates), len(test_dates))

    results.log_metrics({
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
    })

    # ------------------------------------------------------------------
    # 2. Initialize model
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Initializing model...")
    logger.info("=" * 60)

    module = DyFOModule(config=config, num_nodes=num_nodes, readout_strategy="mean")

    # Decoder: regression (predict rho) or classification (predict edge)
    is_regression = mode == "regression"
    if is_regression:
        decoder = CorrelationRegressor(embedding_dim=config.embedding_dim, hidden_dim=64, dropout=config.dropout)
        loss_fn = nn.SmoothL1Loss()  # Huber loss — robust to outlier correlations
        logger.info("Mode: REGRESSION (predict continuous rho, Huber loss)")
    else:
        decoder = LinkPredictor(embedding_dim=config.embedding_dim, hidden_dim=64, dropout=config.dropout)
        logger.info("Mode: CLASSIFICATION (predict binary edge, BCE loss)")

    # Optimise TGN encoder + decoder jointly
    all_params = list(module.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    # Static graph info
    edge_index = data["graph"].get_full_edge_index()
    edge_type_ids = data["graph"].get_edge_type_ids()
    edge_timestamps = torch.zeros(edge_index.shape[1])

    total_params = sum(p.numel() for p in all_params)
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
    logger.info("  Total parameters: %d", total_params)
    logger.info("  Trainable parameters: %d", trainable_params)
    results.log_metric("total_params", total_params)

    # ------------------------------------------------------------------
    # Helper: get node features for a date
    # ------------------------------------------------------------------
    nf_dates = sorted(data["node_features_by_date"].keys())

    def get_node_features(date_key: int) -> torch.Tensor:
        closest = nf_dates[0]
        for d in nf_dates:
            if d <= str(date_key):
                closest = d
            else:
                break
        return data["node_features_by_date"][closest]

    # ------------------------------------------------------------------
    # Helper: run one split (train, val, or test)
    # ------------------------------------------------------------------
    def run_split(
        dates: List[int],
        split_name: str,
        train_mode: bool = False,
        collect_predictions: bool = False,
        logit_threshold: float = 0.0,
    ) -> dict:
        if train_mode:
            module.train()
            decoder.train()
        else:
            module.eval()
            decoder.eval()

        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        num_batches = 0
        all_preds = []
        all_targets = []

        for d_idx in range(len(dates) - 1):
            today = dates[d_idx]
            tomorrow = dates[d_idx + 1]

            # Get events for today
            day_events = data["events_by_date"].get(today, [])
            node_feat = get_node_features(today)
            current_time = float(today) + 0.99

            # Get correlation labels — use mode-appropriate source
            if is_regression:
                corr_tomorrow = data["corr_labels_by_date"].get(tomorrow, {})
            else:
                corr_tomorrow = data["corr_by_date"].get(tomorrow, {})

            if not corr_tomorrow:
                with torch.no_grad():
                    module.process_day_events(day_events)
                continue

            # Build labels
            if is_regression:
                src, dst, targets = build_regression_labels(
                    corr_tomorrow, num_nodes,
                )
            else:
                corr_today = data["corr_by_date"].get(today, {})
                src, dst, targets = build_link_labels(
                    corr_today, corr_tomorrow, num_nodes, corr_threshold, neg_ratio,
                )

            if len(src) == 0:
                with torch.no_grad():
                    module.process_day_events(day_events)
                continue

            # Forward pass
            if train_mode:
                module.process_day_events(day_events)
                z = module.encoder.compute_embeddings(
                    node_features=node_feat,
                    edge_index=edge_index,
                    edge_type_ids=edge_type_ids,
                    edge_timestamps=edge_timestamps,
                    current_time=current_time,
                )
                preds = decoder(z[src], z[dst])
                if is_regression:
                    loss = loss_fn(preds, targets)
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        preds, targets, pos_weight=torch.tensor(pos_weight),
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                module.encoder.memory = module.encoder.memory.detach()
            else:
                with torch.no_grad():
                    module.process_day_events(day_events)
                    z = module.encoder.compute_embeddings(
                        node_features=node_feat,
                        edge_index=edge_index,
                        edge_type_ids=edge_type_ids,
                        edge_timestamps=edge_timestamps,
                        current_time=current_time,
                    )
                    preds = decoder(z[src], z[dst])

            if is_regression:
                metrics = compute_regression_metrics(preds.detach(), targets)
            else:
                metrics = compute_metrics(preds.detach(), targets, threshold=logit_threshold)
            epoch_loss += metrics["loss"]
            for k, v in metrics.items():
                epoch_metrics[k] += v
            num_batches += 1
            if collect_predictions:
                all_preds.append(preds.detach())
                all_targets.append(targets)

        # Average metrics
        if num_batches > 0:
            avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        else:
            if is_regression:
                avg_metrics = {"loss": 0, "mae": 0, "r_squared": 0, "spearman": 0,
                               "cls_accuracy": 0, "cls_precision": 0, "cls_recall": 0, "cls_f1": 0}
            else:
                avg_metrics = {"loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

        if collect_predictions and all_preds:
            avg_metrics["_all_preds"] = torch.cat(all_preds)
            avg_metrics["_all_targets"] = torch.cat(all_targets)

        return avg_metrics

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training...")
    logger.info("=" * 60)

    best_val_score = -float("inf")  # higher is better: R² for regression, AUC for classification
    best_epoch = 0
    patience_counter = 0
    history = {"train": [], "val": []}
    score_key = "r_squared" if is_regression else "auc"

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Reset memory at start of each epoch
        module.reset_memory()

        # Train
        train_metrics = run_split(train_dates, "train", train_mode=True)

        # Save memory checkpoint before validation
        mem_ckpt = module.encoder.get_memory_checkpoint()

        # Validation (memory inherited from training — no reset per manual §5.3)
        val_metrics = run_split(val_dates, "val", train_mode=False)

        elapsed = time.time() - t0

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if is_regression:
            logger.info(
                "Epoch %d/%d [%.1fs] | Train: loss=%.4f R2=%.3f MAE=%.3f spearman=%.3f | Val: loss=%.4f R2=%.3f MAE=%.3f spearman=%.3f",
                epoch, num_epochs, elapsed,
                train_metrics["loss"], train_metrics["r_squared"], train_metrics["mae"], train_metrics["spearman"],
                val_metrics["loss"], val_metrics["r_squared"], val_metrics["mae"], val_metrics["spearman"],
            )
        else:
            logger.info(
                "Epoch %d/%d [%.1fs] | Train: loss=%.4f acc=%.3f auc=%.3f f1=%.3f | Val: loss=%.4f acc=%.3f auc=%.3f f1=%.3f",
                epoch, num_epochs, elapsed,
                train_metrics["loss"], train_metrics["accuracy"], train_metrics["auc"], train_metrics["f1"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["auc"], val_metrics["f1"],
            )

        # Track best
        val_score = val_metrics[score_key]
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "mode": mode,
                "module_state": module.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, results.run_dir / "best_model.pt")
            logger.info("  -> New best model saved (val %s=%.4f)", score_key, best_val_score)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, early_stopping_patience)
                break

        # Restore memory to post-training state for next epoch's reset
        module.encoder.load_memory_checkpoint(mem_ckpt)

    logger.info("Best epoch: %d (val %s=%.4f)", best_epoch, score_key, best_val_score)

    # ------------------------------------------------------------------
    # 4. Test evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Test evaluation...")
    logger.info("=" * 60)

    # Load best model
    ckpt = torch.load(results.run_dir / "best_model.pt", weights_only=False)
    module.load_state_dict(ckpt["module_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    # Reset and replay train+val to get memory state
    module.reset_memory()
    _ = run_split(train_dates, "train_replay", train_mode=False)
    val_replay = run_split(val_dates, "val_replay", train_mode=False, collect_predictions=True)

    # ------------------------------------------------------------------
    # 4b. Threshold tuning (classification mode only)
    # ------------------------------------------------------------------
    best_threshold = 0.0
    if not is_regression and "_all_preds" in val_replay:
        val_logits = val_replay["_all_preds"]
        val_labels = val_replay["_all_targets"]
        best_f1_thresh = 0.0
        for t in torch.linspace(-2.0, 2.0, 41):
            t_val = t.item()
            m = compute_metrics(val_logits, val_labels, threshold=t_val)
            if m["f1"] > best_f1_thresh:
                best_f1_thresh = m["f1"]
                best_threshold = t_val
        logger.info("Optimal logit threshold: %.2f (val F1=%.4f)", best_threshold, best_f1_thresh)

    # Test with inherited memory
    test_metrics = run_split(test_dates, "test", train_mode=False, logit_threshold=best_threshold)

    if is_regression:
        logger.info(
            "Test: loss=%.4f MAE=%.3f R2=%.3f spearman=%.3f | cls_prec=%.3f cls_rec=%.3f cls_f1=%.3f",
            test_metrics["loss"], test_metrics["mae"],
            test_metrics["r_squared"], test_metrics["spearman"],
            test_metrics["cls_precision"], test_metrics["cls_recall"], test_metrics["cls_f1"],
        )
    else:
        logger.info(
            "Test: loss=%.4f acc=%.3f precision=%.3f recall=%.3f f1=%.3f auc=%.3f",
            test_metrics["loss"], test_metrics["accuracy"],
            test_metrics["precision"], test_metrics["recall"],
            test_metrics["f1"], test_metrics["auc"],
        )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results.log_metrics({
        "best_epoch": best_epoch,
        f"best_val_{score_key}": best_val_score,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    })

    # Save training history
    import json
    with open(results.run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    results_path = results.save()

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Mode:             %s", mode)
    logger.info("  Best epoch:       %d / %d", best_epoch, num_epochs)
    logger.info("  Best val %s:  %.4f", score_key, best_val_score)
    if is_regression:
        logger.info("  Test MSE:         %.4f", test_metrics["loss"])
        logger.info("  Test MAE:         %.4f", test_metrics["mae"])
        logger.info("  Test R²:          %.4f", test_metrics["r_squared"])
        logger.info("  Test Spearman:    %.4f", test_metrics["spearman"])
        logger.info("  Test cls F1(@0.5):%.4f", test_metrics["cls_f1"])
    else:
        logger.info("  Test AUC:         %.4f", test_metrics["auc"])
        logger.info("  Test F1:          %.4f", test_metrics["f1"])
        logger.info("  Test Accuracy:    %.4f", test_metrics["accuracy"])
    logger.info("  Results:          %s", results_path)
    logger.info("  Best model:       %s", results.run_dir / "best_model.pt")

    return test_metrics


if __name__ == "__main__":
    # 30 S&P 500 tickers by liquidity, covering all 11 GICS sectors (BL-01)
    # Tech(5): AAPL MSFT GOOGL NVDA AVGO | Fin(4): JPM GS MA BRK-B
    # Health(3): JNJ UNH LLY | Disc(3): AMZN TSLA HD | Staples(2): PG KO
    # Energy(2): XOM CVX | Industrials(3): CAT BA RTX | Comm(3): META GOOG DIS
    # Materials(2): LIN APD | Utilities(2): NEE DUK | Real Estate(1): PLD
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

    test_metrics = train_link_prediction(
        tickers=TICKERS_30,
        start="2020-01-01",
        end="2024-12-31",
        benchmark="SPY",
        num_epochs=10,
        lr=1e-3,
        corr_threshold=0.3,
        neg_ratio=1.0,
        early_stopping_patience=5,
        weight_decay=1e-4,
        pos_weight=1.0,
        mode="regression",
    )
