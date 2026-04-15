"""Training script — Self-supervised link prediction pre-training for DyFO.

Supports three encoder variants (--model_variant / model_variant= parameter):
  tgn         — Temporal Graph Network (original DyFO encoder, default)
  ra_htgn     — relation-aware heterogeneous TGN (BL-17)
  gat_static  — 2-layer GAT on a static mean-correlation graph (BL-02)
  roland      — ROLAND-like monthly snapshot GNN with EMA state (BL-02)
  temporal_kg — interpretable temporal KG ablation arm (BL-18)

All variants share the same decoder (CorrelationRegressor / LinkPredictor)
and the same walk-forward 60/20/20 evaluation protocol.

Walk-forward protocol
---------------------
  - Train : first 60 % of trading days
  - Val   : next  20 %
  - Test  : last  20 %
  - Memory is inherited across splits (not zeroed at split boundaries)

Running each variant
--------------------
  python scripts/train_link_prediction.py                    # tgn (default)
  python scripts/train_link_prediction.py --variant tgn
  python scripts/train_link_prediction.py --variant gat_static
  python scripts/train_link_prediction.py --variant roland

Or programmatically:
  train_link_prediction(..., model_variant="gat_static")
"""

from __future__ import annotations

import math
import os
import time
import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.edge_features import (
    build_sector_edges,
    compute_dcc_garch_correlations,
    compute_factor_edges,
    compute_rolling_correlations,
)

# Epoch used by timestamp_to_float (must match event_stream.py)
_EPOCH = datetime.date(2000, 1, 1)
from dyfo.core.event_stream import EventStreamBuilder, FinancialEvent, timestamp_to_float
from dyfo.core.graph_builder import GraphBuilder
from dyfo.core.link_prediction import (
    LinkPredictor, build_link_labels, compute_metrics,
    CorrelationRegressor, build_regression_labels, compute_regression_metrics,
)
from dyfo.core.model_variants import BaseGraphEncoder, build_encoder
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
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data preparation
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
    # Prioritize environment, then DataConfig
    fred_key = os.environ.get("FRED_API_KEY", data_config.fred_api_key)
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

    # Factor edges (Fama-French 5)
    from dyfo.data.ff_adapter import download_ff5_factors
    logger.info("Downloading Fama-French 5 factors...")
    ff5_factors = download_ff5_factors(start, end)
    if ff5_factors is not None:
        ff5_returns = ff5_factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        factor_edges = compute_factor_edges(
            prices, ff5_returns, ticker_to_idx,
            loading_window=252, threshold=0.5,
        )
    else:
        logger.warning("FF5 factors unavailable; skipping FACT edges")
        factor_edges = []

    # Correlation method: DCC-GARCH (Engle 2002) or rolling Pearson
    use_dcc = config.correlation_method == "dcc_garch"
    if use_dcc:
        logger.info("Computing DCC-GARCH correlations...")
        corr_series_all, corr_pairs_all = compute_dcc_garch_correlations(
            prices, window=config.dcc_garch_window, threshold=0.0,
        )
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
    graph = builder.build_initial_graph(sector_edges=sector_edges, supply_chain_edges=[], factor_edges=factor_edges)

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
    corr_by_date: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)
    for ev in corr_events:
        date_key = int(ev.timestamp)
        i, j = ev.source_node, ev.target_node
        rho = ev.features[0].item()
        corr_by_date[date_key][(i, j)] = rho
        corr_by_date[date_key][(j, i)] = rho

    # Unsparsified correlations for regression labels (continuous ρ prediction)
    if not use_dcc:
        corr_series_all, corr_pairs_all = compute_rolling_correlations(
            prices, window=config.rolling_corr_window, threshold=0.0,
        )
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
    model_variant: str = "tgn",
    seed: int = 42,
    prepared_data: dict = None,
    config: "DyFOConfig" = None,
    decoder_hidden_dim: int = 64,
    use_cosine_schedule: bool = False,
    test_dates: List[int] = None,
    val_dates: List[int] = None,
    train_dates: List[int] = None,
):
    """Full training pipeline for link prediction pre-training.

    Parameters
    ----------
    model_variant : str
        Encoder variant: ``"tgn"`` (default), ``"ra_htgn"``, ``"gat_static"``,
        ``"roland"``, or ``"temporal_kg"``.
    seed : int
        RNG seed for reproducibility.  Applied after data preparation so
        deterministic downloads do not consume RNG state.
    prepared_data : dict, optional
        Pre-loaded data dict from ``prepare_data()``.  Pass this when running
        multiple seeds to avoid re-downloading data for every run.
    config : DyFOConfig, optional
        Custom architecture config. If None, uses DyFOConfig defaults.
        Useful for tuning embedding_dim, memory_dim, num_attention_heads.
    decoder_hidden_dim : int
        Hidden layer width for the CorrelationRegressor / LinkPredictor MLP.
        Default 64; increase to 128 for more decoder capacity.
    use_cosine_schedule : bool
        If True, replaces the flat-after-warmup LR with cosine annealing
        (warmup 2 ep → cosine decay to 10 % of peak). Recommended for
        runs with num_epochs > 10.
    test_dates, val_dates, train_dates : List[int], optional
        Custom date lists for splits. If provided, the internal 60/20/20
        split logic is bypassed.
    """

    run_tag = f"link_pred_{model_variant}_s{seed}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging("dyfo", run_tag=run_tag)
    results = ResultLogger(run_tag=run_tag)

    if config is None:
        config = DyFOConfig(model_variant=model_variant)
    data_config = DataConfig(tickers=tickers, benchmark_ticker=benchmark, start_date=start, end_date=end)

    results.log_params({
        "task": "link_prediction",
        "mode": mode,
        "model_variant": model_variant,
        "seed": seed,
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
    # 1. Prepare data (skip if caller already provided it)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Preparing data...")
    logger.info("=" * 60)
    if prepared_data is not None:
        data = prepared_data
        logger.info("Using pre-loaded data (%d dates)", len(data["sorted_dates"]))
    else:
        data = prepare_data(tickers, start, end, benchmark, config, data_config, logger)

    # Seed AFTER data preparation — downloads are deterministic; only model
    # initialisation and dropout need to be seeded per run.
    set_seed(seed)
    logger.info("Seed set to %d (variant=%s)", seed, model_variant)

    sorted_dates = data["sorted_dates"]
    num_nodes = len(tickers)

    # Walk-forward split: 60/20/20 (or custom if provided)
    if train_dates is not None and val_dates is not None and test_dates is not None:
        logger.info("Using CUSTOM walk-forward splits provided by caller.")
    else:
        n = len(sorted_dates)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        train_dates = sorted_dates[:train_end]
        val_dates = sorted_dates[train_end:val_end]
        test_dates = sorted_dates[val_end:]

    logger.info(
        "Walk-forward split: train=%d, val=%d, test=%d days",
        len(train_dates), len(val_dates), len(test_dates),
    )

    results.log_metrics({
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates),
    })

    # ------------------------------------------------------------------
    # 2. Initialize encoder + decoder
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Initializing model (variant=%s)...", model_variant)
    logger.info("=" * 60)

    # Select compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build encoder via factory
    encoder: BaseGraphEncoder = build_encoder(config, num_nodes, variant=model_variant)
    encoder = encoder.to(device)

    # Variant-specific post-init setup
    if model_variant == "gat_static":
        from dyfo.core.gat_static_baseline import GATStaticEncoder
        assert isinstance(encoder, GATStaticEncoder)
        logger.info("Building static graph from training correlations...")
        encoder.set_static_graph_from_correlations(
            data["corr_labels_by_date"], train_dates
        )
        num_static_edges = encoder.static_edge_index.shape[1] // 2
        logger.info("  Static graph: %d undirected edges", num_static_edges)

    elif model_variant == "roland":
        from dyfo.core.roland_baseline import ROLANDLikeEncoder
        assert isinstance(encoder, ROLANDLikeEncoder)
        logger.info("Precomputing monthly snapshots (ROLAND-like)...")
        encoder.precompute_monthly_snapshots(data["corr_labels_by_date"])
        logger.info("  %d monthly snapshots available", len(encoder._monthly_snapshots))

    # Decoder: regression (predict rho) or classification (predict edge)
    is_regression = mode == "regression"
    if is_regression:
        decoder = CorrelationRegressor(
            embedding_dim=config.embedding_dim, hidden_dim=decoder_hidden_dim, dropout=config.dropout
        ).to(device)
        loss_fn = nn.SmoothL1Loss()
        logger.info("Mode: REGRESSION (predict continuous rho, Huber loss)")
    else:
        decoder = LinkPredictor(
            embedding_dim=config.embedding_dim, hidden_dim=decoder_hidden_dim, dropout=config.dropout
        ).to(device)
        logger.info("Mode: CLASSIFICATION (predict binary edge, BCE loss)")

    # Optimise encoder + decoder jointly
    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    # LR schedule: cosine annealing with warmup (recommended for long runs)
    # or plain linear warmup that plateaus (legacy default).
    warmup_epochs = min(2, num_epochs)
    if use_cosine_schedule and num_epochs > 4:
        def _cosine_lr(ep: int, _w: int = warmup_epochs, _n: int = num_epochs) -> float:
            if ep < _w:
                return (ep + 1) / max(1, _w)
            progress = (ep - _w) / max(1, _n - _w)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lr)
        logger.info("LR schedule: cosine annealing with %d-epoch warmup", warmup_epochs)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs),
        )
        logger.info("LR schedule: linear warmup (%d epochs), then flat", warmup_epochs)

    # Static graph info (used by TGN; ignored by GAT-Static and ROLAND)
    edge_index = data["graph"].get_full_edge_index().to(device)
    edge_type_ids = data["graph"].get_edge_type_ids().to(device)
    edge_timestamps = torch.zeros(edge_index.shape[1], device=device)

    total_params = sum(p.numel() for p in all_params)
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
    logger.info("  Total parameters:     %d", total_params)
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
    # Helper: Portfolio optimization (GMV)
    # ------------------------------------------------------------------
    def optimize_min_variance(cov_matrix: np.ndarray) -> np.ndarray:
        """Find Global Minimum Variance weights: min w' Sigma w s.t. sum(w)=1, w >= 0."""
        n = cov_matrix.shape[0]
        # Equal weight as initial guess
        init_w = np.ones(n) / n
        bounds = [(0, 1) for _ in range(n)]
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})

        def obj(w):
            return w.T @ cov_matrix @ w

        res = minimize(obj, init_w, bounds=bounds, constraints=cons, method="SLSQP", tol=1e-6)
        return res.x if res.success else init_w

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
            encoder.train()
            decoder.train()
        else:
            encoder.eval()
            decoder.eval()

        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        num_batches = 0
        all_preds = []
        all_targets = []
        realized_returns = []

        # For Sharpe calculation: get daily returns
        price_returns = data["prices"].pct_change().fillna(0)
        # 21-day rolling volatility for covariance reconstruction
        price_vols = price_returns.rolling(window=21).std() * np.sqrt(252)
        price_vols = price_vols.fillna(0.15)  # 15% default vol if too early

        for d_idx in range(len(dates) - 1):
            today = dates[d_idx]
            tomorrow = dates[d_idx + 1]

            day_events = data["events_by_date"].get(today, [])
            node_feat = get_node_features(today).to(device)
            current_time = float(today) + 0.99

            # Correlation labels — mode-appropriate source
            if is_regression:
                corr_tomorrow = data["corr_labels_by_date"].get(tomorrow, {})
            else:
                corr_tomorrow = data["corr_by_date"].get(tomorrow, {})

            if not corr_tomorrow:
                # Still advance temporal state even when no labels available
                with torch.no_grad():
                    encoder.advance_day(
                        day_events, node_feat,
                        edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                continue

            # Build labels
            if is_regression:
                src, dst, targets = build_regression_labels(corr_tomorrow, num_nodes)
            else:
                corr_today = data["corr_by_date"].get(today, {})
                src, dst, targets = build_link_labels(
                    corr_today, corr_tomorrow, num_nodes, corr_threshold, neg_ratio,
                )
            src = src.to(device)
            dst = dst.to(device)
            targets = targets.to(device)

            if len(src) == 0:
                with torch.no_grad():
                    encoder.advance_day(
                        day_events, node_feat,
                        edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                continue

            # Forward pass
            if train_mode:
                encoder.advance_day(
                    day_events, node_feat,
                    edge_index, edge_type_ids, edge_timestamps, current_time,
                )
                z = encoder.get_node_embeddings(
                    node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
                )
                preds = decoder(z[src], z[dst])
                if is_regression:
                    loss = loss_fn(preds, targets)
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        preds, targets, pos_weight=torch.tensor(pos_weight, device=device),
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                optimizer.step()
                # TBPTT: detach temporal state from computation graph
                encoder.detach_state()
            else:
                with torch.no_grad():
                    encoder.advance_day(
                        day_events, node_feat,
                        edge_index, edge_type_ids, edge_timestamps, current_time,
                    )
                    z = encoder.get_node_embeddings(
                        node_feat, edge_index, edge_type_ids, edge_timestamps, current_time,
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

            # --- Economic Utility (Sharpe Proxy) ---
            # Only for test split or specific validation requests
            if split_name == "test" and is_regression:
                with torch.no_grad():
                    # Preds: shape (num_pairs,) in [-1, 1]
                    # We need to build the full correlation matrix
                    corr_matrix = np.eye(num_nodes)
                    idx = 0
                    pairs_processed = set()
                    
                    # reconstruct indices from BuildRegressionLabels logic (sorted pairs)
                    # For simplicity, we assume preds correspond to the pairs in tomorrow's dict
                    p_np = preds.detach().cpu().numpy()
                    
                    # We need the original pair mapping
                    # Since we don't have it explicitly, we recreate the build logic here
                    seen = set()
                    pair_list = []
                    for (i, j) in corr_tomorrow.keys():
                        pair = (min(i, j), max(i, j))
                        if pair not in seen:
                            seen.add(pair)
                            pair_list.append(pair)
                    
                    if len(p_np) == len(pair_list):
                        for p_idx, (i, j) in enumerate(pair_list):
                            rho = p_np[p_idx]
                            corr_matrix[i, j] = rho
                            corr_matrix[j, i] = rho
                        
                        # Build Covariance Matrix
                        date_str = str(_EPOCH + datetime.timedelta(days=tomorrow))
                        # Use today's vol to predict tomorrow's risk
                        vols = price_vols.loc[date_str].values if date_str in price_vols.index else np.full(num_nodes, 0.15)
                        cov = np.diag(vols) @ corr_matrix @ np.diag(vols)
                        
                        # Add small regularization for stability
                        cov += np.eye(num_nodes) * 1e-4
                        
                        # Optimize weights
                        weights = optimize_min_variance(cov)
                        
                        # Realized return tomorrow
                        # Convert tomorrow (float days since 2000) back to price index
                        day_returns = price_returns.loc[date_str].values if date_str in price_returns.index else np.zeros(num_nodes)
                        realized_ret = np.dot(weights, day_returns)
                        realized_returns.append(realized_ret)

        # Average metrics
        if num_batches > 0:
            avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
            
            # --- Aggregated Sharpe ---
            if realized_returns:
                rets = np.array(realized_returns)
                if len(rets) > 1 and rets.std() > 0:
                    sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
                else:
                    sharpe = 0.0
                avg_metrics["sharpe_proxy"] = sharpe
        else:
            if is_regression:
                avg_metrics = {
                    "loss": 0, "mae": 0, "r_squared": 0, "spearman": 0,
                    "cls_accuracy": 0, "cls_precision": 0, "cls_recall": 0, "cls_f1": 0,
                }
            else:
                avg_metrics = {
                    "loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0,
                }

        if collect_predictions and all_preds:
            avg_metrics["_all_preds"] = torch.cat(all_preds)
            avg_metrics["_all_targets"] = torch.cat(all_targets)
            if realized_returns:
                avg_metrics["_realized_returns"] = realized_returns

        return avg_metrics

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training...")
    logger.info("=" * 60)

    best_val_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train": [], "val": []}
    score_key = "r_squared" if is_regression else "auc"

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Reset temporal state at start of each epoch
        encoder.reset_state()

        # Train
        train_metrics = run_split(train_dates, "train", train_mode=True)

        # Validation (memory/state inherited from training — no reset per manual §5.3)
        val_metrics = run_split(val_dates, "val", train_mode=False)

        elapsed = time.time() - t0

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if is_regression:
            logger.info(
                "Epoch %d/%d [%.1fs] | Train: loss=%.4f R2=%.3f MAE=%.3f spearman=%.3f"
                " | Val: loss=%.4f R2=%.3f MAE=%.3f spearman=%.3f",
                epoch, num_epochs, elapsed,
                train_metrics["loss"], train_metrics["r_squared"],
                train_metrics["mae"], train_metrics["spearman"],
                val_metrics["loss"], val_metrics["r_squared"],
                val_metrics["mae"], val_metrics["spearman"],
            )
        else:
            logger.info(
                "Epoch %d/%d [%.1fs] | Train: loss=%.4f acc=%.3f auc=%.3f f1=%.3f"
                " | Val: loss=%.4f acc=%.3f auc=%.3f f1=%.3f",
                epoch, num_epochs, elapsed,
                train_metrics["loss"], train_metrics["accuracy"],
                train_metrics["auc"], train_metrics["f1"],
                val_metrics["loss"], val_metrics["accuracy"],
                val_metrics["auc"], val_metrics["f1"],
            )

        # Track best
        val_score = val_metrics[score_key]
        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "mode": mode,
                "model_variant": model_variant,
                "encoder_state": encoder.state_dict(),
                "decoder_state": decoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, results.run_dir / "best_model.pt")
            logger.info("  -> New best model saved (val %s=%.4f)", score_key, best_val_score)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, early_stopping_patience,
                )
                break

        scheduler.step()
        logger.info("  LR after epoch %d: %.2e", epoch, scheduler.get_last_lr()[0])

    logger.info("Best epoch: %d (val %s=%.4f)", best_epoch, score_key, best_val_score)

    # ------------------------------------------------------------------
    # 4. Test evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Test evaluation...")
    logger.info("=" * 60)

    # Load best model
    ckpt = torch.load(results.run_dir / "best_model.pt", weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    # Reset and replay train+val to get correct temporal state at test time
    encoder.reset_state()
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
        logger.info(
            "Optimal logit threshold: %.2f (val F1=%.4f)", best_threshold, best_f1_thresh
        )

    # Test with inherited temporal state
    test_metrics = run_split(
        test_dates, "test", train_mode=False, logit_threshold=best_threshold,
        collect_predictions=True
    )
    if model_variant == "temporal_kg" and hasattr(encoder, "export_temporal_kg_artifacts"):
        test_metrics["_temporal_kg_artifacts"] = encoder.export_temporal_kg_artifacts(top_k=10)

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
        **{f"test_{k}": v for k, v in test_metrics.items() if not k.startswith("_")},
    })

    import json
    with open(results.run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    results_path = results.save()

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Variant:          %s", model_variant)
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

    test_metrics["_best_val_r_squared"] = best_val_score
    test_metrics["_best_epoch"] = best_epoch
    return test_metrics


if __name__ == "__main__":
    import argparse

    # 30 S&P 500 tickers by liquidity, covering all 11 GICS sectors (BL-01)
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

    parser = argparse.ArgumentParser(description="DyFO link prediction pre-training")
    parser.add_argument(
        "--variant",
        choices=["tgn", "ra_htgn", "gat_static", "roland", "temporal_kg"],
        default="tgn",
        help="Encoder variant (default: tgn)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_metrics = train_link_prediction(
        tickers=TICKERS_30,
        start="2020-01-01",
        end="2024-12-31",
        benchmark="SPY",
        num_epochs=args.epochs,
        lr=args.lr,
        corr_threshold=0.3,
        neg_ratio=1.0,
        early_stopping_patience=5,
        weight_decay=1e-4,
        pos_weight=1.0,
        mode="regression",
        model_variant=args.variant,
        seed=args.seed,
    )
