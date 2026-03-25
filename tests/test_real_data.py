"""Integration test — runs the full DyFO pipeline with real market data.

Downloads data from yfinance + FRED, builds the event stream,
processes through TGN, and logs all results.
"""

from __future__ import annotations

import time

import pandas as pd
import torch

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.dyfo_module import DyFOModule
from dyfo.core.edge_features import build_sector_edges, compute_rolling_correlations
from dyfo.core.event_stream import EventStreamBuilder
from dyfo.core.graph_builder import GraphBuilder
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


def main():
    run_tag = "real_data_test"
    logger = setup_logging("dyfo", run_tag=run_tag)
    results = ResultLogger(run_tag=run_tag)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "MA"]
    START = "2023-01-01"
    END = "2024-12-31"
    BENCHMARK = "SPY"

    config = DyFOConfig()
    data_config = DataConfig(
        tickers=TICKERS,
        benchmark_ticker=BENCHMARK,
        start_date=START,
        end_date=END,
    )

    results.log_params({
        "tickers": TICKERS,
        "benchmark": BENCHMARK,
        "start": START,
        "end": END,
        "memory_dim": config.memory_dim,
        "embedding_dim": config.embedding_dim,
        "num_heads": config.num_attention_heads,
        "num_neighbors": config.num_neighbors,
    })

    ticker_to_idx = {t: i for i, t in enumerate(TICKERS)}
    num_nodes = len(TICKERS)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading market data")
    logger.info("=" * 60)

    t0 = time.time()

    logger.info("Downloading prices for %d tickers...", len(TICKERS))
    prices = download_prices(TICKERS, START, END)
    logger.info("  Prices shape: %s, date range: %s to %s", prices.shape, prices.index[0].date(), prices.index[-1].date())

    logger.info("Downloading OHLCV...")
    ohlcv = download_ohlcv(TICKERS, START, END)
    # Extract volumes
    volumes = pd.DataFrame({t: ohlcv[t]["Volume"] for t in TICKERS if t in ohlcv})
    volumes = volumes.reindex(prices.index)
    logger.info("  Volumes shape: %s", volumes.shape)

    logger.info("Downloading benchmark (%s)...", BENCHMARK)
    bench_prices = download_prices([BENCHMARK], START, END)
    bench_series = bench_prices[BENCHMARK] if BENCHMARK in bench_prices.columns else None

    logger.info("Fetching ticker info (sector, market_cap, beta)...")
    ticker_info = get_ticker_info(TICKERS)
    for t, info in ticker_info.items():
        logger.info("  %s: sector=%s, mcap=%s, beta=%s", t, info["sector"], info.get("market_cap"), info.get("beta"))

    logger.info("Fetching earnings dates...")
    earnings_df = get_earnings_dates(TICKERS, START, END)
    logger.info("  Earnings events: %d", len(earnings_df))

    logger.info("Fetching corporate actions...")
    actions_df = get_corporate_actions(TICKERS, START, END)
    logger.info("  Corporate action events: %d", len(actions_df))

    logger.info("Downloading FRED macro series...")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    fred_key = os.environ.get("FRED_API_KEY", "")
    macro_df = download_fred_series(data_config.fred_series, START, END, api_key=fred_key)
    logger.info("  Macro series shape: %s", macro_df.shape)
    macro_events_df = detect_macro_events(macro_df, threshold_std=1.5)
    logger.info("  Significant macro events detected: %d", len(macro_events_df))

    download_time = time.time() - t0
    logger.info("Data download completed in %.1f seconds", download_time)
    results.log_metric("download_time_sec", round(download_time, 1))
    results.log_metric("num_tickers", len(TICKERS))
    results.log_metric("price_rows", len(prices))
    results.log_metric("earnings_events_raw", len(earnings_df))
    results.log_metric("corp_actions_raw", len(actions_df))
    results.log_metric("macro_events_detected", len(macro_events_df))

    # ------------------------------------------------------------------
    # 2. Build features and edges
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Building features and edges")
    logger.info("=" * 60)

    t0 = time.time()

    # Node features
    nf_builder = NodeFeatureBuilder(
        tickers=TICKERS,
        ticker_to_idx=ticker_to_idx,
        gics_sectors=data_config.gics_sectors,
        num_regimes=config.num_regimes,
    )
    node_features_by_date = nf_builder.build_daily_features(
        prices=prices,
        volumes=volumes,
        benchmark_prices=bench_series,
        ticker_info=ticker_info,
    )
    logger.info("  Node features: %d dates, dim=%d", len(node_features_by_date), nf_builder.feature_dim)

    # Sector edges
    sector_edges = build_sector_edges(ticker_info, ticker_to_idx)
    logger.info("  Sector edges: %d", len(sector_edges))

    # Rolling correlations
    corr_series, corr_pairs = compute_rolling_correlations(
        prices, window=63, threshold=config.corr_sparsify_threshold,
    )
    logger.info("  Correlation pairs (surviving sparsification): %d", len(corr_pairs))

    feat_time = time.time() - t0
    logger.info("Feature/edge building completed in %.1f seconds", feat_time)
    results.log_metric("feature_build_time_sec", round(feat_time, 1))
    results.log_metric("node_feature_dim", nf_builder.feature_dim)
    results.log_metric("sector_edges", len(sector_edges))
    results.log_metric("corr_pairs", len(corr_pairs))

    # ------------------------------------------------------------------
    # 3. Build event stream
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Building event stream")
    logger.info("=" * 60)

    t0 = time.time()

    builder = GraphBuilder(config=config, tickers=TICKERS)
    graph = builder.build_initial_graph(
        sector_edges=sector_edges,
        supply_chain_edges=[],  # no SUPL data for now
        factor_edges=[],        # no FF5 data for now
    )

    esb = EventStreamBuilder(ticker_to_idx)
    price_events = esb.build_price_events(prices, volumes)
    earnings_events = esb.build_earnings_events(earnings_df)
    action_events = esb.build_corp_action_events(actions_df)
    macro_events = esb.build_macro_events(macro_events_df, num_nodes)
    corr_events = esb.build_correlation_events(corr_series, corr_pairs)

    all_events = EventStreamBuilder.merge_and_sort(
        price_events, earnings_events, action_events, macro_events, corr_events,
    )

    event_time = time.time() - t0
    logger.info("Event stream built in %.1f seconds", event_time)
    logger.info("  Total events: %d", len(all_events))
    logger.info("    PRICE_UPDATE:       %d", len(price_events))
    logger.info("    EARNINGS_REPORT:    %d", len(earnings_events))
    logger.info("    CORP_ACTION:        %d", len(action_events))
    logger.info("    MACRO (broadcast):  %d", len(macro_events))
    logger.info("    CORRELATION_UPDATE: %d", len(corr_events))

    results.log_metrics({
        "event_stream_build_time_sec": round(event_time, 1),
        "total_events": len(all_events),
        "price_events": len(price_events),
        "earnings_events": len(earnings_events),
        "corp_action_events": len(action_events),
        "macro_events": len(macro_events),
        "corr_events": len(corr_events),
    })

    # ------------------------------------------------------------------
    # 4. Run TGN forward pass
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Running TGN forward pass (day by day)")
    logger.info("=" * 60)

    t0 = time.time()

    module = DyFOModule(config=config, num_nodes=num_nodes, readout_strategy="mean")
    module.eval()

    # Prepare static graph tensors
    edge_index = graph.get_full_edge_index()
    edge_type_ids = graph.get_edge_type_ids()
    edge_timestamps = torch.zeros(edge_index.shape[1])

    # Group events by date
    from collections import defaultdict
    events_by_date = defaultdict(list)
    for ev in all_events:
        # Convert timestamp back to date string for grouping
        date_approx = int(ev.timestamp)
        events_by_date[date_approx].append(ev)

    sorted_dates = sorted(events_by_date.keys())
    logger.info("  Processing %d unique timestamps...", len(sorted_dates))

    embeddings = []
    embedding_dates = []

    with torch.no_grad():
        for i, date_key in enumerate(sorted_dates):
            day_events = events_by_date[date_key]

            # Get node features for this day (use latest available)
            # Find closest date key in node_features_by_date
            nf_dates = sorted(node_features_by_date.keys())
            closest_nf_date = nf_dates[0]
            for nf_d in nf_dates:
                if nf_d <= str(date_key):
                    closest_nf_date = nf_d
                else:
                    break
            node_feat = node_features_by_date[closest_nf_date]

            current_time = float(date_key) + 0.99  # end of day

            e_t = module(
                events=day_events,
                node_features=node_feat,
                edge_index=edge_index,
                edge_type_ids=edge_type_ids,
                edge_timestamps=edge_timestamps,
                current_time=current_time,
            )
            embeddings.append(e_t.clone())
            embedding_dates.append(date_key)

            if (i + 1) % 100 == 0 or i == 0 or i == len(sorted_dates) - 1:
                logger.info(
                    "  Day %d/%d | events=%d | e_t norm=%.4f | stale_nodes=%d",
                    i + 1,
                    len(sorted_dates),
                    len(day_events),
                    e_t.norm().item(),
                    len(module.get_stale_nodes()),
                )

    forward_time = time.time() - t0
    logger.info("TGN forward pass completed in %.1f seconds", forward_time)

    # Stack embeddings
    embedding_matrix = torch.stack(embeddings)  # (T, d)
    logger.info("  Embedding matrix shape: %s", embedding_matrix.shape)

    results.log_metrics({
        "forward_pass_time_sec": round(forward_time, 1),
        "num_days_processed": len(sorted_dates),
        "embedding_shape": list(embedding_matrix.shape),
        "embedding_norm_mean": round(embedding_matrix.norm(dim=1).mean().item(), 4),
        "embedding_norm_std": round(embedding_matrix.norm(dim=1).std().item(), 4),
        "embedding_norm_min": round(embedding_matrix.norm(dim=1).min().item(), 4),
        "embedding_norm_max": round(embedding_matrix.norm(dim=1).max().item(), 4),
    })

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Saving results")
    logger.info("=" * 60)

    results.log_tensor("embedding_matrix", embedding_matrix)
    results.log_tensor("memory_final", module.encoder.memory.clone())

    # Save embedding dates mapping
    import json
    dates_path = results.run_dir / "embedding_dates.json"
    with open(dates_path, "w") as f:
        json.dump(embedding_dates, f)

    results_path = results.save()
    logger.info("Results saved to: %s", results_path)
    logger.info("Artifacts in: %s", results.run_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("  Tickers:          %s", TICKERS)
    logger.info("  Date range:       %s to %s", START, END)
    logger.info("  Total events:     %d", len(all_events))
    logger.info("  Days processed:   %d", len(sorted_dates))
    logger.info("  Embedding dim:    %d", config.embedding_dim)
    logger.info("  e_t norm (mean):  %.4f", embedding_matrix.norm(dim=1).mean().item())
    logger.info("  Total time:       %.1f sec", download_time + feat_time + event_time + forward_time)
    logger.info("")
    logger.info("Log file:    logs/%s.log", run_tag)
    logger.info("Results:     results/%s/results.json", run_tag)
    logger.info("Embeddings:  results/%s/embedding_matrix.pt", run_tag)


if __name__ == "__main__":
    main()
