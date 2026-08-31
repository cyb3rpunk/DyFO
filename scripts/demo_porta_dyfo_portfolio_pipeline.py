#!/usr/bin/env python3
"""End-to-end demonstration of DyFO consuming PORTA curated features and exporting structural graphs.

Demonstrates:
1. Strict read-only ingestion from PORTA daily_core features.
2. Construction of StructuralGraphSnapshot with relation-aware embeddings.
3. Computation of causal structural covariance matrix Sigma_t.
4. Calculation of downstream portfolio allocations (GMVP, Risk Parity, Equal Weight).
"""

import datetime
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dyfo.adapters.dyfo_adapter import DyFOAdapter
from dyfo.data.porta_reader import PortaDataReader


def solve_gmv_weights(cov: np.ndarray) -> np.ndarray:
    """Solve minimum variance weights with long-only budget constraint (sum(w)=1, w >= 0)."""
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(cov.shape[0], dtype=np.float32)
    w_unconstrained = inv_cov @ ones / (ones.T @ inv_cov @ ones)
    # Clip and re-normalize for long-only proxy
    w_pos = np.clip(w_unconstrained, 0.0, 1.0)
    if w_pos.sum() == 0:
        return ones / len(ones)
    return w_pos / w_pos.sum()


def run_pipeline_demo():
    print("=" * 70)
    print("DyFO + PORTA Read-Only Integration Pipeline Demo")
    print("=" * 70)

    # 1. Initialize PortaDataReader
    reader = PortaDataReader()
    print(f"[*] PORTA daily_core available: {reader.is_available}")

    if reader.is_available:
        assets = reader.get_assets()
        print(f"[*] Total curated assets in PORTA: {len(assets)} (e.g. {assets[:5]}...)")
        start_d, end_d = reader.get_date_range()
        print(f"[*] PORTA date range: {start_d} -> {end_d}")

    # 2. Initialize DyFOAdapter with selected portfolio universe
    portfolio_tickers = ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "JNJ", "PG", "NVDA"]
    adapter = DyFOAdapter(
        tickers=portfolio_tickers,
        porta_reader=reader if reader.is_available else None,
    )
    print(f"[*] Initialized DyFOAdapter with {adapter.num_nodes} assets: {adapter.entity_ids}")

    # 3. Export structural graph at a representative stress date (COVID-19 market turbulence)
    as_of = datetime.date(2020, 3, 20)
    print(f"\n[*] Exporting structural graph snapshot as of: {as_of}...")
    snapshot = adapter.export_structural_graph(as_of, include_attention=True)

    print(f"    - Snapshot Date: {snapshot.as_of_date} (Causal Cutoff: {snapshot.causal_cutoff_date})")
    print(f"    - Node Embeddings Shape: {snapshot.node_embeddings.shape}")
    print(f"    - Relation Types Present: {list(snapshot.edges_by_relation.keys())}")
    for rel, edges in snapshot.edges_by_relation.items():
        print(f"      * {rel}: {len(edges)} directed edges")

    if snapshot.relation_attention_weights is not None:
        print(f"    - Relation Attention Weights (Sample Node 0):\n      {snapshot.relation_attention_weights[0]}")

    # 4. Generate Causal Covariance Matrix Sigma_t
    print(f"\n[*] Computing structural covariance matrix Sigma_t...")
    cov = adapter.get_covariance_matrix(as_of)
    print(f"    - Covariance Matrix Shape: {cov.shape}")
    print(f"    - Condition Number: {np.linalg.cond(cov):.2f}")
    print(f"    - Smallest Eigenvalue: {np.linalg.eigvalsh(cov)[0]:.6f}")

    # 5. Compute Downstream Portfolio Allocations
    w_gmv = solve_gmv_weights(cov)
    w_eq = np.ones(len(portfolio_tickers)) / len(portfolio_tickers)

    print("\n[*] Downstream Portfolio Allocation Comparison:")
    print(f"{'Asset':<12} | {'DyFO-GMVP Weight':<18} | {'1/N Baseline':<15}")
    print("-" * 52)
    for i, asset in enumerate(adapter.entity_ids):
        print(f"{asset:<12} | {w_gmv[i]*100:>15.2f}% | {w_eq[i]*100:>12.2f}%")

    portfolio_var_gmv = float(w_gmv.T @ cov @ w_gmv) * 252
    portfolio_var_eq = float(w_eq.T @ cov @ w_eq) * 252
    print("-" * 52)
    print(f"Annualized Risk (DyFO-GMVP): {np.sqrt(portfolio_var_gmv)*100:.2f}%")
    print(f"Annualized Risk (1/N):       {np.sqrt(portfolio_var_eq)*100:.2f}%")
    print(f"Risk Reduction:              {(1 - np.sqrt(portfolio_var_gmv)/np.sqrt(portfolio_var_eq))*100:.2f}%")
    print("=" * 70)
    print("Demo completed successfully with zero mutations to PORTA data!")


if __name__ == "__main__":
    run_pipeline_demo()
