"""Unit and integration tests for advanced econometric and dynamic graph improvements.

Covers:
  - REQ-IMP1: Delta / Residual regression metrics and conditioning
  - REQ-IMP2: SPD Higham projection and hybrid graph-shrinkage
  - REQ-IMP3: EvolveGCN dynamic baseline encoder
  - REQ-IMP4: cDCC-GARCH and DECO econometric baselines
"""

import numpy as np
import pandas as pd
import pytest
import torch

from dyfo.config import DyFOConfig
from dyfo.core.edge_features import (
    compute_cdcc_garch_correlations,
    compute_deco_correlations,
)
from dyfo.core.evolvegcn_baseline import EvolveGCNEncoder
from dyfo.core.link_prediction import (
    CorrelationRegressor,
    compute_graph_shrinkage_covariance,
    compute_regression_metrics,
    project_to_spd_correlation,
)
from dyfo.core.model_variants import build_encoder


# ---------------------------------------------------------------------------
# REQ-IMP2: SPD Higham Projection & Shrinkage Tests
# ---------------------------------------------------------------------------


def test_project_to_spd_correlation_numpy():
    """Verify that a non-SPD symmetric matrix with unit diagonal is projected to SPD."""
    # Construct a 3x3 matrix with negative eigenvalues
    # [1.0,  0.9,  0.9]
    # [0.9,  1.0, -0.9]
    # [0.9, -0.9,  1.0]
    # Determinant = 1*(1 - 0.81) - 0.9*(0.9 - (-0.81)) + 0.9*(-0.81 - 0.9) < 0
    mat = np.array([
        [1.0, 0.95, 0.95],
        [0.95, 1.0, -0.95],
        [0.95, -0.95, 1.0],
    ])
    raw_eigvals = np.linalg.eigvalsh(mat)
    assert raw_eigvals.min() < 0, "Input matrix must have negative eigenvalues for test validity"

    spd_mat = project_to_spd_correlation(mat, epsilon=1e-4)

    # Check properties
    assert isinstance(spd_mat, np.ndarray)
    np.testing.assert_allclose(np.diag(spd_mat), 1.0, atol=1e-6)
    assert np.allclose(spd_mat, spd_mat.T, atol=1e-6)
    post_eigvals = np.linalg.eigvalsh(spd_mat)
    assert post_eigvals.min() >= 1e-4 - 1e-7


def test_project_to_spd_correlation_torch():
    """Verify PyTorch tensor SPD projection."""
    mat = torch.tensor([
        [1.0, 0.95, 0.95],
        [0.95, 1.0, -0.95],
        [0.95, -0.95, 1.0],
    ], dtype=torch.float32)

    spd_mat = project_to_spd_correlation(mat, epsilon=1e-3)
    assert isinstance(spd_mat, torch.Tensor)
    np.testing.assert_allclose(torch.diag(spd_mat).numpy(), 1.0, atol=1e-5)
    post_eigvals = torch.linalg.eigvalsh(spd_mat)
    assert post_eigvals.min().item() >= 1e-3 - 1e-6


def test_compute_graph_shrinkage_covariance():
    """Verify hybrid graph-shrinkage covariance blending."""
    np.random.seed(42)
    T, N = 100, 5
    returns = np.random.randn(T, N) * 0.01
    cov_gnn = np.corrcoef(returns.T) * 0.0004

    cov_hybrid, alpha = compute_graph_shrinkage_covariance(cov_gnn, returns, alpha=0.6)
    assert cov_hybrid.shape == (N, N)
    assert alpha == 0.6
    eigvals = np.linalg.eigvalsh(cov_hybrid)
    assert eigvals.min() > 0


# ---------------------------------------------------------------------------
# REQ-IMP3: EvolveGCN Baseline Tests
# ---------------------------------------------------------------------------


def test_evolvegcn_encoder_forward_and_evolution():
    """Verify EvolveGCN forward pass and weight evolution."""
    config = DyFOConfig(embedding_dim=32, node_feature_dim=10)
    num_nodes = 4

    encoder = EvolveGCNEncoder(config, num_nodes=num_nodes, variant="EvolveGCN-O", hidden_dim=16)
    assert isinstance(encoder, EvolveGCNEncoder)

    node_feat = torch.randn(num_nodes, 10)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_type_ids = torch.zeros(4, dtype=torch.long)
    edge_ts = torch.zeros(4, dtype=torch.float32)

    # Initial embeddings
    emb0 = encoder.get_node_embeddings(node_feat, edge_index, edge_type_ids, edge_ts, 0.0)
    assert emb0.shape == (num_nodes, 32)

    # Advance one day -> weights evolve
    encoder.advance_day([], node_feat, edge_index, edge_type_ids, edge_ts, 1.0)
    emb1 = encoder.get_node_embeddings(node_feat, edge_index, edge_type_ids, edge_ts, 1.0)
    assert emb1.shape == (num_nodes, 32)

    # Reset state
    encoder.reset_state()


def test_build_encoder_factory_evolvegcn():
    """Verify factory registration of evolvegcn."""
    config = DyFOConfig(model_variant="evolvegcn", embedding_dim=16, node_feature_dim=8)
    encoder = build_encoder(config, num_nodes=5)
    assert isinstance(encoder, EvolveGCNEncoder)


# ---------------------------------------------------------------------------
# REQ-IMP4: cDCC-GARCH and DECO Tests
# ---------------------------------------------------------------------------


def test_compute_cdcc_and_deco_econometric_baselines():
    """Verify cDCC and DECO executions and structure."""
    dates = pd.date_range("2022-01-01", periods=60, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    np.random.seed(42)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(np.random.randn(60, 3) * 0.01, axis=0)),
        index=dates,
        columns=tickers,
    )

    # 1. cDCC-GARCH
    cdcc_df, cdcc_pairs, cdcc_meta = compute_cdcc_garch_correlations(
        prices, window=40, threshold=0.1, mode="causal_filter"
    )
    assert not cdcc_df.empty
    assert cdcc_meta["method"] == "cdcc_garch_aielli_2013"

    # 2. DECO
    deco_df, deco_pairs, deco_meta = compute_deco_correlations(
        prices, window=40, threshold=0.0, mode="causal_filter"
    )
    assert not deco_df.empty
    assert deco_meta["method"] == "deco_engle_kelly_2012"
    # All pairs should have equal correlation at each row
    for row_idx in range(len(deco_df)):
        vals = deco_df.iloc[row_idx].values
        assert np.allclose(vals, vals[0], atol=1e-5)


# ---------------------------------------------------------------------------
# REQ-IMP1: Delta Target & Conditioning Tests
# ---------------------------------------------------------------------------


def test_correlation_regressor_delta_and_rho_conditioning():
    """Verify CorrelationRegressor in delta mode with rho conditioning."""
    dim = 16
    regressor = CorrelationRegressor(
        embedding_dim=dim, hidden_dim=32, output_mode="delta", use_rho_conditioning=True
    )
    z_src = torch.randn(10, dim)
    z_dst = torch.randn(10, dim)
    rho_today = torch.linspace(-0.5, 0.5, 10)

    delta_preds = regressor(z_src, z_dst, rho_today=rho_today)
    assert delta_preds.shape == (10,)

    # Metric reconstruction check
    targets = torch.randn(10) * 0.05
    metrics = compute_regression_metrics(delta_preds, targets, rho_today=rho_today)
    assert "r_squared_reconstructed" in metrics
    assert "mae_reconstructed" in metrics
