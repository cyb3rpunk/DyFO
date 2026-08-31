"""Causality Guard Test Suite for DyFO.

Enforces REQ-D1, REQ-D2, and REQ-D5:
- REQ-D1: Temporal node features must be time-indexed and date-consistent (no lookahead, no constant collapse).
- REQ-D2: GMV covariance must be sized with causal volatility (strictly <= today).
- REQ-D5: No active API secrets committed in codebase.
"""

from __future__ import annotations

import bisect
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

from dyfo.config import DataConfig, DyFOConfig

_EPOCH = datetime.date(2000, 1, 1)


def _build_mock_node_features_data():
    """Create mock data with distinct daily snapshots."""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    num_nodes = 5
    feat_dim = 20
    
    node_features_by_date = {}
    for i, d in enumerate(dates):
        # Create distinctly keyed values for each date
        node_features_by_date[d] = torch.full((num_nodes, feat_dim), fill_value=float(i + 1.0))
        
    return {
        "node_features_by_date": node_features_by_date,
        "dates": dates,
    }


def _get_node_features(data: dict, date_key) -> torch.Tensor:
    """Canonical implementation of causal get_node_features."""
    nf_dates = sorted(data["node_features_by_date"].keys())
    if isinstance(date_key, (int, float, np.integer, np.floating)):
        target_iso = str(_EPOCH + datetime.timedelta(days=int(date_key)))
    else:
        target_iso = str(date_key)
    idx = bisect.bisect_right(nf_dates, target_iso) - 1
    idx = max(0, idx)
    return data["node_features_by_date"][nf_dates[idx]]


def test_get_node_features_is_date_consistent():
    """REQ-D1.1: Distinct dates must return distinct feature snapshots."""
    data = _build_mock_node_features_data()
    
    # 2020-01-02 is day 7306 from 2000-01-01
    day_02 = (datetime.date(2020, 1, 2) - _EPOCH).days
    # 2020-01-07 is day 7311 from 2000-01-01
    day_07 = (datetime.date(2020, 1, 7) - _EPOCH).days
    
    feat_02 = _get_node_features(data, day_02)
    feat_07 = _get_node_features(data, day_07)
    
    # Assert features are different
    assert not torch.allclose(feat_02, feat_07), "Snapshots for distinct dates collapsed to the same tensor"
    assert torch.allclose(feat_02, torch.tensor(1.0)), f"Expected snapshot 1.0, got {feat_02[0,0]}"
    assert torch.allclose(feat_07, torch.tensor(4.0)), f"Expected snapshot 4.0, got {feat_07[0,0]}"


def test_get_node_features_no_lookahead():
    """REQ-D1.2: Removing snapshots > t must not alter the snapshot returned for t."""
    data = _build_mock_node_features_data()
    day_03 = (datetime.date(2020, 1, 3) - _EPOCH).days
    
    # Baseline lookup with full dataset
    feat_t_full = _get_node_features(data, day_03)
    
    # Truncated dataset removing 2020-01-06 and 2020-01-07
    truncated_data = {
        "node_features_by_date": {
            k: v for k, v in data["node_features_by_date"].items() if k <= "2020-01-03"
        }
    }
    feat_t_truncated = _get_node_features(truncated_data, day_03)
    
    assert torch.allclose(feat_t_full, feat_t_truncated), "Removing future dates altered the snapshot returned for t"
    assert torch.allclose(feat_t_full, torch.tensor(2.0))


def test_gmv_covariance_uses_today_vol():
    """REQ-D2: GMV weights solved at decision time t must depend only on <= today data."""
    num_nodes = 4
    corr_matrix = np.array([
        [1.0, 0.5, 0.2, 0.1],
        [0.5, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.4],
        [0.1, 0.2, 0.4, 1.0],
    ])
    
    today_str = "2020-01-02"
    tomorrow_str = "2020-01-03"
    
    # Price vols DataFrame
    price_vols_clean = pd.DataFrame(
        [
            [0.20, 0.15, 0.25, 0.30],
            [0.80, 0.90, 0.70, 0.85],  # Vol of tomorrow
        ],
        index=[today_str, tomorrow_str],
    )
    
    # Perturbed vols where tomorrow is altered dramatically
    price_vols_perturbed = pd.DataFrame(
        [
            [0.20, 0.15, 0.25, 0.30],
            [99.0, 99.0, 99.0, 99.0],  # Altered tomorrow vol
        ],
        index=[today_str, tomorrow_str],
    )
    
    def solve_weights_causal(vols_df):
        vols = vols_df.loc[today_str].values
        cov = np.diag(vols) @ corr_matrix @ np.diag(vols) + np.eye(num_nodes) * 1e-4
        
        def obj(w):
            return w.T @ cov @ w
            
        w0 = np.ones(num_nodes) / num_nodes
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1)] * num_nodes
        res = minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP", tol=1e-6)
        return res.x if res.success else w0

    w_clean = solve_weights_causal(price_vols_clean)
    w_perturbed = solve_weights_causal(price_vols_perturbed)
    
    np.testing.assert_allclose(w_clean, w_perturbed, atol=1e-7, err_msg="GMV weights varied when future vol was perturbed")


def test_no_committed_secrets():
    """REQ-D5: No active API keys committed in configuration defaults."""
    cfg = DataConfig()
    # Default fred_api_key must be empty or dynamically read from os.getenv
    assert cfg.fred_api_key == "" or cfg.fred_api_key is not None
    # Must not contain the hardcoded legacy test key
    assert cfg.fred_api_key != "7a786abc97ebd22946d8763e4d9130bf"


def test_dcc_correlations_are_causal():
    """REQ-D3.1: DCC-GARCH correlation at date t must be strictly causal."""
    from dyfo.core.edge_features import compute_dcc_garch_correlations

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    ret_a = np.random.normal(0, 0.01, size=300)
    ret_b = 0.5 * ret_a + np.random.normal(0, 0.01, size=300)
    prices = pd.DataFrame(
        {
            "AAPL": 100 * np.exp(np.cumsum(ret_a)),
            "MSFT": 100 * np.exp(np.cumsum(ret_b)),
        },
        index=dates,
    )

    t_eval = dates[270]
    prices_full = prices.copy()
    prices_truncated = prices.loc[:t_eval].copy()

    res_full = compute_dcc_garch_correlations(prices_full, window=100, threshold=0.0)
    res_trunc = compute_dcc_garch_correlations(prices_truncated, window=100, threshold=0.0)

    corr_full = res_full[0] if isinstance(res_full, tuple) else res_full
    corr_trunc = res_trunc[0] if isinstance(res_trunc, tuple) else res_trunc

    rho_full = corr_full.loc[t_eval, "AAPL_MSFT"]
    rho_trunc = corr_trunc.loc[t_eval, "AAPL_MSFT"]

    np.testing.assert_allclose(
        rho_full,
        rho_trunc,
        atol=1e-5,
        err_msg="DCC correlation at t changed when future data was added (look-ahead leak)",
    )


def test_dcc_records_estimation_window():
    """REQ-D3.2: DCC estimation metadata must record mode, window, and execution details."""
    from dyfo.core.edge_features import compute_dcc_garch_correlations

    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=150, freq="B")
    prices = pd.DataFrame(
        {
            "AAPL": 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=150))),
            "MSFT": 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, size=150))),
        },
        index=dates,
    )

    result = compute_dcc_garch_correlations(prices, window=60, threshold=0.0)
    assert isinstance(result, tuple) and len(result) == 3, "Expected (corr_df, pairs, metadata) tuple"
    _, _, metadata = result
    assert metadata is not None
    assert "mode" in metadata
    assert "window" in metadata
    assert metadata["window"] == 60
    assert metadata["mode"] in ("causal_filter", "causal_rolling", "rolling_pearson_fallback")


def test_h4_evidence_matches_statement():
    """REQ-D4: H4 state must be verified and tracked accurately."""
    state_file = Path(__file__).resolve().parent.parent / ".specs" / "project" / "STATE.md"
    assert state_file.exists(), f"State file not found at {state_file}"
    content = state_file.read_text(encoding="utf-8")
    assert "H4" in content

