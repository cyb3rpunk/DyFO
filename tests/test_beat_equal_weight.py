"""Unit tests for Beating Equal-Weight (1/N) Alpha Engine components."""

import numpy as np
import pytest
import torch

from dyfo.portfolio.graph_hrp import GraphHRP, cov_to_corr
from dyfo.portfolio.portfolio_accounting import (
    PortfolioLedger,
    compute_post_drift_weights,
    compute_turnover_and_cost,
)
from dyfo.portfolio.tangency_optimizer import (
    DyFOTangency,
    compute_causal_excess_return_signal,
)
from dyfo.portfolio.vol_target import VolTargetingEngine, VolTargetResult
from dyfo.drl.tactical_drl_tilt import (
    TacticalDRLPolicy,
    DifferentialSharpeReward,
    project_to_l1_simplex_ball,
)


def test_portfolio_accounting_drift_and_ledger():
    """Verify exact post-drift accounting, turnover, transaction cost and ledger metrics."""
    w_prev = np.array([0.5, 0.5])
    # Asset 0 gains +10%, Asset 1 loses -10%
    returns_today = np.array([0.10, -0.10])
    w_drifted = compute_post_drift_weights(w_prev, returns_today)

    # 0.5 * 1.10 = 0.55; 0.5 * 0.90 = 0.45 => sum = 1.00 => drifted weights = [0.55, 0.45]
    assert np.isclose(w_drifted[0], 0.55)
    assert np.isclose(w_drifted[1], 0.45)
    assert np.isclose(np.sum(w_drifted), 1.0)

    # Rebalance back to [0.5, 0.5]
    w_target = np.array([0.5, 0.5])
    turnover, tx_cost = compute_turnover_and_cost(w_target, w_drifted, tx_cost_bps=10.0)
    # Turnover = |0.50 - 0.55| + |0.50 - 0.45| = 0.10
    assert np.isclose(turnover, 0.10)
    # Cost = 0.10 * 0.0010 = 0.0001
    assert np.isclose(tx_cost, 0.0001)

    # Ledger step test
    ledger = PortfolioLedger(tx_cost_bps=10.0)
    net_ret1 = ledger.step(w_target, realized_asset_returns_next=np.array([0.01, 0.02]))
    assert len(ledger.net_returns) == 1
    metrics = ledger.compute_summary_metrics()
    assert "annualized_net_return" in metrics
    assert "net_sharpe_ratio" in metrics
    assert "annualized_cost_drag_bps" in metrics


def test_graph_hrp_allocation():
    """Verify GraphHRP tree clustering, quasi-diagonalization and recursive bisection."""
    np.random.seed(42)
    n = 10
    # Create synthetic structured covariance
    a = np.random.randn(n, n)
    cov = a @ a.T + np.eye(n) * 0.1

    hrp = GraphHRP(linkage_method="ward")
    weights = hrp.allocate(cov)

    assert weights.shape == (n,)
    assert np.all(weights >= 0.0)
    assert np.isclose(np.sum(weights), 1.0)
    # Weights should not be strictly equal, reflecting cluster risk parity
    assert not np.allclose(weights, 1.0 / n)


def test_tangency_convex_optimizer():
    """Verify homogeneous tangency QP, excess return signal and turnover smoothing."""
    np.random.seed(42)
    t, n = 50, 10
    returns_history = np.random.randn(t, n) * 0.01 + 0.0005
    node_embeddings = np.random.randn(n, 64)

    mu_excess = compute_causal_excess_return_signal(
        returns_history,
        node_embeddings=node_embeddings,
        lookback_mom=20,
    )
    assert mu_excess.shape == (n,)
    assert np.all(mu_excess > 0.0)

    cov = np.cov(returns_history, rowvar=False) + np.eye(n) * 1e-4
    solver = DyFOTangency(max_weight=0.25, turnover_smoothing=0.85)

    w_drift = np.full(n, 1.0 / n)
    w_opt = solver.solve(cov, mu_excess, prev_weights_drifted=w_drift)

    assert w_opt.shape == (n,)
    assert np.all(w_opt >= 0.0)
    assert np.isclose(np.sum(w_opt), 1.0)
    assert np.all(w_opt <= 0.25 + 1e-5)


def test_vol_targeting_engine():
    """Verify decoupled volatility targeting, cash buffer and spectral stress triggers."""
    np.random.seed(42)
    n = 10
    w = np.full(n, 1.0 / n)

    # Moderate volatility covariance (~10% annual)
    cov_low = np.eye(n) * (0.10 / np.sqrt(252)) ** 2
    engine = VolTargetingEngine(target_vol_annual=0.12, allow_leverage=True, max_leverage=1.25)
    res = engine.scale_portfolio(w, cov_low)

    assert isinstance(res, VolTargetResult)
    assert res.vol_scale_factor >= 1.0
    assert np.isclose(np.sum(res.scaled_weights) + res.cash_weight, 1.0)

    # High correlation stress covariance (dominant eigenvalue)
    cov_stress = np.ones((n, n)) * 0.001 + np.eye(n) * 0.0001
    engine_stress = VolTargetingEngine(target_vol_annual=0.12, spectral_stress_threshold=0.38)
    res_stress = engine_stress.scale_portfolio(w, cov_stress)

    assert res_stress.is_spectral_stress is True
    assert res_stress.cash_weight >= 0.30 - 1e-5


def test_tactical_drl_policy_and_dsr():
    """Verify TacticalDRLPolicy L1 simplex projection and online DSR recursion."""
    b, n, f = 2, 10, 105
    state = torch.randn(b, n, f)
    anchor = torch.full((b, n), 1.0 / n)

    policy = TacticalDRLPolicy(node_feature_dim=f, delta_max=0.30)
    weights, values = policy(state)

    assert weights.shape == (b, n)
    assert values.shape == (b,)
    # Check sum to 1
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(b), atol=1e-5)
    # Check L1 deviation bound <= delta_max
    l1_dev = torch.sum(torch.abs(weights - anchor), dim=-1)
    assert torch.all(l1_dev <= 0.30 + 1e-4)

    # Differential Sharpe Reward
    dsr_calc = DifferentialSharpeReward(eta=0.05, turnover_penalty=0.50)
    r1 = dsr_calc.compute_reward(realized_return=0.01, turnover=0.02)
    r2 = dsr_calc.compute_reward(realized_return=0.02, turnover=0.01)
    assert isinstance(r1, float)
    assert isinstance(r2, float)
