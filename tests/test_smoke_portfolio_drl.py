#!/usr/bin/env python3
"""Smoke test: DRL portfolio optimizer with 3 state representations.

The three conditions isolate exactly what the DyFO embedding contributes:

  A) DyFO-DRL  -- state = e_t in R^100 from DyFOModule (full graph embedding)
  B) Raw-DRL   -- state = raw price features in R^(N*3), no graph structure
  C) EWMA-GMVP -- no learning; closed-form minimum-variance portfolio from
                  EWMA covariance (alpha=0.05)

All three conditions run on the same synthetic price series (N assets, T days,
GBM with heterogeneous vol/correlation structure).  The test verifies:
  - valid portfolio weights at every step (non-negative, sum=1)
  - gradient flows through the DRL variants (params actually change)
  - reward (log portfolio return) can be computed and backpropagated
  - no NaN/Inf at any point
  - episode-level cumulative return is comparable across conditions

Usage
-----
  python tests/test_smoke_portfolio_drl.py          # standalone
  pytest tests/test_smoke_portfolio_drl.py -v       # pytest

Notes
-----
  - DyFOModule weights are randomly initialised; this test validates the
    *interface* (shape, gradient flow, episode loop), not model quality.
  - T=40 days, N=10 assets — fast enough to run in < 10 s on CPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dyfo.config import DyFOConfig
from dyfo.core.dyfo_module import DyFOModule
from dyfo.core.event_stream import EventType, FinancialEvent

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic market data
# ─────────────────────────────────────────────────────────────────────────────

N_ASSETS = 10
N_DAYS = 40          # episode length
SEED = 42
ALPHA_EWMA = 0.05    # EWMA decay (matches project convention)


def make_synthetic_returns(n_assets: int, n_days: int, seed: int) -> torch.Tensor:
    """GBM daily returns with a latent factor structure.

    Returns shape (n_days, n_assets).  One common market factor (beta) plus
    idiosyncratic noise creates realistic cross-asset correlations.
    """
    g = torch.Generator().manual_seed(seed)
    market = torch.randn(n_days, 1, generator=g) * 0.01        # ~16% ann vol
    betas = torch.rand(1, n_assets, generator=g) * 0.8 + 0.2   # (0.2, 1.0)
    idio = torch.randn(n_days, n_assets, generator=g) * 0.008
    return market * betas + idio   # (T, N)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_simplex(weights: torch.Tensor) -> torch.Tensor:
    """Project to probability simplex (non-negative, sum=1)."""
    w = torch.clamp(weights, min=0.0)
    s = w.sum()
    if s < 1e-9:
        return torch.full_like(w, 1.0 / len(w))
    return w / s


def portfolio_log_return(weights: torch.Tensor, next_returns: torch.Tensor) -> torch.Tensor:
    """Log-return of the portfolio for one period."""
    gross = 1.0 + (weights * next_returns).sum()
    return torch.log(gross.clamp(min=1e-8))


def ewma_cov(returns_history: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponentially weighted sample covariance, shape (N, N).

    Standard RiskMetrics EWMA:  Sigma_t = alpha * r_t r_t' + (1-alpha) * Sigma_{t-1}
    """
    n = returns_history.shape[1]
    cov = torch.eye(n) * 1e-4   # initial diagonal
    for r in returns_history:
        r = r.unsqueeze(-1)     # (N, 1)
        cov = alpha * (r @ r.T) + (1 - alpha) * cov
    return cov


def gmvp_weights(cov: torch.Tensor) -> torch.Tensor:
    """Closed-form long-only Global Minimum Variance Portfolio.

    w = Sigma^{-1} 1 / (1' Sigma^{-1} 1), then clamp to non-negative.
    """
    n = cov.shape[0]
    reg = cov + 1e-5 * torch.eye(n)
    cov_inv = torch.linalg.inv(reg)
    ones = torch.ones(n)
    raw = cov_inv @ ones
    return to_simplex(raw)


# ─────────────────────────────────────────────────────────────────────────────
# DRL components
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioPolicy(nn.Module):
    """Simple MLP policy: state -> portfolio weights (simplex via softmax)."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_assets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns portfolio weight vector in the probability simplex."""
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)   # (N,)


class EpisodeResult(NamedTuple):
    cumulative_log_return: float
    mean_step_entropy: float           # weight entropy — higher = more diversified
    param_l2_delta: float              # L2 distance of params before/after update
    weights_valid: bool                # all weights >= 0 and sum ~= 1


# ─────────────────────────────────────────────────────────────────────────────
# DyFO state builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_dyfo_module(n_assets: int) -> DyFOModule:
    config = DyFOConfig(model_variant="tgn")
    return DyFOModule(config=config, num_nodes=n_assets, readout_strategy="mean")


def _synthetic_events(day: int, returns_t: torch.Tensor, n_assets: int) -> list:
    """One PRICE_UPDATE event per asset for day `day`."""
    t = float(day) + 0.5
    events = []
    for i in range(n_assets):
        r = float(returns_t[i])
        events.append(
            FinancialEvent(
                event_type=EventType.PRICE_UPDATE,
                timestamp=t + i * 1e-4,
                source_node=i,
                target_node=-1,
                edge_type=None,
                features=torch.tensor([r, abs(r), 1.0]),   # return, |return|, flag
            )
        )
    return events


def _static_graph(n_assets: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fully-connected graph (correlation edge type = 0)."""
    src, dst = [], []
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type_ids = torch.zeros(edge_index.shape[1], dtype=torch.long)  # CORR edges
    edge_timestamps = torch.zeros(edge_index.shape[1])
    return edge_index, edge_type_ids, edge_timestamps


def _node_features(n_assets: int, config: DyFOConfig) -> torch.Tensor:
    return torch.randn(n_assets, config.node_feature_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Condition A: DyFO-DRL
# ─────────────────────────────────────────────────────────────────────────────

def run_dyfo_drl(returns: torch.Tensor) -> EpisodeResult:
    """DRL episode where the state is the DyFO graph embedding e_t in R^100.

    The policy receives the full temporal graph embedding at each step and
    outputs portfolio weights.  One REINFORCE gradient update per episode.
    """
    T, N = returns.shape
    config = DyFOConfig(model_variant="tgn")
    dyfo = _make_dyfo_module(N)
    policy = PortfolioPolicy(state_dim=config.embedding_dim, n_assets=N)
    optimizer = optim.Adam(list(dyfo.parameters()) + list(policy.parameters()), lr=1e-3)

    edge_index, edge_type_ids, edge_timestamps = _static_graph(N)
    node_feat = _node_features(N, config)

    params_before = torch.cat([p.detach().flatten() for p in policy.parameters()])

    log_returns, log_probs_list = [], []
    entropy_sum = 0.0
    weights_valid = True

    dyfo.reset_memory()

    for t in range(T - 1):
        events = _synthetic_events(t, returns[t], N)
        e_t = dyfo(
            events=events,
            node_features=node_feat,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_timestamps=edge_timestamps,
            current_time=float(t) + 0.99,
        )  # (100,)

        weights = policy(e_t.detach())   # detach DyFO; policy is the learner here
        weights_valid &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

        r = portfolio_log_return(weights, returns[t + 1])
        log_returns.append(r)
        entropy_sum += float(-(weights.detach() * (weights.detach() + 1e-8).log()).sum())

        # surrogate log-prob for REINFORCE
        log_probs_list.append(torch.log(weights + 1e-8).mean())

    cum_return = float(sum(r.item() for r in log_returns))

    # REINFORCE: single backward pass per episode
    rewards_t = torch.stack([r.detach() for r in log_returns])
    baseline = rewards_t.mean()
    loss = -torch.stack(log_probs_list) @ (rewards_t - baseline)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    params_after = torch.cat([p.detach().flatten() for p in policy.parameters()])
    delta = float((params_after - params_before).norm())

    assert not math.isnan(cum_return), "DyFO-DRL: NaN in cumulative return"
    assert not math.isnan(delta), "DyFO-DRL: NaN in param delta"

    return EpisodeResult(
        cumulative_log_return=cum_return,
        mean_step_entropy=entropy_sum / (T - 1),
        param_l2_delta=delta,
        weights_valid=weights_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Condition B: Raw-DRL (no graph embedding)
# ─────────────────────────────────────────────────────────────────────────────

def _raw_state(returns_window: torch.Tensor) -> torch.Tensor:
    """Per-asset raw features: [latest return, 5-day mean, 5-day std].

    Shape: (N * 3,) — same information the DRL agent would have without DyFO.
    """
    latest = returns_window[-1]                    # (N,)
    momentum = returns_window.mean(dim=0)          # (N,)
    vol = returns_window.std(dim=0).clamp(min=1e-6)  # (N,)
    return torch.cat([latest, momentum, vol])      # (N*3,)


def run_raw_drl(returns: torch.Tensor) -> EpisodeResult:
    """DRL episode where the state is raw price features per asset (no graph).

    Ablation: same policy architecture as DyFO-DRL, but the state has no
    cross-asset graph structure — just individual asset statistics.
    """
    T, N = returns.shape
    state_dim = N * 3
    policy = PortfolioPolicy(state_dim=state_dim, n_assets=N)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    params_before = torch.cat([p.detach().flatten() for p in policy.parameters()])
    window = max(5, T // 8)

    log_returns, log_probs_list = [], []
    entropy_sum = 0.0
    weights_valid = True

    for t in range(window, T - 1):
        state = _raw_state(returns[t - window: t])
        weights = policy(state)
        weights_valid &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

        r = portfolio_log_return(weights, returns[t + 1])
        log_returns.append(r)
        entropy_sum += float(-(weights.detach() * (weights.detach() + 1e-8).log()).sum())
        log_probs_list.append(torch.log(weights + 1e-8).mean())

    cum_return = float(sum(r.item() for r in log_returns))

    rewards_t = torch.stack([r.detach() for r in log_returns])
    baseline = rewards_t.mean()
    loss = -torch.stack(log_probs_list) @ (rewards_t - baseline)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    params_after = torch.cat([p.detach().flatten() for p in policy.parameters()])
    delta = float((params_after - params_before).norm())

    assert not math.isnan(cum_return), "Raw-DRL: NaN in cumulative return"
    assert not math.isnan(delta), "Raw-DRL: NaN in param delta"

    return EpisodeResult(
        cumulative_log_return=cum_return,
        mean_step_entropy=entropy_sum / max(1, len(log_returns)),
        param_l2_delta=delta,
        weights_valid=weights_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Condition C: EWMA-GMVP (no learning)
# ─────────────────────────────────────────────────────────────────────────────

def run_ewma_gmvp(returns: torch.Tensor, alpha: float = ALPHA_EWMA) -> EpisodeResult:
    """Deterministic EWMA minimum-variance portfolio (no gradient, no policy).

    At each step the GMVP weights are derived analytically from the EWMA
    covariance of all available returns up to that day.  This is the strongest
    purely-statistical baseline that does not use any graph or DRL.
    """
    T, N = returns.shape
    window = max(10, T // 4)   # minimum history before allocating

    log_returns = []
    entropy_sum = 0.0
    weights_valid = True

    for t in range(window, T - 1):
        history = returns[:t]
        cov = ewma_cov(history, alpha=alpha)
        weights = gmvp_weights(cov)          # (N,) — no grad
        weights_valid &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

        with torch.no_grad():
            r = portfolio_log_return(weights, returns[t + 1])
        log_returns.append(float(r))
        entropy_sum += float(-(weights * (weights + 1e-8).log()).sum())

    cum_return = sum(log_returns)
    assert not math.isnan(cum_return), "EWMA-GMVP: NaN in cumulative return"

    return EpisodeResult(
        cumulative_log_return=cum_return,
        mean_step_entropy=entropy_sum / max(1, len(log_returns)),
        param_l2_delta=0.0,   # no learnable parameters
        weights_valid=weights_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(result: EpisodeResult) -> str:
    valid_str = "OK" if result.weights_valid else "INVALID"
    return (
        f"cum_log_ret={result.cumulative_log_return:+.4f}  "
        f"entropy={result.mean_step_entropy:.3f}  "
        f"param_delta={result.param_l2_delta:.4f}  "
        f"weights={valid_str}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pytest entry points
# ─────────────────────────────────────────────────────────────────────────────

def test_dyfo_drl_smoke():
    """DyFO-DRL: valid weights, gradient flows, no NaN."""
    returns = make_synthetic_returns(N_ASSETS, N_DAYS, SEED)
    result = run_dyfo_drl(returns)
    assert result.weights_valid, "DyFO-DRL produced invalid portfolio weights"
    assert result.param_l2_delta > 0, "DyFO-DRL: no parameter update (gradient did not flow)"
    assert not math.isnan(result.cumulative_log_return), "DyFO-DRL: NaN in return"
    print(f"[PASS] DyFO-DRL     {_fmt(result)}")


def test_raw_drl_smoke():
    """Raw-DRL (no graph): valid weights, gradient flows, no NaN."""
    returns = make_synthetic_returns(N_ASSETS, N_DAYS, SEED)
    result = run_raw_drl(returns)
    assert result.weights_valid, "Raw-DRL produced invalid portfolio weights"
    assert result.param_l2_delta > 0, "Raw-DRL: no parameter update (gradient did not flow)"
    assert not math.isnan(result.cumulative_log_return), "Raw-DRL: NaN in return"
    print(f"[PASS] Raw-DRL      {_fmt(result)}")


def test_ewma_gmvp_smoke():
    """EWMA-GMVP: valid weights, deterministic (no gradient), no NaN."""
    returns = make_synthetic_returns(N_ASSETS, N_DAYS, SEED)
    result = run_ewma_gmvp(returns, alpha=ALPHA_EWMA)
    assert result.weights_valid, "EWMA-GMVP produced invalid portfolio weights"
    assert result.param_l2_delta == 0.0, "EWMA-GMVP should have no param delta"
    assert not math.isnan(result.cumulative_log_return), "EWMA-GMVP: NaN in return"
    print(f"[PASS] EWMA-GMVP    {_fmt(result)}")


def test_all_conditions_comparable():
    """All three conditions run on the same data and produce finite results."""
    returns = make_synthetic_returns(N_ASSETS, N_DAYS, SEED)

    dyfo_res = run_dyfo_drl(returns)
    raw_res  = run_raw_drl(returns)
    ewma_res = run_ewma_gmvp(returns, alpha=ALPHA_EWMA)

    for name, res in [("DyFO-DRL", dyfo_res), ("Raw-DRL", raw_res), ("EWMA-GMVP", ewma_res)]:
        assert res.weights_valid, f"{name}: invalid weights"
        assert math.isfinite(res.cumulative_log_return), f"{name}: non-finite return"

    print("\nSummary - same synthetic episode, N=%d assets, T=%d days" % (N_ASSETS, N_DAYS))
    print(f"  DyFO-DRL   (state=e_t, graph embed)  {_fmt(dyfo_res)}")
    print(f"  Raw-DRL    (state=raw features)       {_fmt(raw_res)}")
    print(f"  EWMA-GMVP  (no learning, closed-form) {_fmt(ewma_res)}")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Smoke test: DRL portfolio optimizer - 3 conditions")
    print(f"  N={N_ASSETS} assets, T={N_DAYS} days, seed={SEED}")
    print("=" * 70)

    returns = make_synthetic_returns(N_ASSETS, N_DAYS, SEED)

    print("\n[A] DyFO-DRL  (state = DyFOModule graph embedding e_t in R^100)")
    dyfo_res = run_dyfo_drl(returns)
    print(f"    {_fmt(dyfo_res)}")

    print("\n[B] Raw-DRL   (state = raw price features in R^(N*3), no graph)")
    raw_res = run_raw_drl(returns)
    print(f"    {_fmt(raw_res)}")

    print("\n[C] EWMA-GMVP (no learning; closed-form min-variance portfolio)")
    ewma_res = run_ewma_gmvp(returns, alpha=ALPHA_EWMA)
    print(f"    {_fmt(ewma_res)}")

    print("\n" + "-" * 70)
    print("Condition        cum_log_ret  entropy  param_delta  weights")
    print("-" * 70)
    for label, res in [
        ("DyFO-DRL   [A]", dyfo_res),
        ("Raw-DRL    [B]", raw_res),
        ("EWMA-GMVP  [C]", ewma_res),
    ]:
        valid = "valid" if res.weights_valid else "INVALID"
        print(
            f"  {label}  "
            f"{res.cumulative_log_return:+.4f}       "
            f"{res.mean_step_entropy:.3f}    "
            f"{res.param_l2_delta:.4f}       "
            f"{valid}"
        )
    print("-" * 70)
    print("\n[PASS] All conditions completed without errors.")
    print(
        "\nNote: with randomly initialised weights this test validates the\n"
        "interface contract (shapes, gradient flow, weight validity),\n"
        "NOT model quality.  Train DyFO first, then load weights here."
    )
