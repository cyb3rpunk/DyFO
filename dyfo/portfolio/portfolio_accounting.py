"""Audited Continuous Portfolio Accounting & Transaction Cost Engine.

Implements exact asset price drift accounting and proportional transaction costs
for empirical out-of-sample portfolio evaluation:
  1. Price drift: w_{t-1}^+ = w_{t-1} * (1 + r_t) / sum(w_{t-1} * (1 + r_t))
  2. Turnover: ||w_t - w_{t-1}^+||_1
  3. Net return: r_{net, t} = r_{gross, t} - c_{tx} * Turnover_t
  4. Continuous wealth compounding: W_t = W_{t-1} * (1 + r_{net, t})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_post_drift_weights(
    weights_prev: np.ndarray,
    realized_returns_today: np.ndarray,
) -> np.ndarray:
    """Calculate asset portfolio weights after daily price drift before rebalancing.

    Parameters
    ----------
    weights_prev : np.ndarray (N,)
        Allocated weights at close of previous trading day t-1.
    realized_returns_today : np.ndarray (N,)
        Asset price returns from t-1 close to t close.

    Returns
    -------
    np.ndarray (N,)
        Evolved weights w_{t-1}^+ immediately prior to execution at day t.
    """
    w_prev = np.asarray(weights_prev, dtype=np.float64)
    r_today = np.asarray(realized_returns_today, dtype=np.float64)
    gross_growth = w_prev * (1.0 + r_today)
    total_wealth_growth = float(np.sum(gross_growth))
    if total_wealth_growth <= 1e-8:
        return np.full_like(w_prev, 1.0 / len(w_prev))
    return gross_growth / total_wealth_growth


def compute_turnover_and_cost(
    target_weights: np.ndarray,
    post_drift_weights: np.ndarray,
    tx_cost_bps: float = 10.0,
) -> Tuple[float, float]:
    """Compute exact two-way portfolio turnover and monetary transaction cost.

    Parameters
    ----------
    target_weights : np.ndarray (N,)
        New target allocation w_t to be traded into.
    post_drift_weights : np.ndarray (N,)
        Current holding weights w_{t-1}^+ after price drift.
    tx_cost_bps : float, default 10.0
        Institutional transaction cost in basis points (10 bps = 0.0010).

    Returns
    -------
    Tuple[float, float]
        (Turnover in [0, 2], Transaction cost drag in decimal return units).
    """
    w_tgt = np.asarray(target_weights, dtype=np.float64)
    w_drift = np.asarray(post_drift_weights, dtype=np.float64)
    turnover = float(np.sum(np.abs(w_tgt - w_drift)))
    cost_rate = tx_cost_bps * 1e-4
    tx_cost = turnover * cost_rate
    return turnover, tx_cost


@dataclass
class PortfolioLedger:
    """Continuous accounting ledger tracking gross/net returns and wealth."""
    tx_cost_bps: float = 10.0
    gross_returns: List[float] = field(default_factory=list)
    net_returns: List[float] = field(default_factory=list)
    turnovers: List[float] = field(default_factory=list)
    tx_costs: List[float] = field(default_factory=list)
    weights_history: List[np.ndarray] = field(default_factory=list)

    def step(
        self,
        target_weights: np.ndarray,
        realized_asset_returns_next: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Record one rebalancing step and return net realized portfolio return.

        Parameters
        ----------
        target_weights : np.ndarray (N,)
            Target weights w_t chosen at decision time t.
        realized_asset_returns_next : np.ndarray (N,)
            Asset returns realized from t to t+1.
        prev_weights : Optional[np.ndarray]
            Holding weights at t-1 (defaults to last in history).

        Returns
        -------
        float : Net realized portfolio return r_{net, t}.
        """
        w_tgt = np.asarray(target_weights, dtype=np.float64)
        r_next = np.asarray(realized_asset_returns_next, dtype=np.float64)

        if prev_weights is not None:
            w_prior = prev_weights
        elif self.weights_history:
            w_prior = self.weights_history[-1]
        else:
            w_prior = w_tgt  # First step initialization: zero initial drift

        # Compute price-drifted holding weights before execution
        w_drift = compute_post_drift_weights(w_prior, np.zeros_like(r_next)) if not self.weights_history else w_prior
        turnover, tx_cost = compute_turnover_and_cost(w_tgt, w_drift, self.tx_cost_bps)

        gross_ret = float(w_tgt @ r_next)
        net_ret = gross_ret - tx_cost

        self.gross_returns.append(gross_ret)
        self.net_returns.append(net_ret)
        self.turnovers.append(turnover)
        self.tx_costs.append(tx_cost)
        self.weights_history.append(w_tgt)

        return net_ret

    def compute_summary_metrics(self) -> Dict[str, float]:
        """Compute full 10-KPI audit summary table."""
        if not self.net_returns:
            return {}

        ann_factor = 252.0
        g_rets = np.array(self.gross_returns, dtype=np.float64)
        n_rets = np.array(self.net_returns, dtype=np.float64)
        t_overs = np.array(self.turnovers, dtype=np.float64)
        c_drags = np.array(self.tx_costs, dtype=np.float64)

        # Gross Metrics
        gross_ret = float(np.mean(g_rets)) * ann_factor
        gross_vol = float(np.std(g_rets)) * np.sqrt(ann_factor)
        gross_sharpe = gross_ret / (gross_vol + 1e-8)
        gross_wealth = np.cumprod(1.0 + g_rets)
        gross_peak = np.maximum.accumulate(gross_wealth)
        gross_mdd = float(np.min((gross_wealth - gross_peak) / gross_peak))

        # Net Metrics
        net_ret = float(np.mean(n_rets)) * ann_factor
        net_vol = float(np.std(n_rets)) * np.sqrt(ann_factor)
        net_sharpe = net_ret / (net_vol + 1e-8)
        net_wealth = np.cumprod(1.0 + n_rets)
        net_peak = np.maximum.accumulate(net_wealth)
        net_mdd = float(np.min((net_wealth - net_peak) / net_peak))

        mean_turnover = float(np.mean(t_overs))
        ann_cost_drag_bps = float(np.mean(c_drags) * ann_factor * 10000.0)

        return {
            "annualized_gross_return": gross_ret,
            "annualized_net_return": net_ret,
            "annualized_volatility": net_vol,
            "gross_sharpe_ratio": gross_sharpe,
            "net_sharpe_ratio": net_sharpe,
            "max_drawdown": net_mdd,
            "turnover": mean_turnover,
            "annualized_cost_drag_bps": ann_cost_drag_bps,
            "final_gross_wealth": float(gross_wealth[-1]),
            "final_net_wealth": float(net_wealth[-1]),
            # Default standard aliases
            "annualized_return": net_ret,
            "sharpe_ratio": net_sharpe,
            "final_wealth": float(net_wealth[-1]),
        }
