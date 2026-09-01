"""Homogeneous Tangency / Maximum Sharpe Convex Quadratic Optimizer.

Solves the exact homogeneous formulation of the Tangency (Maximum Sharpe) portfolio:
  min (1/2) y^T Sigma_t y
  subject to (mu^e)^T y = 1,  y >= 0
  Weights: w* = y* / sum(y*)

Where mu^e = mu - r_f * 1 is strictly positive excess return, derived causally
from momentum and DyFO topological node centrality.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from dyfo.core.link_prediction import project_to_spd_covariance
from dyfo.portfolio.portfolio_accounting import compute_post_drift_weights


def compute_causal_excess_return_signal(
    returns_history: np.ndarray,
    node_embeddings: Optional[np.ndarray] = None,
    lookback_mom: int = 20,
    alpha_mom: float = 0.60,
    alpha_cent: float = 0.40,
    rf_daily: float = 0.0001,
) -> np.ndarray:
    """Compute strictly causal composite expected excess return vector mu^e.

    Parameters
    ----------
    returns_history : np.ndarray (T, N)
        Past realized asset returns up to day t.
    node_embeddings : Optional[np.ndarray] (N, D)
        DyFO temporal graph node embeddings Z_t.
    lookback_mom : int, default 20
        Momentum lookback window.
    alpha_mom : float, default 0.60
        Weight for normalized momentum.
    alpha_cent : float, default 0.40
        Weight for normalized embedding centrality.
    rf_daily : float, default 0.0001
        Daily risk-free rate (~2.5% annual).

    Returns
    -------
    np.ndarray (N,)
        Expected excess return vector mu^e with guaranteed positive ray.
    """
    n = returns_history.shape[1]
    t = returns_history.shape[0]

    # 1. Causal Momentum: mean return over lookback
    if t >= lookback_mom:
        mom = np.mean(returns_history[-lookback_mom:], axis=0)
    else:
        mom = np.mean(returns_history, axis=0) if t > 0 else np.zeros(n)

    mom_std = np.std(mom)
    z_mom = (mom - np.mean(mom)) / (mom_std + 1e-6)

    # 2. DyFO Embedding Centrality (L2 norm)
    if node_embeddings is not None and node_embeddings.shape[0] == n:
        cent = np.linalg.norm(node_embeddings, axis=1)
        cent_std = np.std(cent)
        z_cent = (cent - np.mean(cent)) / (cent_std + 1e-6)
    else:
        z_cent = np.zeros(n)

    # Composite signal (annualized base + relative z-score tilt)
    signal = alpha_mom * z_mom + alpha_cent * z_cent
    # Map to positive expected daily excess return ray (e.g. 0.05% to 0.20% daily)
    mu_excess = np.maximum(0.0005 + 0.0003 * signal, 0.0001)
    return mu_excess


class DyFOTangency:
    """Exact Homogeneous Convex Tangency Portfolio Solver."""

    def __init__(
        self,
        max_weight: float = 0.20,
        turnover_smoothing: float = 0.90,
        ridge_reg: float = 1e-5,
    ):
        self.max_weight = max_weight
        self.turnover_smoothing = turnover_smoothing
        self.ridge_reg = ridge_reg

    def solve(
        self,
        cov_matrix: np.ndarray,
        excess_returns: np.ndarray,
        prev_weights_drifted: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve exact homogeneous tangency QP: min (1/2) y^T Sigma y s.t. (mu^e)^T y = 1, y >= 0.

        Parameters
        ----------
        cov_matrix : np.ndarray (N, N)
            Covariance matrix.
        excess_returns : np.ndarray (N,)
            Excess return vector mu^e > 0.
        prev_weights_drifted : Optional[np.ndarray] (N,)
            Previous holding portfolio w_{t-1}^+ after price drift.

        Returns
        -------
        np.ndarray (N,)
            Optimal smoothed allocation w_t on Delta^N.
        """
        n = cov_matrix.shape[0]
        sigma_spd = project_to_spd_covariance(cov_matrix, epsilon=1e-5) + np.eye(n) * self.ridge_reg
        mu_e = np.asarray(excess_returns, dtype=np.float64)
        mu_e = np.maximum(mu_e, 1e-5)

        # Objective: (1/2) y^T Sigma y
        def objective(y: np.ndarray) -> float:
            return float(0.5 * y.T @ sigma_spd @ y)

        def gradient(y: np.ndarray) -> np.ndarray:
            return sigma_spd @ y

        # Initial point on the hyperplane (mu^e)^T y = 1
        y0 = np.ones(n) / (np.sum(mu_e))

        # Bounds: y_i >= 0
        bounds = [(0.0, None) for _ in range(n)]

        # Constraint: (mu^e)^T y = 1
        cons = [{"type": "eq", "fun": lambda y: np.dot(mu_e, y) - 1.0}]

        try:
            res = minimize(
                fun=objective,
                x0=y0,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 150, "ftol": 1e-8},
            )
            if res.success and np.sum(res.x) > 1e-6:
                w_raw = res.x / np.sum(res.x)
            else:
                w_raw = np.ones(n) / n
        except Exception:
            w_raw = np.ones(n) / n

        # Apply maximum asset bound clipping
        w_clipped = np.clip(w_raw, 0.0, self.max_weight)
        w_tangency = w_clipped / np.sum(w_clipped)

        # Apply turnover smoothing against post-drift portfolio
        if prev_weights_drifted is not None:
            w_final = (1.0 - self.turnover_smoothing) * w_tangency + self.turnover_smoothing * prev_weights_drifted
            return w_final / np.sum(w_final)

        return w_tangency
