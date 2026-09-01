"""Constrained Quadratic Portfolio Solver with Symbolic Bounds and Higham SPD Projection.

Solves the quadratic minimum-variance optimization problem subject to
linear inequality constraints and sector bounds derived from LLM symbolic reasoning:
    min_w (1/2) w^T Sigma w
    s.t.  sum(w) = 1 - cash_buffer
          w >= 0, w <= max_asset_weight
          A_ub w <= b_ub
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from dyfo.core.link_prediction import project_to_spd_covariance
from dyfo.neurosymbolic.symbolic_parser import ParsedConstraints

logger = logging.getLogger("DyFO.NeuroSymbolicSolver")


class ConstrainedPortfolioSolver:
    """Solves convex quadratic portfolio optimization over DyFO covariance with symbolic constraints."""

    def __init__(self, ridge_reg: float = 1e-5):
        self.ridge_reg = ridge_reg

    def solve(
        self,
        cov_matrix: np.ndarray,
        constraints: Optional[ParsedConstraints] = None,
        target_cash_buffer: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve constrained portfolio optimization: min (1/2) w^T Sigma w s.t. A_ub w <= b_ub, sum(w) = 1 - cash.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Predicted covariance matrix in shape (N, N).
        constraints : Optional[ParsedConstraints]
            Symbolic linear inequality constraints and bounds.
        target_cash_buffer : Optional[float]
            Override cash buffer (e.g. 0.10 for 10% cash).

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Optimized weights w (length N) and optimization metadata.
        """
        n = cov_matrix.shape[0]

        # 1. Guarantee Strict SPD via Covariance Projection & Regularization
        sigma_spd = project_to_spd_covariance(cov_matrix, epsilon=1e-5) + np.eye(n) * self.ridge_reg

        # Determine target risky asset sum (1 - cash_buffer)
        cash_buffer = target_cash_buffer if target_cash_buffer is not None else (
            constraints.cash_buffer if constraints is not None else 0.0
        )
        risky_total = max(0.10, 1.0 - cash_buffer)

        # Base Bounds
        if constraints is not None and constraints.bounds:
            bounds = constraints.bounds
        else:
            bounds = [(0.0, 0.25) for _ in range(n)]

        # Objective Function: (1/2) w^T Sigma w
        def objective(w: np.ndarray) -> float:
            return float(0.5 * w.T @ sigma_spd @ w)

        def gradient(w: np.ndarray) -> np.ndarray:
            return sigma_spd @ w

        # Equality Constraint: sum(w) = risky_total
        cons_list = [{"type": "eq", "fun": lambda w: np.sum(w) - risky_total}]

        # Inequality Constraints: A_ub w <= b_ub  =>  b_ub - A_ub w >= 0
        if constraints is not None and constraints.A_ub.shape[0] > 0:
            for row_idx in range(constraints.A_ub.shape[0]):
                a_row = constraints.A_ub[row_idx]
                b_val = constraints.b_ub[row_idx]
                cons_list.append({
                    "type": "ineq",
                    "fun": lambda w, a=a_row, b=b_val: b - np.dot(a, w),
                })

        # Initial Guess: Equal Weight normalized to risky_total
        w0 = np.full(n, risky_total / n)

        try:
            res = minimize(
                fun=objective,
                x0=w0,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=cons_list,
                options={"maxiter": 200, "ftol": 1e-7, "disp": False},
            )

            if res.success and not np.any(np.isnan(res.x)):
                w_opt = res.x
                status = "OPTIMAL"
            else:
                # Fallback: Projected heuristic within bounds
                logger.warning("SLSQP did not fully converge (%s). Applying projected fallback.", res.message)
                w_opt = self._projected_fallback(sigma_spd, bounds, risky_total)
                status = "FALLBACK_PROJECTED"
        except Exception as ex:
            logger.warning("Solver exception (%s). Applying equal-weight bounded fallback.", ex)
            w_opt = self._projected_fallback(sigma_spd, bounds, risky_total)
            status = "FALLBACK_EXCEPTION"

        # Final Feasibility Normalization
        w_opt = np.clip(w_opt, 0.0, None)
        sum_w = np.sum(w_opt)
        if sum_w > 0:
            w_opt = (w_opt / sum_w) * risky_total

        var_opt = float(w_opt.T @ sigma_spd @ w_opt)
        ann_vol = float(np.sqrt(max(var_opt, 1e-8)))

        meta = {
            "status": status,
            "cash_buffer": cash_buffer,
            "risky_weight_sum": float(np.sum(w_opt)),
            "annualized_risk": ann_vol,
            "active_assets_count": int(np.sum(w_opt > 1e-4)),
        }
        return w_opt, meta

    def _projected_fallback(
        self,
        sigma: np.ndarray,
        bounds: List[Tuple[float, float]],
        target_sum: float,
    ) -> np.ndarray:
        """Project unconstrained analytical GMVP onto box bounds."""
        n = sigma.shape[0]
        try:
            inv_s = np.linalg.pinv(sigma)
            raw = inv_s @ np.ones(n)
            raw = np.clip(raw, 0.0, None)
            if np.sum(raw) > 0:
                w = raw / np.sum(raw) * target_sum
            else:
                w = np.full(n, target_sum / n)
        except Exception:
            w = np.full(n, target_sum / n)

        # Apply box bounds
        for i, (b_min, b_max) in enumerate(bounds):
            w[i] = np.clip(w[i], b_min, b_max)

        w_sum = np.sum(w)
        if w_sum > 0:
            w = (w / w_sum) * target_sum
        return w


def solve_symbolically_constrained_gmvp(
    cov_matrix: np.ndarray,
    constraints: Optional[ParsedConstraints] = None,
) -> np.ndarray:
    """Convenience functional wrapper for constrained GMVP."""
    solver = ConstrainedPortfolioSolver()
    w, _ = solver.solve(cov_matrix, constraints)
    return w
