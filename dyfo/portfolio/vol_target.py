"""Decoupled Dynamic Volatility Targeting Engine.

Scales ex-ante portfolio risk to a target annual volatility (e.g., 12%) while
strictly decoupling cash buffering (k_t <= 1) from leverage (k_t > 1) and
monitoring spectral stress concentration from DyFO eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from dyfo.core.link_prediction import project_to_spd_covariance


@dataclass
class VolTargetResult:
    """Output of volatility targeting engine."""
    scaled_weights: np.ndarray
    cash_weight: float
    ex_ante_vol_annual: float
    vol_scale_factor: float
    is_spectral_stress: bool


class VolTargetingEngine:
    """Dynamic Volatility Targeting and Cash Buffer Management."""

    def __init__(
        self,
        target_vol_annual: float = 0.12,
        allow_leverage: bool = False,
        max_leverage: float = 1.25,
        min_scale: float = 0.50,
        spectral_stress_threshold: float = 0.38,
        stress_cash_floor: float = 0.30,
    ):
        self.target_vol_annual = target_vol_annual
        self.allow_leverage = allow_leverage
        self.max_leverage = max_leverage
        self.min_scale = min_scale
        self.spectral_stress_threshold = spectral_stress_threshold
        self.stress_cash_floor = stress_cash_floor

    def scale_portfolio(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> VolTargetResult:
        """Compute scaled weights and cash buffer.

        Parameters
        ----------
        weights : np.ndarray (N,)
            Base portfolio weights on Delta^N.
        cov_matrix : np.ndarray (N, N)
            Causal daily covariance matrix from DyFO.

        Returns
        -------
        VolTargetResult
        """
        w = np.asarray(weights, dtype=np.float64)
        sigma_spd = project_to_spd_covariance(cov_matrix, epsilon=1e-5)
        n = w.shape[0]

        # 1. Causal Ex-Ante Portfolio Volatility
        port_var_daily = float(w.T @ sigma_spd @ w)
        port_vol_annual = float(np.sqrt(max(port_var_daily, 1e-8) * 252.0))

        # 2. Base Volatility Scale Factor
        k_raw = self.target_vol_annual / (port_vol_annual + 1e-6)

        # 3. Apply Leverage Bounds
        if self.allow_leverage:
            k_bounded = float(np.clip(k_raw, self.min_scale, self.max_leverage))
        else:
            k_bounded = float(np.clip(k_raw, self.min_scale, 1.0))

        # 4. Spectral Stress Monitoring: lambda_1 / Tr(Sigma)
        eigvals = np.linalg.eigvalsh(sigma_spd)
        spectral_ratio = float(eigvals[-1] / (np.sum(eigvals) + 1e-8))
        is_stress = spectral_ratio > self.spectral_stress_threshold

        if is_stress:
            # Enforce defensive cash buffer
            max_risky = 1.0 - self.stress_cash_floor
            k_final = min(k_bounded, max_risky)
        else:
            k_final = k_bounded

        scaled_w = w * k_final
        cash_w = 1.0 - float(np.sum(scaled_w))

        return VolTargetResult(
            scaled_weights=scaled_w,
            cash_weight=cash_w,
            ex_ante_vol_annual=port_vol_annual,
            vol_scale_factor=k_final,
            is_spectral_stress=is_stress,
        )
