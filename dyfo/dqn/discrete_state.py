"""Discrete MDP State Constructor & Action Space for DQN Dynamic Hedging.

Transforms high-dimensional DyFO graph embeddings, spectral eigenvalue concentration,
macro regimes, and portfolio drawdowns into compact state vectors for Deep Q-Learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING

# 4 Canonical Discrete Regime Actions
REGIME_ACTIONS = {
    0: "ALPHA_GMVP",        # Maximize Return / GMVP on top central assets
    1: "DEFENSIVE_ERC",     # Equal Risk Contribution / Inverse Volatility
    2: "TAIL_RISK_HEDGE",   # 50% Defensive Assets + 50% Cash Buffer
    3: "SECTOR_ROTATION",   # Overweight top-2 least correlated sectors
}


@dataclass
class DiscreteDQNState:
    """Compact state vector for DQN decision step."""
    features: torch.Tensor  # 1D Tensor of shape (state_dim,)
    eigen_concentration: float
    top_centrality_score: float
    current_drawdown: float
    macro_regime_prob: float
    date: str

    @property
    def state_dim(self) -> int:
        return int(self.features.shape[0])


class DiscreteStateConstructor:
    """Builds compact state vectors from DyFO graph and portfolio observables."""

    def __init__(
        self,
        tickers: Optional[Sequence[str]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        state_dim: int = 16,
        device: str = "cpu",
    ):
        self.tickers = list(tickers) if tickers is not None else list(TICKERS_30)
        self.num_nodes = len(self.tickers)
        self.sector_mapping = sector_mapping or TICKER_GICS_MAPPING
        self.state_dim = state_dim
        self.device = torch.device(device)

    def build_state(
        self,
        cov_matrix: np.ndarray,
        node_embeddings: np.ndarray,
        current_drawdown: float = 0.0,
        realized_vol_30d: float = 0.15,
        macro_prob: float = 0.0,
        date_str: str = "2024-01-01",
    ) -> DiscreteDQNState:
        """Construct compact state representation s_t in R^16.

        Components:
        - Top-5 normalized eigenvalues of Covariance Matrix (5 dims)
        - Spectral gap lambda_1 / lambda_2 (1 dim)
        - Mean and Max Graph Centrality from Node Embeddings (2 dims)
        - Top-3 Sector dispersion metrics (3 dims)
        - Portfolio Current Drawdown & Peak (2 dims)
        - 30-Day Realized Volatility & Trend (2 dims)
        - Macro regime probability (1 dim)
        Total = 16 dimensions.
        """
        n = cov_matrix.shape[0]

        # 1. Spectral Analysis of Covariance Matrix
        eigvals = np.linalg.eigvalsh(cov_matrix)
        eigvals = np.sort(np.maximum(eigvals, 1e-6))[::-1]
        sum_eig = float(np.sum(eigvals))
        norm_eigs = eigvals[:5] / (sum_eig + 1e-8)  # 5 dims
        if len(norm_eigs) < 5:
            norm_eigs = np.pad(norm_eigs, (0, 5 - len(norm_eigs)))

        spectral_gap = float(eigvals[0] / max(eigvals[1], 1e-6))  # 1 dim
        spectral_conc = float(norm_eigs[0])

        # 2. Graph Centrality from Node Embeddings
        emb_norms = np.linalg.norm(node_embeddings, axis=1)  # (N,)
        mean_centrality = float(np.mean(emb_norms))
        max_centrality = float(np.max(emb_norms))  # 2 dims

        # 3. Sector Dispersion
        sector_vols: List[float] = []
        for sector in ["Information Technology", "Financials", "Health Care"]:
            idx = [i for i, t in enumerate(self.tickers) if self.sector_mapping.get(t) == sector]
            if idx:
                sec_sub = cov_matrix[np.ix_(idx, idx)]
                sector_vols.append(float(np.mean(np.diag(sec_sub))))
            else:
                sector_vols.append(0.04)
        sec_disp = np.array(sector_vols[:3], dtype=np.float32)  # 3 dims

        # 4. Portfolio & Market Regime Observables
        dd_norm = float(np.clip(current_drawdown, -0.50, 0.0))
        vol_norm = float(np.clip(realized_vol_30d, 0.05, 0.80))
        vol_stress = float(1.0 if realized_vol_30d > 0.25 else 0.0)
        macro_val = float(np.clip(macro_prob, 0.0, 1.0))

        feature_vector = np.concatenate([
            norm_eigs,                      # 5 dims
            [spectral_gap * 0.1],           # 1 dim
            [mean_centrality, max_centrality], # 2 dims
            sec_disp,                       # 3 dims
            [dd_norm, abs(dd_norm) ** 2],   # 2 dims
            [vol_norm, vol_stress],         # 2 dims
            [macro_val],                    # 1 dim
        ]).astype(np.float32)  # Total: 16 dims

        feat_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self.device)

        return DiscreteDQNState(
            features=feat_tensor,
            eigen_concentration=spectral_conc,
            top_centrality_score=max_centrality,
            current_drawdown=dd_norm,
            macro_regime_prob=macro_val,
            date=date_str,
        )
