"""Continuous Relational State Constructor for Portfolio DRL.

Constructs state tensors combining DyFO's relation-aware graph node embeddings
(Z_t in R^(N x 100)), previous portfolio allocations (w_{t-1} in R^N), and
macro regime vectors (pi_t) to break the permutation symmetry bottleneck.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING


@dataclass
class ContinuousDRLState:
    """Container for state tensors at decision step t."""
    node_features: torch.Tensor  # Shape (N, feature_dim = 100 + 1 + d_macro)
    graph_embeddings: torch.Tensor  # Shape (N, 100)
    current_weights: torch.Tensor  # Shape (N,)
    macro_features: torch.Tensor  # Shape (d_macro,)
    tickers: List[str]
    date: str

    @property
    def num_assets(self) -> int:
        return len(self.tickers)

    @property
    def feature_dim(self) -> int:
        return self.node_features.shape[1]


class ContinuousStateConstructor:
    """Builds relational DRL state representations without look-ahead bias."""

    def __init__(
        self,
        tickers: Optional[Sequence[str]] = None,
        embedding_dim: int = 100,
        macro_dim: int = 4,
        device: str = "cpu",
    ):
        self.tickers = list(tickers) if tickers is not None else list(TICKERS_30)
        self.num_nodes = len(self.tickers)
        self.embedding_dim = embedding_dim
        self.macro_dim = macro_dim
        self.device = torch.device(device)
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

    def build_state(
        self,
        graph_embeddings: np.ndarray,
        current_weights: np.ndarray,
        macro_vector: Optional[np.ndarray] = None,
        date_str: str = "2024-01-01",
    ) -> ContinuousDRLState:
        """Construct ContinuousDRLState tensor from numpy inputs.

        Parameters
        ----------
        graph_embeddings : np.ndarray
            DyFO node embeddings in shape (N, embedding_dim).
        current_weights : np.ndarray
            Previous portfolio weights in shape (N,).
        macro_vector : Optional[np.ndarray]
            Macro regime features (length macro_dim).
        date_str : str
            Date identifier.

        Returns
        -------
        ContinuousDRLState
            Structured PyTorch state container.
        """
        n = self.num_nodes
        assert graph_embeddings.shape[0] == n, f"Graph embeddings shape {graph_embeddings.shape} mismatch with N={n}"

        emb_t = torch.tensor(graph_embeddings, dtype=torch.float32, device=self.device)
        w_t = torch.tensor(current_weights, dtype=torch.float32, device=self.device).view(n, 1)

        if macro_vector is not None:
            macro_arr = np.asarray(macro_vector, dtype=np.float32)
            if macro_arr.ndim == 1 and len(macro_arr) < self.macro_dim:
                # Pad to macro_dim
                macro_arr = np.pad(macro_arr, (0, self.macro_dim - len(macro_arr)))
            elif len(macro_arr) > self.macro_dim:
                macro_arr = macro_arr[:self.macro_dim]
        else:
            # Default neutral macro vector
            macro_arr = np.zeros(self.macro_dim, dtype=np.float32)

        macro_t = torch.tensor(macro_arr, dtype=torch.float32, device=self.device)
        macro_broadcast = macro_t.unsqueeze(0).repeat(n, 1)  # (N, macro_dim)

        # Concatenate asset token features: [embedding (100), current_weight (1), macro (d_macro)]
        node_feats = torch.cat([emb_t, w_t, macro_broadcast], dim=1)  # (N, 100 + 1 + d_macro)

        return ContinuousDRLState(
            node_features=node_feats,
            graph_embeddings=emb_t,
            current_weights=w_t.squeeze(1),
            macro_features=macro_t,
            tickers=self.tickers,
            date=date_str,
        )

    def build_raw_state(
        self,
        prices_history: np.ndarray,
        current_weights: np.ndarray,
        date_str: str = "2024-01-01",
    ) -> ContinuousDRLState:
        """Construct Raw-DRL state (flat historical return features, NO graph embedding).

        Used as an ablation baseline to evaluate the value of relational graph embeddings.
        """
        n = self.num_nodes
        # Extract 5-day, 20-day returns and 20-day volatility per asset
        if prices_history.shape[0] >= 21:
            ret_5d = (prices_history[-1] / (prices_history[-6] + 1e-6)) - 1.0
            ret_20d = (prices_history[-1] / (prices_history[-21] + 1e-6)) - 1.0
            vol_20d = np.std(prices_history[-20:] / prices_history[-21:-1] - 1.0, axis=0) * np.sqrt(252)
        else:
            ret_5d = np.zeros(n, dtype=np.float32)
            ret_20d = np.zeros(n, dtype=np.float32)
            vol_20d = np.full(n, 0.15, dtype=np.float32)

        raw_feats = np.stack([ret_5d, ret_20d, vol_20d], axis=1)  # (N, 3)
        # Pad to embedding_dim to match policy input dimension
        padded_feats = np.pad(raw_feats, ((0, 0), (0, self.embedding_dim - 3)))

        return self.build_state(
            graph_embeddings=padded_feats,
            current_weights=current_weights,
            date_str=date_str,
        )
