"""Readout functions — convert per-node embeddings to a single graph embedding e_t.

Implements the three strategies from DyFO Manual §3.5:
  - Mean pooling (default)
  - Market-cap weighted
  - Attention readout
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanReadout(nn.Module):
    """e_t = mean(z_i for i in portfolio)."""

    def forward(
        self,
        node_embeddings: torch.Tensor,       # (N, d)
        mask: Optional[torch.Tensor] = None,  # (N,) bool — active nodes
    ) -> torch.Tensor:
        if mask is not None:
            active = node_embeddings[mask]
        else:
            active = node_embeddings
        return active.mean(dim=0)  # (d,)


class WeightedReadout(nn.Module):
    """e_t = sum(w_i * z_i) / sum(w_i), weighted by market cap or other."""

    def forward(
        self,
        node_embeddings: torch.Tensor,  # (N, d)
        weights: torch.Tensor,          # (N,) — e.g. log market cap
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            node_embeddings = node_embeddings[mask]
            weights = weights[mask]
        w = F.softmax(weights, dim=0).unsqueeze(-1)  # (N, 1)
        return (w * node_embeddings).sum(dim=0)  # (d,)


class AttentionReadout(nn.Module):
    """e_t = softmax(q * Z^T) * Z, with learnable query vector."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embedding_dim))

    def forward(
        self,
        node_embeddings: torch.Tensor,       # (N, d)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            node_embeddings = node_embeddings[mask]
        scores = node_embeddings @ self.query  # (N,)
        alpha = F.softmax(scores, dim=0).unsqueeze(-1)  # (N, 1)
        return (alpha * node_embeddings).sum(dim=0)  # (d,)


def get_readout(strategy: str, embedding_dim: int = 100) -> nn.Module:
    """Factory for readout modules."""
    if strategy == "mean":
        return MeanReadout()
    elif strategy == "weighted":
        return WeightedReadout()
    elif strategy == "attention":
        return AttentionReadout(embedding_dim)
    else:
        raise ValueError(f"Unknown readout strategy: {strategy}")
