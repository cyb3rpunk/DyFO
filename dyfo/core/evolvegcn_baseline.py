"""EvolveGCN Baseline Encoder (Pareja et al., AAAI 2020) for DyFO.

EvolveGCN adapts Graph Convolutional Networks (GCN) along the temporal dimension
by using an RNN (GRU) to evolve the GCN weight matrices over time, rather than
relying on node memory vectors.

Supports:
  - EvolveGCN-O (Weight-evolving GCN via GRU on weight parameters)
  - EvolveGCN-H (Weight-evolving GCN with node embedding summary conditioning)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent
from dyfo.core.model_variants import BaseGraphEncoder

logger = logging.getLogger(__name__)


def _normalize_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Compute symmetric normalized adjacency matrix with self-loops: D^{-1/2} (A + I) D^{-1/2}."""
    if edge_index.numel() == 0:
        return torch.eye(num_nodes, device=device)

    # Add self-loops
    loop_idx = torch.arange(num_nodes, device=device, dtype=torch.long)
    full_edge_index = torch.cat([edge_index, torch.stack([loop_idx, loop_idx], dim=0)], dim=1)

    # Dense adjacency
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[full_edge_index[0], full_edge_index[1]] = 1.0
    adj = torch.maximum(adj, adj.T)  # Ensure symmetric

    # Degree normalization
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return norm_adj


class EvolveGCNLayer(nn.Module):
    """Single EvolveGCN layer with weight evolution via GRU."""

    def __init__(self, in_features: int, out_features: int, variant: str = "EvolveGCN-O"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.variant = variant

        # Initial canonical GCN weight
        self.initial_weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.initial_weight)

        # GRU for evolving weights column-wise (or flattened)
        self.weight_gru = nn.GRUCell(out_features, out_features)

        # Current live weight buffer (detached for TBPTT)
        self.current_weight: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Reset weight to initial learned parameter."""
        self.current_weight = self.initial_weight.clone()

    def advance(self, summary_vec: Optional[torch.Tensor] = None) -> None:
        """Evolve weights forward by one step using GRU."""
        if self.current_weight is None:
            self.reset_state()

        # Evolve each row of weight matrix through GRU
        # current_weight shape: (in_features, out_features)
        h_prev = self.current_weight
        if summary_vec is not None and self.variant == "EvolveGCN-H":
            x_in = summary_vec.unsqueeze(0).expand(self.in_features, -1)
        else:
            x_in = h_prev  # autonomous evolution (EvolveGCN-O)

        new_h = self.weight_gru(x_in, h_prev)
        self.current_weight = new_h.detach()

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """Apply GCN layer: norm_adj @ x @ W."""
        w = self.current_weight if self.current_weight is not None else self.initial_weight
        ax = torch.matmul(norm_adj, x)
        return torch.matmul(ax, w)


class EvolveGCNEncoder(BaseGraphEncoder):
    """EvolveGCN Graph Encoder conforming to BaseGraphEncoder.

    Parameters
    ----------
    config : DyFOConfig
    num_nodes : int
    variant : str, default "EvolveGCN-O"
    """

    def __init__(
        self,
        config: DyFOConfig,
        num_nodes: int,
        variant: str = "EvolveGCN-O",
        hidden_dim: int = 64,
    ):
        super().__init__(config, num_nodes)
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.embedding_dim = config.embedding_dim
        self.node_feature_dim = config.node_feature_dim

        # Input feature projection if needed
        self.input_proj = nn.Linear(self.node_feature_dim, hidden_dim)

        # 2-layer EvolveGCN
        self.layer1 = EvolveGCNLayer(hidden_dim, hidden_dim, variant=variant)
        self.layer2 = EvolveGCNLayer(hidden_dim, self.embedding_dim, variant=variant)
        self.dropout = nn.Dropout(0.1)

        self.reset_state()

    def reset_state(self) -> None:
        """Reset internal weights of all evolving layers."""
        self.layer1.reset_state()
        self.layer2.reset_state()

    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> None:
        """Process one trading day and evolve GCN weights."""
        summary = None
        if self.variant == "EvolveGCN-H" and node_features is not None:
            # Summary vector is mean over node features
            proj_x = self.input_proj(node_features)
            summary = proj_x.mean(dim=0)

        self.layer1.advance(summary)
        self.layer2.advance(None)

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        """Compute (num_nodes, embedding_dim) embeddings for today."""
        device = node_features.device if node_features is not None else next(self.parameters()).device
        norm_adj = _normalize_adj(edge_index, self.num_nodes, device)

        if node_features is None:
            x = torch.eye(self.num_nodes, self.hidden_dim, device=device)
        else:
            x = F.relu(self.input_proj(node_features))

        h1 = F.relu(self.layer1(x, norm_adj))
        h1 = self.dropout(h1)
        h2 = self.layer2(h1, norm_adj)
        return h2

    def detach_state(self) -> None:
        """Detach live weight buffers."""
        if self.layer1.current_weight is not None:
            self.layer1.current_weight = self.layer1.current_weight.detach()
        if self.layer2.current_weight is not None:
            self.layer2.current_weight = self.layer2.current_weight.detach()
