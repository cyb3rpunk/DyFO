"""GAT Static baseline (BL-02) — message-passing on a time-invariant graph.

The graph is built **once** from the mean absolute pairwise correlation
over the training window and is held fixed for all splits.  No temporal
state is maintained; every call to ``get_node_embeddings()`` runs a fresh
2-layer multi-head GAT forward pass.

Architecture
------------
  x_0 = node_features                              (N, node_feature_dim)
  x_1 = MultiHeadGAT(x_0, E_static)               (N, num_heads * hidden)
  x_2 = MultiHeadGAT(x_1, E_static)               (N, embedding_dim)

where E_static is the static edge index computed during setup.

Usage
-----
  encoder = GATStaticEncoder(config, num_nodes)
  encoder.set_static_graph_from_correlations(corr_labels_by_date, train_dates)
  # ... BaseGraphEncoder interface from here ...
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent
from dyfo.core.model_variants import BaseGraphEncoder


# ---------------------------------------------------------------------------
# GAT building blocks
# ---------------------------------------------------------------------------

class _GATLayer(nn.Module):
    """Single-head graph attention layer (Veličković et al., ICLR 2018).

    Attention: e_ij = LeakyReLU(a^T [W h_i || W h_j])
    Aggregate: h_i' = ELU( sum_j  softmax_j(e_ij) * W h_j )
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(
        self,
        x: torch.Tensor,           # (N, in_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:             # (N, out_dim)
        N = x.size(0)
        h = self.W(x)  # (N, out_dim)

        if edge_index.size(1) == 0:
            # No edges — return feature-projected values without aggregation
            return F.elu(h)

        src, dst = edge_index[0], edge_index[1]

        # Attention logits
        e = self.a(torch.cat([h[src], h[dst]], dim=-1)).squeeze(-1)  # (E,)
        e = F.leaky_relu(e, negative_slope=0.2)

        # Numerically-stable softmax over in-edges per destination node
        # Using scatter_softmax manually to avoid external dependencies
        e_max = torch.full((N,), float("-inf"), device=x.device)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        e_shifted = e - e_max[dst]
        exp_e = torch.exp(e_shifted)
        denom = torch.zeros(N, device=x.device).scatter_add_(0, dst, exp_e)
        alpha = exp_e / (denom[dst] + 1e-8)
        alpha = self.dropout(alpha)

        # Weighted neighbourhood aggregation
        out = torch.zeros(N, h.size(-1), device=x.device)
        out.scatter_add_(
            0,
            dst.unsqueeze(-1).expand(-1, h.size(-1)),
            alpha.unsqueeze(-1) * h[src],
        )
        return F.elu(out)


class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT layer.

    Parameters
    ----------
    concat
        If True, concatenate head outputs (→ num_heads * out_dim).
        If False, average head outputs (→ out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.concat = concat
        self.heads = nn.ModuleList(
            [_GATLayer(in_dim, out_dim, dropout=dropout) for _ in range(num_heads)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        outs = [h(x, edge_index) for h in self.heads]
        if self.concat:
            return torch.cat(outs, dim=-1)   # (N, num_heads * out_dim)
        return torch.stack(outs, dim=0).mean(0)  # (N, out_dim)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GATStaticEncoder(BaseGraphEncoder):
    """Graph Attention Network over a static mean-correlation graph.

    The static adjacency is built from the mean |rho| over training dates
    and thresholded at ``config.corr_sparsify_threshold``.

    Call ``set_static_graph_from_correlations(corr_labels_by_date, train_dates)``
    once after constructing the encoder and before the first training epoch.
    """

    def __init__(self, config: DyFOConfig, num_nodes: int):
        super().__init__(config, num_nodes)

        node_feat_dim = config.node_feature_dim
        num_heads = config.num_attention_heads
        # Per-head hidden dimension; ensures total post-concat dim is reasonable
        hidden_per_head = max(32, config.embedding_dim // num_heads)

        # Layer 1: concat → (N, num_heads * hidden_per_head)
        self.gat1 = MultiHeadGATLayer(
            in_dim=node_feat_dim,
            out_dim=hidden_per_head,
            num_heads=num_heads,
            dropout=config.dropout,
            concat=True,
        )
        # Layer 2: mean over heads → (N, embedding_dim)
        self.gat2 = MultiHeadGATLayer(
            in_dim=hidden_per_head * num_heads,
            out_dim=config.embedding_dim,
            num_heads=num_heads,
            dropout=config.dropout,
            concat=False,
        )

        # Static graph — set via set_static_graph_from_correlations()
        self.register_buffer("static_edge_index", torch.zeros(2, 0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_static_graph_from_correlations(
        self,
        corr_labels_by_date: Dict[int, Dict[Tuple[int, int], float]],
        train_dates: List[int],
        threshold: Optional[float] = None,
    ) -> None:
        """Build the static adjacency from mean |rho| over training dates.

        Parameters
        ----------
        corr_labels_by_date
            Mapping date_key → {(i, j): rho}.
        train_dates
            Date keys belonging to the training split.  Only correlations
            from these dates are used, preventing look-ahead leakage.
        threshold
            Minimum mean |rho| required for an edge.  Defaults to
            ``config.corr_sparsify_threshold``.
        """
        if threshold is None:
            threshold = self.config.corr_sparsify_threshold

        sums: Dict[Tuple[int, int], float] = {}
        counts: Dict[Tuple[int, int], int] = {}

        for d in train_dates:
            for (i, j), rho in corr_labels_by_date.get(d, {}).items():
                pair = (min(i, j), max(i, j))
                sums[pair] = sums.get(pair, 0.0) + abs(rho)
                counts[pair] = counts.get(pair, 0) + 1

        src_list: List[int] = []
        dst_list: List[int] = []
        for (i, j), total in sums.items():
            mean_abs_rho = total / counts[(i, j)]
            if mean_abs_rho >= threshold:
                src_list += [i, j]   # undirected → add both directions
                dst_list += [j, i]

        if src_list:
            ei = torch.tensor([src_list, dst_list], dtype=torch.long)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)

        self.static_edge_index = ei

    # ------------------------------------------------------------------
    # BaseGraphEncoder interface
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """No-op — static graph carries no temporal state."""

    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> None:
        """No-op — static graph has no state to update."""

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        """Run 2-layer GAT on the static graph and return node embeddings.

        The ``edge_index`` argument is intentionally ignored; the static
        graph built during setup is used instead.

        Returns
        -------
        Tensor of shape (num_nodes, embedding_dim).
        """
        ei = self.static_edge_index.to(node_features.device)
        h = self.gat1(node_features, ei)
        h = self.gat2(h, ei)
        return h
