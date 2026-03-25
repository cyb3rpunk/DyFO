"""TGN encoder — Memory, Message, Aggregation, and Temporal Graph Attention.

Implements the Temporal Graph Network (Rossi et al., 2020) adapted for
the heterogeneous financial graph of DyFO.

Pipeline per batch:
  1. Message function:  msg = concat[s_i, s_j, phi(dt), f_e, edge_type_emb]
  2. Aggregation:       m_bar_i = agg({msg : events involving i})
  3. Memory update:     s_i = GRU(m_bar_i, s_i)
  4. Embedding (GAT):   z_i = GAT(s_i + v_i, neighbours)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time encoding (Time2Vec)
# ---------------------------------------------------------------------------

class Time2Vec(nn.Module):
    """Learnable time encoding: one linear + (dim-1) periodic components."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.periodic = nn.Linear(1, dim - 1, bias=True)
        self.dim = dim

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: shape (...,) → output: shape (..., dim)."""
        dt = dt.unsqueeze(-1).float()
        lin = self.linear(dt)                      # (..., 1)
        per = torch.sin(self.periodic(dt))         # (..., dim-1)
        return torch.cat([lin, per], dim=-1)       # (..., dim)


# ---------------------------------------------------------------------------
# Message function
# ---------------------------------------------------------------------------

class MessageFunction(nn.Module):
    """Computes raw messages for each event.

    msg_i(t) = [s_i || s_j || phi(dt) || f_e || edge_type_emb]

    For single-node events (target_node == -1), s_j is replaced with zeros.
    """

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        event_feature_dim: int,
        edge_type_emb_dim: int,
        num_edge_types: int,
        num_event_types: int,
    ):
        super().__init__()
        self.time_enc = Time2Vec(time_dim)
        self.edge_type_emb = nn.Embedding(num_edge_types + 1, edge_type_emb_dim)  # +1 for "none"
        self.event_type_emb = nn.Embedding(num_event_types, edge_type_emb_dim)
        self.memory_dim = memory_dim

        # Output dimension of a raw message
        self.output_dim = (
            memory_dim      # s_i
            + memory_dim    # s_j
            + time_dim      # phi(dt)
            + event_feature_dim  # f_e
            + edge_type_emb_dim  # edge_type
            + edge_type_emb_dim  # event_type
        )

    def forward(
        self,
        memory_src: torch.Tensor,   # (B, memory_dim)
        memory_tgt: torch.Tensor,   # (B, memory_dim) — zeros for node-only events
        delta_t: torch.Tensor,      # (B,)
        event_features: torch.Tensor,  # (B, event_feature_dim)
        edge_type_ids: torch.Tensor,   # (B,) int
        event_type_ids: torch.Tensor,  # (B,) int
    ) -> torch.Tensor:
        """Returns raw messages of shape (B, output_dim)."""
        time_emb = self.time_enc(delta_t)                  # (B, time_dim)
        edge_emb = self.edge_type_emb(edge_type_ids)       # (B, edge_type_emb_dim)
        event_emb = self.event_type_emb(event_type_ids)    # (B, edge_type_emb_dim)
        return torch.cat(
            [memory_src, memory_tgt, time_emb, event_features, edge_emb, event_emb],
            dim=-1,
        )


# ---------------------------------------------------------------------------
# Message aggregator
# ---------------------------------------------------------------------------

class MessageAggregator(nn.Module):
    """Aggregates multiple messages per node within a batch.

    Supports 'mean' and 'last' strategies.
    """

    def __init__(self, method: str = "mean"):
        super().__init__()
        assert method in ("mean", "last"), f"Unknown aggregation: {method}"
        self.method = method

    def forward(
        self,
        node_ids: torch.Tensor,       # (B,) node indices
        messages: torch.Tensor,         # (B, msg_dim)
        timestamps: torch.Tensor,       # (B,)
        num_nodes: int,
    ) -> torch.Tensor:
        """Returns aggregated message per node, shape (N, msg_dim)."""
        msg_dim = messages.shape[1]
        agg = torch.zeros(num_nodes, msg_dim, device=messages.device)
        counts = torch.zeros(num_nodes, 1, device=messages.device)

        if self.method == "mean":
            agg.index_add_(0, node_ids, messages)
            counts.index_add_(0, node_ids, torch.ones(len(node_ids), 1, device=messages.device))
            safe_counts = counts.clamp(min=1)
            return agg / safe_counts

        else:  # "last"
            # Sort by timestamp, then scatter — last write wins
            order = timestamps.argsort()
            ordered_ids = node_ids[order]
            ordered_msgs = messages[order]
            agg[ordered_ids] = ordered_msgs
            return agg


# ---------------------------------------------------------------------------
# Memory updater (GRU)
# ---------------------------------------------------------------------------

class MemoryUpdater(nn.Module):
    """Updates per-node memory via GRU."""

    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def forward(
        self,
        aggregated_messages: torch.Tensor,  # (N, msg_dim)
        memory: torch.Tensor,               # (N, memory_dim)
        update_mask: torch.Tensor,          # (N,) bool — which nodes received messages
    ) -> torch.Tensor:
        """Returns updated memory, shape (N, memory_dim)."""
        new_memory = memory.clone()
        if update_mask.any():
            updated = self.gru(
                aggregated_messages[update_mask],
                memory[update_mask],
            )
            new_memory[update_mask] = updated
        return new_memory


# ---------------------------------------------------------------------------
# Temporal Graph Attention (embedding layer)
# ---------------------------------------------------------------------------

class TemporalGraphAttention(nn.Module):
    """Single-layer multi-head attention over temporal neighbourhood.

    z_i(t) = MLP( h_i || MultiHeadAttn(h_i, neighbors) )

    where h_i = s_i + v_i and neighbours include edge features + time encoding.
    """

    def __init__(
        self,
        memory_dim: int,
        node_feat_dim: int,
        time_dim: int,
        edge_feat_dim: int,
        embedding_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.time_enc = Time2Vec(time_dim)

        # Input dim for queries and keys
        d_model = memory_dim + node_feat_dim
        self.d_model = d_model

        # Key/value dimension includes neighbour memory + edge features + time
        d_neighbor = d_model + edge_feat_dim + time_dim
        self.d_neighbor = d_neighbor

        # Project q, k, v to same head dimension
        self.head_dim = d_model // num_heads if d_model >= num_heads else d_model
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, self.head_dim * num_heads)
        self.k_proj = nn.Linear(d_neighbor, self.head_dim * num_heads)
        self.v_proj = nn.Linear(d_neighbor, self.head_dim * num_heads)

        self.attn_dropout = nn.Dropout(dropout)

        # Final MLP: concat(h_i, attn_output) -> embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(d_model + self.head_dim * num_heads, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,          # (N, memory_dim)
        node_features: torch.Tensor,    # (N, node_feat_dim)
        edge_index: torch.Tensor,       # (2, E) — full graph edges
        edge_attr: torch.Tensor,        # (E, edge_feat_dim)
        edge_timestamps: torch.Tensor,  # (E,) — timestamps of last interaction
        current_time: float,
    ) -> torch.Tensor:
        """Compute node embeddings z_i(t). Returns (N, embedding_dim)."""
        N = memory.shape[0]

        # h_i = memory + node features (with padding if dims differ)
        if node_features.shape[1] < self.node_feat_dim:
            pad = torch.zeros(N, self.node_feat_dim - node_features.shape[1], device=memory.device)
            node_features = torch.cat([node_features, pad], dim=1)
        h = torch.cat([memory, node_features[:, : self.node_feat_dim]], dim=-1)  # (N, d_model)

        # For nodes with no neighbours, return MLP(h_i, zeros)
        attn_out = torch.zeros(N, self.head_dim * self.num_heads, device=memory.device)

        if edge_index.shape[1] == 0:
            return self.mlp(torch.cat([h, attn_out], dim=-1))

        # Time encoding for edges
        dt = current_time - edge_timestamps.float()
        time_emb = self.time_enc(dt)  # (E, time_dim)

        src, tgt = edge_index  # src -> tgt
        # For each target node, gather its neighbours
        h_src = h[src]  # (E, d_model)
        neighbor_repr = torch.cat([h_src, edge_attr, time_emb], dim=-1)  # (E, d_neighbor)

        # Compute Q, K, V
        q = self.q_proj(h)                    # (N, head_dim * num_heads)
        k = self.k_proj(neighbor_repr)        # (E, head_dim * num_heads)
        v = self.v_proj(neighbor_repr)        # (E, head_dim * num_heads)

        # Per-target scatter attention
        # Gather query for each edge's target
        q_edge = q[tgt]  # (E, head_dim * num_heads)

        # Scale dot-product attention scores
        scale = math.sqrt(self.head_dim)
        scores = (q_edge * k).sum(dim=-1) / scale  # (E,)

        # Softmax per target node (scatter)
        scores_max = torch.zeros(N, device=scores.device)
        scores_max.scatter_reduce_(0, tgt, scores, reduce="amax", include_self=True)
        scores_exp = torch.exp(scores - scores_max[tgt])

        scores_sum = torch.zeros(N, device=scores.device)
        scores_sum.scatter_add_(0, tgt, scores_exp)
        alpha = scores_exp / scores_sum[tgt].clamp(min=1e-8)  # (E,)
        alpha = self.attn_dropout(alpha.unsqueeze(-1))

        # Weighted sum of values
        weighted_v = alpha * v  # (E, head_dim * num_heads)
        attn_out.scatter_add_(0, tgt.unsqueeze(-1).expand_as(weighted_v), weighted_v)

        z = self.mlp(torch.cat([h, attn_out], dim=-1))  # (N, embedding_dim)
        return z


# ---------------------------------------------------------------------------
# Full TGN Encoder
# ---------------------------------------------------------------------------

class TGNEncoder(nn.Module):
    """Complete TGN encoder for the DyFO module.

    Manages memory, processes event batches, and produces node embeddings.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = 172,
        embedding_dim: int = 100,
        time_dim: int = 100,
        event_feature_dim: int = 3,
        edge_type_emb_dim: int = 16,
        num_edge_types: int = 4,
        num_event_types: int = 7,
        node_feat_dim: int = 20,
        num_heads: int = 2,
        dropout: float = 0.1,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        # Memory buffer (not a parameter — updated in-place)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update_time", torch.zeros(num_nodes))

        # Sub-modules
        self.message_fn = MessageFunction(
            memory_dim=memory_dim,
            time_dim=time_dim,
            event_feature_dim=event_feature_dim,
            edge_type_emb_dim=edge_type_emb_dim,
            num_edge_types=num_edge_types,
            num_event_types=num_event_types,
        )
        self.aggregator = MessageAggregator(method=aggregation)
        self.memory_updater = MemoryUpdater(
            memory_dim=memory_dim,
            message_dim=self.message_fn.output_dim,
        )
        self.embedding_layer = TemporalGraphAttention(
            memory_dim=memory_dim,
            node_feat_dim=node_feat_dim,
            time_dim=time_dim,
            edge_feat_dim=edge_type_emb_dim,  # using edge type embedding as feature
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Edge type embedding for the GAT layer
        self.edge_type_emb_gat = nn.Embedding(num_edge_types + 1, edge_type_emb_dim)

    def reset_memory(self):
        """Zero out all node memories and timestamps. Call at episode start."""
        self.memory.zero_()
        self.last_update_time.zero_()

    def process_events(
        self,
        source_nodes: torch.Tensor,     # (B,)
        target_nodes: torch.Tensor,      # (B,)  -1 for node-only
        timestamps: torch.Tensor,        # (B,)
        event_features: torch.Tensor,    # (B, event_feature_dim)
        edge_type_ids: torch.Tensor,     # (B,)
        event_type_ids: torch.Tensor,    # (B,)
    ):
        """Process a batch of events: compute messages and update memory.

        This implements the Raw Message Store pattern:
        - Messages are computed from CURRENT memory
        - Memory is updated with the computed messages
        """
        B = source_nodes.shape[0]
        device = source_nodes.device

        # Gather source and target memories
        mem_src = self.memory[source_nodes]

        # For node-only events (target == -1), use zeros
        valid_target = target_nodes.clone()
        is_node_only = target_nodes == -1
        valid_target[is_node_only] = 0
        mem_tgt = self.memory[valid_target]
        mem_tgt[is_node_only] = 0.0

        # Delta time since last update
        dt = timestamps - self.last_update_time[source_nodes]

        # Compute raw messages
        raw_messages = self.message_fn(
            memory_src=mem_src,
            memory_tgt=mem_tgt,
            delta_t=dt,
            event_features=event_features,
            edge_type_ids=edge_type_ids,
            event_type_ids=event_type_ids,
        )

        # Aggregate messages per source node
        agg_messages = self.aggregator(
            node_ids=source_nodes,
            messages=raw_messages,
            timestamps=timestamps,
            num_nodes=self.num_nodes,
        )

        # Determine which nodes received messages
        update_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=device)
        update_mask[source_nodes.unique()] = True

        # Also process target nodes (bidirectional update for pair events)
        real_targets = target_nodes[~is_node_only]
        if len(real_targets) > 0:
            # Build messages for target side
            tgt_mem_src = self.memory[real_targets]
            src_for_tgt = source_nodes[~is_node_only]
            tgt_mem_tgt = self.memory[src_for_tgt]
            tgt_dt = timestamps[~is_node_only] - self.last_update_time[real_targets]

            tgt_messages = self.message_fn(
                memory_src=tgt_mem_src,
                memory_tgt=tgt_mem_tgt,
                delta_t=tgt_dt,
                event_features=event_features[~is_node_only],
                edge_type_ids=edge_type_ids[~is_node_only],
                event_type_ids=event_type_ids[~is_node_only],
            )

            tgt_agg = self.aggregator(
                node_ids=real_targets,
                messages=tgt_messages,
                timestamps=timestamps[~is_node_only],
                num_nodes=self.num_nodes,
            )
            agg_messages = agg_messages + tgt_agg
            update_mask[real_targets.unique()] = True

        # Update memory
        self.memory = self.memory_updater(agg_messages, self.memory, update_mask)

        # Update last-event timestamps
        for i in range(B):
            src = source_nodes[i].item()
            self.last_update_time[src] = max(
                self.last_update_time[src].item(), timestamps[i].item()
            )
            tgt = target_nodes[i].item()
            if tgt >= 0:
                self.last_update_time[tgt] = max(
                    self.last_update_time[tgt].item(), timestamps[i].item()
                )

    def compute_embeddings(
        self,
        node_features: torch.Tensor,       # (N, node_feat_dim)
        edge_index: torch.Tensor,           # (2, E)
        edge_type_ids: torch.Tensor,        # (E,)
        edge_timestamps: torch.Tensor,      # (E,)
        current_time: float,
    ) -> torch.Tensor:
        """Compute node embeddings z_i(t) using current memory + GAT.

        Returns (N, embedding_dim).
        """
        edge_attr = self.edge_type_emb_gat(edge_type_ids)
        return self.embedding_layer(
            memory=self.memory,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_timestamps=edge_timestamps,
            current_time=current_time,
        )

    def get_memory_checkpoint(self) -> Dict[str, torch.Tensor]:
        """Return current memory state for persistence."""
        return {
            "memory": self.memory.clone(),
            "last_update_time": self.last_update_time.clone(),
        }

    def load_memory_checkpoint(self, checkpoint: Dict[str, torch.Tensor]):
        """Restore memory from a checkpoint."""
        self.memory.copy_(checkpoint["memory"])
        self.last_update_time.copy_(checkpoint["last_update_time"])
