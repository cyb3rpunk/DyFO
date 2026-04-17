"""TGAT encoder — Temporal Graph Attention Network (Xu et al., ICLR 2020).

This module adapts the TGAT architecture for the DyFO financial graph setting.
It is the canonical transformer baseline to compare against TGN: both process
continuous-time dynamic graphs, but differ architecturally in how they
aggregate temporal context:

  TGN:  GRU(persistent memory) → stateful, long-range context
  TGAT: Multi-Head Attention over k recent events → stateless, local context

That difference makes the TGN vs TGAT comparison a clean ablation of
*recurrent memory* vs *attention-based temporal aggregation*.

Architecture (per Xu et al. 2020, adapted for DyFO)
----------------------------------------------------
1. Time encoding : Time2Vec Φ(Δt) ∈ R^{d_time}
2. Event buffer  : ring buffer of the k most recent events per node
                   (source and target), stored as (feature, edge_emb, time_emb)
3. Temporal attention : for each node i, attend over its k buffered events
                        Q = v_i · W_Q
                        K = [f_e || Φ(Δt)] · W_K     (event context)
                        V = [f_e || Φ(Δt)] · W_V
                        h_i = Concat(heads) · W_O
4. GAT readout   : 1-layer GAT over the static structural graph using h_i
5. Output        : z_i = MLP(h_i + GAT_out_i)  ∈ R^{embedding_dim}

Design choices for DyFO
------------------------
- k = 20 events per node  (configurable via config.num_neighbors)
- Φ(Δt) dim = config.time_encoding_dim
- Attention heads = config.num_attention_heads
- The event buffer is cleared at reset_state() and detach_state()
- Node features v_i(t) are used as queries (Q) following the original paper
- TGAT is stateless between trading days: advance_day() only fills the buffer

References
----------
Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., & Achan, K. (2020).
  Inductive representation learning on temporal graphs.
  ICLR 2020. https://arxiv.org/abs/2002.07962
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent
from dyfo.core.model_variants import BaseGraphEncoder


# ---------------------------------------------------------------------------
# Time encoding (shared with TGN — re-implemented here to avoid circular deps)
# ---------------------------------------------------------------------------

class _Time2Vec(nn.Module):
    """Learnable time encoding: 1 linear + (dim-1) harmonic components.

    Φ(t) = [ω₀·t + φ₀, sin(ω₁·t + φ₁), …, sin(ω_{d-1}·t + φ_{d-1})]
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.periodic = nn.Linear(1, dim - 1, bias=True)
        self.dim = dim

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """dt: (B,) or scalar → (B, dim)."""
        dt = dt.reshape(-1, 1).float()
        lin = self.linear(dt)               # (B, 1)
        per = torch.sin(self.periodic(dt))  # (B, dim-1)
        return torch.cat([lin, per], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Event ring buffer (per node)
# ---------------------------------------------------------------------------

class _EventBuffer:
    """Fixed-capacity rolling buffer of recent events for one node.

    Stores ``(feature_vec, edge_type_id, timestamp)`` tuples.
    The feature_vec is the raw event feature tensor (detached from graph).
    """

    def __init__(self, capacity: int):
        self._cap = capacity
        self._buf: deque = deque(maxlen=capacity)

    def push(self, feature: torch.Tensor, edge_type_id: int, timestamp: float):
        self._buf.append((feature.detach().cpu(), edge_type_id, timestamp))

    def clear(self):
        self._buf.clear()

    def get(self) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        """Return all buffered (features, edge_types, timestamps)."""
        if not self._buf:
            return [], [], []
        feats, etypes, times = zip(*self._buf)
        return list(feats), list(etypes), list(times)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Temporal Self-Attention module (per node)
# ---------------------------------------------------------------------------

class _TemporalAttention(nn.Module):
    """Multi-head temporal attention for one node over its recent events.

    Query : node feature v_i projected to d_model
    Key   : event context [f_e || Φ(Δt)] projected to d_model
    Value : same as Key

    Output: attended context h_i ∈ R^{d_model}
    """

    def __init__(
        self,
        node_feat_dim: int,
        event_feat_dim: int,
        time_dim: int,
        edge_type_emb_dim: int,
        num_heads: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # Context dimension: event features + time encoding + edge_type embedding
        ctx_dim = event_feat_dim + time_dim + edge_type_emb_dim

        self.W_q = nn.Linear(node_feat_dim, d_model, bias=False)
        self.W_k = nn.Linear(ctx_dim, d_model, bias=False)
        self.W_v = nn.Linear(ctx_dim, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,    # (d_node)
        keys: torch.Tensor,     # (k, ctx_dim)
        values: torch.Tensor,   # (k, ctx_dim)
    ) -> torch.Tensor:          # (d_model)
        """Single-node temporal attention.  k = number of buffered events."""
        k = keys.shape[0]
        q = self.W_q(query).unsqueeze(0)  # (1, d_model)
        K = self.W_k(keys)                 # (k, d_model)
        V = self.W_v(values)               # (k, d_model)

        # Reshape to multi-head: (num_heads, 1, d_k) / (num_heads, k, d_k)
        q = q.view(1, self.num_heads, self.d_k).transpose(0, 1)  # (H, 1, d_k)
        K = K.view(k, self.num_heads, self.d_k).transpose(0, 1)  # (H, k, d_k)
        V = V.view(k, self.num_heads, self.d_k).transpose(0, 1)  # (H, k, d_k)

        scale = math.sqrt(self.d_k)
        attn = torch.bmm(q, K.transpose(1, 2)) / scale  # (H, 1, k)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, V)                          # (H, 1, d_k)
        out = out.transpose(0, 1).contiguous().view(1, self.d_model)  # (1, d_model)
        out = self.W_o(out).squeeze(0)                    # (d_model,)
        return self.norm(out + self.W_q(query))           # residual


# ---------------------------------------------------------------------------
# Main TGAT Encoder
# ---------------------------------------------------------------------------

class TGATEncoder(BaseGraphEncoder):
    """Temporal Graph Attention Network encoder for DyFO.

    Conforms to ``BaseGraphEncoder`` — stateless between windows (buffer is
    cleared at ``reset_state()``), making it fair to compare against TGN
    which carries GRU memory.

    Parameters
    ----------
    config : DyFOConfig
        DyFO configuration.  Used fields:
        - ``embedding_dim``      : output embedding dim (= d_model here)
        - ``num_attention_heads``: multi-head attention heads
        - ``time_encoding_dim``  : Time2Vec output dim
        - ``num_neighbors``      : event buffer capacity k per node
        - ``edge_type_embedding_dim``: edge type embedding dim
        - ``dropout``            : attention dropout
        - ``node_feature_dim``   : node feature dim (from NodeFeatureBuilder)
    num_nodes : int
        Number of asset nodes.
    """

    def __init__(self, config: DyFOConfig, num_nodes: int):
        super().__init__(config, num_nodes)

        self._num_nodes = num_nodes
        self._d_model = config.embedding_dim          # 100
        self._time_dim = config.time_encoding_dim     # 100
        self._n_heads = config.num_attention_heads    # 2
        self._k = config.num_neighbors                # 20
        self._et_dim = config.edge_type_embedding_dim # 16
        self._node_feat_dim = config.node_feature_dim # 20
        self._dropout_p = config.dropout

        num_edge_types = len(config.edge_types)       # 4 (CORR, SECT, SUPL, FACT)
        num_event_types = len(config.event_types)     # 7

        # --- Sub-modules ---
        self.time_enc = _Time2Vec(self._time_dim)
        self.edge_type_emb = nn.Embedding(num_edge_types + 1, self._et_dim)
        self.event_type_emb = nn.Embedding(num_event_types + 1, self._et_dim)

        # Event feature dim: raw feature from FinancialEvent (variable → project to 32)
        self._evt_proj_dim = 32
        self.event_proj = nn.Sequential(
            nn.Linear(20, self._evt_proj_dim),  # 20 = max raw event feature dim
            nn.ReLU(),
        )

        # Context dim used in attention: evt_proj + time_dim + et_dim
        ctx_dim = self._evt_proj_dim + self._time_dim + self._et_dim

        self.temporal_attn = _TemporalAttention(
            node_feat_dim=self._node_feat_dim,
            event_feat_dim=self._evt_proj_dim,
            time_dim=self._time_dim,
            edge_type_emb_dim=self._et_dim,
            num_heads=self._n_heads,
            d_model=self._d_model,
            dropout=self._dropout_p,
        )

        # 1-layer GAT for structural readout
        self.gat = GATConv(
            in_channels=self._d_model + self._node_feat_dim,
            out_channels=self._d_model // self._n_heads,
            heads=self._n_heads,
            dropout=self._dropout_p,
            concat=True,
        )  # output: (N, d_model)

        # Final projection MLP
        self.output_proj = nn.Sequential(
            nn.Linear(self._d_model * 2, self._d_model),
            nn.ReLU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._d_model, self._d_model),
        )

        # Event type string → int (order matches config.event_types)
        self._event_type_to_id: Dict[str, int] = {
            et: i for i, et in enumerate(config.event_types)
        }
        self._edge_type_to_id: Dict[str, int] = {
            et: i for i, et in enumerate(config.edge_types)
        }

        # --- State: one ring buffer per node ---
        self._buffers_list: List[_EventBuffer] = [
            _EventBuffer(self._k) for _ in range(num_nodes)
        ]
        # Cache of last call to _build_node_context (for advance_day path)
        self._last_time: float = 0.0

    # ------------------------------------------------------------------
    # BaseGraphEncoder interface
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear all event buffers (call at epoch start or between windows)."""
        for buf in self._buffers_list:
            buf.clear()
        self._last_time = 0.0

    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> None:
        """Push today's events into the per-node buffers (no gradient)."""
        self._last_time = current_time
        if not events:
            return

        for ev in events:
            # Normalise raw event feature to fixed dim=20 (pad/truncate)
            feat = ev.features
            if feat is None or feat.numel() == 0:
                feat = torch.zeros(20)
            else:
                feat = feat.detach().float()
                if feat.numel() < 20:
                    feat = F.pad(feat.view(-1), (0, 20 - feat.numel()))
                else:
                    feat = feat.view(-1)[:20]

            et_id = self._edge_type_to_id.get(ev.edge_type, len(self._edge_type_to_id))
            src = ev.source_node
            if 0 <= src < self._num_nodes:
                self._buffers_list[src].push(feat, et_id, current_time)
            tgt = ev.target_node
            if tgt is not None and 0 <= tgt < self._num_nodes:
                self._buffers_list[tgt].push(feat, et_id, current_time)

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,   # (N, node_feat_dim)
        edge_index: torch.Tensor,       # (2, E)
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:                  # (N, embedding_dim)
        """Compute TGAT embeddings (with gradient)."""
        device = node_features.device
        N = self._num_nodes

        # --- Step 1: temporal attention per node ---
        h_temporal = torch.zeros(N, self._d_model, device=device)

        for i in range(N):
            buf = self._buffers_list[i]
            if len(buf) == 0:
                # No events → zero context; node feature projected directly
                q = node_features[i]
                # project through W_q only as fallback
                h_temporal[i] = self.temporal_attn.W_q(q.to(device))
                continue

            feats_list, etypes_list, times_list = buf.get()
            k = len(feats_list)

            # Project event features
            raw_feats = torch.stack(feats_list, dim=0).to(device)  # (k, 20)
            evt_proj = self.event_proj(raw_feats)                   # (k, evt_proj_dim)

            # Time encoding: Δt = current_time - event_time
            dt = torch.tensor(
                [current_time - t for t in times_list], dtype=torch.float32, device=device
            )
            time_emb = self.time_enc(dt)  # (k, time_dim)

            # Edge-type embeddings
            et_ids = torch.tensor(etypes_list, dtype=torch.long, device=device)
            et_emb = self.edge_type_emb(et_ids)  # (k, et_dim)

            # Context: [evt_proj || time_emb || et_emb]
            ctx = torch.cat([evt_proj, time_emb, et_emb], dim=-1)  # (k, ctx_dim)

            query = node_features[i].to(device)  # (node_feat_dim,)
            h_i = self.temporal_attn(query, ctx, ctx)  # (d_model,)
            h_temporal[i] = h_i

        # --- Step 2: GAT structural readout ---
        # Input to GAT: concat temporal context + raw node features
        gat_in = torch.cat([h_temporal, node_features], dim=-1)  # (N, d_model + node_feat_dim)
        edge_index_dev = edge_index.to(device)
        gat_out = self.gat(gat_in, edge_index_dev)  # (N, d_model)
        gat_out = F.elu(gat_out)

        # --- Step 3: Final projection ---
        combined = torch.cat([h_temporal, gat_out], dim=-1)  # (N, d_model * 2)
        z = self.output_proj(combined)                        # (N, embedding_dim)
        return z

    def detach_state(self) -> None:
        """No-op — TGAT is stateless (buffers hold detached CPU tensors)."""
