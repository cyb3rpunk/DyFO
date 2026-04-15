"""Relation-aware heterogeneous TGN building blocks for BL-17.

This module now covers Sessions 2 and 3 of BL-17:
  - semantic-group-specific message projections
  - intra-relation aggregation with deterministic mean pooling
  - inter-relation fusion with semantic attention
  - shared GRU updates over the fused relation-aware message
  - relation-aware temporal attention with projected real edge features
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent
from dyfo.core.model_variants import BaseGraphEncoder
from dyfo.core.relation_semantic_attention import RelationSemanticAttention
from dyfo.core.tgn_encoder import MemoryUpdater, Time2Vec


SEMANTIC_GROUP_ORDER: Tuple[str, ...] = (
    "node_event",
    "system_event",
    "pair_relation",
    "static_relation",
)

EVENT_TYPE_TO_GROUP: Dict[str, str] = {
    "PRICE_UPDATE": "node_event",
    "EARNINGS_REPORT": "node_event",
    "CREDIT_DOWNGRADE": "node_event",
    "CORP_ACTION": "node_event",
    "FED_DECISION": "system_event",
    "MACRO_RELEASE": "system_event",
    "CORRELATION_UPDATE": "pair_relation",
}

STATIC_RELATION_TYPES: Tuple[str, ...] = ("CORR", "SECT", "SUPL", "FACT")


@dataclass
class GroupedMessages:
    """Container for the active rows of one semantic group."""

    node_ids: torch.Tensor
    messages: torch.Tensor
    timestamps: torch.Tensor


class _GroupProjection(nn.Module):
    """Group-specific projection block with its own normalization."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class RelationAwareMessageFunction(nn.Module):
    """Encode event messages into semantic-group-specific latent spaces."""

    def __init__(
        self,
        memory_dim: int,
        time_dim: int,
        event_feature_dim: int,
        edge_type_emb_dim: int,
        num_edge_types: int,
        event_type_names: Sequence[str],
        relation_dim: int,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.relation_dim = relation_dim
        self.time_enc = Time2Vec(time_dim)
        self.event_type_names = list(event_type_names)

        self.group_name_to_idx = {
            group_name: idx for idx, group_name in enumerate(SEMANTIC_GROUP_ORDER)
        }
        self.event_type_to_group_idx = {
            idx: self.group_name_to_idx[EVENT_TYPE_TO_GROUP[name]]
            for idx, name in enumerate(self.event_type_names)
            if name in EVENT_TYPE_TO_GROUP
        }

        self.event_type_embeddings = nn.ModuleDict(
            {
                group_name: nn.Embedding(len(self.event_type_names), edge_type_emb_dim)
                for group_name in SEMANTIC_GROUP_ORDER
            }
        )
        self.edge_type_embeddings = nn.ModuleDict(
            {
                group_name: nn.Embedding(num_edge_types + 1, edge_type_emb_dim)
                for group_name in SEMANTIC_GROUP_ORDER
            }
        )

        raw_dim = (
            memory_dim
            + memory_dim
            + time_dim
            + event_feature_dim
            + edge_type_emb_dim
            + edge_type_emb_dim
        )
        self.projections = nn.ModuleDict(
            {
                group_name: _GroupProjection(raw_dim, relation_dim)
                for group_name in SEMANTIC_GROUP_ORDER
            }
        )

    def event_group_ids(self, event_type_ids: torch.Tensor) -> torch.Tensor:
        """Map each event type id to its semantic-group index."""
        group_ids = torch.full_like(
            event_type_ids, fill_value=-1, dtype=torch.long, device=event_type_ids.device
        )
        for event_id, group_id in self.event_type_to_group_idx.items():
            group_ids[event_type_ids == event_id] = group_id
        if (group_ids < 0).any():
            missing_ids = event_type_ids[group_ids < 0].unique().tolist()
            raise ValueError(
                f"Found event_type ids without semantic group mapping: {missing_ids}."
            )
        return group_ids

    def forward(
        self,
        memory_src: torch.Tensor,
        memory_tgt: torch.Tensor,
        delta_t: torch.Tensor,
        event_features: torch.Tensor,
        edge_type_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode active batch rows into one tensor per semantic group."""
        time_emb = self.time_enc(delta_t)
        group_ids = self.event_group_ids(event_type_ids)

        grouped: Dict[str, torch.Tensor] = {}
        for group_name, group_idx in self.group_name_to_idx.items():
            if group_name == "static_relation":
                continue

            group_mask = group_ids == group_idx
            if not group_mask.any():
                continue

            event_emb = self.event_type_embeddings[group_name](event_type_ids[group_mask])
            edge_emb = self.edge_type_embeddings[group_name](edge_type_ids[group_mask])
            raw = torch.cat(
                [
                    memory_src[group_mask],
                    memory_tgt[group_mask],
                    time_emb[group_mask],
                    event_features[group_mask],
                    edge_emb,
                    event_emb,
                ],
                dim=-1,
            )
            grouped[group_name] = self.projections[group_name](raw)

        return grouped


class StaticRelationEncoder(nn.Module):
    """Encode structural relation edges into the shared relation space."""

    def __init__(
        self,
        memory_dim: int,
        edge_type_emb_dim: int,
        num_edge_types: int,
        relation_dim: int,
        max_edge_feature_dim: int = 5,
    ):
        super().__init__()
        self.max_edge_feature_dim = max_edge_feature_dim
        self.edge_type_embedding = nn.Embedding(num_edge_types, edge_type_emb_dim)
        input_dim = 2 * memory_dim + edge_type_emb_dim + max_edge_feature_dim
        self.projection = _GroupProjection(input_dim, relation_dim)

    def forward(
        self,
        memory_src: torch.Tensor,
        memory_tgt: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape (E, d_edge).")
        if edge_features.size(1) > self.max_edge_feature_dim:
            raise ValueError(
                f"edge feature dim {edge_features.size(1)} exceeds max_edge_feature_dim={self.max_edge_feature_dim}."
            )

        if edge_features.size(1) < self.max_edge_feature_dim:
            pad = torch.zeros(
                edge_features.size(0),
                self.max_edge_feature_dim - edge_features.size(1),
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
            edge_features = torch.cat([edge_features, pad], dim=-1)

        edge_emb = self.edge_type_embedding(edge_type_ids)
        raw = torch.cat([memory_src, memory_tgt, edge_emb, edge_features], dim=-1)
        return self.projection(raw)


class IntraRelationAggregator(nn.Module):
    """Aggregate each semantic group independently with deterministic mean pooling."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        grouped_messages: Dict[str, GroupedMessages],
        num_nodes: int,
    ) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}

        for group_name, payload in grouped_messages.items():
            if payload.messages.ndim != 2:
                raise ValueError(f"{group_name} messages must have shape (B, d_rel).")

            msg_dim = payload.messages.shape[1]
            agg = torch.zeros(num_nodes, msg_dim, device=payload.messages.device)
            counts = torch.zeros(num_nodes, 1, device=payload.messages.device)

            agg.index_add_(0, payload.node_ids, payload.messages)
            counts.index_add_(
                0,
                payload.node_ids,
                torch.ones(payload.node_ids.shape[0], 1, device=payload.messages.device),
            )
            aggregated[group_name] = agg / counts.clamp(min=1.0)

        return aggregated


def _default_raw_edge_features(
    edge_type_ids: torch.Tensor,
    edge_type_names: Sequence[str],
    max_raw_edge_feature_dim: int,
) -> torch.Tensor:
    """Create a padded raw edge-feature tensor when callers provide none."""
    features = torch.zeros(
        edge_type_ids.shape[0],
        max_raw_edge_feature_dim,
        device=edge_type_ids.device,
        dtype=torch.float32,
    )
    edge_type_to_id = {name: idx for idx, name in enumerate(edge_type_names)}

    for type_name in ("SECT", "SUPL"):
        type_id = edge_type_to_id.get(type_name)
        if type_id is None:
            continue
        features[edge_type_ids == type_id, 0] = 1.0

    return features


class RelationAwareEdgeFeatureProjector(nn.Module):
    """Project raw edge features into a shared space according to edge type."""

    _TYPE_DIMS: Dict[str, int] = {
        "CORR": 3,
        "FACT": 5,
        "SUPL": 1,
        "SECT": 1,
    }

    def __init__(
        self,
        edge_type_names: Sequence[str],
        output_dim: int = 16,
        max_raw_edge_feature_dim: int = 5,
    ):
        super().__init__()
        self.edge_type_names = list(edge_type_names)
        self.output_dim = output_dim
        self.max_raw_edge_feature_dim = max_raw_edge_feature_dim
        self.edge_type_to_id = {name: idx for idx, name in enumerate(self.edge_type_names)}

        self.projectors = nn.ModuleDict(
            {
                type_name: nn.Linear(input_dim, output_dim)
                for type_name, input_dim in self._TYPE_DIMS.items()
            }
        )

    def _normalize_raw_features(
        self,
        edge_type_ids: torch.Tensor,
        edge_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_features is None:
            return _default_raw_edge_features(
                edge_type_ids=edge_type_ids,
                edge_type_names=self.edge_type_names,
                max_raw_edge_feature_dim=self.max_raw_edge_feature_dim,
            )

        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape (E, d_edge).")
        if edge_features.size(1) > self.max_raw_edge_feature_dim:
            raise ValueError(
                f"edge feature dim {edge_features.size(1)} exceeds max_raw_edge_feature_dim={self.max_raw_edge_feature_dim}."
            )

        if edge_features.size(1) == self.max_raw_edge_feature_dim:
            return edge_features.float()

        pad = torch.zeros(
            edge_features.size(0),
            self.max_raw_edge_feature_dim - edge_features.size(1),
            device=edge_features.device,
            dtype=edge_features.dtype,
        )
        return torch.cat([edge_features.float(), pad], dim=-1)

    def forward(
        self,
        edge_type_ids: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw = self._normalize_raw_features(edge_type_ids, edge_features)
        projected = torch.zeros(
            raw.size(0),
            self.output_dim,
            device=raw.device,
            dtype=raw.dtype,
        )

        for type_name, input_dim in self._TYPE_DIMS.items():
            type_id = self.edge_type_to_id.get(type_name)
            if type_id is None:
                continue
            mask = edge_type_ids == type_id
            if mask.any():
                projected[mask] = self.projectors[type_name](raw[mask, :input_dim])

        return projected


class RelationAwareTemporalGraphAttention(nn.Module):
    """Temporal graph attention using projected real edge features."""

    def __init__(
        self,
        memory_dim: int,
        node_feat_dim: int,
        time_dim: int,
        edge_type_names: Sequence[str],
        edge_feat_dim: int,
        embedding_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_raw_edge_feature_dim: int = 5,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.node_feat_dim = node_feat_dim
        self.time_enc = Time2Vec(time_dim)
        self.edge_projector = RelationAwareEdgeFeatureProjector(
            edge_type_names=edge_type_names,
            output_dim=edge_feat_dim,
            max_raw_edge_feature_dim=max_raw_edge_feature_dim,
        )

        d_model = memory_dim + node_feat_dim
        d_neighbor = d_model + edge_feat_dim + time_dim
        self.head_dim = d_model // num_heads if d_model >= num_heads else d_model
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, self.head_dim * num_heads)
        self.k_proj = nn.Linear(d_neighbor, self.head_dim * num_heads)
        self.v_proj = nn.Linear(d_neighbor, self.head_dim * num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + self.head_dim * num_heads, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = memory.shape[0]

        if node_features.shape[1] < self.node_feat_dim:
            pad = torch.zeros(
                num_nodes,
                self.node_feat_dim - node_features.shape[1],
                device=memory.device,
            )
            node_features = torch.cat([node_features, pad], dim=1)
        h = torch.cat([memory, node_features[:, : self.node_feat_dim]], dim=-1)

        attn_out = torch.zeros(
            num_nodes,
            self.head_dim * self.num_heads,
            device=memory.device,
        )

        if edge_index.shape[1] == 0:
            return self.mlp(torch.cat([h, attn_out], dim=-1))

        dt = current_time - edge_timestamps.float()
        time_emb = self.time_enc(dt)
        edge_attr = self.edge_projector(edge_type_ids=edge_type_ids, edge_features=edge_features)

        src, tgt = edge_index
        h_src = h[src]
        neighbor_repr = torch.cat([h_src, edge_attr, time_emb], dim=-1)

        q = self.q_proj(h)
        k = self.k_proj(neighbor_repr)
        v = self.v_proj(neighbor_repr)
        q_edge = q[tgt]

        scale = math.sqrt(self.head_dim)
        scores = (q_edge * k).sum(dim=-1) / scale

        scores_max = torch.zeros(num_nodes, device=scores.device)
        scores_max.scatter_reduce_(0, tgt, scores, reduce="amax", include_self=True)
        scores_exp = torch.exp(scores - scores_max[tgt])

        scores_sum = torch.zeros(num_nodes, device=scores.device)
        scores_sum.scatter_add_(0, tgt, scores_exp)
        alpha = scores_exp / scores_sum[tgt].clamp(min=1e-8)
        alpha = self.attn_dropout(alpha.unsqueeze(-1))

        weighted_v = alpha * v
        attn_out.scatter_add_(0, tgt.unsqueeze(-1).expand_as(weighted_v), weighted_v)

        self.last_src = src.detach().cpu()
        self.last_tgt = tgt.detach().cpu()
        self.last_alpha = alpha.detach().cpu().squeeze(-1)

        return self.mlp(torch.cat([h, attn_out], dim=-1))


class RelationAwareTGNEncoder(nn.Module):
    """Relation-aware heterogeneous TGN encoder core for BL-17."""

    def __init__(
        self,
        num_nodes: int,
        event_type_names: Sequence[str],
        memory_dim: int = 172,
        relation_dim: int = 172,
        time_dim: int = 100,
        event_feature_dim: int = 3,
        edge_type_emb_dim: int = 16,
        num_edge_types: int = 4,
        edge_type_names: Optional[Sequence[str]] = None,
        embedding_dim: int = 100,
        node_feat_dim: int = 20,
        edge_feat_dim: int = 16,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_static_edge_feature_dim: int = 5,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.relation_dim = relation_dim
        self.edge_type_names = list(edge_type_names or STATIC_RELATION_TYPES)
        self.last_attn_weights: Optional[torch.Tensor] = None

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update_time", torch.zeros(num_nodes))

        self.message_fn = RelationAwareMessageFunction(
            memory_dim=memory_dim,
            time_dim=time_dim,
            event_feature_dim=event_feature_dim,
            edge_type_emb_dim=edge_type_emb_dim,
            num_edge_types=num_edge_types,
            event_type_names=event_type_names,
            relation_dim=relation_dim,
        )
        self.static_relation_encoder = StaticRelationEncoder(
            memory_dim=memory_dim,
            edge_type_emb_dim=edge_type_emb_dim,
            num_edge_types=num_edge_types,
            relation_dim=relation_dim,
            max_edge_feature_dim=max_static_edge_feature_dim,
        )
        self.intra_relation_aggregator = IntraRelationAggregator()
        self.semantic_attention = RelationSemanticAttention(
            relation_dim=relation_dim,
            num_relations=len(SEMANTIC_GROUP_ORDER),
        )
        self.memory_updater = MemoryUpdater(
            memory_dim=memory_dim,
            message_dim=relation_dim,
        )
        self.embedding_layer = RelationAwareTemporalGraphAttention(
            memory_dim=memory_dim,
            node_feat_dim=node_feat_dim,
            time_dim=time_dim,
            edge_type_names=self.edge_type_names,
            edge_feat_dim=edge_feat_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_raw_edge_feature_dim=max_static_edge_feature_dim,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.orthogonal_(self.memory_updater.gru.weight_ih)
        nn.init.orthogonal_(self.memory_updater.gru.weight_hh)
        nn.init.zeros_(self.memory_updater.gru.bias_ih)
        nn.init.zeros_(self.memory_updater.gru.bias_hh)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset_memory(self) -> None:
        self.memory.zero_()
        self.last_update_time.zero_()
        self.last_attn_weights = None

    def build_intra_relation_messages(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        event_features: torch.Tensor,
        edge_type_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encode and aggregate event messages separately for each semantic group."""
        valid_target = target_nodes.clone()
        pair_mask = target_nodes != -1
        valid_target[~pair_mask] = 0

        mem_src = self.memory[source_nodes]
        mem_tgt = self.memory[valid_target].clone()
        mem_tgt[~pair_mask] = 0.0

        dt_src = timestamps - self.last_update_time[source_nodes]
        src_group_messages = self.message_fn(
            memory_src=mem_src,
            memory_tgt=mem_tgt,
            delta_t=dt_src,
            event_features=event_features,
            edge_type_ids=edge_type_ids,
            event_type_ids=event_type_ids,
        )

        grouped_payloads: Dict[str, GroupedMessages] = {}
        src_group_ids = self.message_fn.event_group_ids(event_type_ids)
        for group_name, group_idx in self.message_fn.group_name_to_idx.items():
            if group_name == "static_relation":
                continue
            group_mask = src_group_ids == group_idx
            if not group_mask.any():
                continue
            grouped_payloads[group_name] = GroupedMessages(
                node_ids=source_nodes[group_mask],
                messages=src_group_messages[group_name],
                timestamps=timestamps[group_mask],
            )

        if pair_mask.any():
            real_targets = target_nodes[pair_mask]
            src_for_tgt = source_nodes[pair_mask]
            dt_tgt = timestamps[pair_mask] - self.last_update_time[real_targets]
            tgt_group_messages = self.message_fn(
                memory_src=self.memory[real_targets],
                memory_tgt=self.memory[src_for_tgt],
                delta_t=dt_tgt,
                event_features=event_features[pair_mask],
                edge_type_ids=edge_type_ids[pair_mask],
                event_type_ids=event_type_ids[pair_mask],
            )
            tgt_group_ids = self.message_fn.event_group_ids(event_type_ids[pair_mask])

            for group_name, group_idx in self.message_fn.group_name_to_idx.items():
                if group_name == "static_relation":
                    continue
                group_mask = tgt_group_ids == group_idx
                if not group_mask.any():
                    continue
                payload = GroupedMessages(
                    node_ids=real_targets[group_mask],
                    messages=tgt_group_messages[group_name],
                    timestamps=timestamps[pair_mask][group_mask],
                )
                if group_name in grouped_payloads:
                    existing = grouped_payloads[group_name]
                    grouped_payloads[group_name] = GroupedMessages(
                        node_ids=torch.cat([existing.node_ids, payload.node_ids], dim=0),
                        messages=torch.cat([existing.messages, payload.messages], dim=0),
                        timestamps=torch.cat([existing.timestamps, payload.timestamps], dim=0),
                    )
                else:
                    grouped_payloads[group_name] = payload

        return self.intra_relation_aggregator(grouped_payloads, num_nodes=self.num_nodes)

    def encode_static_relations(
        self,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Project structural edges into the shared relation space."""
        if edge_features is None:
            edge_features = _default_raw_edge_features(
                edge_type_ids=edge_type_ids,
                edge_type_names=self.edge_type_names,
                max_raw_edge_feature_dim=self.static_relation_encoder.max_edge_feature_dim,
            )
        src, tgt = edge_index
        return self.static_relation_encoder(
            memory_src=self.memory[src],
            memory_tgt=self.memory[tgt],
            edge_type_ids=edge_type_ids,
            edge_features=edge_features,
        )

    def aggregate_static_relation(
        self,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate structural relation messages into a node-level state."""
        if edge_index.shape[1] == 0:
            return torch.zeros(self.num_nodes, self.relation_dim, device=self.memory.device)

        encoded = self.encode_static_relations(
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_features=edge_features,
        )
        src, tgt = edge_index
        payload = GroupedMessages(
            node_ids=torch.cat([src, tgt], dim=0),
            messages=torch.cat([encoded, encoded], dim=0),
            timestamps=torch.zeros(src.shape[0] * 2, device=encoded.device),
        )
        aggregated = self.intra_relation_aggregator(
            {"static_relation": payload},
            num_nodes=self.num_nodes,
        )
        return aggregated["static_relation"]

    def _collect_relation_states(
        self,
        grouped_messages: Dict[str, torch.Tensor],
        static_relation_state: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        relation_states: List[torch.Tensor] = []
        relation_indices: List[int] = []

        for group_name, group_idx in self.message_fn.group_name_to_idx.items():
            if group_name == "static_relation":
                continue
            group_state = grouped_messages.get(group_name)
            if group_state is None:
                continue
            relation_states.append(group_state)
            relation_indices.append(group_idx)

        if static_relation_state is not None:
            relation_states.append(static_relation_state)
            relation_indices.append(self.message_fn.group_name_to_idx["static_relation"])

        return relation_states, relation_indices

    def process_events(
        self,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        timestamps: torch.Tensor,
        event_features: torch.Tensor,
        edge_type_ids: torch.Tensor,
        event_type_ids: torch.Tensor,
        static_edge_index: Optional[torch.Tensor] = None,
        static_edge_type_ids: Optional[torch.Tensor] = None,
        static_edge_features: Optional[torch.Tensor] = None,
    ) -> None:
        """Fuse relation groups and update node memory with a shared GRU."""
        if source_nodes.numel() == 0:
            return

        grouped_messages = self.build_intra_relation_messages(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            timestamps=timestamps,
            event_features=event_features,
            edge_type_ids=edge_type_ids,
            event_type_ids=event_type_ids,
        )

        static_relation_state = None
        if static_edge_index is not None and static_edge_type_ids is not None:
            static_relation_state = self.aggregate_static_relation(
                edge_index=static_edge_index,
                edge_type_ids=static_edge_type_ids,
                edge_features=static_edge_features,
            )

        relation_states, relation_indices = self._collect_relation_states(
            grouped_messages=grouped_messages,
            static_relation_state=static_relation_state,
        )
        if not relation_states:
            return

        fused_messages, _ = self.semantic_attention(
            relation_states,
            relation_indices=relation_indices,
        )
        self.last_attn_weights = self.semantic_attention.last_attn_weights

        update_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=source_nodes.device)
        update_mask[source_nodes.unique()] = True
        real_targets = target_nodes[target_nodes >= 0]
        if real_targets.numel() > 0:
            update_mask[real_targets.unique()] = True

        self.memory = self.memory_updater(
            aggregated_messages=fused_messages,
            memory=self.memory,
            update_mask=update_mask,
        )

        # Vectorised timestamp update — replaces a Python loop with N .item() syncs.
        # scatter_reduce_ 'amax' does max-reduction in one GPU kernel per call.
        self.last_update_time.scatter_reduce_(
            0, source_nodes, timestamps, reduce="amax", include_self=True
        )
        real_targets = target_nodes[target_nodes >= 0]
        if real_targets.numel() > 0:
            ts_for_tgt = timestamps[target_nodes >= 0]
            self.last_update_time.scatter_reduce_(
                0, real_targets, ts_for_tgt, reduce="amax", include_self=True
            )

    def compute_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run relation-aware temporal attention using projected real edge features."""
        return self.embedding_layer(
            memory=self.memory,
            node_features=node_features,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_timestamps=edge_timestamps,
            current_time=current_time,
            edge_features=edge_features,
        )

    def get_memory_checkpoint(self) -> Dict[str, torch.Tensor]:
        return {
            "memory": self.memory.clone(),
            "last_update_time": self.last_update_time.clone(),
        }

    def load_memory_checkpoint(self, checkpoint: Dict[str, torch.Tensor]) -> None:
        self.memory.copy_(checkpoint["memory"])
        self.last_update_time.copy_(checkpoint["last_update_time"])


class RAHTGNEncoder(BaseGraphEncoder):
    """BaseGraphEncoder-compatible wrapper for the relation-aware TGN core."""

    def __init__(self, config: DyFOConfig, num_nodes: int):
        super().__init__(config, num_nodes)
        self._event_type_to_id = {event_type: idx for idx, event_type in enumerate(config.event_types)}
        self._edge_type_to_id = {edge_type: idx for idx, edge_type in enumerate(config.edge_types)}
        self._no_edge_id = len(config.edge_types)

        self.encoder = RelationAwareTGNEncoder(
            num_nodes=num_nodes,
            event_type_names=config.event_types,
            edge_type_names=config.edge_types,
            memory_dim=config.memory_dim,
            relation_dim=config.memory_dim,
            embedding_dim=config.embedding_dim,
            time_dim=config.time_encoding_dim,
            event_feature_dim=3,
            edge_type_emb_dim=config.edge_type_embedding_dim,
            num_edge_types=len(config.edge_types),
            node_feat_dim=config.node_feature_dim,
            edge_feat_dim=16,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
        )

    def _events_to_tensors(
        self,
        events: List[FinancialEvent],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        source_nodes = torch.tensor(
            [event.source_node for event in events],
            dtype=torch.long,
            device=device,
        )
        target_nodes = torch.tensor(
            [event.target_node for event in events],
            dtype=torch.long,
            device=device,
        )
        timestamps = torch.tensor(
            [event.timestamp for event in events],
            dtype=torch.float32,
            device=device,
        )
        # Stack on CPU first, then move in a single transfer — avoids N individual
        # .to(device) calls that each incur a GPU kernel launch + sync overhead.
        event_features = torch.stack([event.features for event in events]).to(device)
        edge_type_ids = torch.tensor(
            [
                self._edge_type_to_id.get(event.edge_type, self._no_edge_id)
                for event in events
            ],
            dtype=torch.long,
            device=device,
        )
        event_type_ids = torch.tensor(
            [self._event_type_to_id[event.event_type.value] for event in events],
            dtype=torch.long,
            device=device,
        )
        return {
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "timestamps": timestamps,
            "event_features": event_features,
            "edge_type_ids": edge_type_ids,
            "event_type_ids": event_type_ids,
        }

    @property
    def last_attn_weights(self) -> Optional[torch.Tensor]:
        return self.encoder.last_attn_weights

    def reset_state(self) -> None:
        self.encoder.reset_memory()

    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
        edge_features: Optional[torch.Tensor] = None,
    ) -> None:
        del node_features, edge_timestamps, current_time
        if not events:
            return
        tensors = self._events_to_tensors(events, self.encoder.memory.device)
        self.encoder.process_events(
            **tensors,
            static_edge_index=edge_index,
            static_edge_type_ids=edge_type_ids,
            static_edge_features=edge_features,
        )

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder.compute_embeddings(
            node_features=node_features,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_timestamps=edge_timestamps,
            current_time=current_time,
            edge_features=edge_features,
        )

    def detach_state(self) -> None:
        self.encoder.memory = self.encoder.memory.detach()
