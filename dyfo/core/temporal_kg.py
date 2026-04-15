"""BL-18 Temporal KG encoder and interpretability helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent
from dyfo.core.model_variants import BaseGraphEncoder
from dyfo.core.temporal_kg_adapter import CANONICAL_RELATIONS, TemporalFact, TemporalKGAdapter
from dyfo.core.tgn_encoder import Time2Vec


MAX_FACT_ATTRS = 5


class TemporalKGCore(nn.Module):
    """Simple recurrent temporal KG scorer focused on interpretability."""

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int,
        embedding_dim: int,
        node_feat_dim: int,
        relation_names: Sequence[str] = CANONICAL_RELATIONS,
        time_dim: int = 16,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim
        self.node_feat_dim = node_feat_dim
        self.relation_names = list(relation_names)
        self.relation_to_id = {name: idx for idx, name in enumerate(self.relation_names)}
        self.adapter = TemporalKGAdapter(num_asset_nodes=num_nodes)

        self.register_buffer("node_state", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update_time", torch.zeros(num_nodes))
        self.relation_embeddings = nn.Embedding(len(self.relation_names), memory_dim)
        self.pseudo_entity_embeddings = nn.Embedding(16, memory_dim)
        self.time_encoder = Time2Vec(time_dim)
        self.attr_proj = nn.Linear(MAX_FACT_ATTRS + time_dim, memory_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.node_feature_proj = nn.Linear(node_feat_dim, memory_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(memory_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.gru = nn.GRUCell(memory_dim, memory_dim)

        self.fact_history: List[TemporalFact] = []
        self.last_fact_batch: List[TemporalFact] = []
        self.last_explanations: List[dict] = []
        self.last_relation_scores: Dict[str, float] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset_state(self) -> None:
        self.node_state.zero_()
        self.last_update_time.zero_()
        self.fact_history = []
        self.last_fact_batch = []
        self.last_explanations = []
        self.last_relation_scores = {}

    def _pseudo_entity_id(self, fact: TemporalFact) -> int:
        stable_key = f"{fact.relation}|{fact.tail}|{fact.source}"
        return sum(ord(ch) for ch in stable_key) % self.pseudo_entity_embeddings.num_embeddings

    def _fact_attr_tensor(self, fact: TemporalFact, device: torch.device) -> torch.Tensor:
        values = list(fact.attributes.values())[:MAX_FACT_ATTRS]
        if len(values) < MAX_FACT_ATTRS:
            values.extend([0.0] * (MAX_FACT_ATTRS - len(values)))
        return torch.tensor(values, dtype=torch.float32, device=device)

    def _fact_message_batch(
        self,
        head_indices: torch.Tensor,       # (F,) long
        tail_combined_ids: torch.Tensor,  # (F,) long  — asset idx OR num_nodes+pseudo_id
        relation_ids: torch.Tensor,       # (F,) long
        attrs: torch.Tensor,              # (F, MAX_FACT_ATTRS) float
        dts: torch.Tensor,               # (F,) float
    ) -> torch.Tensor:                    # (F, memory_dim)
        """Vectorised batch replacement for the old per-fact _fact_message loop."""
        # Combined lookup: asset node states followed by pseudo-entity embeddings
        combined = torch.cat(
            [self.node_state, self.pseudo_entity_embeddings.weight], dim=0
        )  # (num_nodes + num_pseudo, memory_dim)

        head_states = self.node_state[head_indices]          # (F, memory_dim)
        tail_states = combined[tail_combined_ids]            # (F, memory_dim)
        relation_states = self.relation_embeddings(relation_ids)  # (F, memory_dim)

        # Time2Vec already handles batched input of shape (F,) → (F, time_dim)
        dt_enc = self.time_encoder(dts)                      # (F, time_dim)
        attr_repr = self.attr_proj(torch.cat([attrs, dt_enc], dim=-1))  # (F, memory_dim)

        mlp_input = torch.cat(
            [head_states, relation_states, tail_states + attr_repr], dim=-1
        )  # (F, 3*memory_dim)
        return self.message_mlp(mlp_input)                   # (F, memory_dim)

    # kept for scoring/explain paths that operate on single facts
    def _fact_message(self, fact: TemporalFact, device: torch.device) -> torch.Tensor:
        head_idx = int(fact.head.split(":")[-1])
        head_state = self.node_state[head_idx]
        relation_id = self.relation_to_id[fact.relation]
        relation_state = self.relation_embeddings.weight[relation_id]

        if fact.tail.startswith("asset:"):
            tail_idx = int(fact.tail.split(":")[-1])
            tail_state = self.node_state[tail_idx]
        else:
            tail_state = self.pseudo_entity_embeddings.weight[self._pseudo_entity_id(fact)]

        dt = torch.tensor(
            [fact.timestamp - float(self.last_update_time[head_idx].item())],
            dtype=torch.float32,
            device=device,
        )
        attr = self._fact_attr_tensor(fact, device=device)
        attr_repr = self.attr_proj(torch.cat([attr, self.time_encoder(dt).squeeze(0)], dim=0))
        return self.message_mlp(torch.cat([head_state, relation_state, tail_state + attr_repr], dim=0))

    def score_fact(self, fact: TemporalFact, device: torch.device) -> float:
        head_idx = int(fact.head.split(":")[-1])
        relation_id = self.relation_to_id[fact.relation]
        head_state = self.node_state[head_idx]
        relation_state = self.relation_embeddings.weight[relation_id]
        if fact.tail.startswith("asset:"):
            tail_state = self.node_state[int(fact.tail.split(":")[-1])]
        else:
            tail_state = self.pseudo_entity_embeddings.weight[self._pseudo_entity_id(fact)]
        score = torch.dot(head_state + relation_state, tail_state) / max(1, self.memory_dim)
        return float(score.detach().cpu().item())

    def _update_explanations(self, facts: Iterable[TemporalFact], device: torch.device) -> None:
        scored = []
        relation_scores: Dict[str, List[float]] = defaultdict(list)
        for fact in facts:
            score = self.score_fact(fact, device=device)
            relation_scores[fact.relation].append(score)
            scored.append(
                {
                    "head": fact.head,
                    "relation": fact.relation,
                    "tail": fact.tail,
                    "timestamp": float(fact.timestamp),
                    "source": fact.source,
                    "plausibility": score,
                    "attributes": {k: float(v) for k, v in fact.attributes.items()},
                }
            )

        scored.sort(key=lambda item: (-item["plausibility"], item["timestamp"], item["head"], item["tail"]))
        self.last_explanations = scored[:10]
        self.last_relation_scores = {
            relation: float(sum(values) / len(values))
            for relation, values in relation_scores.items()
            if values
        }

    def process_day(
        self,
        events: List[FinancialEvent],
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_type_names: Sequence[str],
        current_time: float,
        edge_features: Optional[torch.Tensor] = None,
    ) -> None:
        facts = self.adapter.events_to_facts(events)
        if edge_index.numel() > 0:
            facts.extend(
                self.adapter.static_graph_to_facts(
                    edge_index=edge_index,
                    edge_type_ids=edge_type_ids,
                    edge_type_names=edge_type_names,
                    timestamp=current_time,
                    edge_features=edge_features,
                )
            )
        facts.sort(key=lambda fact: (fact.timestamp, fact.head, fact.relation, fact.tail))
        self.last_fact_batch = facts
        self.fact_history.extend(facts)

        if not facts:
            self.last_explanations = []
            self.last_relation_scores = {}
            return

        device = self.node_state.device
        num_pseudo = self.pseudo_entity_embeddings.num_embeddings

        # ------------------------------------------------------------------
        # Step 1: preprocess facts into tensors (pure Python, no CUDA ops)
        # ------------------------------------------------------------------
        head_indices_list: list[int] = []
        tail_combined_list: list[int] = []   # asset idx OR num_nodes+pseudo_id
        relation_ids_list: list[int] = []
        attrs_list: list[list[float]] = []
        dts_list: list[float] = []
        latest_time_by_node: Dict[int, float] = {}

        # For aggregation: which node receives which message index
        recipient_list: list[int] = []   # node idx
        msg_idx_list: list[int] = []     # index into the (F,) messages tensor

        for f_idx, fact in enumerate(facts):
            head_idx = int(fact.head.split(":")[-1])
            relation_id = self.relation_to_id[fact.relation]

            if fact.tail.startswith("asset:"):
                tail_combined = int(fact.tail.split(":")[-1])
            else:
                tail_combined = self.num_nodes + (self._pseudo_entity_id(fact) % num_pseudo)

            values = list(fact.attributes.values())[:MAX_FACT_ATTRS]
            if len(values) < MAX_FACT_ATTRS:
                values.extend([0.0] * (MAX_FACT_ATTRS - len(values)))

            head_indices_list.append(head_idx)
            tail_combined_list.append(tail_combined)
            relation_ids_list.append(relation_id)
            attrs_list.append(values)
            dts_list.append(fact.timestamp - float(self.last_update_time[head_idx].item()))

            # head always receives its own message
            recipient_list.append(head_idx)
            msg_idx_list.append(f_idx)

            latest_time_by_node[head_idx] = max(
                latest_time_by_node.get(head_idx, fact.timestamp), fact.timestamp
            )

            # asset tail also receives the same message
            if fact.tail.startswith("asset:"):
                tail_idx = tail_combined
                recipient_list.append(tail_idx)
                msg_idx_list.append(f_idx)
                latest_time_by_node[tail_idx] = max(
                    latest_time_by_node.get(tail_idx, fact.timestamp), fact.timestamp
                )

        # ------------------------------------------------------------------
        # Step 2: single batched forward pass for all F messages
        # ------------------------------------------------------------------
        head_t = torch.tensor(head_indices_list, dtype=torch.long, device=device)
        tail_t = torch.tensor(tail_combined_list, dtype=torch.long, device=device)
        rel_t = torch.tensor(relation_ids_list, dtype=torch.long, device=device)
        attrs_t = torch.tensor(attrs_list, dtype=torch.float32, device=device)   # (F, MAX_FACT_ATTRS)
        dts_t = torch.tensor(dts_list, dtype=torch.float32, device=device)        # (F,)

        messages = self._fact_message_batch(head_t, tail_t, rel_t, attrs_t, dts_t)  # (F, memory_dim)

        # ------------------------------------------------------------------
        # Step 3: scatter-mean — average messages per recipient node
        # ------------------------------------------------------------------
        R = len(recipient_list)
        recip_t = torch.tensor(recipient_list, dtype=torch.long, device=device)   # (R,)
        msg_sel_t = torch.tensor(msg_idx_list, dtype=torch.long, device=device)   # (R,)
        selected_msgs = messages[msg_sel_t]                                        # (R, memory_dim)

        msg_sum = torch.zeros(self.num_nodes, self.memory_dim, device=device)
        msg_count = torch.zeros(self.num_nodes, device=device)
        msg_sum.index_add_(0, recip_t, selected_msgs)
        msg_count.index_add_(0, recip_t, torch.ones(R, device=device))

        update_mask = msg_count > 0
        update_idx_t = update_mask.nonzero(as_tuple=True)[0]           # (K,)

        # ------------------------------------------------------------------
        # Step 4: single batched GRU call for all K updated nodes
        # ------------------------------------------------------------------
        if update_idx_t.numel() > 0:
            mean_msgs = msg_sum[update_idx_t] / msg_count[update_idx_t].unsqueeze(-1)  # (K, memory_dim)
            hidden = self.node_state[update_idx_t]                                      # (K, memory_dim)
            new_states = self.gru(mean_msgs, hidden)                                    # (K, memory_dim)
            # non-inplace index_put avoids autograd version-counter conflict
            self._buffers["node_state"] = self.node_state.index_put((update_idx_t,), new_states)

        # timestamp updates are outside the grad path — plain Python dict is fine
        for node_idx, t in latest_time_by_node.items():
            self.last_update_time[node_idx] = t

        # explanations are only useful outside training (and expensive: O(F))
        if not self.training:
            self._update_explanations(facts, device=device)

    def compute_embeddings(self, node_features: torch.Tensor) -> torch.Tensor:
        if node_features.shape[1] < self.node_feat_dim:
            pad = torch.zeros(
                node_features.shape[0],
                self.node_feat_dim - node_features.shape[1],
                dtype=node_features.dtype,
                device=node_features.device,
            )
            node_features = torch.cat([node_features, pad], dim=1)
        feature_repr = self.node_feature_proj(node_features[:, : self.node_feat_dim])
        return self.output_proj(torch.cat([self.node_state, feature_repr], dim=-1))

    def export_artifacts(self, top_k: int = 10) -> dict:
        relation_histogram: Dict[str, int] = defaultdict(int)
        for fact in self.fact_history:
            relation_histogram[fact.relation] += 1

        return {
            "n_facts_seen": len(self.fact_history),
            "last_fact_batch_size": len(self.last_fact_batch),
            "relation_histogram": dict(sorted(relation_histogram.items())),
            "last_relation_scores": self.last_relation_scores,
            "top_explanations": self.last_explanations[:top_k],
            "recent_facts": [fact.to_dict() for fact in self.last_fact_batch[:top_k]],
        }


class TemporalKGEncoder(BaseGraphEncoder):
    """BaseGraphEncoder wrapper for the BL-18 Temporal KG arm."""

    def __init__(self, config: DyFOConfig, num_nodes: int):
        super().__init__(config, num_nodes)
        self.encoder = TemporalKGCore(
            num_nodes=num_nodes,
            memory_dim=config.memory_dim,
            embedding_dim=config.embedding_dim,
            node_feat_dim=config.node_feature_dim,
            time_dim=max(8, min(config.time_encoding_dim, 32)),
        )

    @property
    def last_explanations(self) -> List[dict]:
        return self.encoder.last_explanations

    def export_temporal_kg_artifacts(self, top_k: int = 10) -> dict:
        return self.encoder.export_artifacts(top_k=top_k)

    def reset_state(self) -> None:
        self.encoder.reset_state()

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
        del node_features, edge_timestamps
        self.encoder.process_day(
            events=events,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_type_names=self.config.edge_types,
            current_time=current_time,
            edge_features=edge_features,
        )

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        del edge_index, edge_type_ids, edge_timestamps, current_time
        return self.encoder.compute_embeddings(node_features=node_features)

    def detach_state(self) -> None:
        self.encoder._buffers["node_state"] = self.encoder.node_state.detach()
