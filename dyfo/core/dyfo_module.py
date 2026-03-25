"""DyFO Module — public interface for the Dynamic Financial Ontology.

This is the top-level nn.Module that:
  1. Receives financial events for a day (or batch of days)
  2. Processes them through the TGN pipeline
  3. Returns the graph embedding e_t ∈ R^d

Future MATTS integration: State Constructor consumes e_t as input.
Standalone usage: e_t can feed any downstream task (portfolio optimisation,
link prediction, regime classification, etc.).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import EventType, FinancialEvent
from dyfo.core.readout import get_readout
from dyfo.core.tgn_encoder import TGNEncoder

logger = logging.getLogger(__name__)


class DyFOModule(nn.Module):
    """Top-level DyFO module.

    Parameters
    ----------
    config : DyFOConfig
        Full configuration.
    num_nodes : int
        Number of asset nodes in the universe.
    readout_strategy : str
        One of 'mean', 'weighted', 'attention'.
    """

    def __init__(
        self,
        config: DyFOConfig,
        num_nodes: int,
        readout_strategy: str = "mean",
    ):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes

        # Maps event type string -> integer id
        self._event_type_to_id = {et: i for i, et in enumerate(config.event_types)}
        # Maps edge type string -> integer id  (None → last id)
        self._edge_type_to_id = {et: i for i, et in enumerate(config.edge_types)}
        self._no_edge_id = len(config.edge_types)  # for node-only events

        self.encoder = TGNEncoder(
            num_nodes=num_nodes,
            memory_dim=config.memory_dim,
            embedding_dim=config.embedding_dim,
            time_dim=config.time_encoding_dim,
            event_feature_dim=3,  # all event types use 3-dim features
            edge_type_emb_dim=config.edge_type_embedding_dim,
            num_edge_types=len(config.edge_types),
            num_event_types=len(config.event_types),
            node_feat_dim=config.node_feature_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            aggregation="mean",
        )

        self.readout = get_readout(readout_strategy, config.embedding_dim)

        # Staleness tracking (not a nn parameter)
        self.register_buffer(
            "_staleness_counter",
            torch.zeros(num_nodes, dtype=torch.long),
        )

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _events_to_tensors(
        self, events: List[FinancialEvent], device: torch.device,
    ) -> dict:
        """Convert a list of FinancialEvent to batched tensors."""
        source_nodes = torch.tensor(
            [e.source_node for e in events], dtype=torch.long, device=device
        )
        target_nodes = torch.tensor(
            [e.target_node for e in events], dtype=torch.long, device=device
        )
        timestamps = torch.tensor(
            [e.timestamp for e in events], dtype=torch.float32, device=device
        )
        features = torch.stack([e.features.to(device) for e in events])
        edge_type_ids = torch.tensor(
            [
                self._edge_type_to_id.get(e.edge_type, self._no_edge_id)
                for e in events
            ],
            dtype=torch.long,
            device=device,
        )
        event_type_ids = torch.tensor(
            [self._event_type_to_id[e.event_type.value] for e in events],
            dtype=torch.long,
            device=device,
        )
        return {
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "timestamps": timestamps,
            "event_features": features,
            "edge_type_ids": edge_type_ids,
            "event_type_ids": event_type_ids,
        }

    def process_day_events(self, events: List[FinancialEvent]):
        """Process all events for a single day (updates memory in-place)."""
        if not events:
            return
        device = self.encoder.memory.device
        tensors = self._events_to_tensors(events, device)
        self.encoder.process_events(**tensors)

        # Update staleness counters
        active_nodes = set()
        for e in events:
            active_nodes.add(e.source_node)
            if e.target_node >= 0:
                active_nodes.add(e.target_node)

        for nid in range(self.num_nodes):
            if nid in active_nodes:
                self._staleness_counter[nid] = 0
            else:
                self._staleness_counter[nid] += 1

    def get_stale_nodes(self) -> List[int]:
        """Return indices of nodes exceeding staleness threshold."""
        threshold = self.config.staleness_threshold_days
        return (self._staleness_counter >= threshold).nonzero(as_tuple=True)[0].tolist()

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def forward(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,       # (N, node_feat_dim)
        edge_index: torch.Tensor,          # (2, E) — static + dynamic edges
        edge_type_ids: torch.Tensor,       # (E,)
        edge_timestamps: torch.Tensor,     # (E,)
        current_time: float,
        active_mask: Optional[torch.Tensor] = None,  # (N,) bool
        readout_weights: Optional[torch.Tensor] = None,  # (N,) for weighted readout
    ) -> torch.Tensor:
        """Full forward pass: process events → compute embeddings → readout.

        Returns e_t of shape (embedding_dim,).
        """
        # 1. Process events (update memory)
        self.process_day_events(events)

        # 2. Compute per-node embeddings via GAT
        z = self.encoder.compute_embeddings(
            node_features=node_features,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_timestamps=edge_timestamps,
            current_time=current_time,
        )

        # 3. Global readout
        if readout_weights is not None and hasattr(self.readout, "forward"):
            # WeightedReadout signature
            try:
                e_t = self.readout(z, readout_weights, active_mask)
            except TypeError:
                e_t = self.readout(z, active_mask)
        else:
            e_t = self.readout(z, active_mask)

        return e_t

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def reset_memory(self):
        """Zero all memories. Call at start of each training episode."""
        self.encoder.reset_memory()
        self._staleness_counter.zero_()

    def save_memory_checkpoint(self, path: str):
        """Save memory state to disk."""
        ckpt = self.encoder.get_memory_checkpoint()
        ckpt["staleness_counter"] = self._staleness_counter.clone()
        torch.save(ckpt, path)

    def load_memory_checkpoint(self, path: str):
        """Load memory state from disk."""
        ckpt = torch.load(path, weights_only=True)
        staleness = ckpt.pop("staleness_counter", None)
        self.encoder.load_memory_checkpoint(ckpt)
        if staleness is not None:
            self._staleness_counter.copy_(staleness)
