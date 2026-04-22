"""Graph builder — assembles the heterogeneous financial graph.

Combines node features, edge features, and the event stream into
PyG-compatible data structures for the TGN encoder.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.event_stream import (
    EventStreamBuilder,
    EventType,
    FinancialEvent,
    timestamp_to_float,
)

logger = logging.getLogger(__name__)


class FinancialGraph:
    """Container for the heterogeneous financial graph at a point in time.

    Stores node indices, edge index per type, edge features, and node features.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_types: List[str],
    ):
        self.num_nodes = num_nodes
        self.edge_types = edge_types

        # Per edge-type adjacency: dict[type] -> (2, E_type) LongTensor
        self.edge_index: Dict[str, torch.Tensor] = {
            et: torch.zeros(2, 0, dtype=torch.long) for et in edge_types
        }
        # Per edge-type features: dict[type] -> (E_type, feat_dim) Tensor
        self.edge_attr: Dict[str, torch.Tensor] = {
            et: torch.zeros(0, 0) for et in edge_types
        }
        # Node features: (N, node_feat_dim)
        self.node_features: Optional[torch.Tensor] = None

    def add_edges(
        self,
        edge_type: str,
        sources: List[int],
        targets: List[int],
        features: Optional[torch.Tensor] = None,
    ):
        """Add edges of a given type (bidirectional)."""
        if not sources:
            return
        src = torch.tensor(sources + targets, dtype=torch.long)
        tgt = torch.tensor(targets + sources, dtype=torch.long)
        new_index = torch.stack([src, tgt])

        existing = self.edge_index[edge_type]
        if existing.shape[1] > 0:
            self.edge_index[edge_type] = torch.cat([existing, new_index], dim=1)
        else:
            self.edge_index[edge_type] = new_index

        if features is not None:
            feat_bi = torch.cat([features, features], dim=0)
            existing_attr = self.edge_attr[edge_type]
            if existing_attr.numel() > 0:
                self.edge_attr[edge_type] = torch.cat([existing_attr, feat_bi], dim=0)
            else:
                self.edge_attr[edge_type] = feat_bi

    def get_full_edge_index(self) -> torch.Tensor:
        """Concatenated edge index across all types. Shape (2, E_total)."""
        parts = [ei for ei in self.edge_index.values() if ei.shape[1] > 0]
        if not parts:
            return torch.zeros(2, 0, dtype=torch.long)
        return torch.cat(parts, dim=1)

    def get_edge_type_names(self) -> List[str]:
        """Edge type names in the same order as get_full_edge_index / get_edge_type_ids."""
        return list(self.edge_types)

    def get_edge_type_ids(self) -> torch.Tensor:
        """Integer ID per edge, matching the order of get_full_edge_index()."""
        ids = []
        for i, et in enumerate(self.edge_types):
            ei = self.edge_index[et]
            if ei.shape[1] > 0:
                ids.append(torch.full((ei.shape[1],), i, dtype=torch.long))
        if not ids:
            return torch.zeros(0, dtype=torch.long)
        return torch.cat(ids)

    @property
    def total_edges(self) -> int:
        return sum(ei.shape[1] for ei in self.edge_index.values())


class GraphBuilder:
    """Orchestrates construction of the FinancialGraph from raw data.

    Usage:
        builder = GraphBuilder(config, data_config, tickers)
        graph = builder.build_initial_graph(ticker_info, sector_edges, supl_edges, factor_edges)
        events = builder.build_event_stream(prices, volumes, earnings, ...)
    """

    def __init__(
        self,
        config: DyFOConfig,
        tickers: List[str],
    ):
        self._config = config
        self._tickers = tickers
        self._ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        self._event_builder = EventStreamBuilder(self._ticker_to_idx)

    @property
    def ticker_to_idx(self) -> Dict[str, int]:
        return self._ticker_to_idx

    @property
    def num_nodes(self) -> int:
        return len(self._tickers)

    def build_initial_graph(
        self,
        sector_edges: List[Tuple[int, int, str]],
        supply_chain_edges: List[Tuple[int, int, float]],
        factor_edges: List[Tuple[int, int, np.ndarray]],
    ) -> FinancialGraph:
        """Build the initial static graph structure (SECT, SUPL, FACT edges).

        CORR edges are added dynamically via CORRELATION_UPDATE events.
        """
        graph = FinancialGraph(
            num_nodes=self.num_nodes,
            edge_types=self._config.edge_types,
        )

        # SECT edges
        if sector_edges:
            src = [e[0] for e in sector_edges]
            tgt = [e[1] for e in sector_edges]
            feat = torch.ones(len(sector_edges), 1)
            graph.add_edges("SECT", src, tgt, feat)

        # SUPL edges
        if supply_chain_edges:
            src = [e[0] for e in supply_chain_edges]
            tgt = [e[1] for e in supply_chain_edges]
            feat = torch.tensor([[e[2]] for e in supply_chain_edges], dtype=torch.float32)
            graph.add_edges("SUPL", src, tgt, feat)

        # FACT edges
        if factor_edges:
            src = [e[0] for e in factor_edges]
            tgt = [e[1] for e in factor_edges]
            feat = torch.tensor(
                np.array([e[2] for e in factor_edges]), dtype=torch.float32
            )
            graph.add_edges("FACT", src, tgt, feat)

        logger.info(
            "Initial graph: %d nodes, %d edges (SECT=%d, SUPL=%d, FACT=%d)",
            graph.num_nodes,
            graph.total_edges,
            graph.edge_index["SECT"].shape[1],
            graph.edge_index["SUPL"].shape[1],
            graph.edge_index["FACT"].shape[1],
        )
        return graph

    def build_event_stream(
        self,
        prices,
        volumes,
        earnings_df,
        actions_df,
        macro_events_df,
        corr_series,
        corr_pairs,
    ) -> List[FinancialEvent]:
        """Build the full sorted event stream from all data sources."""
        price_events = self._event_builder.build_price_events(prices, volumes)
        earnings_events = self._event_builder.build_earnings_events(earnings_df)
        action_events = self._event_builder.build_corp_action_events(actions_df)
        macro_events = self._event_builder.build_macro_events(
            macro_events_df, self.num_nodes
        )
        corr_events = self._event_builder.build_correlation_events(
            corr_series, corr_pairs
        )

        all_events = EventStreamBuilder.merge_and_sort(
            price_events,
            earnings_events,
            action_events,
            macro_events,
            corr_events,
        )
        logger.info("Total event stream: %d events", len(all_events))
        return all_events
