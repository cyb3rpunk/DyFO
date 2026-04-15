"""Deterministic adapters from DyFO events/graph edges to temporal KG facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from dyfo.core.event_stream import EventType, FinancialEvent


CANONICAL_RELATIONS: Sequence[str] = (
    "correlated_with",
    "in_sector",
    "supply_link",
    "similar_factor_profile",
    "affected_by_event",
    "exposed_to_macro",
)


EVENT_RELATION_MAP: Dict[EventType, str] = {
    EventType.PRICE_UPDATE: "affected_by_event",
    EventType.EARNINGS_REPORT: "affected_by_event",
    EventType.CREDIT_DOWNGRADE: "affected_by_event",
    EventType.CORP_ACTION: "affected_by_event",
    EventType.CORRELATION_UPDATE: "correlated_with",
    EventType.FED_DECISION: "exposed_to_macro",
    EventType.MACRO_RELEASE: "exposed_to_macro",
}


STATIC_EDGE_RELATION_MAP: Dict[str, str] = {
    "CORR": "correlated_with",
    "SECT": "in_sector",
    "SUPL": "supply_link",
    "FACT": "similar_factor_profile",
}

EDGE_TYPE_CODE_MAP: Dict[str, float] = {
    "CORR": 1.0,
    "SECT": 2.0,
    "SUPL": 3.0,
    "FACT": 4.0,
}


EVENT_TAIL_ENTITY_MAP: Dict[EventType, str] = {
    EventType.PRICE_UPDATE: "event:price_update",
    EventType.EARNINGS_REPORT: "event:earnings_report",
    EventType.CREDIT_DOWNGRADE: "event:credit_downgrade",
    EventType.CORP_ACTION: "event:corp_action",
    EventType.FED_DECISION: "macro:fed_funds_rate",
    EventType.MACRO_RELEASE: "macro:macro_release",
}


@dataclass(frozen=True)
class TemporalFact:
    """Canonical temporal fact used by the BL-18 Temporal KG arm."""

    head: str
    relation: str
    tail: str
    timestamp: float
    attributes: Dict[str, float] = field(default_factory=dict)
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "timestamp": float(self.timestamp),
            "attributes": {k: float(v) for k, v in self.attributes.items()},
            "source": self.source,
        }


def _feature_names_for_event(event_type: EventType) -> List[str]:
    if event_type == EventType.PRICE_UPDATE:
        return ["delta_ret", "vol_1d", "volume_norm"]
    if event_type == EventType.EARNINGS_REPORT:
        return ["surprise_eps", "revenue_beat", "guidance_delta"]
    if event_type == EventType.FED_DECISION:
        return ["surprise_z", "change", "sentiment"]
    if event_type == EventType.CREDIT_DOWNGRADE:
        return ["notch_delta", "outlook_code", "sector_contagion"]
    if event_type == EventType.CORP_ACTION:
        return ["event_type_code", "deal_value_norm", "premium"]
    if event_type == EventType.CORRELATION_UPDATE:
        return ["rho", "delta_rho", "significance"]
    if event_type == EventType.MACRO_RELEASE:
        return ["surprise_z", "revision", "vol_impact"]
    return [f"feature_{idx}" for idx in range(3)]


def _feature_names_for_edge_type(edge_type: str, feature_dim: int) -> List[str]:
    if edge_type == "CORR":
        return ["rho", "delta_rho", "significance"][:feature_dim]
    if edge_type == "SECT":
        return ["sector_overlap"][:feature_dim]
    if edge_type == "SUPL":
        return ["strength"][:feature_dim]
    if edge_type == "FACT":
        return [f"d_beta_{idx}" for idx in range(1, feature_dim + 1)]
    return [f"feature_{idx}" for idx in range(feature_dim)]


class TemporalKGAdapter:
    """Deterministic conversion from DyFO structures to TemporalFact objects."""

    def __init__(self, num_asset_nodes: int):
        self.num_asset_nodes = num_asset_nodes

    def asset_entity(self, node_idx: int) -> str:
        return f"asset:{node_idx}"

    def event_to_fact(self, event: FinancialEvent) -> TemporalFact:
        relation = EVENT_RELATION_MAP[event.event_type]
        head = self.asset_entity(event.source_node)
        if event.event_type == EventType.CORRELATION_UPDATE and event.target_node >= 0:
            tail = self.asset_entity(event.target_node)
        else:
            tail = EVENT_TAIL_ENTITY_MAP.get(event.event_type, f"event:{event.event_type.value.lower()}")

        feature_names = _feature_names_for_event(event.event_type)
        attributes = {
            name: float(event.features[idx].item())
            for idx, name in enumerate(feature_names)
            if idx < int(event.features.shape[0])
        }
        if event.edge_type is not None:
            attributes["edge_type_code"] = EDGE_TYPE_CODE_MAP.get(event.edge_type, 0.0)

        return TemporalFact(
            head=head,
            relation=relation,
            tail=tail,
            timestamp=float(event.timestamp),
            attributes=attributes,
            source=event.event_type.value,
        )

    def events_to_facts(self, events: Iterable[FinancialEvent]) -> List[TemporalFact]:
        facts = [self.event_to_fact(event) for event in events]
        facts.sort(key=lambda fact: (fact.timestamp, fact.head, fact.relation, fact.tail))
        return facts

    def static_graph_to_facts(
        self,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_type_names: Sequence[str],
        timestamp: float,
        edge_features: Optional[torch.Tensor] = None,
    ) -> List[TemporalFact]:
        facts: List[TemporalFact] = []
        if edge_index.numel() == 0 or edge_type_ids.numel() == 0:
            return facts

        seen = set()
        src_nodes = edge_index[0].tolist()
        dst_nodes = edge_index[1].tolist()
        type_ids = edge_type_ids.tolist()

        for idx, (src, dst, type_id) in enumerate(zip(src_nodes, dst_nodes, type_ids)):
            edge_type = edge_type_names[type_id]
            relation = STATIC_EDGE_RELATION_MAP.get(edge_type)
            if relation is None:
                continue

            dedupe_key = (min(src, dst), max(src, dst), relation)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            feature_values = {}
            if edge_features is not None and idx < edge_features.shape[0]:
                raw = edge_features[idx]
                for name, value in zip(_feature_names_for_edge_type(edge_type, raw.shape[0]), raw.tolist()):
                    feature_values[name] = float(value)

            facts.append(
                TemporalFact(
                    head=self.asset_entity(src),
                    relation=relation,
                    tail=self.asset_entity(dst),
                    timestamp=float(timestamp),
                    attributes=feature_values,
                    source=f"static:{edge_type}",
                )
            )

        facts.sort(key=lambda fact: (fact.timestamp, fact.head, fact.relation, fact.tail))
        return facts
