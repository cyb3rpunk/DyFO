"""Structural Graph Export Data Models for DyFO_LITE.

Formalizes the relational graph snapshot data structures for cross-repository
integration with PORTA (covariance/shrinkage) and ORION (multimodal state constructor).
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RelationEdge:
    """A typed relational edge connecting two financial asset entities."""

    source_entity_id: str  # e.g., "AAPL.US" (REQ-G4)
    target_entity_id: str  # e.g., "MSFT.US" (REQ-G4)
    weight: float  # Raw relational weight (e.g. correlation rho)
    attributes: Dict[str, Any] = field(default_factory=dict)  # e.g. {"gics_sector": "45"} (REQ-G5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "weight": float(self.weight),
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RelationEdge:
        return cls(
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            weight=float(data["weight"]),
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class StructuralGraphSnapshot:
    """Snapshot of the decomposed multi-relational financial graph at a specific causal cutoff date."""

    as_of_date: datetime.date
    entity_ids: List[str]  # (N,) Standardized entity IDs (e.g. ["AAPL.US", "MSFT.US", ...])
    node_embeddings: np.ndarray  # (N, embedding_dim=100)
    edges_by_relation: Dict[str, List[RelationEdge]]  # {"CORR": [...], "SECT": [...], "SUPL": [...], "FACT": [...]}
    relation_attention_weights: Optional[np.ndarray] = None  # (N, 4) Optional relation attention weights
    causal_cutoff_date: Optional[datetime.date] = None  # Causal audit timestamp (matches as_of_date)

    def __post_init__(self):
        if self.causal_cutoff_date is None:
            object.__setattr__(self, "causal_cutoff_date", self.as_of_date)

    @property
    def num_nodes(self) -> int:
        return len(self.entity_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date.isoformat(),
            "causal_cutoff_date": self.causal_cutoff_date.isoformat() if self.causal_cutoff_date else None,
            "entity_ids": list(self.entity_ids),
            "node_embeddings": self.node_embeddings.tolist(),
            "edges_by_relation": {
                rel: [edge.to_dict() for edge in edges]
                for rel, edges in self.edges_by_relation.items()
            },
            "relation_attention_weights": (
                self.relation_attention_weights.tolist()
                if self.relation_attention_weights is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuralGraphSnapshot:
        as_of = datetime.date.fromisoformat(data["as_of_date"])
        causal_cutoff = (
            datetime.date.fromisoformat(data["causal_cutoff_date"])
            if data.get("causal_cutoff_date")
            else as_of
        )
        entity_ids = list(data["entity_ids"])
        node_embeddings = np.array(data["node_embeddings"], dtype=np.float32)
        
        edges_by_relation = {
            rel: [RelationEdge.from_dict(e) for e in edge_list]
            for rel, edge_list in data.get("edges_by_relation", {}).items()
        }
        
        rel_attn = (
            np.array(data["relation_attention_weights"], dtype=np.float32)
            if data.get("relation_attention_weights") is not None
            else None
        )

        return cls(
            as_of_date=as_of,
            entity_ids=entity_ids,
            node_embeddings=node_embeddings,
            edges_by_relation=edges_by_relation,
            relation_attention_weights=rel_attn,
            causal_cutoff_date=causal_cutoff,
        )
