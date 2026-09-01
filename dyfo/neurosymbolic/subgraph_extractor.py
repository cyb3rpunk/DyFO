"""Causal Subgraph Extraction & Semantic Triples Serializer for GraphRAG LLMs.

Extracts causal subgraphs and topological structures from DyFO's temporal knowledge graph,
converting co-movement innovations, fundamental links, and macro events into
JSON-LD, RDF/Turtle, and Human-Readable Prompt Triples.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dyfo.core.ticker_registry import TICKER_GICS_MAPPING, TICKERS_30


@dataclass
class SemanticTriple:
    """Represents a single typed relationship in the financial temporal knowledge graph."""
    source: str
    relation: str
    target: str
    timestamp: str
    weight: float = 1.0
    delta_rho: Optional[float] = None
    implication: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_turtle(self) -> str:
        """Format as RDF/Turtle triple."""
        s = f"dyfo:{self.source}"
        p = f"dyfo:{self.relation}"
        o = f"dyfo:{self.target}"
        return f"{s} {p} {o} ; dyfo:weight {self.weight:.4f} ; dyfo:timestamp \"{self.timestamp}\" ."

    def to_text_line(self) -> str:
        """Format as concise human-readable text line."""
        shock_str = f" [Δρ={self.delta_rho:+.2f}]" if self.delta_rho is not None else ""
        impl_str = f" -> ({self.implication})" if self.implication else ""
        return f"({self.source}) --[{self.relation}{shock_str}]--> ({self.target}){impl_str}"


@dataclass
class CausalSubgraph:
    """Container for an extracted temporal ego-network and macro-regime state."""
    date: str
    center_tickers: List[str]
    triples: List[SemanticTriple]
    macro_regime: str = "NEUTRAL"
    eigen_concentration: float = 0.0
    top_eigenvalues: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_ld(self) -> Dict[str, Any]:
        """Convert subgraph to structured JSON-LD specification."""
        return {
            "@context": {
                "dyfo": "https://w3id.org/dyfo/ontology#",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "source": "dyfo:sourceAsset",
                "relation": "dyfo:relationType",
                "target": "dyfo:targetAsset",
                "weight": "dyfo:edgeWeight",
                "delta_rho": "dyfo:correlationInnovation",
                "timestamp": "dyfo:validDate",
            },
            "@id": f"dyfo:subgraph_{self.date}",
            "@type": "dyfo:TemporalCausalSubgraph",
            "date": self.date,
            "macro_regime": self.macro_regime,
            "eigen_concentration": round(self.eigen_concentration, 4),
            "top_eigenvalues": [round(v, 4) for v in self.top_eigenvalues[:3]],
            "active_assets": self.center_tickers,
            "triples": [t.to_dict() for t in self.triples],
        }

    def to_rdf_turtle(self) -> str:
        """Serialize subgraph to RDF Turtle format."""
        header = (
            "@prefix dyfo: <https://w3id.org/dyfo/ontology#> .\n"
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n"
            f"# DyFO Temporal Causal Subgraph for {self.date}\n"
            f"# Macro Regime: {self.macro_regime} | Eigen Concentration: {self.eigen_concentration:.4f}\n\n"
        )
        body = "\n".join(t.to_turtle() for t in self.triples)
        return header + body

    def to_natural_text(self, max_triples: int = 25) -> str:
        """Serialize into clean, categorized text for LLM Prompt context."""
        lines = [
            f"=== DYFO CAUSAL GRAPH CONTEXT ({self.date}) ===",
            f"Market Macro Regime: {self.macro_regime}",
            f"Eigenvalue Market Concentration (Top-1 / Tr(Sigma)): {self.eigen_concentration:.2%}",
        ]
        if self.top_eigenvalues:
            ev_str = ", ".join(f"{ev:.2f}" for ev in self.top_eigenvalues[:3])
            lines.append(f"Top-3 Spectral Eigenvalues: [{ev_str}]")

        lines.append("\n--- Top Relational Shocks & Topological Pathways ---")
        for idx, t in enumerate(self.triples[:max_triples]):
            lines.append(f"{idx + 1}. {t.to_text_line()}")

        if len(self.triples) > max_triples:
            lines.append(f"... and {len(self.triples) - max_triples} more causal graph edges.")

        return "\n".join(lines)


class CausalSubgraphExtractor:
    """Extracts causal ego-networks and macro triples around target portfolio assets."""

    def __init__(
        self,
        tickers: Optional[Sequence[str]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
    ):
        self.tickers = list(tickers) if tickers is not None else list(TICKERS_30)
        self.sector_mapping = sector_mapping or TICKER_GICS_MAPPING
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        # Predefined supply-chain / cross-sector knowledge base (factual ground-truth triples)
        self.fundamental_relations = [
            ("NVDA", "SUPPLIER_TO", "MSFT", "AI GPU compute hardware supply"),
            ("NVDA", "SUPPLIER_TO", "GOOGL", "Tensor Processing / GPU cluster supply"),
            ("AVGO", "SUPPLIER_TO", "AAPL", "RF Front-end and custom silicon supplier"),
            ("XOM", "ENERGY_INPUT_TO", "CAT", "Heavy industrial machinery fuel and petrochemicals"),
            ("JPM", "FINANCIAL_UNDERWRITER_TO", "AMZN", "Syndicated debt credit facility"),
            ("LIN", "INDUSTRIAL_GAS_TO", "NVDA", "Semiconductor fabrication specialty gases"),
            ("LLY", "COMPETITOR_IN_GLP1", "NVO", "GLP-1 obesity therapeutics duopoly"),
        ]

    def extract_subgraph(
        self,
        date_str: str,
        predicted_delta_rho: np.ndarray,
        correlation_matrix: np.ndarray,
        center_tickers: Optional[Sequence[str]] = None,
        macro_events: Optional[List[Dict[str, Any]]] = None,
        top_k_shocks: int = 15,
    ) -> CausalSubgraph:
        """Extract top co-movement innovations, fundamental ties, and macro regime into CausalSubgraph.

        Parameters
        ----------
        date_str : str
            ISO date string (YYYY-MM-DD).
        predicted_delta_rho : np.ndarray
            Matrix of predicted correlation innovations Delta rho in shape (N, N).
        correlation_matrix : np.ndarray
            Current or predicted SPD correlation matrix in shape (N, N).
        center_tickers : Optional[Sequence[str]]
            List of focus tickers (default: all tickers).
        macro_events : Optional[List[Dict]]
            List of macro/calendar events active on this date.
        top_k_shocks : int
            Number of highest co-movement shocks to include.

        Returns
        -------
        CausalSubgraph
            Extracted semantic subgraph with typed triples.
        """
        n = len(self.tickers)
        active_tickers = list(center_tickers) if center_tickers is not None else self.tickers

        # 1. Compute Spectral Metrics (Market Eigenvalue Concentration)
        eigvals = np.linalg.eigvalsh(correlation_matrix)
        eigvals = np.sort(eigvals)[::-1]
        eigen_concentration = float(eigvals[0] / max(np.sum(eigvals), 1e-6))
        top_eigvals = [float(v) for v in eigvals[:5]]

        # Determine Macro Regime Heuristic if not provided
        macro_regime = "MODERATE_NORMAL"
        if eigen_concentration > 0.45:
            macro_regime = "HIGH_STRESS_CONTAGION"
        elif eigen_concentration < 0.20:
            macro_regime = "HIGH_DISPERSION_DECORRELATED"

        triples: List[SemanticTriple] = []

        # 2. Extract Top Correlation Innovation Shocks (Delta rho_{ij})
        shock_candidates: List[Tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                d_rho = float(predicted_delta_rho[i, j])
                shock_candidates.append((abs(d_rho), i, j))

        shock_candidates.sort(key=lambda x: x[0], reverse=True)

        for abs_val, i, j in shock_candidates[:top_k_shocks]:
            t_i = self.tickers[i]
            t_j = self.tickers[j]
            d_rho = float(predicted_delta_rho[i, j])
            corr_val = float(correlation_matrix[i, j])

            sec_i = self.sector_mapping.get(t_i, "GICS")
            sec_j = self.sector_mapping.get(t_j, "GICS")

            rel_type = "DYNAMIC_CORRELATION_SHOCK"
            if sec_i == sec_j:
                rel_type = "INTRA_SECTOR_CO_MOVEMENT"
            else:
                rel_type = "CROSS_SECTOR_SPILLOVER"

            implication = (
                f"Rising co-movement (rho={corr_val:.2f})"
                if d_rho > 0
                else f"Decorrelating divergence (rho={corr_val:.2f})"
            )

            triples.append(
                SemanticTriple(
                    source=t_i,
                    relation=rel_type,
                    target=t_j,
                    timestamp=date_str,
                    weight=corr_val,
                    delta_rho=d_rho,
                    implication=implication,
                    metadata={"sector_source": sec_i, "sector_target": sec_j},
                )
            )

        # 3. Add Fundamental Supply-Chain / Corporate Triples for Active Universe
        active_set = set(self.tickers)
        for src, rel, tgt, desc in self.fundamental_relations:
            if src in active_set and tgt in active_set:
                triples.append(
                    SemanticTriple(
                        source=src,
                        relation=rel,
                        target=tgt,
                        timestamp=date_str,
                        weight=1.0,
                        delta_rho=None,
                        implication=desc,
                        metadata={"type": "fundamental_ontology"},
                    )
                )

        # 4. Add Macro Events
        if macro_events:
            for event in macro_events:
                triples.append(
                    SemanticTriple(
                        source=event.get("entity", "FED_DECISION"),
                        relation="MACRO_IMPACT",
                        target=event.get("impact_sector", "MARKET_GLOBAL"),
                        timestamp=date_str,
                        weight=float(event.get("severity", 1.0)),
                        implication=event.get("description", "Macro interest rate release"),
                        metadata=event,
                    )
                )
        else:
            # Add synthetic macro node if regime is high stress
            if macro_regime == "HIGH_STRESS_CONTAGION":
                triples.append(
                    SemanticTriple(
                        source="MACRO_VOLATILITY_SURGE",
                        relation="SYSTEMIC_SPILLOVER",
                        target="ALL_SECTORS",
                        timestamp=date_str,
                        weight=eigen_concentration,
                        implication="Systemic eigenvalue concentration exceeds risk threshold (>45%)",
                    )
                )

        return CausalSubgraph(
            date=date_str,
            center_tickers=active_tickers,
            triples=triples,
            macro_regime=macro_regime,
            eigen_concentration=eigen_concentration,
            top_eigenvalues=top_eigvals,
            metadata={"num_assets": n, "top_k_shocks": top_k_shocks},
        )
