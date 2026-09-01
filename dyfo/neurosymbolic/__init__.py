"""DyFO Neuro-Symbolic AI & LLM GraphRAG Package.

This package bridges DyFO's continuous temporal graph predictions with
Large Language Models (LLMs) and Symbolic Constraint Solvers.
"""

from dyfo.neurosymbolic.subgraph_extractor import (
    CausalSubgraph,
    CausalSubgraphExtractor,
    SemanticTriple,
)
from dyfo.neurosymbolic.graphrag_prompt_engine import (
    GraphRAGPromptEngine,
    LLMReasoner,
    RiskExplanation,
)
from dyfo.neurosymbolic.symbolic_parser import (
    ParsedConstraints,
    SymbolicConstraintParser,
)
from dyfo.neurosymbolic.constrained_solver import (
    ConstrainedPortfolioSolver,
    solve_symbolically_constrained_gmvp,
)

__all__ = [
    "SemanticTriple",
    "CausalSubgraph",
    "CausalSubgraphExtractor",
    "RiskExplanation",
    "GraphRAGPromptEngine",
    "LLMReasoner",
    "ParsedConstraints",
    "SymbolicConstraintParser",
    "ConstrainedPortfolioSolver",
    "solve_symbolically_constrained_gmvp",
]
