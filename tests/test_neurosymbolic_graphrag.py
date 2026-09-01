"""Unit tests for DyFO Neuro-Symbolic AI & GraphRAG LLM Reasoning."""

import json
import numpy as np
import pytest

from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING
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


@pytest.fixture
def sample_market_data():
    np.random.seed(42)
    n = len(TICKERS_30)
    # Generate synthetic correlation matrix
    A = np.random.randn(n, n)
    cov = A @ A.T + np.eye(n) * 2.0
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)

    # Synthetic delta rho
    delta_rho = np.random.randn(n, n) * 0.15
    np.fill_diagonal(delta_rho, 0.0)
    delta_rho = 0.5 * (delta_rho + delta_rho.T)
    return corr, delta_rho


def test_subgraph_extraction_and_serialization(sample_market_data):
    corr, delta_rho = sample_market_data
    extractor = CausalSubgraphExtractor(tickers=TICKERS_30, sector_mapping=TICKER_GICS_MAPPING)

    subgraph = extractor.extract_subgraph(
        date_str="2024-06-15",
        predicted_delta_rho=delta_rho,
        correlation_matrix=corr,
        top_k_shocks=10,
    )

    assert isinstance(subgraph, CausalSubgraph)
    assert subgraph.date == "2024-06-15"
    assert len(subgraph.triples) >= 10
    assert 0.0 <= subgraph.eigen_concentration <= 1.0

    # JSON-LD serialization
    json_ld = subgraph.to_json_ld()
    assert "@context" in json_ld
    assert json_ld["@type"] == "dyfo:TemporalCausalSubgraph"
    assert len(json_ld["triples"]) == len(subgraph.triples)

    # RDF Turtle serialization
    turtle_str = subgraph.to_rdf_turtle()
    assert "@prefix dyfo:" in turtle_str
    assert "dyfo:weight" in turtle_str

    # Natural text context
    text_prompt = subgraph.to_natural_text(max_triples=5)
    assert "DYFO CAUSAL GRAPH CONTEXT" in text_prompt
    assert "Market Macro Regime:" in text_prompt


def test_graphrag_prompt_engine_and_mock_llm(sample_market_data):
    corr, delta_rho = sample_market_data
    extractor = CausalSubgraphExtractor()
    subgraph = extractor.extract_subgraph("2024-06-15", delta_rho, corr)

    prompt_engine = GraphRAGPromptEngine()
    prompt = prompt_engine.build_prompt(subgraph, portfolio_context={"Current Sharpe": 2.1, "Drawdown": "-2.5%"})

    assert "DYFO CAUSAL GRAPH CONTEXT" in prompt
    assert "Current Portfolio State" in prompt

    # Test Mock Reasoner
    reasoner = LLMReasoner(backend="mock")
    explanation = reasoner.reason(subgraph)

    assert isinstance(explanation, RiskExplanation)
    assert explanation.date == "2024-06-15"
    assert len(explanation.macro_rationale) > 20
    assert isinstance(explanation.recommended_sector_caps, dict)
    assert explanation.hedging_action in ["NONE", "MILD_HEDGE", "STRONG_HEDGE", "DEFENSIVE_ROTATE"]


def test_symbolic_constraint_parser():
    explanation = RiskExplanation(
        date="2024-06-15",
        macro_rationale="High concentration in Tech",
        spillover_risks=["NVDA supply shock"],
        recommended_sector_caps={"Information Technology": 0.20, "Financials": 0.25},
        exclude_tickers=["CAT"],
        hedging_action="MILD_HEDGE",
        min_cash_buffer=0.05,
    )

    parser = SymbolicConstraintParser(tickers=TICKERS_30, sector_mapping=TICKER_GICS_MAPPING)
    constraints = parser.parse(explanation)

    assert isinstance(constraints, ParsedConstraints)
    assert constraints.A_ub.shape[0] >= 2  # Tech and Financials caps
    assert constraints.A_ub.shape[1] == len(TICKERS_30)
    assert constraints.cash_buffer >= 0.05

    # Check CAT is excluded (max bound 0.0)
    cat_idx = parser.ticker_to_idx["CAT"]
    assert constraints.bounds[cat_idx] == (0.0, 0.0)


def test_constrained_portfolio_solver(sample_market_data):
    corr, delta_rho = sample_market_data
    n = len(TICKERS_30)

    # Construct test constraints
    explanation = RiskExplanation(
        date="2024-06-15",
        macro_rationale="Defensive test",
        recommended_sector_caps={"Information Technology": 0.20},
        exclude_tickers=["TSLA"],
        hedging_action="MILD_HEDGE",
        min_cash_buffer=0.10,
    )

    parser = SymbolicConstraintParser(tickers=TICKERS_30, sector_mapping=TICKER_GICS_MAPPING)
    constraints = parser.parse(explanation)

    solver = ConstrainedPortfolioSolver()
    w_opt, meta = solver.solve(corr, constraints)

    assert len(w_opt) == n
    assert np.all(w_opt >= -1e-7), "Weights must be non-negative"
    assert abs(np.sum(w_opt) - 0.90) < 1e-4, "Risky sum must equal 1 - cash_buffer (0.90)"

    # Verify TSLA is excluded
    tsla_idx = parser.ticker_to_idx["TSLA"]
    assert w_opt[tsla_idx] < 1e-6, "Excluded asset must have weight ~ 0.0"

    # Verify Tech cap <= 0.20
    tech_indices = [i for i, t in enumerate(TICKERS_30) if TICKER_GICS_MAPPING.get(t) == "Information Technology"]
    tech_weight = float(np.sum(w_opt[tech_indices]))
    assert tech_weight <= 0.20 + 1e-4, f"Tech weight {tech_weight} exceeded 0.20 cap"
