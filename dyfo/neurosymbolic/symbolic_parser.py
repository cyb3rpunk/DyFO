"""Symbolic Constraint Parser for Neuro-Symbolic Portfolio Optimization.

Translates LLM natural language explanations and structured risk decisions
into mathematical linear inequality constraints (A w <= b, bounds) for
Quadratic Programming and GMVP solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dyfo.core.ticker_registry import TICKER_GICS_MAPPING, TICKERS_30
from dyfo.neurosymbolic.graphrag_prompt_engine import RiskExplanation


@dataclass
class ParsedConstraints:
    """Mathematical linear inequality bounds compiled from LLM symbolic constraints."""
    A_ub: np.ndarray  # Shape (M, N)
    b_ub: np.ndarray  # Shape (M,)
    bounds: List[Tuple[float, float]]  # Length N: (min_w_i, max_w_i)
    cash_buffer: float = 0.0
    sector_caps: Dict[str, float] = field(default_factory=dict)
    excluded_tickers: List[str] = field(default_factory=list)
    constraint_descriptions: List[str] = field(default_factory=list)


class SymbolicConstraintParser:
    """Compiles high-level LLM risk decisions into strict numerical bounds."""

    def __init__(
        self,
        tickers: Optional[Sequence[str]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        default_max_asset_weight: float = 0.20,
    ):
        self.tickers = list(tickers) if tickers is not None else list(TICKERS_30)
        self.n = len(self.tickers)
        self.sector_mapping = sector_mapping or TICKER_GICS_MAPPING
        self.ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        self.default_max_asset_weight = default_max_asset_weight

    def parse(self, explanation: RiskExplanation) -> ParsedConstraints:
        """Translate RiskExplanation into ParsedConstraints linear matrices."""
        A_rows: List[np.ndarray] = []
        b_vals: List[float] = []
        descriptions: List[str] = []

        # 1. Base Individual Asset Bounds
        bounds: List[Tuple[float, float]] = [(0.0, self.default_max_asset_weight) for _ in range(self.n)]

        # 2. Excluded Assets (Hard 0% Cap)
        for ticker in explanation.exclude_tickers:
            if ticker in self.ticker_to_idx:
                idx = self.ticker_to_idx[ticker]
                bounds[idx] = (0.0, 0.0)
                descriptions.append(f"Hard exclusion for asset {ticker} (max_w = 0.0)")

        # 3. Sector Weight Constraints: \sum_{i \in Sector} w_i <= Cap
        for sector_name, cap in explanation.recommended_sector_caps.items():
            cap_val = float(np.clip(cap, 0.05, 1.0))
            sector_indices = [
                i for i, t in enumerate(self.tickers)
                if self.sector_mapping.get(t, "").lower() == sector_name.lower()
            ]

            if sector_indices:
                row = np.zeros(self.n, dtype=np.float64)
                for idx in sector_indices:
                    row[idx] = 1.0
                A_rows.append(row)
                b_vals.append(cap_val)
                descriptions.append(f"Sector Cap: {sector_name} <= {cap_val:.1%}")

        # 4. Handle Cash Buffer & Hedging Action
        cash_buffer = float(np.clip(explanation.min_cash_buffer, 0.0, 0.50))
        if explanation.hedging_action == "STRONG_HEDGE" and cash_buffer < 0.15:
            cash_buffer = 0.15
        elif explanation.hedging_action == "MILD_HEDGE" and cash_buffer < 0.05:
            cash_buffer = 0.05

        if A_rows:
            A_ub = np.vstack(A_rows)
            b_ub = np.array(b_vals, dtype=np.float64)
        else:
            A_ub = np.zeros((0, self.n), dtype=np.float64)
            b_ub = np.zeros(0, dtype=np.float64)

        return ParsedConstraints(
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            cash_buffer=cash_buffer,
            sector_caps=explanation.recommended_sector_caps,
            excluded_tickers=explanation.exclude_tickers,
            constraint_descriptions=descriptions,
        )
