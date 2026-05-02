"""Ticker registries for DyFO experiments.

This module centralises the S&P 500 ticker lists used across all evaluation
scripts.  Import the appropriate constant rather than hard-coding tickers in
individual scripts.

Universe sizes
--------------
TICKERS_30  — Original BL-01 universe (30 tickers, all 11 GICS sectors).
              Used in every published result up to rev1.
TICKERS_50  — Extended universe (50 tickers).  Adds depth to each sector
              without crossing the ≤50-asset threshold where simple threshold
              sparsification is still valid (see spec/02_graph_spec.md).
TICKERS_100 — Large universe (100 tickers).  Requires TMFG sparsification for
              the CORR graph (51–200 range per spec/02_graph_spec.md).

Usage
-----
    from dyfo.core.ticker_registry import get_tickers
    tickers = get_tickers(30)   # or 50 or 100
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# 30 tickers — BL-01 canonical universe (all 11 GICS sectors)
# ---------------------------------------------------------------------------
TICKERS_30: List[str] = [
    # Information Technology (5)
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM",
    # Financials (4)
    "JPM", "GS", "MA", "BRK-B",
    # Health Care (3)
    "JNJ", "UNH", "LLY",
    # Consumer Discretionary (3)
    "AMZN", "TSLA", "HD",
    # Consumer Staples (2)
    "PG", "KO",
    # Energy (2)
    "XOM", "CVX",
    # Industrials (3)
    "CAT", "BA", "RTX",
    # Communication Services (3)
    "META", "GOOGL", "DIS",
    # Materials (2)
    "LIN", "APD",
    # Utilities (2)
    "NEE", "DUK",
    # Real Estate (1)
    "PLD",
]

# ---------------------------------------------------------------------------
# 50 tickers — extended S&P 500 universe (≤50 → simple threshold still valid)
# ---------------------------------------------------------------------------
TICKERS_50: List[str] = TICKERS_30 + [
    # Information Technology (+4)
    "AMD", "INTC", "QCOM", "ADBE",
    # Financials (+3)
    "BAC", "C", "WFC",
    # Health Care (+3)
    "ABBV", "MRK", "PFE",
    # Consumer Discretionary (+2)
    "NKE", "MCD",
    # Consumer Staples (+1)
    "COST",
    # Energy (+1)
    "SLB",
    # Industrials (+1)
    "HON",
    # Communication Services (+2)
    "NFLX", "T",
    # Materials (+1)
    "FCX",
    # Utilities (+1)
    "SO",
    # Real Estate (+1)
    "AMT",
]

# ---------------------------------------------------------------------------
# 100 tickers — large universe (TMFG sparsification required for CORR graph)
# ---------------------------------------------------------------------------
TICKERS_100: List[str] = TICKERS_50 + [
    # Information Technology (+10)
    "TXN", "MU", "NOW", "PANW", "SNOW",
    "AMAT", "LRCX", "KLAC", "HPQ", "IBM",
    # Financials (+7)
    "AXP", "BLK", "MS", "SCHW", "PNC",
    "USB", "TFC",
    # Health Care (+5)
    "BMY", "AMGN", "GILD", "TMO", "ISRG",
    # Consumer Discretionary (+5)
    "LOW", "TGT", "SBUX", "GM", "F",
    # Consumer Staples (+3)
    "MO", "PM", "CL",
    # Energy (+4)
    "EOG", "COP", "MPC", "VLO",
    # Industrials (+4)
    "GE", "MMM", "DE", "FDX",
    # Communication Services (+3)
    "CMCSA", "VZ", "CHTR",
    # Materials (+3)
    "NEM", "DOW", "DD",
    # Utilities (+2)
    "AEP", "EXC",
    # Real Estate (+4)
    "SPG", "EQR", "O", "WELL",
]

assert len(TICKERS_50) == 50, f"TICKERS_50 has {len(TICKERS_50)} tickers"
assert len(TICKERS_100) == 100, f"TICKERS_100 has {len(TICKERS_100)} tickers"

# Sparsification strategy per universe size (from spec/02_graph_spec.md)
SPARSIFICATION_STRATEGY = {
    30: "threshold",   # |ρ| > 0.3
    50: "threshold",   # |ρ| > 0.3
    100: "tmfg",       # TMFG (51-200 range)
}


def get_tickers(n: int) -> List[str]:
    """Return the canonical ticker list for a given universe size.

    Parameters
    ----------
    n : int
        Universe size.  Must be 30, 50, or 100.

    Returns
    -------
    List[str]
        Ordered list of ticker symbols.

    Raises
    ------
    ValueError
        If ``n`` is not a supported universe size.
    """
    if n == 30:
        return list(TICKERS_30)
    if n == 50:
        return list(TICKERS_50)
    if n == 100:
        return list(TICKERS_100)
    raise ValueError(
        f"Unsupported universe size: {n}. Choose from 30, 50, or 100."
    )


def get_sparsification(n: int) -> str:
    """Return the recommended CORR sparsification strategy for universe size n."""
    return SPARSIFICATION_STRATEGY.get(n, "tmfg")
