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

# ---------------------------------------------------------------------------
# GICS Sector Mapping for all 100 Universe Tickers
# ---------------------------------------------------------------------------
TICKER_GICS_MAPPING: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AVGO": "Information Technology", "CRM": "Information Technology", "AMD": "Information Technology",
    "INTC": "Information Technology", "QCOM": "Information Technology", "ADBE": "Information Technology",
    "TXN": "Information Technology", "MU": "Information Technology", "NOW": "Information Technology",
    "PANW": "Information Technology", "SNOW": "Information Technology", "AMAT": "Information Technology",
    "LRCX": "Information Technology", "KLAC": "Information Technology", "HPQ": "Information Technology",
    "IBM": "Information Technology",
    # Financials
    "JPM": "Financials", "GS": "Financials", "MA": "Financials", "BRK-B": "Financials",
    "BAC": "Financials", "C": "Financials", "WFC": "Financials", "AXP": "Financials",
    "BLK": "Financials", "MS": "Financials", "SCHW": "Financials", "PNC": "Financials",
    "USB": "Financials", "TFC": "Financials",
    # Health Care
    "JNJ": "Health Care", "UNH": "Health Care", "LLY": "Health Care", "ABBV": "Health Care",
    "MRK": "Health Care", "PFE": "Health Care", "BMY": "Health Care", "AMGN": "Health Care",
    "GILD": "Health Care", "TMO": "Health Care", "ISRG": "Health Care",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "MCD": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "SBUX": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "COST": "Consumer Staples",
    "MO": "Consumer Staples", "PM": "Consumer Staples", "CL": "Consumer Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "SLB": "Energy", "EOG": "Energy",
    "COP": "Energy", "MPC": "Energy", "VLO": "Energy",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "RTX": "Industrials", "HON": "Industrials",
    "GE": "Industrials", "MMM": "Industrials", "DE": "Industrials", "FDX": "Industrials",
    # Communication Services
    "META": "Communication Services", "GOOGL": "Communication Services", "DIS": "Communication Services",
    "NFLX": "Communication Services", "T": "Communication Services", "CMCSA": "Communication Services",
    "VZ": "Communication Services", "CHTR": "Communication Services",
    # Materials
    "LIN": "Materials", "APD": "Materials", "FCX": "Materials",
    "NEM": "Materials", "DOW": "Materials", "DD": "Materials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "SPG": "Real Estate",
    "EQR": "Real Estate", "O": "Real Estate", "WELL": "Real Estate",
}


def get_sector_distribution(tickers: list[str]) -> dict[str, int]:
    """Return the GICS sector count breakdown for a given list of tickers."""
    dist: dict[str, int] = {}
    for t in tickers:
        sec = TICKER_GICS_MAPPING.get(t, "Unknown")
        dist[sec] = dist.get(sec, 0) + 1
    return dist


def select_balanced_gics_universe(
    prices: any,
    volumes: any,
    target_n: int = 30,
    min_per_sector: int = 1,
) -> list[str]:
    """Select a liquidity-balanced basket across all 11 GICS sectors.

    Computes causal Average Daily Dollar Volume (ADTV = Price * Volume) and ranks
    tickers within each sector, picking the top liquid assets to compose target_n assets.

    Parameters
    ----------
    prices : DataFrame (T, N)
    volumes : DataFrame (T, N)
    target_n : int, default 30
    min_per_sector : int, default 1

    Returns
    -------
    selected_tickers : List[str] of length target_n
    """
    import pandas as pd
    import numpy as np

    dollar_volume = (prices * volumes).mean(axis=0).dropna()
    available_tickers = list(dollar_volume.index)

    # Group available tickers by GICS sector
    by_sector: dict[str, list[tuple[str, float]]] = {}
    for t in available_tickers:
        sec = TICKER_GICS_MAPPING.get(t, "Other")
        by_sector.setdefault(sec, []).append((t, float(dollar_volume[t])))

    # Sort each sector by liquidity descending
    for sec in by_sector:
        by_sector[sec].sort(key=lambda x: x[1], reverse=True)

    # First pass: guarantee min_per_sector for each GICS sector
    selected: list[str] = []
    for sec, tkr_list in by_sector.items():
        for t, _ in tkr_list[:min_per_sector]:
            if t not in selected and len(selected) < target_n:
                selected.append(t)

    # Second pass: allocate remaining slots to highest overall liquidity
    remaining_slots = target_n - len(selected)
    if remaining_slots > 0:
        candidates = []
        for sec, tkr_list in by_sector.items():
            for t, dv in tkr_list[min_per_sector:]:
                if t not in selected:
                    candidates.append((t, dv))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for t, _ in candidates[:remaining_slots]:
            selected.append(t)

    return selected


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

