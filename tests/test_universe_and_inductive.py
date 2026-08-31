"""Unit tests for GICS universe selection, liquidity rules, and inductive ticker enrollment.

Covers:
  - REQ-IMP7: GICS balanced universe selection by causal ADTV
  - Inductive ticker enrollment in DyFOAdapter
  - Sector distribution integrity
"""

import datetime
import numpy as np
import pandas as pd
import pytest

from dyfo.adapters.dyfo_adapter import DyFOAdapter
from dyfo.core.ticker_registry import (
    TICKER_GICS_MAPPING,
    TICKERS_30,
    TICKERS_50,
    TICKERS_100,
    get_sector_distribution,
    select_balanced_gics_universe,
)


def test_gics_mapping_covers_all_tickers():
    """Verify that all 100 tickers have explicit GICS mappings."""
    assert len(TICKER_GICS_MAPPING) >= 100
    for t in TICKERS_100:
        assert t in TICKER_GICS_MAPPING, f"Ticker {t} missing from GICS mapping"
        assert len(TICKER_GICS_MAPPING[t]) > 3


def test_get_sector_distribution():
    """Verify sector breakdown distribution counts."""
    dist30 = get_sector_distribution(TICKERS_30)
    assert len(dist30) == 11, "TICKERS_30 must cover all 11 GICS sectors"
    assert sum(dist30.values()) == 30

    dist50 = get_sector_distribution(TICKERS_50)
    assert len(dist50) == 11
    assert sum(dist50.values()) == 50


def test_select_balanced_gics_universe_causal():
    """Verify causal liquidity-based GICS universe selection."""
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    tickers = list(TICKERS_50)
    np.random.seed(42)

    # Simulate price and volume
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(np.random.randn(100, 50) * 0.01, axis=0)),
        index=dates,
        columns=tickers,
    )
    volumes = pd.DataFrame(
        np.random.uniform(1e5, 1e7, size=(100, 50)),
        index=dates,
        columns=tickers,
    )

    # Select top 30 balanced
    selected = select_balanced_gics_universe(prices, volumes, target_n=30, min_per_sector=1)
    assert len(selected) == 30
    assert len(set(selected)) == 30

    # Ensure all 11 sectors are represented
    dist = get_sector_distribution(selected)
    assert len(dist) == 11, "Selected universe must maintain all 11 GICS sectors"


def test_dyfo_adapter_inductive_ticker_enrollment():
    """Verify dynamic inductive enrollment of an out-of-universe asset."""
    adapter = DyFOAdapter(tickers=list(TICKERS_30))
    assert adapter.num_nodes == 30
    assert "UBER" not in adapter.ticker_to_idx

    # Dynamically enroll new ticker (e.g. UBER)
    new_idx = adapter.enroll_inductive_ticker("UBER", sector="Industrials")
    assert new_idx == 30
    assert adapter.num_nodes == 31
    assert "UBER" in adapter.ticker_to_idx
    assert "UBER.US" in adapter.entity_to_idx

    # Generating covariance matrix should now produce (31, 31) SPD matrix
    cov = adapter.get_covariance_matrix("2023-06-15")
    assert cov.shape == (31, 31)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > 0, "Inductively extended covariance matrix must be strictly SPD"
