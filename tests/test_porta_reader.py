"""Tests for PortaDataReader — Verifying strict read-only consumption of PORTA data."""

import datetime
from pathlib import Path
import numpy as np
import pytest

from dyfo.adapters.dyfo_adapter import DyFOAdapter
from dyfo.data.porta_reader import PortaDataReader, DEFAULT_PORTA_PATH


def test_porta_reader_readonly_contract():
    """Verify that PortaDataReader NEVER modifies or creates files in PORTA workspace."""
    porta_root = Path(DEFAULT_PORTA_PATH)
    if not porta_root.exists():
        pytest.skip("PORTA repository not present at default path")

    features_dir = porta_root / "data" / "features" / "daily_core"
    if not features_dir.exists():
        pytest.skip("PORTA daily_core features not present")

    # Record modification times of files in PORTA daily_core
    mtimes_before = {}
    for p in features_dir.iterdir():
        if p.is_file():
            mtimes_before[p.name] = p.stat().st_mtime_ns

    reader = PortaDataReader(porta_root)
    assert reader.is_available

    # Perform various read operations
    assets = reader.get_assets()
    assert len(assets) > 0
    
    date_start, date_end = reader.get_date_range()
    assert date_start < date_end
    
    mid_date = datetime.date(2020, 6, 15)
    feats = reader.get_features_at_date(mid_date, assets=["AAPL.US", "MSFT.US"])
    assert feats is not None
    assert feats.shape == (2, 35)

    returns = reader.get_returns_history(mid_date, lookback_days=50, assets=["AAPL.US"])
    assert returns is not None

    regimes = reader.get_regime_probabilities_at_date(mid_date)
    assert regimes.shape == (3,)
    np.testing.assert_allclose(regimes.sum(), 1.0, atol=1e-5)

    # Verify modification times AFTER reads: NO file must have been altered
    for p in features_dir.iterdir():
        if p.is_file():
            assert p.name in mtimes_before, f"Unexpected new file created in PORTA: {p.name}"
            mtime_after = p.stat().st_mtime_ns
            assert mtime_after == mtimes_before[p.name], f"PORTA file was modified! {p.name}"


def test_porta_reader_causality():
    """Verify that PortaDataReader lookups strictly enforce t <= as_of_date."""
    reader = PortaDataReader()
    if not reader.is_available:
        pytest.skip("PORTA daily_core features not present")

    # Query before start of dataset
    early_date = datetime.date(1990, 1, 1)
    feats_early = reader.get_features_at_date(early_date)
    assert feats_early is None

    # Query within dataset
    test_date = datetime.date(2021, 1, 15)
    t_idx = reader._get_time_index_at_date(test_date)
    assert t_idx is not None

    # Verify that the indexed date is <= test_date
    indexed_date = reader._date_index.iloc[t_idx]["date"]
    assert indexed_date <= test_date


def test_dyfo_adapter_with_porta_reader():
    """Verify that DyFOAdapter cleanly consumes PortaDataReader in read-only mode."""
    reader = PortaDataReader()
    adapter = DyFOAdapter(
        tickers=["AAPL", "MSFT", "JPM"],
        porta_reader=reader if reader.is_available else None,
    )

    as_of = datetime.date(2020, 1, 15)
    snapshot = adapter.export_structural_graph(as_of, include_attention=True)

    assert snapshot.as_of_date == as_of
    assert snapshot.num_nodes == 3
    assert snapshot.node_embeddings.shape == (3, 100)
    assert not np.isnan(snapshot.node_embeddings).any()
    assert len(snapshot.edges_by_relation["CORR"]) >= 0

    cov = adapter.get_covariance_matrix(as_of)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov, cov.T, atol=1e-5)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > 0).all()
