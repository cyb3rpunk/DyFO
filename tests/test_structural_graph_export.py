"""Tests for StructuralGraphSnapshot and DyFOAdapter."""

import datetime
import numpy as np
import pytest

from dyfo.adapters.dyfo_adapter import DyFOAdapter, _to_us_ticker
from dyfo.adapters.structural_graph_export import RelationEdge, StructuralGraphSnapshot
from dyfo.config import DyFOConfig


def test_to_us_ticker_normalization():
    assert _to_us_ticker("aapl") == "AAPL.US"
    assert _to_us_ticker("MSFT.US") == "MSFT.US"
    assert _to_us_ticker("  jpm  ") == "JPM.US"


def test_relation_edge_serialization():
    edge = RelationEdge(
        source_entity_id="AAPL.US",
        target_entity_id="MSFT.US",
        weight=0.75,
        attributes={"gics_sector": "45"},
    )
    d = edge.to_dict()
    assert d["source_entity_id"] == "AAPL.US"
    assert d["weight"] == 0.75
    
    edge_restored = RelationEdge.from_dict(d)
    assert edge_restored == edge


def test_dyfo_adapter_export_structural_graph_schema():
    adapter = DyFOAdapter(tickers=["AAPL", "MSFT", "JPM", "XOM"])
    as_of = datetime.date(2023, 6, 15)
    
    snapshot = adapter.export_structural_graph(as_of, include_attention=True)
    
    assert snapshot.as_of_date == as_of
    assert snapshot.causal_cutoff_date == as_of
    assert len(snapshot.entity_ids) == 4
    assert snapshot.entity_ids == ["AAPL.US", "MSFT.US", "JPM.US", "XOM.US"]
    assert snapshot.node_embeddings.shape == (4, 100)
    assert not np.isnan(snapshot.node_embeddings).any()
    
    # Check all 4 relation keys exist explicitly (REQ-G3)
    for rel_key in ["CORR", "SECT", "SUPL", "FACT"]:
        assert rel_key in snapshot.edges_by_relation
        assert isinstance(snapshot.edges_by_relation[rel_key], list)
        
    assert snapshot.relation_attention_weights is not None
    assert snapshot.relation_attention_weights.shape == (4, 4)
    np.testing.assert_allclose(snapshot.relation_attention_weights.sum(axis=1), np.ones(4), atol=1e-5)


def test_dyfo_adapter_determinism():
    adapter = DyFOAdapter(tickers=["AAPL", "MSFT", "JPM"])
    as_of = datetime.date(2023, 1, 10)
    
    snap1 = adapter.export_structural_graph(as_of)
    snap2 = adapter.export_structural_graph(as_of)
    
    np.testing.assert_array_equal(snap1.node_embeddings, snap2.node_embeddings)
    assert len(snap1.edges_by_relation["CORR"]) == len(snap2.edges_by_relation["CORR"])


def test_dyfo_adapter_covariance_matrix():
    adapter = DyFOAdapter(tickers=["AAPL", "MSFT", "JPM"])
    as_of = datetime.date(2023, 1, 10)
    
    cov = adapter.get_covariance_matrix(as_of)
    assert cov.shape == (3, 3)
    # Must be positive semi-definite and symmetric
    np.testing.assert_allclose(cov, cov.T, atol=1e-6)
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals > 0).all(), "Covariance matrix is not positive definite"


def test_snapshot_to_from_dict():
    adapter = DyFOAdapter(tickers=["AAPL", "MSFT"])
    as_of = datetime.date(2023, 1, 10)
    snapshot = adapter.export_structural_graph(as_of, include_attention=True)
    
    snap_dict = snapshot.to_dict()
    restored = StructuralGraphSnapshot.from_dict(snap_dict)
    
    assert restored.as_of_date == snapshot.as_of_date
    assert restored.entity_ids == snapshot.entity_ids
    np.testing.assert_allclose(restored.node_embeddings, snapshot.node_embeddings)
    assert len(restored.edges_by_relation["CORR"]) == len(snapshot.edges_by_relation["CORR"])
