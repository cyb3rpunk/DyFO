import torch

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import EventType, FinancialEvent
from dyfo.core.temporal_kg import TemporalKGEncoder
from dyfo.core.temporal_kg_adapter import TemporalKGAdapter


def test_temporal_kg_adapter_is_deterministic_for_events():
    adapter = TemporalKGAdapter(num_asset_nodes=4)
    event = FinancialEvent(
        event_type=EventType.CORRELATION_UPDATE,
        timestamp=123.0,
        source_node=0,
        target_node=1,
        edge_type="CORR",
        features=torch.tensor([0.8, 0.1, 1.0]),
    )

    fact_a = adapter.event_to_fact(event)
    fact_b = adapter.event_to_fact(event)

    assert fact_a == fact_b
    assert fact_a.relation == "correlated_with"
    assert fact_a.head == "asset:0"
    assert fact_a.tail == "asset:1"


def test_temporal_kg_encoder_exports_interpretability_artifacts():
    encoder = TemporalKGEncoder(DyFOConfig(model_variant="temporal_kg"), num_nodes=4)
    events = [
        FinancialEvent(
            event_type=EventType.EARNINGS_REPORT,
            timestamp=50.0,
            source_node=0,
            target_node=-1,
            edge_type=None,
            features=torch.tensor([0.3, 0.2, 0.0]),
        ),
        FinancialEvent(
            event_type=EventType.CORRELATION_UPDATE,
            timestamp=50.0,
            source_node=0,
            target_node=1,
            edge_type="CORR",
            features=torch.tensor([0.7, 0.1, 1.0]),
        ),
    ]

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type_ids = torch.tensor([0, 0], dtype=torch.long)
    edge_timestamps = torch.zeros(2)
    node_features = torch.randn(4, 20)

    encoder.advance_day(events, node_features, edge_index, edge_type_ids, edge_timestamps, current_time=50.99)
    z = encoder.get_node_embeddings(node_features, edge_index, edge_type_ids, edge_timestamps, current_time=50.99)
    artifacts = encoder.export_temporal_kg_artifacts()

    assert z.shape == (4, encoder.embedding_dim)
    assert artifacts["n_facts_seen"] >= 2
    assert artifacts["top_explanations"]
    assert "correlated_with" in artifacts["relation_histogram"]
