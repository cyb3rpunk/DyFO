"""Quick smoke test — verifies that all modules import and a forward pass runs."""

import torch

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import EventType, FinancialEvent, timestamp_to_float
from dyfo.core.dyfo_module import DyFOModule


def test_forward_pass():
    """Minimal forward pass with synthetic data."""
    config = DyFOConfig()
    num_nodes = 10

    module = DyFOModule(config=config, num_nodes=num_nodes, readout_strategy="mean")

    # Synthetic events
    events = [
        FinancialEvent(
            event_type=EventType.PRICE_UPDATE,
            timestamp=1000.0 + i * 0.01,
            source_node=i % num_nodes,
            target_node=-1,
            edge_type=None,
            features=torch.randn(3),
        )
        for i in range(20)
    ]
    # Add a pair event (correlation update)
    events.append(
        FinancialEvent(
            event_type=EventType.CORRELATION_UPDATE,
            timestamp=1000.5,
            source_node=0,
            target_node=1,
            edge_type="CORR",
            features=torch.tensor([0.65, 0.05, 1.0]),
        )
    )

    node_features = torch.randn(num_nodes, config.node_feature_dim)

    # Simple edge index (ring graph)
    src = list(range(num_nodes))
    tgt = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([src + tgt, tgt + src], dtype=torch.long)
    edge_type_ids = torch.zeros(edge_index.shape[1], dtype=torch.long)
    edge_timestamps = torch.full((edge_index.shape[1],), 999.0)

    # Forward pass
    e_t = module(
        events=events,
        node_features=node_features,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_timestamps=edge_timestamps,
        current_time=1001.0,
    )

    assert e_t.shape == (config.embedding_dim,), f"Expected ({config.embedding_dim},), got {e_t.shape}"
    assert not torch.isnan(e_t).any(), "e_t contains NaN"
    print(f"[PASS] Forward pass OK — e_t shape: {e_t.shape}, norm: {e_t.norm():.4f}")

    # Test memory reset
    module.reset_memory()
    assert module.encoder.memory.abs().sum() == 0, "Memory not zeroed"
    print("[PASS] Memory reset OK")


def test_readout_strategies():
    """Test all three readout strategies."""
    config = DyFOConfig()
    num_nodes = 5

    for strategy in ["mean", "weighted", "attention"]:
        module = DyFOModule(config=config, num_nodes=num_nodes, readout_strategy=strategy)
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type_ids = torch.zeros(0, dtype=torch.long)
        edge_timestamps = torch.zeros(0)

        weights = torch.randn(num_nodes) if strategy == "weighted" else None

        e_t = module(
            events=[],
            node_features=node_features,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_timestamps=edge_timestamps,
            current_time=0.0,
            readout_weights=weights,
        )
        assert e_t.shape == (config.embedding_dim,)
        print(f"[PASS] Readout '{strategy}' OK — e_t norm: {e_t.norm():.4f}")


if __name__ == "__main__":
    test_forward_pass()
    test_readout_strategies()
    print("\n=== All smoke tests passed ===")
