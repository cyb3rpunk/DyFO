import torch

from dyfo.config import DyFOConfig
from dyfo.core.relation_aware_tgn import (
    GroupedMessages,
    IntraRelationAggregator,
    RelationAwareTGNEncoder,
)


def test_intra_relation_aggregator_uses_mean_per_group():
    aggregator = IntraRelationAggregator()
    grouped_messages = {
        "system_event": GroupedMessages(
            node_ids=torch.tensor([0, 0, 2]),
            messages=torch.tensor(
                [
                    [1.0, 3.0],
                    [3.0, 5.0],
                    [10.0, 20.0],
                ]
            ),
            timestamps=torch.tensor([1.0, 1.0, 1.0]),
        )
    }

    aggregated = aggregator(grouped_messages, num_nodes=4)

    assert aggregated["system_event"].shape == (4, 2)
    assert torch.allclose(aggregated["system_event"][0], torch.tensor([2.0, 4.0]))
    assert torch.allclose(aggregated["system_event"][2], torch.tensor([10.0, 20.0]))
    assert torch.allclose(aggregated["system_event"][1], torch.zeros(2))


def test_relation_aware_encoder_builds_grouped_messages_from_events():
    config = DyFOConfig()
    encoder = RelationAwareTGNEncoder(
        num_nodes=4,
        event_type_names=config.event_types,
        memory_dim=12,
        relation_dim=12,
        time_dim=8,
        event_feature_dim=3,
        edge_type_emb_dim=4,
        num_edge_types=len(config.edge_types),
    )

    no_edge_id = len(config.edge_types)
    corr_edge_id = config.edge_types.index("CORR")

    source_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    target_nodes = torch.tensor([-1, -1, 3], dtype=torch.long)
    timestamps = torch.tensor([5.0, 5.0, 5.0])
    event_features = torch.tensor(
        [
            [0.2, 0.1, 0.3],   # PRICE_UPDATE
            [1.0, 0.0, -0.5],  # FED_DECISION
            [0.8, -0.1, 1.0],  # CORRELATION_UPDATE
        ]
    )
    edge_type_ids = torch.tensor([no_edge_id, no_edge_id, corr_edge_id], dtype=torch.long)
    event_type_ids = torch.tensor(
        [
            config.event_types.index("PRICE_UPDATE"),
            config.event_types.index("FED_DECISION"),
            config.event_types.index("CORRELATION_UPDATE"),
        ],
        dtype=torch.long,
    )

    aggregated = encoder.build_intra_relation_messages(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        timestamps=timestamps,
        event_features=event_features,
        edge_type_ids=edge_type_ids,
        event_type_ids=event_type_ids,
    )

    assert set(aggregated) == {"node_event", "system_event", "pair_relation"}
    assert aggregated["node_event"].shape == (4, 12)
    assert aggregated["system_event"].shape == (4, 12)
    assert aggregated["pair_relation"].shape == (4, 12)
    assert torch.count_nonzero(aggregated["node_event"][0]).item() > 0
    assert torch.count_nonzero(aggregated["system_event"][1]).item() > 0
    assert torch.count_nonzero(aggregated["pair_relation"][2]).item() > 0
    assert torch.count_nonzero(aggregated["pair_relation"][3]).item() > 0


def test_relation_aware_encoder_encodes_static_relations():
    config = DyFOConfig()
    encoder = RelationAwareTGNEncoder(
        num_nodes=4,
        event_type_names=config.event_types,
        memory_dim=10,
        relation_dim=10,
        time_dim=6,
        event_feature_dim=3,
        edge_type_emb_dim=4,
        num_edge_types=len(config.edge_types),
        max_static_edge_feature_dim=5,
    )

    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    edge_type_ids = torch.tensor(
        [config.edge_types.index("SECT"), config.edge_types.index("FACT")],
        dtype=torch.long,
    )
    edge_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, -0.4, 0.5],
        ]
    )

    encoded = encoder.encode_static_relations(
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_features=edge_features,
    )

    assert encoded.shape == (2, 10)


def test_relation_aware_encoder_process_events_updates_memory_and_attention():
    config = DyFOConfig()
    encoder = RelationAwareTGNEncoder(
        num_nodes=4,
        event_type_names=config.event_types,
        edge_type_names=config.edge_types,
        memory_dim=12,
        relation_dim=12,
        embedding_dim=10,
        node_feat_dim=config.node_feature_dim,
        time_dim=8,
        event_feature_dim=3,
        edge_type_emb_dim=4,
        num_edge_types=len(config.edge_types),
    )

    no_edge_id = len(config.edge_types)
    corr_edge_id = config.edge_types.index("CORR")

    source_nodes = torch.tensor([0, 1, 2], dtype=torch.long)
    target_nodes = torch.tensor([-1, -1, 3], dtype=torch.long)
    timestamps = torch.tensor([5.0, 5.0, 5.0])
    event_features = torch.tensor(
        [
            [0.2, 0.1, 0.3],
            [1.0, 0.0, -0.5],
            [0.8, -0.1, 1.0],
        ]
    )
    event_edge_type_ids = torch.tensor(
        [no_edge_id, no_edge_id, corr_edge_id],
        dtype=torch.long,
    )
    event_type_ids = torch.tensor(
        [
            config.event_types.index("PRICE_UPDATE"),
            config.event_types.index("FED_DECISION"),
            config.event_types.index("CORRELATION_UPDATE"),
        ],
        dtype=torch.long,
    )

    static_edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    static_edge_type_ids = torch.tensor(
        [config.edge_types.index("SECT"), config.edge_types.index("FACT")],
        dtype=torch.long,
    )
    static_edge_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, -0.4, 0.5],
        ]
    )

    encoder.process_events(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        timestamps=timestamps,
        event_features=event_features,
        edge_type_ids=event_edge_type_ids,
        event_type_ids=event_type_ids,
        static_edge_index=static_edge_index,
        static_edge_type_ids=static_edge_type_ids,
        static_edge_features=static_edge_features,
    )

    assert encoder.last_attn_weights is not None
    assert encoder.last_attn_weights.shape == (4, 4)
    assert torch.count_nonzero(encoder.memory[0]).item() > 0
    assert torch.count_nonzero(encoder.memory[1]).item() > 0
    assert torch.count_nonzero(encoder.memory[2]).item() > 0
    assert torch.count_nonzero(encoder.memory[3]).item() > 0


def test_relation_aware_encoder_compute_embeddings_uses_real_edge_features():
    config = DyFOConfig()
    encoder = RelationAwareTGNEncoder(
        num_nodes=4,
        event_type_names=config.event_types,
        edge_type_names=config.edge_types,
        memory_dim=12,
        relation_dim=12,
        embedding_dim=9,
        node_feat_dim=config.node_feature_dim,
        time_dim=8,
        event_feature_dim=3,
        edge_type_emb_dim=4,
        num_edge_types=len(config.edge_types),
    )

    node_features = torch.randn(4, config.node_feature_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_type_ids = torch.tensor(
        [
            config.edge_types.index("CORR"),
            config.edge_types.index("FACT"),
            config.edge_types.index("SUPL"),
            config.edge_types.index("SECT"),
        ],
        dtype=torch.long,
    )
    edge_timestamps = torch.tensor([1.0, 1.0, 1.0, 1.0])
    edge_features = torch.tensor(
        [
            [0.5, 0.1, 1.0, 0.0, 0.0],
            [0.1, -0.2, 0.3, -0.4, 0.5],
            [0.7, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    z = encoder.compute_embeddings(
        node_features=node_features,
        edge_index=edge_index,
        edge_type_ids=edge_type_ids,
        edge_timestamps=edge_timestamps,
        current_time=2.0,
        edge_features=edge_features,
    )

    assert z.shape == (4, 9)
    assert not torch.isnan(z).any()
    assert encoder.embedding_layer.last_alpha.shape == (4,)
