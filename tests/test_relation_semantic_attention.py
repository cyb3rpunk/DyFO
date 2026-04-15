import torch

from dyfo.core.relation_semantic_attention import RelationSemanticAttention


def test_relation_semantic_attention_returns_dense_weights_for_active_groups():
    module = RelationSemanticAttention(relation_dim=8, num_relations=4)

    relation_states = [
        torch.randn(5, 8),
        torch.randn(5, 8),
        torch.randn(5, 8),
    ]

    fused, attn = module(relation_states, relation_indices=[0, 2, 3])

    assert fused.shape == (5, 8)
    assert attn.shape == (5, 4)
    assert torch.allclose(attn[:, 1], torch.zeros(5), atol=1e-6)
    assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-6)
    assert module.last_attn_weights is not None
    assert not module.last_attn_weights.requires_grad
    assert torch.allclose(module.last_attn_weights, attn.detach())


def test_relation_semantic_attention_single_group_becomes_identity_fusion():
    module = RelationSemanticAttention(relation_dim=6, num_relations=4)
    only_group = torch.randn(3, 6)

    fused, attn = module([only_group], relation_indices=[2])

    assert torch.allclose(fused, only_group, atol=1e-6)
    expected = torch.zeros(3, 4)
    expected[:, 2] = 1.0
    assert torch.allclose(attn, expected, atol=1e-6)
