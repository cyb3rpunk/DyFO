"""Semantic attention fusion across relation-specific node representations.

BL-17 Session 1 introduces the inter-relation fusion block used by the
relation-aware heterogeneous TGN. Each active semantic group produces a
node-level representation ``h_i^r`` and this module learns a scalar score
per group before applying a softmax across relations.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class RelationSemanticAttention(nn.Module):
    """Fuse relation-specific node states with semantic attention.

    Parameters
    ----------
    relation_dim
        Shared dimensionality ``d_rel`` for each relation-specific tensor.
    num_relations
        Total number of semantic groups in the model. BL-17 fixes this at 4.

    Notes
    -----
    ``forward`` accepts only the active groups for the current batch/day.
    To keep diagnostics stable for downstream analysis, ``last_attn_weights``
    is always stored with shape ``(N, num_relations)`` and zeros for inactive
    groups.
    """

    def __init__(self, relation_dim: int, num_relations: int = 4):
        super().__init__()
        if relation_dim <= 0:
            raise ValueError("relation_dim must be positive.")
        if num_relations <= 0:
            raise ValueError("num_relations must be positive.")

        self.relation_dim = relation_dim
        self.num_relations = num_relations
        self.score = nn.Linear(relation_dim, 1, bias=False)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        relation_states: List[torch.Tensor],
        relation_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fused node states and attention weights.

        Parameters
        ----------
        relation_states
            Active relation tensors, each with shape ``(N, relation_dim)``.
        relation_indices
            Optional mapping from each active tensor to its semantic-group id
            in ``[0, num_relations)``. When omitted, groups are mapped to the
            first ``len(relation_states)`` slots.

        Returns
        -------
        fused_states
            Tensor with shape ``(N, relation_dim)``.
        attn_weights
            Dense tensor with shape ``(N, num_relations)`` containing the
            softmax weight for each active group and zeros elsewhere.
        """
        if not relation_states:
            raise ValueError("relation_states must contain at least one tensor.")

        num_active = len(relation_states)
        if num_active > self.num_relations:
            raise ValueError(
                f"Received {num_active} active groups, but num_relations={self.num_relations}."
            )

        if relation_indices is None:
            relation_indices = list(range(num_active))
        elif len(relation_indices) != num_active:
            raise ValueError("relation_indices must match relation_states length.")

        if len(set(relation_indices)) != len(relation_indices):
            raise ValueError("relation_indices must be unique.")

        base_shape = relation_states[0].shape
        if len(base_shape) != 2:
            raise ValueError("Each relation tensor must have shape (N, relation_dim).")

        num_nodes, relation_dim = base_shape
        if relation_dim != self.relation_dim:
            raise ValueError(
                f"Expected relation_dim={self.relation_dim}, got {relation_dim}."
            )

        for group_id, state in zip(relation_indices, relation_states):
            if state.shape != base_shape:
                raise ValueError("All relation tensors must share the same shape.")
            if group_id < 0 or group_id >= self.num_relations:
                raise ValueError(
                    f"relation index {group_id} is out of range for num_relations={self.num_relations}."
                )

        stacked_states = torch.stack(relation_states, dim=1)  # (N, R_active, d_rel)
        attn_logits = self.score(stacked_states).squeeze(-1)  # (N, R_active)
        active_attn = torch.softmax(attn_logits, dim=1)

        fused_states = torch.sum(active_attn.unsqueeze(-1) * stacked_states, dim=1)

        full_attn = torch.zeros(
            num_nodes,
            self.num_relations,
            device=stacked_states.device,
            dtype=stacked_states.dtype,
        )
        full_attn[:, list(relation_indices)] = active_attn
        self.last_attn_weights = full_attn.detach()

        return fused_states, full_attn
