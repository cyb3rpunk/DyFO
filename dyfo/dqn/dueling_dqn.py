"""Dueling Deep Q-Network (DQN) Architecture.

Separates State Value V(s) and Action Advantage A(s, a) streams to identify
valuable states without having to learn the effect of each action at each state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQNNetwork(nn.Module):
    """Dueling Double-DQN neural network."""

    def __init__(
        self,
        state_dim: int = 16,
        num_actions: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Shared Feature Representation Backbone
        self.feature_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Action Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) = V(s) + (A(s, a) - mean(A(s, a'))).

        Parameters
        ----------
        state : torch.Tensor
            State tensor of shape (batch, state_dim) or (state_dim,).

        Returns
        -------
        torch.Tensor
            Q-values of shape (batch, num_actions).
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)

        features = self.feature_network(state)
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, num_actions)

        # Centered Dueling Formula
        q_values = value + (advantage - torch.mean(advantage, dim=-1, keepdim=True))
        return q_values
