"""Relational Cross-Attention Actor-Critic Policy Network for Continuous Portfolio DRL.

Breaks the asset permutation symmetry problem by processing DyFO relational graph states
through Multi-Head Self-Attention layers, outputting portfolio simplex weights w_t in Delta^N
and state value estimates V(s).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dyfo.drl.continuous_state import ContinuousDRLState


@dataclass
class ActorCriticOutput:
    """Output container for policy and value heads."""
    weights: torch.Tensor  # Simplex allocation weights (batch, N)
    logits: torch.Tensor  # Raw unnormalized action logits (batch, N)
    value: torch.Tensor  # Scalar baseline state value (batch, 1)


class RelationalActorCriticPolicy(nn.Module):
    """Actor-Critic architecture with multi-head asset cross-attention."""

    def __init__(
        self,
        feature_dim: int = 105,  # 100 (DyFO embedding) + 1 (weight) + 4 (macro)
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Asset Token Projection
        self.token_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Multi-Head Cross-Asset Attention Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor (Policy) Head: produces asset preference score -> Softmax simplex
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Critic (Value) Head: evaluates whole-market topological state
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_or_tensor: torch.Tensor | ContinuousDRLState) -> ActorCriticOutput:
        """Forward pass through attention encoder and actor/critic heads.

        Parameters
        ----------
        state_or_tensor : torch.Tensor or ContinuousDRLState
            Input tensor of shape (batch, N, feature_dim) or ContinuousDRLState.

        Returns
        -------
        ActorCriticOutput
            Weights, logits, and value estimate.
        """
        if isinstance(state_or_tensor, ContinuousDRLState):
            x = state_or_tensor.node_features
        else:
            x = state_or_tensor

        # Ensure batch dimension
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, N, feature_dim)

        batch_size, num_nodes, _ = x.shape

        # 1. Project to hidden token representations
        tokens = self.token_proj(x)  # (batch, N, hidden_dim)

        # 2. Relational Cross-Attention over Asset Graph Tokens
        h = self.transformer(tokens)  # (batch, N, hidden_dim)

        # 3. Actor Head (Simplex weights via Softmax)
        logits = self.actor_head(h).squeeze(-1)  # (batch, N)
        weights = F.softmax(logits, dim=-1)  # (batch, N) on simplex Delta^N

        # 4. Critic Head (Global Graph Pooling)
        graph_summary = torch.mean(h, dim=1)  # (batch, hidden_dim)
        value = self.critic_head(graph_summary)  # (batch, 1)

        return ActorCriticOutput(weights=weights, logits=logits, value=value)

    def act(
        self,
        state: ContinuousDRLState,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, float, float]:
        """Compute action numpy vector and scalar value for episode stepping.

        Returns
        -------
        Tuple[np.ndarray, float, float]
            (weights_np, log_prob, value_scalar)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(state)
            logits = out.logits / max(temperature, 1e-4)

            if deterministic:
                weights = F.softmax(logits, dim=-1)
            else:
                # Add mild Gumbel-Softmax perturbation for exploration during training
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
                weights = F.softmax(logits + gumbel_noise * 0.15, dim=-1)

            # Action-weighted log probability under simplex action
            log_prob = float(torch.sum(torch.log(weights + 1e-8) * weights, dim=-1).item())
            val_scalar = float(out.value.item())
            w_np = weights.squeeze(0).cpu().numpy()

        return w_np, log_prob, val_scalar
