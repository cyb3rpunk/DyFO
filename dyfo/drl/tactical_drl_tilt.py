"""Tactical 1/N DRL Policy with Explicit L1 Simplex Projection.

Implements an anchored tactical policy around the 1/N benchmark:
  w_t = ProjectSimplex(1/N * 1 + delta_t)
  subject to sum(delta) = 0, ||w_t - w_{1/N}||_1 <= Delta_max

And recursive Differential Sharpe Ratio (DSR) reward formulation (Moody & Saffell, 2001).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def project_to_l1_simplex_ball(
    weights: torch.Tensor,
    anchor: torch.Tensor,
    delta_max: float = 0.30,
) -> torch.Tensor:
    """Project weights onto the intersection of the probability simplex and an L1 ball around anchor.

    Parameters
    ----------
    weights : Tensor (B, N) or (N,)
        Proposed weights summing to 1.
    anchor : Tensor (B, N) or (N,)
        Anchor weights (e.g. 1/N).
    delta_max : float, default 0.30
        Maximum allowable L1 deviation ||w - w_anchor||_1.

    Returns
    -------
    Tensor of same shape on Delta^N with ||w - w_anchor||_1 <= delta_max.
    """
    diff = weights - anchor
    l1_norm = torch.sum(torch.abs(diff), dim=-1, keepdim=True)
    scale = torch.clamp(delta_max / (l1_norm + 1e-8), max=1.0)
    w_projected = anchor + diff * scale
    # Ensure non-negativity and exact sum to 1
    w_projected = torch.clamp(w_projected, min=0.0)
    return w_projected / torch.sum(w_projected, dim=-1, keepdim=True)


class TacticalDRLPolicy(nn.Module):
    """Relation-aware Actor-Critic Policy generating bounded tilts over 1/N."""

    def __init__(
        self,
        node_feature_dim: int = 105,
        hidden_dim: int = 64,
        num_heads: int = 4,
        delta_max: float = 0.30,
    ):
        super().__init__()
        self.delta_max = delta_max
        self.node_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Cross-Asset Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Tilt Generator Head
        self.tilt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Value Baseline Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass generating tactical weights and value estimate.

        Parameters
        ----------
        state_tensor : Tensor of shape (B, N, F) or (N, F)

        Returns
        -------
        Tuple[Tensor, Tensor]
            (Simplex weights w_t in Delta^N, Value estimate V(s)).
        """
        is_unbatched = state_tensor.ndim == 2
        if is_unbatched:
            x = state_tensor.unsqueeze(0)  # (1, N, F)
        else:
            x = state_tensor

        b, n, f = x.shape
        h = self.node_proj(x)  # (B, N, H)
        attn_out, _ = self.attn(h, h, h)  # (B, N, H)
        h_fused = h + attn_out

        # Raw delta logits
        raw_deltas = self.tilt_head(h_fused).squeeze(-1)  # (B, N)
        # Enforce zero-mean tilt
        centered_deltas = raw_deltas - torch.mean(raw_deltas, dim=-1, keepdim=True)
        bounded_deltas = self.delta_max * torch.tanh(centered_deltas)

        # Apply softmax over log(1/N) + bounded_deltas
        anchor = torch.full((b, n), 1.0 / n, dtype=x.dtype, device=x.device)
        logits = torch.log(anchor) + bounded_deltas
        w_raw = F.softmax(logits, dim=-1)

        # Explicit L1 projection
        w_final = project_to_l1_simplex_ball(w_raw, anchor, self.delta_max)

        # Value baseline from pooled node representations
        h_pool = torch.mean(h_fused, dim=1)  # (B, H)
        v_s = self.value_head(h_pool).squeeze(-1)  # (B,)

        if is_unbatched:
            return w_final.squeeze(0), v_s.squeeze(0)
        return w_final, v_s


class DifferentialSharpeReward:
    """Recursive Differential Sharpe Ratio calculator (Moody & Saffell, 2001)."""

    def __init__(self, eta: float = 0.05, turnover_penalty: float = 0.50):
        self.eta = eta
        self.turnover_penalty = turnover_penalty
        self.A_t = 0.0
        self.B_t = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.A_t = 0.0
        self.B_t = 0.0
        self.initialized = False

    def compute_reward(
        self,
        realized_return: float,
        turnover: float,
    ) -> float:
        """Calculate online DSR step reward."""
        r = float(realized_return)
        to = float(turnover)

        if not self.initialized:
            self.A_t = r
            self.B_t = r * r
            self.initialized = True
            return r - self.turnover_penalty * to

        # DSR Gradient term
        delta_A = r - self.A_t
        delta_B = r * r - self.B_t
        variance = max(self.B_t - self.A_t * self.A_t, 1e-6)

        dsr = (self.B_t * delta_A - 0.5 * self.A_t * delta_B) / (variance ** 1.5)

        # Update EMA moments
        self.A_t += self.eta * delta_A
        self.B_t += self.eta * delta_B

        reward = float(dsr) - self.turnover_penalty * to
        return reward
