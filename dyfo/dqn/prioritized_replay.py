"""Prioritized Experience Replay (PER) for Deep Q-Learning.

Samples transitions proportionally to their Temporal Difference (TD) error magnitude |delta|,
allowing the DQN agent to learn more frequently from rare market shocks and regime transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """Individual transition tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance-sampling correction."""

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 5000,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.frame = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add new transition with maximum current priority."""
        max_prio = np.max(self.priorities[:len(self.buffer)]) if self.buffer else 1.0
        if max_prio <= 0.0:
            max_prio = 1.0

        trans = Transition(
            state=np.asarray(state, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            done=bool(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(trans)
        else:
            self.buffer[self.pos] = trans

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample batch of transitions weighted by priority.

        Returns
        -------
        Tuple containing:
        - states (batch, state_dim)
        - actions (batch,)
        - rewards (batch,)
        - next_states (batch, state_dim)
        - dones (batch,)
        - is_weights (batch,)
        - indices (batch,)
        """
        self.frame += 1
        n = len(self.buffer)
        assert n > 0, "Cannot sample from empty replay buffer."

        # Proportional Sampling Probabilities P(i) = p_i^alpha / sum(p_k^alpha)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= np.sum(probs)

        batch_size = min(batch_size, n)
        indices = np.random.choice(n, size=batch_size, p=probs)

        # Importance-Sampling Weights: w_i = (N * P(i))^(-beta)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (n * probs[indices]) ** (-beta)
        weights /= np.max(weights)  # Normalize by max weight

        sampled_transitions = [self.buffer[idx] for idx in indices]

        states = torch.tensor(np.array([t.state for t in sampled_transitions]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in sampled_transitions], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in sampled_transitions], dtype=torch.float32)
        next_states = torch.tensor(np.array([t.next_state for t in sampled_transitions]), dtype=torch.float32)
        dones = torch.tensor([t.done for t in sampled_transitions], dtype=torch.float32)
        is_weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, is_weights, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update transition priorities based on new TD-errors."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = float(abs(td) + self.epsilon)

    def __len__(self) -> int:
        return len(self.buffer)
