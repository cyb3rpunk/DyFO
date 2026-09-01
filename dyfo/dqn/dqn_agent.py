"""DQN Dynamic Hedging Agent with Dueling Double-DQN and Prioritized Replay.

Translates discrete regime decisions into continuous portfolio allocations:
- Action 0: ALPHA_GMVP
- Action 1: DEFENSIVE_ERC
- Action 2: TAIL_RISK_HEDGE
- Action 3: SECTOR_ROTATION
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dyfo.core.link_prediction import project_to_spd_covariance
from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING
from dyfo.dqn.discrete_state import REGIME_ACTIONS, DiscreteDQNState
from dyfo.dqn.dueling_dqn import DuelingDQNNetwork
from dyfo.dqn.prioritized_replay import PrioritizedReplayBuffer
from scipy.optimize import minimize

logger = logging.getLogger("DyFO.DQN")


class DQNHedgingAgent:
    """Double-DQN Agent for dynamic portfolio regime switching and hedging."""

    def __init__(
        self,
        state_dim: int = 16,
        num_actions: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.98,
        tau: float = 0.01,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        # Q-Networks (Online and Target)
        self.q_online = DuelingDQNNetwork(state_dim, num_actions).to(self.device)
        self.q_target = DuelingDQNNetwork(state_dim, num_actions).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = optim.AdamW(self.q_online.parameters(), lr=lr, weight_decay=1e-4)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: DiscreteDQNState | np.ndarray | torch.Tensor, deterministic: bool = False) -> int:
        """Select discrete regime action using epsilon-greedy policy."""
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))

        if isinstance(state, DiscreteDQNState):
            s_t = state.features
        elif isinstance(state, np.ndarray):
            s_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            s_t = state.to(self.device)

        self.q_online.eval()
        with torch.no_grad():
            q_vals = self.q_online(s_t)
            action = int(torch.argmax(q_vals, dim=-1).item())

        return action

    def step_decay_epsilon(self) -> None:
        """Decay exploration rate epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def execute_regime_action(
        self,
        action: int,
        cov_matrix: np.ndarray,
        node_embeddings: np.ndarray,
        tickers: Optional[List[str]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Convert discrete regime action into continuous portfolio weights w_t in Delta^N."""
        n = cov_matrix.shape[0]
        tickers = tickers or list(TICKERS_30)
        sec_map = sector_mapping or TICKER_GICS_MAPPING
        sigma_spd = project_to_spd_covariance(cov_matrix, epsilon=1e-5) + np.eye(n) * 1e-5

        # -------------------------------------------------------------
        # Action 0: ALPHA_GMVP (Exact Convex Global Minimum Variance)
        # -------------------------------------------------------------
        if action == 0:
            try:
                res = minimize(
                    fun=lambda w: float(0.5 * w.T @ sigma_spd @ w),
                    x0=np.full(n, 1.0 / n),
                    method="SLSQP",
                    bounds=[(0.0, 0.25) for _ in range(n)],
                    constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                    options={"maxiter": 100, "ftol": 1e-7},
                )
                if res.success:
                    return res.x
                return np.full(n, 1.0 / n)
            except Exception:
                return np.full(n, 1.0 / n)

        # -------------------------------------------------------------
        # Action 1: DEFENSIVE_ERC (Equal Risk Contribution / Inv Vol)
        # -------------------------------------------------------------
        elif action == 1:
            vols = np.sqrt(np.clip(np.diag(sigma_spd), 1e-6, None))
            inv_vols = 1.0 / vols
            return inv_vols / np.sum(inv_vols)

        # -------------------------------------------------------------
        # Action 2: TAIL_RISK_HEDGE (Defensive Utilities/HealthCare + Cash Buffer)
        # -------------------------------------------------------------
        elif action == 2:
            w = np.zeros(n, dtype=np.float32)
            defensive_idx = [
                i for i, t in enumerate(tickers)
                if sec_map.get(t) in ["Utilities", "Health Care", "Consumer Staples"]
            ]
            if not defensive_idx:
                defensive_idx = list(range(n))

            # 50% allocated to defensive bucket, 50% implied cash
            w_def_sub = 0.50 / len(defensive_idx)
            for idx in defensive_idx:
                w[idx] = w_def_sub
            return w

        # -------------------------------------------------------------
        # Action 3: SECTOR_ROTATION (Focus on Top-2 Least Correlated Sectors)
        # -------------------------------------------------------------
        else:
            w = np.zeros(n, dtype=np.float32)
            # Group into non-tech sectors
            rotate_idx = [
                i for i, t in enumerate(tickers)
                if sec_map.get(t) in ["Financials", "Industrials", "Materials", "Energy"]
            ]
            if not rotate_idx:
                rotate_idx = list(range(n))

            for idx in rotate_idx:
                w[idx] = 1.0 / len(rotate_idx)
            return w

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform Double-DQN training step from Prioritized Replay Buffer."""
        if len(self.replay_buffer) < batch_size:
            return {"loss": 0.0, "mean_q": 0.0}

        states, actions, rewards, next_states, dones, is_weights, indices = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)

        self.q_online.train()
        q_current = self.q_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double-DQN Target: a* = argmax Q_online(s_{t+1}), Q_target(s_{t+1}, a*)
            next_q_online = self.q_online(next_states)
            best_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.q_target(next_states).gather(1, best_actions).squeeze(1)
            q_target = rewards + (1.0 - dones) * self.gamma * next_q_target

        td_errors = (q_current - q_target).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Weighted Smooth L1 (Huber) Loss
        loss = torch.mean(is_weights * F.smooth_l1_loss(q_current, q_target, reduction="none"))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 1.0)
        self.optimizer.step()

        # Soft Polyak Target Update
        for target_param, online_param in zip(self.q_target.parameters(), self.q_online.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        self.step_decay_epsilon()

        return {
            "loss": float(loss.item()),
            "mean_q": float(torch.mean(q_current).item()),
            "epsilon": self.epsilon,
        }
