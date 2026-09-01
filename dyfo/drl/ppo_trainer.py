"""Risk-Regularized PPO Trainer for Continuous Portfolio DRL.

Implements Generalized Advantage Estimation (GAE), Clipped Policy Gradient,
and Multi-Objective Risk Penalties (Turnover, Portfolio Variance, Drawdown).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dyfo.drl.continuous_state import ContinuousDRLState
from dyfo.drl.relational_actor_critic import RelationalActorCriticPolicy


@dataclass
class PPOConfig:
    """Hyperparameters for Risk-Regularized PPO."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    turnover_penalty: float = 0.005
    volatility_penalty: float = 0.01
    drawdown_penalty: float = 0.05


@dataclass
class EpisodeTrajectory:
    """Container for recorded transition steps in an episode."""
    states: List[ContinuousDRLState] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)  # Portfolio weights (N,)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)


class PPOTrainer:
    """Trainer for continuous relational portfolio policy."""

    def __init__(
        self,
        policy: RelationalActorCriticPolicy,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ):
        self.policy = policy
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=self.config.lr, weight_decay=1e-4)

    def compute_step_reward(
        self,
        realized_return: float,
        curr_weights: np.ndarray,
        prev_weights: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        drawdown: float = 0.0,
    ) -> float:
        """Compute multi-objective risk-penalized reward for DRL transition.

        R_t = ln(1 + r_p) - lambda_turn * ||w_t - w_{t-1}||_1 - lambda_vol * (w^T Sigma w) - lambda_dd * DD
        """
        # 1. Base Log Return
        r_log = float(np.log(max(1.0 + realized_return, 1e-4)))

        # 2. Turnover Penalty
        turnover = float(np.sum(np.abs(curr_weights - prev_weights)))
        pen_turn = self.config.turnover_penalty * turnover

        # 3. Covariance Volatility Penalty
        pen_vol = 0.0
        if cov_matrix is not None:
            port_var = float(curr_weights @ cov_matrix @ curr_weights)
            pen_vol = self.config.volatility_penalty * port_var

        # 4. Drawdown Penalty
        pen_dd = 0.0
        if drawdown > 0.03:
            pen_dd = self.config.drawdown_penalty * (drawdown - 0.03)

        reward = r_log - pen_turn - pen_vol - pen_dd
        return float(reward)

    def train_epoch(self, trajectory: EpisodeTrajectory, ppo_epochs: int = 4) -> Dict[str, float]:
        """Perform PPO update over collected episode trajectory."""
        if len(trajectory.rewards) < 2:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        rewards = np.array(trajectory.rewards, dtype=np.float32)
        values = np.array(trajectory.values, dtype=np.float32)
        old_log_probs = torch.tensor(trajectory.log_probs, dtype=torch.float32, device=self.device)
        old_actions = torch.tensor(np.array(trajectory.actions), dtype=torch.float32, device=self.device)

        # 1. Compute Generalized Advantage Estimation (GAE)
        t_steps = len(rewards)
        advantages = np.zeros(t_steps, dtype=np.float32)
        returns = np.zeros(t_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(t_steps)):
            next_val = values[t + 1] if t + 1 < t_steps else 0.0
            delta = rewards[t] + self.config.gamma * next_val - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            returns[t] = advantages[t] + values[t]

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # Batch states tensor
        state_tensors = torch.stack([s.node_features for s in trajectory.states], dim=0).to(self.device)

        self.policy.train()
        total_loss_val = 0.0
        p_loss_val = 0.0
        v_loss_val = 0.0

        for _ in range(ppo_epochs):
            self.optimizer.zero_grad()
            out = self.policy(state_tensors)
            new_weights = out.weights  # (T, N)
            new_values = out.value  # (T, 1)

            # Compute new log probs: sum log(w_i + 1e-8) * old_actions
            new_log_probs = torch.sum(torch.log(new_weights + 1e-8) * old_actions, dim=1)

            # Probability Ratio with clamping
            log_ratio = torch.clamp(new_log_probs - old_log_probs, -10.0, 10.0)
            ratio = torch.exp(log_ratio)

            # Clipped Surrogate Policy Loss
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv_tensor
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # Value Function MSE Loss
            value_loss = F.mse_loss(new_values, ret_tensor)

            # Entropy Bonus for Simplex Uniformity
            entropy = -torch.mean(torch.sum(new_weights * torch.log(new_weights + 1e-8), dim=1))

            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss_val += float(loss.item())
            p_loss_val += float(policy_loss.item())
            v_loss_val += float(value_loss.item())

        return {
            "loss": total_loss_val / ppo_epochs,
            "policy_loss": p_loss_val / ppo_epochs,
            "value_loss": v_loss_val / ppo_epochs,
            "mean_reward": float(np.mean(rewards)),
        }
