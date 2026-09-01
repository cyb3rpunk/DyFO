"""Unit tests for Continuous Deep Reinforcement Learning (Actor-Critic / PPO)."""

import numpy as np
import pytest
import torch

from dyfo.core.ticker_registry import TICKERS_30
from dyfo.drl.continuous_state import ContinuousDRLState, ContinuousStateConstructor
from dyfo.drl.relational_actor_critic import RelationalActorCriticPolicy
from dyfo.drl.ppo_trainer import EpisodeTrajectory, PPOConfig, PPOTrainer


@pytest.fixture
def sample_drl_env():
    n = len(TICKERS_30)
    embedding_dim = 100
    constructor = ContinuousStateConstructor(tickers=TICKERS_30, embedding_dim=embedding_dim, macro_dim=4)
    rng = np.random.RandomState(42)
    embs = rng.randn(n, embedding_dim).astype(np.float32)
    weights = np.full(n, 1.0 / n, dtype=np.float32)
    macro = np.array([0.2, -0.1, 0.5, 0.0], dtype=np.float32)
    return constructor, embs, weights, macro


def test_continuous_state_construction(sample_drl_env):
    constructor, embs, weights, macro = sample_drl_env
    state = constructor.build_state(embs, weights, macro, date_str="2024-01-02")

    assert isinstance(state, ContinuousDRLState)
    assert state.num_assets == len(TICKERS_30)
    assert state.feature_dim == 100 + 1 + 4  # 105
    assert state.node_features.shape == (len(TICKERS_30), 105)
    assert not torch.isnan(state.node_features).any()


def test_relational_actor_critic_forward(sample_drl_env):
    constructor, embs, weights, macro = sample_drl_env
    state = constructor.build_state(embs, weights, macro)

    policy = RelationalActorCriticPolicy(feature_dim=105, hidden_dim=32, num_heads=2, num_layers=1)
    out = policy(state)

    assert out.weights.shape == (1, len(TICKERS_30))
    assert out.value.shape == (1, 1)

    # Verify simplex properties (w >= 0 and sum(w) == 1)
    w_sum = float(torch.sum(out.weights).item())
    assert abs(w_sum - 1.0) < 1e-5
    assert (out.weights >= 0.0).all()

    # Test act method
    w_act, log_prob, val = policy.act(state, deterministic=True)
    assert len(w_act) == len(TICKERS_30)
    assert abs(np.sum(w_act) - 1.0) < 1e-5
    assert not np.isnan(val)


def test_ppo_reward_and_gae(sample_drl_env):
    constructor, embs, weights, macro = sample_drl_env
    policy = RelationalActorCriticPolicy(feature_dim=105, hidden_dim=32)
    trainer = PPOTrainer(policy=policy, config=PPOConfig())

    curr_w = np.full(len(TICKERS_30), 1.0 / len(TICKERS_30))
    prev_w = np.zeros(len(TICKERS_30))
    prev_w[0] = 1.0

    reward = trainer.compute_step_reward(
        realized_return=0.02,
        curr_weights=curr_w,
        prev_weights=prev_w,
        drawdown=0.04,
    )

    assert isinstance(reward, float)
    assert not np.isnan(reward)


def test_ppo_training_step(sample_drl_env):
    constructor, embs, weights, macro = sample_drl_env
    policy = RelationalActorCriticPolicy(feature_dim=105, hidden_dim=32, num_heads=2, num_layers=1)
    trainer = PPOTrainer(policy=policy, config=PPOConfig(lr=1e-3))

    # Record small trajectory
    traj = EpisodeTrajectory()
    curr_w = weights.copy()

    for step in range(5):
        st = constructor.build_state(embs + step * 0.01, curr_w, macro)
        w_step, lp, val = policy.act(st)
        r = trainer.compute_step_reward(0.01, w_step, curr_w)
        traj.states.append(st)
        traj.actions.append(w_step)
        traj.log_probs.append(lp)
        traj.rewards.append(r)
        traj.values.append(val)
        traj.dones.append(False)
        curr_w = w_step

    metrics = trainer.train_epoch(traj, ppo_epochs=2)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert not np.isnan(metrics["loss"])
