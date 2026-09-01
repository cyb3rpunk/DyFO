"""Unit tests for Discrete Deep Q-Networks (DQN) Dynamic Hedging."""

import numpy as np
import pytest
import torch

from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING
from dyfo.dqn.discrete_state import DiscreteDQNState, DiscreteStateConstructor, REGIME_ACTIONS
from dyfo.dqn.dueling_dqn import DuelingDQNNetwork
from dyfo.dqn.prioritized_replay import PrioritizedReplayBuffer
from dyfo.dqn.dqn_agent import DQNHedgingAgent


@pytest.fixture
def sample_dqn_env():
    n = len(TICKERS_30)
    rng = np.random.RandomState(42)
    A = rng.randn(n, n)
    cov = A @ A.T + np.eye(n) * 1.5
    embs = rng.randn(n, 100).astype(np.float32)
    return cov, embs


def test_discrete_state_construction(sample_dqn_env):
    cov, embs = sample_dqn_env
    constructor = DiscreteStateConstructor(tickers=TICKERS_30, state_dim=16)

    state = constructor.build_state(
        cov_matrix=cov,
        node_embeddings=embs,
        current_drawdown=-0.04,
        realized_vol_30d=0.18,
        macro_prob=0.8,
        date_str="2024-03-01",
    )

    assert isinstance(state, DiscreteDQNState)
    assert state.state_dim == 16
    assert state.features.shape == (16,)
    assert 0.0 <= state.eigen_concentration <= 1.0
    assert not torch.isnan(state.features).any()


def test_dueling_dqn_forward():
    net = DuelingDQNNetwork(state_dim=16, num_actions=4, hidden_dim=32)
    sample_state = torch.randn(8, 16)
    q_vals = net(sample_state)

    assert q_vals.shape == (8, 4)
    assert not torch.isnan(q_vals).any()


def test_prioritized_replay_buffer():
    buf = PrioritizedReplayBuffer(capacity=100)
    assert len(buf) == 0

    for i in range(20):
        s = np.random.randn(16)
        s_next = np.random.randn(16)
        buf.push(s, action=i % 4, reward=0.01 * i, next_state=s_next, done=False)

    assert len(buf) == 20

    states, actions, rewards, next_states, dones, weights, indices = buf.sample(batch_size=8)
    assert states.shape == (8, 16)
    assert actions.shape == (8,)
    assert weights.shape == (8,)
    assert len(indices) == 8

    # Update priorities
    buf.update_priorities(indices, td_errors=np.ones(8) * 0.5)


def test_dqn_agent_training_and_action_execution(sample_dqn_env):
    cov, embs = sample_dqn_env
    agent = DQNHedgingAgent(state_dim=16, num_actions=4, lr=1e-3, buffer_capacity=500)

    # Test all 4 regime action allocations
    for act_idx in range(4):
        w = agent.execute_regime_action(act_idx, cov, embs, tickers=TICKERS_30, sector_mapping=TICKER_GICS_MAPPING)
        assert len(w) == len(TICKERS_30)
        assert np.all(w >= -1e-6), f"Action {act_idx} produced negative weights"
        assert np.sum(w) <= 1.0001, f"Action {act_idx} total weight exceeded 1.0"

    # Fill buffer and test training step
    for _ in range(50):
        s = np.random.randn(16).astype(np.float32)
        s_next = np.random.randn(16).astype(np.float32)
        agent.replay_buffer.push(s, action=np.random.randint(0, 4), reward=0.02, next_state=s_next, done=False)

    metrics = agent.train_step(batch_size=16)
    assert "loss" in metrics
    assert "mean_q" in metrics
    assert not np.isnan(metrics["loss"])
