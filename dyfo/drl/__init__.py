"""DyFO Continuous Deep Reinforcement Learning (DRL) Package.

Implements Relational State Augmentation, Cross-Attention Actor-Critic Policies,
and Risk-Regularized PPO/A2C for continuous portfolio optimization.
"""

from dyfo.drl.continuous_state import (
    ContinuousDRLState,
    ContinuousStateConstructor,
)
from dyfo.drl.relational_actor_critic import (
    RelationalActorCriticPolicy,
    ActorCriticOutput,
)
from dyfo.drl.ppo_trainer import (
    PPOTrainer,
    PPOConfig,
    EpisodeTrajectory,
)

__all__ = [
    "ContinuousDRLState",
    "ContinuousStateConstructor",
    "RelationalActorCriticPolicy",
    "ActorCriticOutput",
    "PPOTrainer",
    "PPOConfig",
    "EpisodeTrajectory",
]
