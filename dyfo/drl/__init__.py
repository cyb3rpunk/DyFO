"""DyFO Deep Reinforcement Learning (DRL) Package.

Contains relational state constructors, transformer-based continuous policy networks,
tactical 1/N policies, and risk-regularized PPO training pipelines.
"""

from dyfo.drl.continuous_state import ContinuousStateConstructor, ContinuousDRLState
from dyfo.drl.relational_actor_critic import RelationalActorCriticPolicy
from dyfo.drl.ppo_trainer import PPOTrainer, EpisodeTrajectory, PPOConfig
from dyfo.drl.tactical_drl_tilt import TacticalDRLPolicy, DifferentialSharpeReward, project_to_l1_simplex_ball

__all__ = [
    "ContinuousStateConstructor",
    "ContinuousDRLState",
    "RelationalActorCriticPolicy",
    "PPOTrainer",
    "EpisodeTrajectory",
    "PPOConfig",
    "TacticalDRLPolicy",
    "DifferentialSharpeReward",
    "project_to_l1_simplex_ball",
]
