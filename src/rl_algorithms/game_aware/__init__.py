"""Game-aware constrained MARL support package."""

from .critic_features import GameAwareCriticFeatures, build_critic_features
from .primal_dual import ConstraintResiduals, PrimalDualState, PrimalDualUpdater
from .reward_design import RewardBreakdown, RewardComponent, compute_interpretable_reward

__all__ = [
    "ConstraintResiduals",
    "GameAwareCriticFeatures",
    "PrimalDualState",
    "PrimalDualUpdater",
    "RewardBreakdown",
    "RewardComponent",
    "build_critic_features",
    "compute_interpretable_reward",
]

