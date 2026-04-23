"""Custom environments for MEC (Multi-Edge Computing)."""

import gymnasium as gym

from .mec_v3 import (
    GameTheoryMECEnv,
    GameTheoryDiscreteMAEnv,
    GameTheoryContinuousMAEnv,
)

__all__ = [
    "GameTheoryMECEnv",
    "GameTheoryDiscreteMAEnv",
    "GameTheoryContinuousMAEnv",
]

# Gymnasium environment registration (GameTheory only)
gym.register(id="MEC-v1-game-theory", entry_point="src.environments.mec_v3:GameTheoryMECEnv")
gym.register(
    id="MEC-v1-game-theory-discrete-ma",
    entry_point="src.environments.mec_v3:GameTheoryDiscreteMAEnv",
)
gym.register(
    id="MEC-v1-game-theory-continuous-ma",
    entry_point="src.environments.mec_v3:GameTheoryContinuousMAEnv",
)
