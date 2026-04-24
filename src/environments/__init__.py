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

def _register_if_missing(env_id: str, entry_point: str):
    if env_id not in gym.envs.registry:
        gym.register(id=env_id, entry_point=entry_point)


# Gymnasium environment registration (GameTheory only)
_register_if_missing("MEC-v1-game-theory", "src.environments.mec_v3:GameTheoryMECEnv")
_register_if_missing(
    "MEC-v1-game-theory-discrete-ma",
    "src.environments.mec_v3:GameTheoryDiscreteMAEnv",
)
_register_if_missing(
    "MEC-v1-game-theory-continuous-ma",
    "src.environments.mec_v3:GameTheoryContinuousMAEnv",
)
