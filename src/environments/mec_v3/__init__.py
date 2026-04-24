"""MEC v3 game-theory environments."""

import gymnasium as gym

from .game_theory_env import GameTheoryMECEnv
from .game_theory_adapters import GameTheoryDiscreteMAEnv, GameTheoryContinuousMAEnv


def _register_if_missing(env_id: str, entry_point: str, max_episode_steps: int = 100):
    if env_id not in gym.envs.registry:
        gym.register(
            id=env_id,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
        )


_register_if_missing(
    "MEC-v1-game-theory",
    "src.environments.mec_v3.game_theory_env:GameTheoryMECEnv",
)
_register_if_missing(
    "MEC-v1-game-theory-discrete-ma",
    "src.environments.mec_v3.game_theory_adapters:GameTheoryDiscreteMAEnv",
)
_register_if_missing(
    "MEC-v1-game-theory-continuous-ma",
    "src.environments.mec_v3.game_theory_adapters:GameTheoryContinuousMAEnv",
)

__all__ = [
    "GameTheoryMECEnv",
    "GameTheoryDiscreteMAEnv",
    "GameTheoryContinuousMAEnv",
]
