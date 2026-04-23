"""
MEC V3 — GameTheory environment only.

1. GameTheoryMECEnv: 多基站协作博弈 + MADRL
   - Stackelberg博弈 + Shapley值
   - 动作空间: Dict({"target": Discrete, "ratio": Box})

2. GameTheoryDiscreteMAEnv: 离散动作适配器
3. GameTheoryContinuousMAEnv: 连续动作适配器
"""

import gymnasium as gym

from .game_theory_env import GameTheoryMECEnv
from .game_theory_adapters import GameTheoryDiscreteMAEnv, GameTheoryContinuousMAEnv

# Gymnasium 注册
gym.register(
    id="MEC-v1-game-theory",
    entry_point="src.environments.mec_v3.game_theory_env:GameTheoryMECEnv",
    max_episode_steps=100,
)

gym.register(
    id="MEC-v1-game-theory-discrete-ma",
    entry_point="src.environments.mec_v3.game_theory_adapters:GameTheoryDiscreteMAEnv",
    max_episode_steps=100,
)

gym.register(
    id="MEC-v1-game-theory-continuous-ma",
    entry_point="src.environments.mec_v3.game_theory_adapters:GameTheoryContinuousMAEnv",
    max_episode_steps=100,
)

__all__ = [
    "GameTheoryMECEnv",
    "GameTheoryDiscreteMAEnv",
    "GameTheoryContinuousMAEnv",
]
