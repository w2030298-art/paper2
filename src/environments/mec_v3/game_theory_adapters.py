"""
GameTheory MEC 环境适配层

将 GameTheoryMECEnv 的 Dict 混合动作空间适配为标准 gym 空间，
供单/多智能体 RL 算法统一使用。

离散适配: action_space = Discrete((K+1) * bins^3)
连续适配: action_space = Box(-1, 1, (4,))

所有适配器保持多智能体接口 (obs -> List[np.ndarray], action -> List[...])。
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from gymnasium import spaces

from .game_theory_env import GameTheoryMECEnv


# 离散量化 5 档 (固定)
QUANTIZED_BINS = 5
QUANTIZED_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)


def _decode_discrete_action(action: int, num_edge_servers: int) -> Dict[str, Any]:
    """将单个离散动作索引解码为 GameTheoryMECEnv 的 Dict 动作。"""
    n_targets = num_edge_servers + 1
    n_ratio_combos = QUANTIZED_BINS ** 3

    target = action // n_ratio_combos
    ratio_idx = action % n_ratio_combos

    # 将 ratio_idx 解码为 3 个 bin 索引
    r0 = ratio_idx // (QUANTIZED_BINS * QUANTIZED_BINS)
    r1 = (ratio_idx // QUANTIZED_BINS) % QUANTIZED_BINS
    r2 = ratio_idx % QUANTIZED_BINS

    ratio = np.array([
        QUANTIZED_VALUES[r0],
        QUANTIZED_VALUES[r1],
        QUANTIZED_VALUES[r2],
    ], dtype=np.float32)

    return {"target": int(np.clip(target, 0, n_targets - 1)), "ratio": ratio}


def _encode_discrete_action(target: int, ratio: np.ndarray, num_edge_servers: int) -> int:
    """将 GameTheoryMECEnv 的 Dict 动作编码为单个离散动作索引。"""
    n_targets = num_edge_servers + 1
    target = int(np.clip(target, 0, n_targets - 1))

    # 找到每个 ratio 维最近的量化值索引
    def nearest_bin(val):
        val = float(np.clip(val, -1.0, 1.0))
        idx = int(np.argmin(np.abs(QUANTIZED_VALUES - val)))
        return idx

    r0 = nearest_bin(ratio[0])
    r1 = nearest_bin(ratio[1])
    r2 = nearest_bin(ratio[2])
    ratio_idx = r0 * QUANTIZED_BINS * QUANTIZED_BINS + r1 * QUANTIZED_BINS + r2
    return target * (QUANTIZED_BINS ** 3) + ratio_idx


def _decode_continuous_action(action: np.ndarray, num_edge_servers: int) -> Dict[str, Any]:
    """将 4D 连续动作向量解码为 GameTheoryMECEnv 的 Dict 动作。"""
    action = np.asarray(action, dtype=np.float32).flatten()
    if action.shape[0] != 4:
        raise ValueError(f"Continuous action must be shape (4,), got {action.shape}")

    # 第 0 维映射到 target: [-1, 1] -> [0, K]
    target_raw = (action[0] + 1.0) * 0.5 * num_edge_servers
    target = int(np.clip(np.round(target_raw), 0, num_edge_servers))
    ratio = np.clip(action[1:4], -1.0, 1.0)

    return {"target": target, "ratio": ratio}


class GameTheoryDiscreteMAEnv:
    """
    GameTheoryMECEnv 离散动作适配器 (多智能体)

    每个智能体的动作空间: Discrete((K+1) * bins^3)
    观测空间与底层环境一致
    """

    def __init__(self, **kwargs):
        self._env = GameTheoryMECEnv(**kwargs)
        self.num_agents = self._env.num_agents
        self.num_edge_servers = self._env._num_edge_servers
        self.max_steps = self._env.max_steps

        n_actions = (self.num_edge_servers + 1) * (QUANTIZED_BINS ** 3)
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = self._env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):
        """
        Args:
            actions: List[int] or int, 每个智能体的离散动作索引
                      单智能体模式下可传入 int，自动包装为 list
        """
        if isinstance(actions, (int, np.integer)):
            actions = [actions]
        elif isinstance(actions, np.ndarray) and actions.ndim == 0:
            actions = [int(actions)]
        dict_actions = [
            _decode_discrete_action(int(a), self.num_edge_servers)
            for a in actions
        ]
        return self._env.step(dict_actions)

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        # 透传底层环境的其他属性 (如 queue_lengths, game_history 等)
        return getattr(self._env, name)


class GameTheoryContinuousMAEnv:
    """
    GameTheoryMECEnv 连续动作适配器 (多智能体)

    每个智能体的动作空间: Box(-1, 1, (4,))
    观测空间与底层环境一致
    """

    def __init__(self, **kwargs):
        self._env = GameTheoryMECEnv(**kwargs)
        self.num_agents = self._env.num_agents
        self.num_edge_servers = self._env._num_edge_servers
        self.max_steps = self._env.max_steps

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = self._env.observation_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):
        """
        Args:
            actions: List[np.ndarray] or np.ndarray, 每个智能体的 4D 连续动作向量
                      单智能体模式下可传入 ndarray，自动包装为 list
        """
        if isinstance(actions, np.ndarray):
            actions = [actions]
        dict_actions = []
        for a in actions:
            a = np.asarray(a, dtype=np.float32)
            dict_actions.append(_decode_continuous_action(a, self.num_edge_servers))
        return self._env.step(dict_actions)

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)
