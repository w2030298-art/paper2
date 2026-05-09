"""
GameTheory MEC 环境适配层

将 GameTheoryMECEnv 的 Dict 混合动作空间适配为标准 gym 空间，
供单/多智能体 RL 算法统一使用。

离散适配: action_space = Discrete((K+1) * bins^3)
连续适配: action_space = Box(-1, 1, (4,))

所有适配器保持多智能体接口 (obs -> List[np.ndarray], action -> List[...])。
适配层负责动作接口容错；Mainline-A / GameTheory 环境核心不在这里修改。
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from gymnasium import spaces

from .game_theory_env import GameTheoryMECEnv


# 离散量化 5 档 (固定)
QUANTIZED_BINS = 5
QUANTIZED_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
CONTINUOUS_ACTION_DIM = 4


def _safe_float_vector(action: Any, expected_dim: int, clip: bool = True) -> np.ndarray:
    """Return a finite 1D float32 action vector with deterministic padding/truncation.

    Some algorithms return shapes such as ``(1, 4)``, flattened joint actions,
    logits-like arrays, or NaN/inf values during early exploration.  The adapter
    should fail closed at the algorithm boundary rather than pushing invalid
    actions into the canonical Mainline-A environment.
    """
    try:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        arr = np.asarray([], dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros(expected_dim, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
    if arr.size < expected_dim:
        arr = np.pad(arr, (0, expected_dim - arr.size), mode="constant", constant_values=0.0)
    elif arr.size > expected_dim:
        arr = arr[:expected_dim]
    if clip:
        arr = np.clip(arr, -1.0, 1.0)
    return arr.astype(np.float32, copy=False)


def _sanitize_discrete_action(action: Any, num_edge_servers: int) -> int:
    """Convert scalar/logit-like algorithm output into a legal discrete action index."""
    n_actions = int((num_edge_servers + 1) * (QUANTIZED_BINS ** 3))
    try:
        arr = np.asarray(action)
    except (TypeError, ValueError):
        return 0
    if arr.ndim > 0 and arr.size > 1:
        # Treat non-scalar vectors as logits/probabilities; this protects against
        # accidental policy-head output passed directly to the environment.
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr = np.nan_to_num(arr, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        if not np.isfinite(arr).any():
            return 0
        return int(np.clip(int(np.nanargmax(arr)), 0, n_actions - 1))
    try:
        value = float(arr.reshape(-1)[0]) if arr.size else 0.0
    except (TypeError, ValueError, IndexError):
        value = 0.0
    if not np.isfinite(value):
        value = 0.0
    return int(np.clip(int(round(value)), 0, n_actions - 1))


def _split_discrete_actions(actions: Any) -> List[Any]:
    """Normalize discrete action payload into a list, preserving scalar intent."""
    if isinstance(actions, (int, np.integer, float, np.floating)):
        return [actions]
    if isinstance(actions, np.ndarray):
        if actions.ndim == 0:
            return [actions.item()]
        if actions.ndim == 2 and actions.shape[-1] == 1:
            return actions.reshape(-1).tolist()
        if actions.ndim == 1:
            return actions.tolist()
    try:
        return list(actions)
    except TypeError:
        return [actions]


def _split_continuous_actions(
    actions: Any,
    action_dim: int = CONTINUOUS_ACTION_DIM,
    num_agents: Optional[int] = None,
) -> List[np.ndarray]:
    """Normalize continuous action payload into one action vector per agent.

    Supported inputs include:
    - single-agent vector: ``(4,)`` or ``(1, 4)``;
    - batched multi-agent matrix: ``(num_agents, 4)``;
    - flattened joint vector: ``(num_agents * 4,)``;
    - list/tuple of per-agent vectors.
    """
    try:
        arr = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError):
        try:
            return [_safe_float_vector(a, action_dim) for a in list(actions)]
        except TypeError:
            return [_safe_float_vector(actions, action_dim)]

    if arr.ndim == 0:
        return [_safe_float_vector(arr, action_dim)]
    if arr.ndim == 1:
        flat = arr.reshape(-1)
        if flat.size == action_dim:
            return [_safe_float_vector(flat, action_dim)]
        if num_agents and flat.size == int(num_agents) * action_dim:
            return [_safe_float_vector(row, action_dim) for row in flat.reshape(int(num_agents), action_dim)]
        return [_safe_float_vector(flat, action_dim)]
    if arr.ndim == 2:
        if arr.shape[1] == action_dim:
            return [_safe_float_vector(row, action_dim) for row in arr]
        if arr.shape[0] == action_dim and arr.shape[1] == 1:
            return [_safe_float_vector(arr.reshape(-1), action_dim)]
        flat = arr.reshape(-1)
        if num_agents and flat.size == int(num_agents) * action_dim:
            return [_safe_float_vector(row, action_dim) for row in flat.reshape(int(num_agents), action_dim)]
        return [_safe_float_vector(flat, action_dim)]

    # Higher-rank tensors are flattened and then split if they encode a joint action.
    flat = arr.reshape(-1)
    if num_agents and flat.size == int(num_agents) * action_dim:
        return [_safe_float_vector(row, action_dim) for row in flat.reshape(int(num_agents), action_dim)]
    return [_safe_float_vector(flat, action_dim)]


def _decode_discrete_action(action: int, num_edge_servers: int) -> Dict[str, Any]:
    """将单个离散动作索引解码为 GameTheoryMECEnv 的 Dict 动作。"""
    action = _sanitize_discrete_action(action, num_edge_servers)
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
    ratio = _safe_float_vector(ratio, 3)

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
    action = _safe_float_vector(action, CONTINUOUS_ACTION_DIM)

    # 第 0 维映射到 target: [-1, 1] -> [0, K]
    target_raw = (float(action[0]) + 1.0) * 0.5 * num_edge_servers
    target = int(np.clip(np.round(target_raw), 0, num_edge_servers))
    ratio = np.clip(action[1:4], -1.0, 1.0).astype(np.float32, copy=False)

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
        raw_actions = _split_discrete_actions(actions)
        dict_actions = [
            _decode_discrete_action(a, self.num_edge_servers)
            for a in raw_actions
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
        raw_actions = _split_continuous_actions(
            actions,
            action_dim=CONTINUOUS_ACTION_DIM,
            num_agents=self.num_agents,
        )
        dict_actions = [
            _decode_continuous_action(a, self.num_edge_servers)
            for a in raw_actions
        ]
        return self._env.step(dict_actions)

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)
