"""
Base Agent Interface for RL Algorithms
All algorithms must inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set
import torch
import numpy as np
import numpy.typing as npt


@dataclass
class ActionResult:
    """Standardized return type for select_action."""

    action: npt.NDArray[np.float32]
    log_prob: float = 0.0
    value: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class UpdateResult:
    """Standardized return type for update."""

    loss: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    extra: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """强化学习算法基类"""

    @abstractmethod
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cuda",
        **kwargs,
    ):
        """
        初始化智能体

        Args:
            state_dim: int - 状态空间维度
            action_dim: int - 动作空间维度
            hidden_dim: int - 网络隐藏层维度
            lr: float - 学习率
            gamma: float - 折扣因子
            device: str - 'cuda' 或 'cpu'
            **kwargs: 算法特有超参数
        """
        pass

    @abstractmethod
    def select_action(self, state: npt.NDArray[np.float32], deterministic: bool = False):
        """
        选择动作

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim] 或 [batch, state_dim]
            deterministic: bool - 是否确定性选择

        Returns:
            action: np.ndarray - 动作
            info: dict - 附加信息，包含 'log_prob' (策略类) 或 'value' (可选)
        """
        pass

    @abstractmethod
    def update(self, batch_data: dict) -> dict:
        """
        更新策略

        Args:
            batch_data: dict - 批量经验数据，包含:
                - states: np.ndarray [batch, state_dim]
                - actions: np.ndarray [batch, action_dim] 或 [batch] (离散)
                - rewards: np.ndarray [batch]
                - next_states: np.ndarray [batch, state_dim]
                - dones: np.ndarray [batch]
                - (可选) log_probs: np.ndarray [batch] - 旧策略对数概率
                - (可选) values: np.ndarray [batch] - 旧状态价值

        Returns:
            info: dict - 训练信息，必须包含 'loss': float
        """
        pass

    # Default attributes that subclasses can override
    required_batch_keys: Set[str] = {"states", "actions", "rewards", "next_states", "dones"}
    is_on_policy: bool = True
    action_type: str = "continuous"
    compatible_env_types: List[str] = ["continuous"]
    multi_agent_mode: str = "shared"  # "shared" | "joint"

    def save(self, path: str):
        """保存模型"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """加载模型"""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))

    def state_dict(self):
        """获取状态字典"""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        raise NotImplementedError
