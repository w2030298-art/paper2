"""
DDQN: Double Deep Q-Network
解决DQN过估计问题的经典价值型算法

核心特点:
1. 目标解耦 — 选择动作和目标Q值使用不同网络
2. Dueling架构 — 分离状态价值和优势函数
3. 经验回放 — 打破数据相关性
4. 目标网络 — 延迟更新，稳定训练

Reference: Hasselt et al. "Deep Reinforcement Learning with Double Q-Learning" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .game_theory_utils import (
    align_first_dim,
    apply_shapley_reward_scaling,
    discrete_imitation_loss,
    eq_actions_to_discrete_indices,
    inject_game_hints,
)
from .utils.networks import DuelingQNetwork
from src.utils.buffer import ReplayBuffer


class DDQNAgent(BaseAgent):
    """
    DDQN 智能体 — Double Deep Q-Network (离散动作)

    适用于：离散信道选择、频谱分配、离散动作决策任务
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=1_000_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10000,
        update_interval=1,
        device="cuda",
        use_game_theory=True,
        use_shapley_credit=False,
        ctde_with_hints=False,
        warm_start_steps=0,
        warm_start_lr_scale=0.5,
    ):
        """
        Args:
            state_dim: int - 状态空间维度
            action_dim: int - 动作空间维度 (离散)
            hidden_dim: int - 网络隐藏层维度 (默认256)
            lr: float - 学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            tau: float - 软更新系数 (默认0.005)
            batch_size: int - 批次大小 (默认256)
            buffer_size: int - 回放缓冲区容量 (默认1_000_000)
            epsilon_start: float - 初始epsilon (默认1.0)
            epsilon_end: float - 最终epsilon (默认0.05)
            epsilon_decay: int - epsilon衰减步数 (默认10000)
            update_interval: int - 每隔几步更新目标网络 (默认1)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 维度
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q 网络
        self.q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # 缓冲区
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=1,
            device=str(self.device),
        )

        # 训练计数
        self.update_count = 0
        self.update_interval = update_interval
        self.is_on_policy = False
        self.action_type = "discrete"
        self.compatible_env_types = ["discrete"]
        self.required_batch_keys = ["states", "actions", "rewards", "next_states", "dones"]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """
        选择动作 (epsilon-greedy)

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 离散动作，shape: [1]
            info: dict - 包含 'q_value': float (最优Q值)
        """
        if not deterministic and np.random.rand() < self.epsilon:
            action_idx = int(np.random.randint(self.action_dim))
            return np.array(action_idx), {"log_prob": 0.0, "q_value": 0.0, "exploration": True}

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=-1)
            max_q = torch.max(q_values)

        return np.array(action.cpu().item()), {
            "log_prob": 0.0,
            "q_value": float(max_q.cpu().item()),
            "exploration": False,
        }

    def _decay_epsilon(self):
        """衰减 epsilon"""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * min(self.update_count, self.epsilon_decay)
            / self.epsilon_decay,
        )

    def update(self, batch_data):
        """
        使用DDQN更新策略
        数据先存入 replay buffer，再采样更新

        Args:
            batch_data: dict - 批量经验数据，包含:
                - states: np.ndarray [batch, state_dim]
                - actions: np.ndarray [batch] 或 [batch, 1] (离散索引)
                - rewards: np.ndarray [batch]
                - next_states: np.ndarray [batch, state_dim]
                - dones: np.ndarray [batch]

        Returns:
            info: dict - 训练信息
                - loss: float - TD损失
                - policy_loss: float - 同loss (Q-network无单独策略损失)
                - td_error: float - 平均TD误差
                - approx_kl: float - 0 (无策略，KL无意义)
                - q_value_mean: float - 平均Q值
                - epsilon: float - 当前探索率
        """
        # 数据存储
        states = np.asarray(batch_data["states"], dtype=np.float32)
        actions = np.asarray(batch_data["actions"])
        if actions.ndim > 1:
            actions = actions.flatten()
        rewards = np.asarray(batch_data["rewards"], dtype=np.float32)
        next_states = np.asarray(batch_data["next_states"], dtype=np.float32)
        dones = np.asarray(batch_data["dones"], dtype=np.float32)

        if self.use_game_theory:
            if self.use_shapley_credit and "shapley_values" in batch_data:
                rewards_t = apply_shapley_reward_scaling(
                    rewards=torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                    shapley_values=batch_data["shapley_values"],
                    enabled=True,
                )
                rewards = rewards_t.detach().cpu().numpy().astype(np.float32)

            if self.ctde_with_hints:
                states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

                if "game_hints" in batch_data:
                    states_t = inject_game_hints(
                        states=states_t,
                        game_hints=batch_data["game_hints"],
                        enabled=True,
                    )
                    next_states_t = inject_game_hints(
                        states=next_states_t,
                        game_hints=batch_data["game_hints"],
                        enabled=True,
                    )

                if "global_states" in batch_data and batch_data["global_states"] is not None:
                    global_states = torch.as_tensor(
                        batch_data["global_states"], dtype=torch.float32, device=self.device
                    )
                    global_states = global_states.view(global_states.shape[0], -1)
                    global_states = align_first_dim(global_states, states_t.shape[0])
                    global_scalar = global_states.mean(dim=-1, keepdim=True)
                    while global_scalar.ndim < states_t.ndim:
                        global_scalar = global_scalar.unsqueeze(-1)
                    states_t = states_t + 0.01 * global_scalar

                    next_global_raw = batch_data.get("next_global_states", batch_data["global_states"])
                    if next_global_raw is None:
                        next_global_raw = batch_data["global_states"]
                    next_global_states = torch.as_tensor(
                        next_global_raw, dtype=torch.float32, device=self.device
                    )
                    next_global_states = next_global_states.view(next_global_states.shape[0], -1)
                    next_global_states = align_first_dim(next_global_states, next_states_t.shape[0])
                    next_global_scalar = next_global_states.mean(dim=-1, keepdim=True)
                    while next_global_scalar.ndim < next_states_t.ndim:
                        next_global_scalar = next_global_scalar.unsqueeze(-1)
                    next_states_t = next_states_t + 0.01 * next_global_scalar

                states = states_t.detach().cpu().numpy().astype(np.float32)
                next_states = next_states_t.detach().cpu().numpy().astype(np.float32)

        batch_size = states.shape[0]
        for i in range(batch_size):
            self.buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # 采样
        if len(self.buffer) < self.batch_size:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "td_error": 0.0,
                "q_value_mean": 0.0,
                "imitation_loss": 0.0,
                "epsilon": self.epsilon,
            }

        b = self.buffer.sample(self.batch_size)

        states_t = b["states"].to(self.device)
        actions_t = b["actions"].long().to(self.device)
        rewards_t = b["rewards"].to(self.device)
        next_states_t = b["next_states"].to(self.device)
        dones_t = b["dones"].to(self.device)

        # ----- 计算DDQN目标 -----
        with torch.no_grad():
            # 当前网络选择动作
            next_actions = self.q_net(next_states_t).argmax(dim=-1)  # [batch]

            # 目标网络评估
            next_q_values = self.target_q_net(next_states_t)
            next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(-1)

            # Bellman 目标
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        # 当前Q值
        q_values = self.q_net(states_t)
        q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(-1)

        # TD 损失
        loss = F.mse_loss(q, target_q)
        imitation_loss = torch.tensor(0.0, device=self.device)
        if (
            self.use_game_theory
            and self.update_count < self.warm_start_steps
            and "eq_actions" in batch_data
        ):
            eq_indices = eq_actions_to_discrete_indices(
                eq_actions=batch_data["eq_actions"],
                batch_size=states_t.shape[0],
                n_agents=1,
                action_dim=self.action_dim,
                device=self.device,
            )
            if eq_indices is not None:
                if eq_indices.ndim > 1:
                    eq_indices = eq_indices[:, 0]
                imitation_loss = discrete_imitation_loss(q_values, eq_indices)
                loss = loss + self.warm_start_lr_scale * imitation_loss

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        # 软更新目标网络
        self._soft_update()
        self._decay_epsilon()

        self.update_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": loss.item(),
            "entropy": 0.0,  # 无策略网络，熵无意义
            "approx_kl": 0.0,
            "td_error": (q - target_q).abs().mean().item(),
            "q_value_mean": q_values.mean().item(),
            "imitation_loss": float(imitation_loss.detach().item()),
            "epsilon": self.epsilon,
        }

    def _soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def state_dict(self):
        """获取状态字典"""
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.q_net.load_state_dict(state_dict["q_net"])
        self.target_q_net.load_state_dict(state_dict["target_q_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.update_count = state_dict.get("update_count", 0)
        self.epsilon = state_dict.get("epsilon", self.epsilon_start)
