"""
GRPO: Group Relative Policy Optimization
DeepSeek-R1 使用的核心算法

核心特点:
1. 无需Critic网络，降低内存开销
2. 组内相对奖励，减少奖励估计方差
3. 基于PPO的裁剪策略，训练稳定

Reference: DeepSeek-R1 paper (2024)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorNetwork
from src.utils.buffer import RolloutBuffer
from .game_theory_utils import (
    align_first_dim,
    apply_shapley_credit_assignment,
    inject_game_hints,
    prepare_eq_actions_continuous,
)


class GRPOAgent(BaseAgent):
    """
    GRPO智能体 - 组相对策略优化

    适用于: 多用户资源分配、边缘计算任务调度等通信场景
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        group_size=64,
        num_epochs=10,
        device="cuda",
        use_game_theory=True,
        use_shapley_credit=True,
        ctde_with_hints=True,
        warm_start_steps=1000,
        warm_start_lr_scale=0.5,
    ):
        """
        Args:
            state_dim: int - 状态空间维度
            action_dim: int - 动作空间维度
            hidden_dim: int - 网络隐藏层维度 (默认256)
            lr: float - 学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            eps_clip: float - PPO裁剪参数 (默认0.2)
            group_size: int - GRPO组大小 (默认64)
            num_epochs: int - 每次更新迭代次数 (默认10)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 维度
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 超参数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.group_size = group_size
        self.num_epochs = num_epochs

        # 策略网络 (GRPO只有Actor，没有Critic)
        self.policy = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 缓冲区
        self.buffer = RolloutBuffer(
            capacity=10000,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )

        # 训练计数
        self.update_count = 0
        self.is_on_policy = True
        self.action_type = "continuous"
        self.compatible_env_types = ["continuous"]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """
        选择动作

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 动作，shape: [action_dim]
            info: dict - 包含 'log_prob': float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.sample(state_tensor, deterministic)

        return action.cpu().numpy()[0], {"log_prob": log_prob.cpu().item()}

    def update(self, batch_data):
        """
        使用GRPO更新策略

        Args:
            batch_data: dict - 批量经验数据，包含:
                - states: np.ndarray [batch, state_dim]
                - actions: np.ndarray [batch, action_dim]
                - rewards: np.ndarray [batch]
                - next_states: np.ndarray [batch, state_dim]
                - dones: np.ndarray [batch]
                - log_probs: np.ndarray [batch] - 旧策略对数概率

        Returns:
            info: dict - 训练信息
                - loss: float - 总损失
                - policy_loss: float - 策略损失
                - entropy: float - 策略熵
                - approx_kl: float - 近似KL散度
        """
        # 数据转移到GPU
        states = torch.FloatTensor(batch_data["states"]).to(self.device)
        actions = torch.FloatTensor(batch_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch_data["rewards"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data["log_probs"]).to(self.device)
        if states.ndim == 3 and states.shape[1] == 1:
            states = states.squeeze(1)
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        rewards = rewards.view(-1)
        old_log_probs = old_log_probs.view(-1)

        if self.use_shapley_credit and "shapley_values" in batch_data:
            rewards = apply_shapley_credit_assignment(
                team_reward=rewards.sum(),
                shapley_values=batch_data["shapley_values"],
            ).view(-1)
            if rewards.shape[0] != old_log_probs.shape[0]:
                rewards = align_first_dim(rewards.view(-1, 1), old_log_probs.shape[0]).view(-1)
        if self.ctde_with_hints and "game_hints" in batch_data:
            states = inject_game_hints(
                states=states,
                game_hints=batch_data["game_hints"],
                enabled=True,
            )

        # 计算组内相对奖励 (GRPO核心)
        # 支持两种分组方式：
        # 1. 如果batch包含group_ids，按group_ids分组
        # 2. 否则按顺序每group_size个样本为一组
        batch_size = states.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        if "group_ids" in batch_data and batch_data["group_ids"] is not None:
            group_ids = torch.LongTensor(batch_data["group_ids"]).to(self.device)
            unique_groups = group_ids.unique()
            for gid in unique_groups:
                mask = group_ids == gid
                group_rewards = rewards[mask]
                group_mean = group_rewards.mean()
                group_std = group_rewards.std() + 1e-8
                advantages[mask] = (group_rewards - group_mean) / group_std
                returns[mask] = group_rewards
        else:
            n = len(rewards)
            for start in range(0, n, self.group_size):
                end = min(start + self.group_size, n)
                if end - start < 2:
                    advantages[start:end] = 0.0
                    continue
                group = rewards[start:end]
                advantages[start:end] = (group - group.mean()) / (group.std() + 1e-8)
            returns = rewards.clone()

        # 多轮更新
        total_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        total_kl = 0
        total_imitation = 0

        for epoch in range(self.num_epochs):
            # 重新计算log_prob和熵
            dist = self.policy.get_distribution(states)
            new_log_probs = dist.log_prob(actions).sum(dim=-1).view(-1)
            entropy = dist.entropy().sum(dim=-1).mean().item()

            # 概率比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 裁剪目标
            adv_flat = advantages.view(-1)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # 总损失 (GRPO不使用价值损失)
            loss = policy_loss - 0.01 * entropy
            imitation_loss = torch.tensor(0.0, device=self.device)
            if self.update_count < self.warm_start_steps and "eq_actions" in batch_data:
                eq_actions = prepare_eq_actions_continuous(
                    eq_actions=batch_data["eq_actions"],
                    batch_size=states.shape[0],
                    action_dim=self.action_dim,
                    device=self.device,
                )
                if eq_actions is not None:
                    mean, _ = self.policy(states)
                    imitation_loss = F.mse_loss(mean, eq_actions)
                    loss = loss + self.warm_start_lr_scale * imitation_loss

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # 统计
            with torch.no_grad():
                kl = (old_log_probs - new_log_probs).mean().item()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_entropy += entropy
            total_kl += kl
            total_imitation += float(imitation_loss.detach().item())

        self.update_count += 1

        # 返回平均统计
        num_updates = self.num_epochs
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_kl / num_updates,
            "imitation_loss": total_imitation / num_updates,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
        }

    def state_dict(self):
        """获取状态字典"""
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.policy.load_state_dict(state_dict["policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.update_count = state_dict.get("update_count", 0)
