"""
PPO: Proximal Policy Optimization
最稳定、最广泛使用的RL算法之一

核心特点:
1. Clipped surrogate objective - 训练稳定
2. 使用GAE(Generalized Advantage Estimation)估计优势
3. Actor-Critic架构，共享特征提取器 (可选)
4. 多epoch更新，高样本效率

Reference: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorNetwork, ActorDiscreteNetwork, CriticNetwork


class PPOAgent(BaseAgent):
    """
    PPO智能体

    适用于: 边缘计算卸载、功率控制、游戏AI控制、移动边缘计算(MEC)任务调度
    """

    is_on_policy = True
    action_type = "continuous"

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.95,
        num_epochs=10,
        clip_grad=0.5,
        value_coeff=0.5,
        ent_coeff=0.01,
        device="cuda",
        discrete=False,
        use_game_theory=True,
        use_shapley_credit=False,
        ctde_with_hints=False,
        warm_start_steps=0,
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
            gae_lambda: float - GAE衰减参数 (默认0.95)
            num_epochs: int - 每次更新迭代次数 (默认10)
            clip_grad: float - 梯度裁剪阈值 (默认0.5)
            value_coeff: float - 价值损失系数 (默认0.5)
            ent_coeff: float - 熵正则系数 (默认0.01)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
            discrete: bool - 是否离散动作空间 (默认False，保持向后兼容)
        """
        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 维度
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        # 超参数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.clip_grad = clip_grad
        self.value_coeff = value_coeff
        self.ent_coeff = ent_coeff

        # Actor (策略网络)
        if self.discrete:
            self.actor = ActorDiscreteNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Critic (价值网络)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

        # 优化器 (共享)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # 训练计数
        self.update_count = 0
        self.compatible_env_types = ["discrete" if discrete else "continuous"]
        self.action_type = "discrete" if discrete else "continuous"
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
            action: np.ndarray - 动作，shape: [action_dim] 或 int (离散)
            info: dict - 包含 'log_prob': float, 'value': float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.sample(state_tensor, deterministic)
            value = self.critic(state_tensor)

        if self.discrete:
            return np.array([action.cpu().item()], dtype=np.int64), {
                "log_prob": log_prob.cpu().item(),
                "value": value.cpu().item(),
            }
        return action.cpu().numpy()[0], {
            "log_prob": log_prob.cpu().item(),
            "value": value.cpu().item(),
        }

    def _compute_gae(self, rewards, values, dones):
        """
        计算GAE优势估计和回报

        Args:
            rewards: torch.Tensor [batch]
            values: torch.Tensor [batch, 1]
            dones: torch.Tensor [batch]

        Returns:
            advantages: torch.Tensor [batch, 1]
            returns: torch.Tensor [batch]
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # 简化: 不做bootstrap
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def update(self, batch_data):
        """
        使用PPO更新策略

        Args:
            batch_data: dict - 批量经验数据，包含:
                - states: np.ndarray [batch, state_dim]
                - actions: np.ndarray [batch, action_dim]
                - rewards: np.ndarray [batch]
                - next_states: np.ndarray [batch, state_dim]
                - dones: np.ndarray [batch]
                - log_probs: np.ndarray [batch] - 旧策略对数概率
                - values: np.ndarray [batch, 1] - 旧状态价值 (可选)

        Returns:
            info: dict - 训练信息
                - loss: float - 总损失
                - policy_loss: float - 策略损失
                - value_loss: float - 价值损失
                - entropy: float - 策略熵
                - approx_kl: float - 近似KL散度
                - clip_fraction: float - 被裁剪的比例
        """
        # 数据转移到 GPU
        states = torch.FloatTensor(batch_data["states"]).to(self.device)
        actions = torch.FloatTensor(batch_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch_data["rewards"]).to(self.device)
        dones = torch.FloatTensor(batch_data["dones"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data["log_probs"]).to(self.device)

        if self.use_shapley_credit and "shapley_values" in batch_data:
            shapley = torch.FloatTensor(batch_data["shapley_values"]).to(self.device)
            if shapley.ndim > 1:
                shapley = shapley.mean(dim=-1)
            shapley = shapley.view(-1)
            if shapley.shape[0] != rewards.shape[0]:
                repeat_factor = max(1, rewards.shape[0] // max(shapley.shape[0], 1))
                shapley = shapley.repeat_interleave(repeat_factor)[: rewards.shape[0]]
            rewards = rewards * (0.5 + torch.clamp(shapley, 0.0, 1.0))

        if self.ctde_with_hints and "game_hints" in batch_data:
            hints = torch.FloatTensor(batch_data["game_hints"]).to(self.device)
            if hints.ndim == 1:
                hints = hints.unsqueeze(0)
            hint_scalar = hints.mean(dim=-1, keepdim=True)
            if hint_scalar.shape[0] != states.shape[0]:
                repeat_factor = max(1, states.shape[0] // max(hint_scalar.shape[0], 1))
                hint_scalar = hint_scalar.repeat_interleave(repeat_factor, dim=0)[: states.shape[0]]
            states = states + 0.01 * hint_scalar

        # 如果有旧价值，用于GAE；否则用当前critic估计
        if batch_data.get("values") is not None:
            old_values = torch.FloatTensor(batch_data["values"]).to(self.device).detach()
        else:
            old_values = self.critic(states).squeeze(-1).detach()

        values = old_values

        # 计算GAE优势
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = advantages.unsqueeze(-1)
        returns = returns

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_frac = 0
        total_imitation = 0

        for epoch in range(self.num_epochs):
            # 重新评估
            dist = self.actor.get_distribution(states)
            if self.discrete:
                new_log_probs = dist.log_prob(actions.squeeze(-1).long()).unsqueeze(-1)
                entropy = dist.entropy()
            else:
                new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1)
            new_values = self.critic(states).squeeze(-1)

            # 概率比率
            ratio = torch.exp(new_log_probs - old_log_probs.unsqueeze(-1))

            # 裁剪策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 裁剪比例统计
            clip_frac = ((ratio - 1).abs() > self.eps_clip).float().mean()

            # 价值损失
            value_loss = F.mse_loss(new_values, returns)

            # 总损失
            loss = policy_loss + self.value_coeff * value_loss - self.ent_coeff * entropy.mean()
            imitation_loss = torch.tensor(0.0, device=self.device)
            if (
                not self.discrete
                and self.update_count < self.warm_start_steps
                and "eq_actions" in batch_data
            ):
                eq_actions = torch.FloatTensor(batch_data["eq_actions"]).to(self.device)
                if eq_actions.ndim == 1:
                    eq_actions = eq_actions.unsqueeze(0)
                if eq_actions.ndim > 2:
                    eq_actions = eq_actions.view(eq_actions.shape[0], -1)
                if eq_actions.shape[-1] != self.action_dim:
                    if eq_actions.shape[-1] > self.action_dim:
                        eq_actions = eq_actions[:, -self.action_dim :]
                    else:
                        pad = torch.zeros(
                            eq_actions.shape[0],
                            self.action_dim - eq_actions.shape[-1],
                            device=self.device,
                        )
                        eq_actions = torch.cat([eq_actions, pad], dim=-1)
                if eq_actions.shape[0] != states.shape[0]:
                    repeat_factor = max(1, states.shape[0] // max(eq_actions.shape[0], 1))
                    eq_actions = eq_actions.repeat_interleave(repeat_factor, dim=0)[: states.shape[0]]
                mean, _ = self.actor(states)
                imitation_loss = F.mse_loss(mean, eq_actions)
                loss = loss + self.warm_start_lr_scale * imitation_loss

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.clip_grad
            )
            self.optimizer.step()

            # 统计
            approx_kl = (old_log_probs - new_log_probs.squeeze(-1)).mean().item()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_kl += approx_kl
            total_clip_frac += clip_frac.item()
            total_imitation += float(imitation_loss.detach().item())

        self.update_count += 1

        num_updates = self.num_epochs
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "approx_kl": total_kl / num_updates,
            "clip_fraction": total_clip_frac / num_updates,
            "imitation_loss": total_imitation / num_updates,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
        }

    def state_dict(self):
        """获取状态字典"""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.update_count = state_dict.get("update_count", 0)
