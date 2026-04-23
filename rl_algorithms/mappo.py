"""
MAPPO: Multi-Agent Proximal Policy Optimization
集中训练分散执行的多智能体 PPO

核心特点:
1. 集中训练 — Critic 使用全局状态信息
2. 分散执行 — Actor 仅使用局部观测做决策
3. 参数共享 — 所有agent使用同一套网络参数
4. PPO裁剪 — 训练稳定，继承PPO优点

Reference: Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorNetwork, ActorDiscreteNetwork, CriticNetwork
from .game_theory_utils import (
    apply_shapley_reward_scaling,
    discrete_imitation_loss,
    eq_actions_to_discrete_indices,
    inject_game_hints,
    prepare_eq_actions_continuous,
)


class MAPPOAgent(BaseAgent):
    """
    MAPPO 智能体 — 集中训练分散执行的多智能体 PPO

    适用于：多智能体资源分配、多用户MEC卸载、合作博弈等
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        gae_lambda=0.95,
        num_agents=1,
        num_epochs=10,
        discrete=False,
        device="cuda",
        use_game_theory=True,
        use_shapley_credit=True,
        ctde_with_hints=True,
        warm_start_steps=1000,
        warm_start_lr_scale=0.5,
    ):
        """
        Args:
            state_dim: int - 单个智能体的状态空间维度
            action_dim: int - 单个智能体的动作空间维度
            hidden_dim: int - 网络隐藏层维度 (默认256)
            lr: float - 学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            eps_clip: float - PPO裁剪参数 (默认0.2)
            gae_lambda: float - GAE参数 (默认0.95)
            num_agents: int - 智能体数量 (默认1)
            num_epochs: int - 每次更新迭代次数 (默认10)
            discrete: bool - 是否离散动作空间 (默认False)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.discrete = discrete

        # 全局状态维度 (所有智能体的状态拼接)
        self.global_state_dim = state_dim * num_agents

        # 超参数
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs

        # Actor (分散 — 每个智能体用局部观测)
        if self.discrete:
            self.actor = ActorDiscreteNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Critic (集中 — 使用全局状态)
        self.critic = CriticNetwork(self.global_state_dim, hidden_dim).to(self.device)

        # 优化器
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.update_count = 0
        self.is_on_policy = True
        self.action_type = "discrete" if self.discrete else "continuous"
        self.compatible_env_types = ["multi_agent"]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """
        选择单个智能体的动作 (分散执行)

        Args:
            state: np.ndarray - 单个智能体的局部状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 动作，shape: [action_dim]
            info: dict - 包含 'log_prob': float, 'value': float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.sample(state_tensor, deterministic)

        return action.cpu().numpy()[0], {
            "log_prob": log_prob.cpu().item(),
        }

    def update(self, batch_data):
        """
        使用 MAPPO 更新共享参数 (集中更新)

        Args:
            batch_data: dict - 批量经验数据，包含:
                - states: np.ndarray [batch, state_dim] - 单个智能体的状态
                - actions: np.ndarray [batch, action_dim] - 单个智能体的动作
                - rewards: np.ndarray [batch] - 全局奖励
                - next_states: np.ndarray [batch, state_dim] - (可选)
                - dones: np.ndarray [batch]
                - log_probs: np.ndarray [batch] - 旧策略对数概率
                - values: np.ndarray [batch] (可选) — 旧状态价值
                - global_states: np.ndarray [batch, global_state_dim] — 全局状态 (可选)

        Returns:
            info: dict - 训练信息
                - loss: float - 总损失
                - policy_loss: float - 策略损失
                - value_loss: float - 价值损失
                - entropy: float - 策略熵
                - approx_kl: float - 近似KL散度
        """
        states = torch.FloatTensor(batch_data["states"]).to(self.device)
        actions = torch.FloatTensor(batch_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch_data["rewards"]).to(self.device)
        dones = torch.FloatTensor(batch_data["dones"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data["log_probs"]).to(self.device)
        if self.use_shapley_credit and "shapley_values" in batch_data:
            rewards = apply_shapley_reward_scaling(
                rewards=rewards,
                shapley_values=batch_data["shapley_values"],
                enabled=True,
            )

        # 获取全局状态 (如果有) 或构造全局状态
        if batch_data.get("global_states") is not None:
            global_states = torch.FloatTensor(batch_data["global_states"]).to(self.device)
        else:
            # 自动构造: 将每 num_agents 步的局部状态拼接为全局状态
            if self.num_agents > 1 and states.shape[0] % self.num_agents == 0:
                batch_per_agent = states.shape[0] // self.num_agents
                # 重组: [agent0_step0, agent1_step0, agent0_step1, ...] -> [step0_global, step1_global, ...]
                states_reshaped = states.view(self.num_agents, batch_per_agent, -1)
                global_states = states_reshaped.permute(1, 0, 2).reshape(batch_per_agent, -1)
            else:
                # 单智能体回退: 直接使用局部状态
                global_states = states

        # 旧价值 (detach 避免 backward 穿过旧图)
        if self.ctde_with_hints and "game_hints" in batch_data:
            global_states = inject_game_hints(
                states=global_states,
                game_hints=batch_data["game_hints"],
                enabled=True,
            )

        if batch_data.get("values") is not None:
            values = torch.FloatTensor(batch_data["values"]).to(self.device).squeeze(-1).detach()
        else:
            values = self.critic(global_states).squeeze(-1).detach()

        # 对齐维度: 如果 values 比 rewards 短 (全局 vs 逐智能体), 扩展 values
        if values.shape[0] != rewards.shape[0] and self.num_agents > 1:
            # values: [n_steps] -> [n_steps * n_agents] (每个时间步的值复制给所有智能体)
            values = values.repeat_interleave(self.num_agents)

        # 计算GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        # 标准化优势
        if advantages.numel() > 1 and advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_imitation = 0

        for epoch in range(max(1, self.num_epochs)):
            # 当前策略
            dist = self.actor.get_distribution(states)
            if self.discrete:
                new_log_probs = dist.log_prob(actions.squeeze(-1).long())
            else:
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # 比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # 裁剪
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失: 在全局状态层面计算
            new_values = self.critic(global_states).squeeze(-1)
            # 对齐: 如果 returns 比 new_values 长, 将 returns 还原为全局维度
            if returns.shape[0] != new_values.shape[0] and self.num_agents > 1:
                # 逐智能体的 returns 还原为逐时间步 (取均值)
                returns_global = returns.view(self.num_agents, -1).mean(dim=0)
                value_loss = F.mse_loss(new_values, returns_global)
            else:
                value_loss = F.mse_loss(new_values, returns)

            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            imitation_loss = torch.tensor(0.0, device=self.device)
            if self.update_count < self.warm_start_steps and "eq_actions" in batch_data:
                if self.discrete:
                    eq_idx = eq_actions_to_discrete_indices(
                        eq_actions=batch_data["eq_actions"],
                        batch_size=states.shape[0],
                        n_agents=1,
                        action_dim=self.action_dim,
                        device=self.device,
                    )
                    if eq_idx is not None:
                        logits = self.actor(states)
                        imitation_loss = discrete_imitation_loss(logits, eq_idx.view(-1))
                        loss = loss + self.warm_start_lr_scale * imitation_loss
                else:
                    eq_actions = prepare_eq_actions_continuous(
                        eq_actions=batch_data["eq_actions"],
                        batch_size=states.shape[0],
                        action_dim=self.action_dim,
                        device=self.device,
                    )
                    if eq_actions is not None:
                        mean, _ = self.actor(states)
                        imitation_loss = F.mse_loss(mean, eq_actions)
                        loss = loss + self.warm_start_lr_scale * imitation_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
            )
            self.optimizer.step()

            approx_kl = (old_log_probs - new_log_probs).mean().item()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += approx_kl
            total_imitation += float(imitation_loss.detach().item())

        self.update_count += 1
        n = max(1, self.num_epochs)

        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
            "imitation_loss": total_imitation / n,
            "reward_mean": rewards.mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "global_state_dim": self.global_state_dim,
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.update_count = sd.get("update_count", 0)
