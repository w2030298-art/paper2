"""
COMA: Counterfactual Multi-Agent Policy Gradients
反事实多智能体策略梯度

核心特点:
1. 集中critic — 输入全局状态和所有agent动作
2. 反事实基线 — 计算每个agent的边际贡献
3. 参数共享Actor — 所有agent共用策略网络
4. 信用分配 — 解决多智能体信用分配问题

Reference: Foerster et al. "Counterfactual Multi-Agent Policy Gradients" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorDiscreteNetwork, ActorNetwork
from .game_theory_utils import (
    apply_shapley_reward_scaling,
    discrete_imitation_loss,
    eq_actions_to_discrete_indices,
    inject_game_hints,
    prepare_eq_actions_multi_continuous,
)


class COMACritic(nn.Module):
    """集中 Critic — 输入全局状态和所有 agent 的动作"""

    def __init__(self, global_state_dim, n_agents, action_dim, hidden_dim=256):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        input_dim = global_state_dim + n_agents * action_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, n_agents * action_dim)

    def forward(self, global_state, all_actions):
        """
        Args:
            global_state: [batch, global_state_dim]
            all_actions: [batch, n_agents] (离散动作的索引)
        Returns:
            q_values: [batch, n_agents, action_dim]
        """
        # one-hot 编码动作
        bs = global_state.shape[0]
        actions_onehot = F.one_hot(all_actions.long(), self.action_dim).float()
        actions_flat = actions_onehot.view(bs, -1)
        x = torch.cat([global_state, actions_flat], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q.view(bs, self.n_agents, self.action_dim)


class COMAAgent(BaseAgent):
    """
    COMA 智能体 — 反事实多智能体策略梯度

    适用于: 合作多智能体任务、多用户MEC卸载
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        num_agents=3,
        num_epochs=10,
        discrete=False,
        device="cuda",
        use_game_theory=True,
        use_shapley_credit=True,
        ctde_with_hints=True,
        warm_start_steps=1000,
        warm_start_lr_scale=0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.discrete = discrete
        self.gamma = gamma
        self.num_epochs = num_epochs

        self.global_state_dim = state_dim * num_agents

        if discrete:
            self.actor = ActorDiscreteNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.critic = COMACritic(self.global_state_dim, num_agents, action_dim, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.update_count = 0
        self.is_on_policy = True
        self.action_type = "discrete" if discrete else "continuous"
        self.compatible_env_types = ["multi_agent"]
        self.multi_agent_mode = "joint"
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """选择单个智能体的动作 (分散执行)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.sample(state_tensor, deterministic)
        return action.cpu().numpy()[0], {"log_prob": log_prob.cpu().item()}

    def update(self, batch_data):
        """COMA 更新 (集中训练)"""
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

        if batch_data.get("global_states") is not None:
            global_states = torch.FloatTensor(batch_data["global_states"]).to(self.device)
        else:
            bs = states.shape[0]
            global_states = states.view(bs, -1)
        if self.ctde_with_hints and "game_hints" in batch_data:
            global_states = inject_game_hints(
                states=global_states,
                game_hints=batch_data["game_hints"],
                enabled=True,
            )

        # 如果 rewards/dones 是 per-agent 的，取均值/any
        if rewards.ndim > 1 and rewards.shape[-1] == self.num_agents:
            rewards = rewards.mean(dim=-1)
        if dones.ndim > 1 and dones.shape[-1] == self.num_agents:
            dones = dones.any(dim=-1).float()

        # 扩展 rewards/dones 到 per-agent (COMA critic 输出 per-agent Q)
        rewards_expanded = rewards.unsqueeze(-1).expand(-1, self.num_agents)
        dones_expanded = dones.unsqueeze(-1).expand(-1, self.num_agents)

        # 计算优势 (简化版 GAE)
        with torch.no_grad():
            q_values = self.critic(global_states, actions.long())
            chosen_q = q_values.gather(2, actions.long().unsqueeze(2)).squeeze(2)
            advantages = rewards_expanded - chosen_q.mean(dim=-1, keepdim=True)
            if advantages.numel() > 1 and advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_imitation = 0

        for _ in range(max(1, self.num_epochs)):
            dist = self.actor.get_distribution(states.view(-1, self.state_dim))
            if self.discrete:
                new_log_probs = dist.log_prob(actions.view(-1).long())
            else:
                new_log_probs = dist.log_prob(actions.view(-1, self.action_dim)).sum(dim=-1)
            new_log_probs = new_log_probs.view(-1, self.num_agents)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic 损失
            q_values = self.critic(global_states, actions.long())
            chosen_q = q_values.gather(2, actions.long().unsqueeze(2)).squeeze(2)
            value_loss = F.mse_loss(chosen_q, rewards_expanded)

            loss = policy_loss + 0.5 * value_loss
            imitation_loss = torch.tensor(0.0, device=self.device)
            if self.update_count < self.warm_start_steps and "eq_actions" in batch_data:
                if self.discrete:
                    eq_idx = eq_actions_to_discrete_indices(
                        eq_actions=batch_data["eq_actions"],
                        batch_size=states.shape[0],
                        n_agents=self.num_agents,
                        action_dim=self.action_dim,
                        device=self.device,
                    )
                    if eq_idx is not None:
                        logits = self.actor(states.view(-1, self.state_dim))
                        imitation_loss = discrete_imitation_loss(logits, eq_idx)
                        loss = loss + self.warm_start_lr_scale * imitation_loss
                else:
                    eq_actions = prepare_eq_actions_multi_continuous(
                        eq_actions=batch_data["eq_actions"],
                        batch_size=states.shape[0],
                        n_agents=self.num_agents,
                        action_dim=self.action_dim,
                        device=self.device,
                    )
                    if eq_actions is not None:
                        mean, _ = self.actor(states.view(-1, self.state_dim))
                        mean = mean.view(-1, self.num_agents, self.action_dim)
                        imitation_loss = F.mse_loss(mean, eq_actions)
                        loss = loss + self.warm_start_lr_scale * imitation_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_imitation += float(imitation_loss.detach().item())

        self.update_count += 1
        n = max(1, self.num_epochs)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "imitation_loss": total_imitation / n,
            "reward_mean": rewards.mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.update_count = sd.get("update_count", 0)
