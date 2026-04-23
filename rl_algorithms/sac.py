"""
SAC (Soft Actor-Critic) — 连续动作空间算法

适用于: MEC-v0-continuous
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional

from .base_agent import BaseAgent
from src.utils.buffer import ReplayBuffer


class SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            action = torch.tanh(mean)
            # Deterministic evaluation path does not need stochastic log_prob.
            log_prob = torch.zeros((state.shape[0], 1), device=state.device)
            return action, log_prob
        std = log_std.exp()
        noise = torch.randn_like(std)
        action = torch.tanh(mean + std * noise)
        log_prob = self._log_prob(mean, std, noise, action)
        return action, log_prob

    @staticmethod
    def _log_prob(mean, std, noise, action):
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(noise)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class SACAgent(BaseAgent):
    """
    Soft Actor-Critic 智能体

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        gamma: 折扣因子 (默认0.99)
        tau: 目标网络软更新系数 (默认0.005)
        lr: 学习率 (默认3e-4)
        hidden_dim: 隐藏层维度 (默认256)
        batch_size: 批大小 (默认256)
        buffer_size: 回放缓冲区大小 (默认1_000_000)
        device: 'cuda' 或 'cpu' (默认'cuda')
    """

    is_on_policy = False
    action_type = "continuous"
    compatible_env_types = ["continuous"]
    required_batch_keys = ["states", "actions", "rewards", "next_states", "dones"]

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        hidden_dim: int = 256,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        device: str = "cuda",
        use_game_theory: bool = True,
        use_shapley_credit: bool = False,
        ctde_with_hints: bool = False,
        warm_start_steps: int = 0,
        warm_start_lr_scale: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # -- Actor --
        self.actor = SACActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # -- Dual Q + Targets --
        self.qf1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf1_t = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.qf2_t = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self._hard_sync(self.qf1_t, self.qf1)
        self._hard_sync(self.qf2_t, self.qf2)

        self.qf1_opt = optim.Adam(self.qf1.parameters(), lr=lr)
        self.qf2_opt = optim.Adam(self.qf2.parameters(), lr=lr)

        # -- Auto alpha --
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.tensor(
                float(action_dim), dtype=torch.float32, device=self.device
            )
            self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = torch.exp(self.log_alpha.detach())
        else:
            self.alpha = torch.tensor(alpha, device=self.device)

        # -- Replay buffer --
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=str(self.device),
        )

        self.update_count = 0
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    @staticmethod
    def _soft_update(target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    @staticmethod
    def _hard_sync(target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            a, lp = self.actor.sample(t, deterministic)
        return a.cpu().numpy()[0], {"log_prob": lp.cpu().item()}

    def update(self, batch_data):
        """
        SAC 是 off-policy：先把接收到的数据存入 buffer，
        再从 buffer 中采样一个 batch 进行更新。
        """
        self._store(batch_data)

        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        return self._update_from_batch(batch)

    def _store(self, batch):
        n = batch["states"].shape[0]
        for i in range(n):
            state_i = batch["states"][i]
            action_i = batch["actions"][i]
            reward_i = float(batch["rewards"][i])
            next_i = batch["next_states"][i]
            done_i = float(batch["dones"][i])
            self.buffer.push(state_i, action_i, reward_i, next_i, done_i)

    def _update_from_batch(self, batch):
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1)
        next_states = batch["next_states"]
        dones = batch["dones"].unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.qf1_t(next_states, next_actions)
            q2_next = self.qf2_t(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Q losses
        q1 = self.qf1(states, actions)
        q2 = self.qf2(states, actions)
        q1_loss = nn.functional.mse_loss(q1, target_q)
        q2_loss = nn.functional.mse_loss(q2, target_q)

        self.qf1_opt.zero_grad()
        q1_loss.backward()
        self.qf1_opt.step()

        self.qf2_opt.zero_grad()
        q2_loss.backward()
        self.qf2_opt.step()

        # Policy loss
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.qf1(states, new_actions)
        q2_new = self.qf2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        # Alpha tuning
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = torch.exp(self.log_alpha.detach())

        # Soft update targets
        self._soft_update(self.qf1_t, self.qf1, self.tau)
        self._soft_update(self.qf2_t, self.qf2, self.tau)

        self.update_count += 1

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha.item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_t": self.qf1_t.state_dict(),
            "qf2_t": self.qf2_t.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.qf1.load_state_dict(state_dict["qf1"])
        self.qf2.load_state_dict(state_dict["qf2"])
        self.qf1_t.load_state_dict(state_dict["qf1_t"])
        self.qf2_t.load_state_dict(state_dict["qf2_t"])
