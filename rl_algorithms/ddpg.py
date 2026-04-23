"""
DDPG: Deep Deterministic Policy Gradient
Actor-Critic 处理连续动作空间

核心特点:
1. 确定性策略 — 直接输出确定性动作 (无需采样)
2. 目标网络 — Actor 和 Critic 都有目标网络，稳定训练
3. 经验回放 — off-policy 学习
4. OU 噪声 — 时序相关的探索噪声

Reference: Lillicrap et al. "Continuous Control with Deep Reinforcement Learning" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from src.utils.buffer import ReplayBuffer


class DDPGActor(nn.Module):
    """DDPG 确定性策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0, action_bias=0.0):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.action(x)) * self.action_scale + self.action_bias


class DDPGQNetwork(nn.Module):
    """Q 网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class OUNoise:
    """Ornstein-Uhlenbeck 噪声 — 时序相关的探索噪声"""

    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu


class DDPGAgent(BaseAgent):
    """
    DDPG 智能体 — 确定性策略梯度 (连续动作)

    适用于：连续资源分配、MIMO预编码、机器人控制
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
        action_scale=1.0,
        action_bias=0.0,
        explore_steps=10000,
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
            action_dim: int - 动作空间维度 (连续)
            hidden_dim: int - 网络隐藏层维度 (默认256)
            lr: float - 学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            tau: float - 软更新系数 (默认0.005)
            batch_size: int - 批次大小 (默认256)
            buffer_size: int - 回放缓冲区容量 (默认1_000_000)
            action_scale: float - 动作幅度缩放 (默认1.0)
            action_bias: float - 动作偏置 (默认0.0)
            explore_steps: int - 噪声探索步数 (默认10000)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor
        self.actor = DDPGActor(state_dim, action_dim, hidden_dim, action_scale, action_bias).to(
            self.device
        )
        self.actor_target = DDPGActor(
            state_dim, action_dim, hidden_dim, action_scale, action_bias
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = DDPGQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = DDPGQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # 噪声
        self.noise = OUNoise(action_dim)
        self.explore_steps = explore_steps
        self.sample_count = 0

        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=str(self.device),
        )
        self.buffer_size = buffer_size

        self.update_count = 0
        self.is_on_policy = False
        self.action_type = "continuous"
        self.compatible_env_types = ["continuous"]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)
        self.required_batch_keys = ["states", "actions", "rewards", "next_states", "dones"]

    def select_action(self, state, deterministic=False):
        """选择动作 — 探索阶段加噪声"""
        self.sample_count += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)

        action_np = action.cpu().numpy()[0]

        if not deterministic and self.sample_count < self.explore_steps:
            noise = self.noise.sample()
            action_np = np.clip(action_np + noise, -1, 1)

        return action_np, {"log_prob": 0.0, "exploration": self.sample_count < self.explore_steps}

    def _store(self, batch):
        """存储经验到缓冲区"""
        n = batch["states"].shape[0]
        for i in range(n):
            self.buffer.push(
                batch["states"][i],
                batch["actions"][i],
                float(batch["rewards"][i]),
                batch["next_states"][i],
                float(batch["dones"][i]),
            )

    def _sample(self):
        """从缓冲区采样"""
        if len(self.buffer) < self.batch_size:
            return None
        return self.buffer.sample(self.batch_size)

    @staticmethod
    def _soft_update(target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

    def update(self, batch_data):
        """
        DDPG 更新 — off-policy，先存buffer再采样更新
        """
        self._store(batch_data)
        s = self._sample()
        if s is None:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }

        states = s["states"].to(self.device)
        actions = s["actions"].to(self.device)
        rewards = s["rewards"].to(self.device)
        next_states = s["next_states"].to(self.device)
        dones = s["dones"].to(self.device)

        # ---- Critic 更新 ----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions).squeeze(-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        q_pred = self.critic(states, actions).squeeze(-1)
        critic_loss = F.mse_loss(q_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        # ---- Actor 更新 ----
        new_actions = self.actor(states)
        q_new = self.critic(states, new_actions).squeeze(-1)
        actor_loss = -q_new.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_opt.step()

        # ---- 软更新 ----
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

        self.update_count += 1

        return {
            "loss": actor_loss.item() + 0.001 * critic_loss.item(),
            "policy_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy": 0.0,
            "approx_kl": 0.0,
            "q_mean": q_pred.mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_t": self.actor_target.state_dict(),
            "critic_t": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self.update_count,
            "sample_count": self.sample_count,
            "buffer": self.buffer,
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_target.load_state_dict(sd.get("actor_t", sd["actor"]))
        self.critic_target.load_state_dict(sd.get("critic_t", sd["critic"]))
        self.actor_opt.load_state_dict(sd.get("actor_opt", {}))
        self.critic_opt.load_state_dict(sd.get("critic_opt", {}))
        self.update_count = sd.get("update_count", 0)
        self.sample_count = sd.get("sample_count", 0)
        self.buffer = sd.get("buffer", [])
