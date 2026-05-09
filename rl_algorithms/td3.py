"""
TD3: Twin Delayed Deep Deterministic Policy Gradient
DDPG 的改进版，更稳定

核心特点:
1. 双 Q 网络取最小值 — 解决过估计 (Twin)
2. 延迟策略更新 — Critic 更新多次后才更新 Actor (Delayed)
3. 目标策略平滑正则 — 为目标动作添加噪声，提高鲁棒性 (Smoothing)
4. 确定性策略 — 连续动作输出

Reference: Fujimoto et al. "Addressing Function Approximation Error in Actor-Critic Methods" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from src.utils.buffer import ReplayBuffer


class TD3Actor(nn.Module):
    """TD3 确定性策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0, action_bias=0.0):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.action_head(x)) * self.action_scale + self.action_bias


class TD3Critic(nn.Module):
    """TD3 Q 网络 (支持双 Q 输出)"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x1 = torch.cat([state, action], dim=-1)
        x1 = F.relu(self.q1_fc1(x1))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)

        x2 = torch.cat([state, action], dim=-1)
        x2 = F.relu(self.q2_fc1(x2))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)

        return q1, q2

    def Q1(self, state, action):
        """只返回 Q1 的值 (用于评估时防泄漏)"""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.q1_fc1(x))
        x = F.relu(self.q1_fc2(x))
        return self.q1_out(x)


class TD3Agent(BaseAgent):
    """
    TD3 智能体 — 双延迟 DDPG (连续动作)

    适用于：连续资源分配、MEC功率控制、机器人控制等需要高稳定性的连续控制任务
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
        noise_std=0.2,
        noise_clip=0.5,
        exploration_noise_std=None,
        policy_delay=2,  # 策略延迟更新间隔
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
            explore_steps: int - 探索噪声步数 (默认10000)
            noise_std: float - 目标策略噪声标准差 (默认0.2)
            noise_clip: float - 目标策略噪声裁剪范围 (默认0.5)
            exploration_noise_std: float - 行为策略探索噪声；默认与 noise_std 一致
            policy_delay: int - 策略延迟更新间隔 (默认2)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_std = float(noise_std)
        self.noise_clip = float(noise_clip)
        self.exploration_noise_std = float(noise_std if exploration_noise_std is None else exploration_noise_std)

        # Actor
        self.actor = TD3Actor(state_dim, action_dim, hidden_dim, action_scale, action_bias).to(
            self.device
        )
        self.actor_t = TD3Actor(state_dim, action_dim, hidden_dim, action_scale, action_bias).to(
            self.device
        )
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic (双 Q 合一个网络)
        self.critic = TD3Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_t = TD3Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=str(self.device),
        )
        self.buffer_size = buffer_size

        # 探索
        self.sample_count = 0
        self.explore_steps = explore_steps

        self.update_count = 0
        self.is_on_policy = False
        self.action_type = "continuous"
        self.compatible_env_types = ["continuous"]
        self.required_batch_keys = ["states", "actions", "rewards", "next_states", "dones"]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """
        选择动作 — 探索阶段加高斯噪声

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 动作，shape: [action_dim]
            info: dict - 包含 'q_value': float
        """
        self.sample_count += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)

        action_np = action.cpu().numpy()[0]

        # 探索噪声
        if not deterministic and self.sample_count < self.explore_steps:
            noise = np.random.normal(0, self.exploration_noise_std, size=self.action_dim)
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
        TD3 更新 — off-policy，先存buffer再采样更新
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
            # 目标动作 + 平滑噪声
            noise = torch.clamp(
                torch.randn_like(actions) * self.noise_std, -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(self.actor_t(next_states) + noise, -1, 1)

            # 双 Q 取最小值 (解决过估计)
            next_q1, next_q2 = self.critic_t(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2).squeeze(-1)

            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        q1_pred, q2_pred = self.critic(states, actions)
        q1_pred = q1_pred.squeeze(-1)
        q2_pred = q2_pred.squeeze(-1)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        # ---- Actor 更新 (延迟) ----
        actor_loss = torch.tensor(0.0, device=self.device)
        if self.update_count % self.policy_delay == 0:
            # 策略梯度：最大化 Q1
            new_actions = self.actor(states)
            actor_loss = -self.critic.Q1(states, new_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            # 软更新目标网络
            self._soft_update(self.actor_t, self.actor, self.tau)
            self._soft_update(self.critic_t, self.critic, self.tau)

        self.update_count += 1

        return {
            "loss": actor_loss.item() + 0.001 * critic_loss.item(),
            "policy_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy": 0.0,
            "approx_kl": 0.0,
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic_t": self.critic_t.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self.update_count,
            "sample_count": self.sample_count,
            "buffer": self.buffer,
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_t.load_state_dict(sd.get("actor_t", sd["actor"]))
        self.critic_t.load_state_dict(sd.get("critic_t", sd["critic"]))
        self.actor_opt.load_state_dict(sd.get("actor_opt", {}))
        self.critic_opt.load_state_dict(sd.get("critic_opt", {}))
        self.update_count = sd.get("update_count", 0)
        self.sample_count = sd.get("sample_count", 0)
        self.buffer = sd.get("buffer", [])
