"""
QMIX: Q-Mixing for Cooperative Multi-Agent RL
值分解方法 — 将每个智能体的Q值混合为联合Q值

核心特点:
1. 单调性约束 — 确保联合Q值对单个Q值的偏导数非负
2. Mixing网络 — 非线性混合单体Q值为全局Q值
3. 集中训练分散执行 — 训练时可用全局信息，执行时仅用局部观测
4. 值分解 — 全局Q = Mixing(Q1, Q2, ..., QN)

Reference: Rashid et al. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .game_theory_utils import (
    apply_shapley_reward_scaling,
    discrete_imitation_loss,
    eq_actions_to_discrete_indices,
    inject_game_hints,
)


class QNetwork(nn.Module):
    """单个智能体的 Q 网络 (离散动作)"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q(x)


class MixingNetwork(nn.Module):
    """
    Mixing 网络 — 将每个智能体的Q值混合为联合Q值
    满足单调性: ∂Q_total / ∂Q_i >= 0
    """

    def __init__(self, n_agents, state_dim=64):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = n_agents

        # Hypernetwork 1: w1 [B, n_agents, embed], b1 [B, 1, embed]
        self.hyper_w1 = nn.Linear(state_dim, self.n_agents * self.embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)

        # Hypernetwork 2: w2 [B, embed, 1], b2 [B, 1, 1]
        self.hyper_w2 = nn.Linear(state_dim, self.embed_dim * 1)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs: torch.Tensor [batch, n_agents]
            states: torch.Tensor [batch, state_dim]
        Returns:
            q_total: torch.Tensor [batch]
        """
        bs = agent_qs.shape[0]

        # First layer
        w1 = torch.abs(self.hyper_w1(states))  # [B, n_agents * embed]
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # [B, n_agents, embed]
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)  # [B, 1, embed]

        # Hidden = f(Q @ W + b) [B, 1, embed]
        hidden = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(bs, self.embed_dim, 1)  # [B, embed, 1]
        b2 = self.hyper_b2(states).view(bs, 1, 1)

        # Q [B, 1, 1]
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(bs)


class QMIXAgent(BaseAgent):
    """
    QMIX 智能体 — 值分解多智能体算法 (离散动作)

    适用于：多用户合作任务、多智能体MEC卸载、合作式资源分配等
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        n_agents=None,
        env=None,
        global_state_dim=None,
        buffer_size=1_000_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10000,
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
            action_dim: int - 单个智能体的动作空间维度 (离散)
            hidden_dim: int - 隐藏层维度 (默认64)
            lr: float - 学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            tau: float - 软更新系数 (默认0.005)
            batch_size: int - 批次大小 (默认256)
            n_agents: int - 智能体数量 (默认2)
            global_state_dim: int - 全局状态维度 (默认: state_dim * n_agents)
            buffer_size: int - 回放缓冲区容量 (默认1_000_000)
            epsilon_start: float - 初始epsilon (默认1.0)
            epsilon_end: float - 最终epsilon (默认0.05)
            epsilon_decay: int - epsilon衰减步数 (默认10000)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        if n_agents is None and env is not None:
            n_agents = getattr(env, "num_agents", 2)
        elif n_agents is None:
            n_agents = 2

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.global_state_dim = global_state_dim if global_state_dim else state_dim * n_agents

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 每个智能体的 Q 网络 (参数共享)
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Mixing 网络
        self.mixer = MixingNetwork(n_agents, self.global_state_dim).to(self.device)
        self.target_mixer = MixingNetwork(n_agents, self.global_state_dim).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # 优化器 (包含所有参数)
        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()), lr=lr
        )

        # 缓冲区
        self.buffer = []
        self.buffer_size = buffer_size

        self.update_count = 0
        self.is_on_policy = False
        self.action_type = "discrete"
        self.compatible_env_types = ["multi_agent"]
        self.multi_agent_mode = "joint"
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
            action: int - 离散动作索引
            info: dict - 包含 'q_value': float
        """
        if not deterministic and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
            return np.array([action], dtype=np.int64), {"log_prob": 0.0, "exploration": True}

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=-1)

        return action.cpu().numpy(), {"log_prob": 0.0, "exploration": False}

    def _store(self, batch):
        """存储经验"""
        n = batch["states"].shape[0]  # states shape: [batch, n_agents, state_dim]
        for i in range(n):
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            self.buffer.append(
                {
                    "states": batch["states"][i],
                    "actions": batch["actions"][i],
                    "rewards": batch["rewards"][i],
                    "next_states": batch["next_states"][i],
                    "dones": batch["dones"][i],
                    "eq_actions": batch["eq_actions"][i] if "eq_actions" in batch else None,
                    "shapley_values": batch["shapley_values"][i] if "shapley_values" in batch else None,
                    "game_hints": batch["game_hints"][i] if "game_hints" in batch else None,
                }
            )

    def _sample(self):
        """从缓冲区采样"""
        if len(self.buffer) < self.batch_size:
            return None
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        batch = {
            "states": np.stack([x["states"] for x in samples]),  # [batch, n_agents, state_dim]
            "actions": np.stack([x["actions"] for x in samples]),  # [batch, n_agents]
            "rewards": np.stack([x["rewards"] for x in samples]),  # [batch, n_agents]
            "next_states": np.stack(
                [x["next_states"] for x in samples]
            ),  # [batch, n_agents, state_dim]
            "dones": np.stack([x["dones"] for x in samples]),  # [batch, n_agents]
        }
        if all(x.get("eq_actions") is not None for x in samples):
            batch["eq_actions"] = np.stack([x["eq_actions"] for x in samples])
        if all(x.get("shapley_values") is not None for x in samples):
            batch["shapley_values"] = np.stack([x["shapley_values"] for x in samples])
        if all(x.get("game_hints") is not None for x in samples):
            batch["game_hints"] = np.stack([x["game_hints"] for x in samples])
        return batch

    @staticmethod
    def _soft_update(target, source, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)

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
        QMIX 更新 — off-policy，先存buffer再采样
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

        states = torch.FloatTensor(s["states"]).to(self.device)  # [batch, n_agents, state_dim]
        actions = torch.LongTensor(s["actions"]).to(self.device)  # [batch, n_agents]
        rewards = torch.FloatTensor(s["rewards"]).to(self.device)  # [batch, n_agents]
        next_states = torch.FloatTensor(s["next_states"]).to(
            self.device
        )  # [batch, n_agents, state_dim]
        dones = torch.FloatTensor(s["dones"]).to(self.device)  # [batch, n_agents]
        if self.use_shapley_credit and "shapley_values" in s:
            rewards = apply_shapley_reward_scaling(
                rewards=rewards,
                shapley_values=s["shapley_values"],
                enabled=True,
            )

        batch_size = states.shape[0]
        global_states = states.view(batch_size, -1)  # [batch, n_agents * state_dim]
        next_global_states = next_states.view(batch_size, -1)  # [batch, n_agents * state_dim]
        if self.ctde_with_hints and "game_hints" in s:
            global_states = inject_game_hints(
                states=global_states,
                game_hints=s["game_hints"],
                enabled=True,
            )
            next_global_states = inject_game_hints(
                states=next_global_states,
                game_hints=s["game_hints"],
                enabled=True,
            )

        # 聚合全局奖励和dones
        rewards = rewards.mean(dim=1)  # [batch]
        dones = dones.any(dim=1).float()  # [batch]

        # ---- 计算 QMIX 目标 ----
        with torch.no_grad():
            # 每个智能体的下一状态Q值
            next_qs = self.target_q_net(next_states.view(-1, self.state_dim))
            next_qs = next_qs.view(batch_size, self.n_agents, self.action_dim)

            # 贪婪动作选择
            next_actions = next_qs.argmax(dim=-1)  # [batch, n_agents]
            next_actions_onehot = torch.zeros_like(next_qs).float()
            next_actions_onehot.scatter_(2, next_actions.unsqueeze(2), 1)

            # Q_i(a*_i) 取每个智能体的最大Q
            max_next_q_values = (next_qs * next_actions_onehot).sum(dim=-1)  # [batch, n_agents]
            target_q_total = self.target_mixer(max_next_q_values, next_global_states)

            target_q = rewards + (1.0 - dones) * self.gamma * target_q_total

        # 当前Q
        current_qs = self.q_net(states.view(-1, self.state_dim))  # [batch*n_agents, action_dim]
        current_qs = current_qs.view(batch_size, self.n_agents, self.action_dim)
        chosen_qs = torch.gather(current_qs, 2, actions.unsqueeze(2))  # [batch, n_agents, 1]
        q_total = self.mixer(chosen_qs.squeeze(2), global_states)  # [batch]

        loss = F.mse_loss(q_total, target_q.detach())
        imitation_loss = torch.tensor(0.0, device=self.device)
        if self.update_count < self.warm_start_steps and "eq_actions" in s:
            eq_idx = eq_actions_to_discrete_indices(
                eq_actions=s["eq_actions"],
                batch_size=batch_size,
                n_agents=self.n_agents,
                action_dim=self.action_dim,
                device=self.device,
            )
            if eq_idx is not None:
                logits = self.q_net(states.view(-1, self.state_dim)).view(batch_size, self.n_agents, self.action_dim)
                imitation_loss = discrete_imitation_loss(logits, eq_idx)
                loss = loss + self.warm_start_lr_scale * imitation_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        nn.utils.clip_grad_norm_(self.mixer.parameters(), 10)
        self.optimizer.step()

        self._soft_update(self.target_q_net, self.q_net, self.tau)
        self._soft_update(self.target_mixer, self.mixer, self.tau)
        self._decay_epsilon()

        self.update_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": 0.0,
            "value_loss": loss.item(),
            "entropy": 0.0,
            "approx_kl": 0.0,
            "imitation_loss": float(imitation_loss.detach().item()),
            "epsilon": self.epsilon,
            "q_total_mean": q_total.mean().item(),
        }

    def state_dict(self):
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "epsilon": self.epsilon,
            "buffer": self.buffer,
        }

    def load_state_dict(self, sd):
        self.q_net.load_state_dict(sd["q_net"])
        self.target_q_net.load_state_dict(sd.get("target_q_net", sd["q_net"]))
        self.mixer.load_state_dict(sd["mixer"])
        self.target_mixer.load_state_dict(sd.get("target_mixer", sd["mixer"]))
        self.optimizer.load_state_dict(sd["optimizer"])
        self.update_count = sd.get("update_count", 0)
        self.epsilon = sd.get("epsilon", self.epsilon_start)
        self.buffer = sd.get("buffer", [])
