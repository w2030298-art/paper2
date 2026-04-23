"""
MATD3: Multi-Agent Twin Delayed Deep Deterministic Policy Gradient
多智能体TD3 — 集中训练分散执行的连续动作算法

核心特点:
1. 双Q网络 — 解决过估计问题
2. 延迟策略更新 — 提高稳定性
3. 集中critic — 输入全局观测和所有agent动作
4. 目标策略平滑 — 添加噪声提高鲁棒性

Reference: Ackermann et al. "Reducing Overestimation Bias in Multi-Agent Actor-Critic" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .game_theory_utils import (
    apply_shapley_reward_scaling,
    inject_game_hints,
    prepare_eq_actions_multi_continuous,
)


class MATD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.action(x))


class MATD3Critic(nn.Module):
    def __init__(self, global_state_dim, n_agents, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = global_state_dim + n_agents * action_dim
        # Q1
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, global_state, all_actions):
        x = torch.cat([global_state, all_actions], dim=-1)
        x1 = F.relu(self.q1_fc1(x))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        x2 = F.relu(self.q2_fc1(x))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        return q1, q2

    def Q1(self, global_state, all_actions):
        x = torch.cat([global_state, all_actions], dim=-1)
        x = F.relu(self.q1_fc1(x))
        x = F.relu(self.q1_fc2(x))
        return self.q1_out(x)


class MATD3Agent(BaseAgent):
    """MATD3 智能体 — 多智能体双延迟确定性策略梯度"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        n_agents=3,
        buffer_size=1_000_000,
        noise_std=0.2,
        noise_clip=0.5,
        policy_delay=2,
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
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.global_state_dim = state_dim * n_agents

        self.actor = MATD3Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = MATD3Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = MATD3Critic(self.global_state_dim, n_agents, action_dim, hidden_dim).to(self.device)
        self.critic_target = MATD3Critic(self.global_state_dim, n_agents, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = []
        self.buffer_size = buffer_size
        self.update_count = 0
        self.is_on_policy = False
        self.action_type = "continuous"
        self.compatible_env_types = ["multi_agent"]
        self.multi_agent_mode = "joint"
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)
        return action.cpu().numpy()[0], {"log_prob": 0.0}

    def _store(self, batch):
        n = batch["states"].shape[0]
        for i in range(n):
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            if "global_states" in batch:
                g_obs = batch["global_states"][i]
                next_g_obs = batch.get("next_global_states", batch["global_states"])[i]
            else:
                g_obs = np.concatenate([np.asarray(o, dtype=np.float32) for o in batch["states"][i]])
                next_g_obs = np.concatenate([np.asarray(o, dtype=np.float32) for o in batch["next_states"][i]])
            self.buffer.append({
                "states": batch["states"][i],
                "actions": batch["actions"][i],
                "rewards": batch["rewards"][i],
                "next_states": batch["next_states"][i],
                "dones": batch["dones"][i],
                "global_states": g_obs,
                "next_global_states": next_g_obs,
                "eq_actions": batch["eq_actions"][i] if "eq_actions" in batch else None,
                "shapley_values": batch["shapley_values"][i] if "shapley_values" in batch else None,
                "game_hints": batch["game_hints"][i] if "game_hints" in batch else None,
            })

    def _sample(self):
        if len(self.buffer) < self.batch_size:
            return None
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        batch = {
            "states": np.stack([x["states"] for x in samples]),
            "actions": np.stack([x["actions"] for x in samples]),
            "rewards": np.stack([x["rewards"] for x in samples]),
            "next_states": np.stack([x["next_states"] for x in samples]),
            "dones": np.stack([x["dones"] for x in samples]),
            "global_states": np.stack([x["global_states"] for x in samples]),
            "next_global_states": np.stack([x["next_global_states"] for x in samples]),
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

    def update(self, batch_data):
        self._store(batch_data)
        s = self._sample()
        if s is None:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        states = torch.FloatTensor(s["states"]).to(self.device)
        actions = torch.FloatTensor(s["actions"]).to(self.device)
        rewards = torch.FloatTensor(s["rewards"]).to(self.device)
        next_states = torch.FloatTensor(s["next_states"]).to(self.device)
        dones = torch.FloatTensor(s["dones"]).to(self.device)
        global_states = torch.FloatTensor(s["global_states"]).to(self.device)
        next_global_states = torch.FloatTensor(s["next_global_states"]).to(self.device)
        if self.use_shapley_credit and "shapley_values" in s:
            rewards = apply_shapley_reward_scaling(
                rewards=rewards,
                shapley_values=s["shapley_values"],
                enabled=True,
            )
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

        batch_size = states.shape[0]

        rewards = rewards.mean(dim=1)
        dones = dones.any(dim=1).float()

        # Critic 更新
        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(actions) * self.noise_std, -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(
                self.actor_target(next_states.view(-1, self.state_dim)).view(batch_size, self.n_agents, self.action_dim) + noise,
                -1, 1
            )
            next_q1, next_q2 = self.critic_target(next_global_states, next_actions.view(batch_size, -1))
            next_q = torch.min(next_q1, next_q2).squeeze(-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        q1_pred, q2_pred = self.critic(global_states, actions.view(batch_size, -1))
        q1_pred = q1_pred.squeeze(-1)
        q2_pred = q2_pred.squeeze(-1)

        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        # Actor 更新 (延迟)
        actor_loss = torch.tensor(0.0, device=self.device)
        imitation_loss = torch.tensor(0.0, device=self.device)
        if self.update_count % self.policy_delay == 0:
            current_actions = self.actor(states.view(-1, self.state_dim)).view(batch_size, self.n_agents, self.action_dim)
            actor_loss = -self.critic.Q1(global_states, current_actions.view(batch_size, -1)).mean()
            if self.update_count < self.warm_start_steps and "eq_actions" in s:
                eq_actions = prepare_eq_actions_multi_continuous(
                    eq_actions=s["eq_actions"],
                    batch_size=batch_size,
                    n_agents=self.n_agents,
                    action_dim=self.action_dim,
                    device=self.device,
                )
                if eq_actions is not None:
                    imitation_loss = F.mse_loss(current_actions, eq_actions)
                    actor_loss = actor_loss + self.warm_start_lr_scale * imitation_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()

            self._soft_update(self.actor_target, self.actor, self.tau)
            self._soft_update(self.critic_target, self.critic, self.tau)

        self.update_count += 1

        return {
            "loss": actor_loss.item() + critic_loss.item(),
            "policy_loss": actor_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy": 0.0,
            "approx_kl": 0.0,
            "imitation_loss": float(imitation_loss.detach().item()),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self.update_count,
            "buffer": self.buffer,
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.actor_target.load_state_dict(sd.get("actor_target", sd["actor"]))
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd.get("critic_target", sd["critic"]))
        self.actor_opt.load_state_dict(sd.get("actor_opt", {}))
        self.critic_opt.load_state_dict(sd.get("critic_opt", {}))
        self.update_count = sd.get("update_count", 0)
        self.buffer = sd.get("buffer", [])
