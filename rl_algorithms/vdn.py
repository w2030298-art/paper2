"""VDN: Value Decomposition Networks."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent
from .game_theory_utils import (
    apply_shapley_reward_scaling,
    discrete_imitation_loss,
    eq_actions_to_discrete_indices,
    inject_game_hints,
)


class VDNQNetwork(nn.Module):
    """Per-agent Q-network for discrete actions."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.q(x)


class VDNAgent(BaseAgent):
    """VDN multi-agent discrete value-decomposition agent."""

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

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = VDNQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net = VDNQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

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
        if not deterministic and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
            return np.array([action], dtype=np.int64), {"log_prob": 0.0, "exploration": True}

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=-1)

        return action.cpu().numpy(), {"log_prob": 0.0, "exploration": False}

    def _store(self, batch):
        n = batch["states"].shape[0]
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
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end) * min(self.update_count, self.epsilon_decay) / self.epsilon_decay,
        )

    def update(self, batch_data):
        self._store(batch_data)
        s = self._sample()
        if s is None:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        states = torch.FloatTensor(s["states"]).to(self.device)
        actions = torch.LongTensor(s["actions"]).to(self.device)
        rewards = torch.FloatTensor(s["rewards"]).to(self.device)
        next_states = torch.FloatTensor(s["next_states"]).to(self.device)
        dones = torch.FloatTensor(s["dones"]).to(self.device)
        if self.use_shapley_credit and "shapley_values" in s:
            rewards = apply_shapley_reward_scaling(
                rewards=rewards,
                shapley_values=s["shapley_values"],
                enabled=True,
            )

        if self.ctde_with_hints and "game_hints" in s:
            states = inject_game_hints(
                states=states,
                game_hints=s["game_hints"],
                enabled=True,
            )
            next_states = inject_game_hints(
                states=next_states,
                game_hints=s["game_hints"],
                enabled=True,
            )

        batch_size = states.shape[0]

        rewards = rewards.mean(dim=1)
        dones = dones.any(dim=1).float()

        with torch.no_grad():
            next_qs = self.target_q_net(next_states.view(-1, self.state_dim))
            next_qs = next_qs.view(batch_size, self.n_agents, self.action_dim)
            next_actions = next_qs.argmax(dim=-1)
            max_next_q = next_qs.gather(2, next_actions.unsqueeze(2)).squeeze(2)
            target_q_total = max_next_q.sum(dim=1)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q_total

        current_qs = self.q_net(states.view(-1, self.state_dim)).view(batch_size, self.n_agents, self.action_dim)
        chosen_qs = current_qs.gather(2, actions.unsqueeze(2)).squeeze(2)
        q_total = chosen_qs.sum(dim=1)

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
                imitation_loss = discrete_imitation_loss(current_qs, eq_idx)
                loss = loss + self.warm_start_lr_scale * imitation_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self._soft_update(self.target_q_net, self.q_net, self.tau)
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
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "epsilon": self.epsilon,
            "buffer": self.buffer,
        }

    def load_state_dict(self, sd):
        self.q_net.load_state_dict(sd["q_net"])
        self.target_q_net.load_state_dict(sd.get("target_q_net", sd["q_net"]))
        self.optimizer.load_state_dict(sd["optimizer"])
        self.update_count = sd.get("update_count", 0)
        self.epsilon = sd.get("epsilon", self.epsilon_start)
        self.buffer = sd.get("buffer", [])
