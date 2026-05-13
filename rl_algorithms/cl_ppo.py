"""Constraint Lyapunov PPO with tail-risk critic and continuous safety layer."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .ppo import PPOAgent


CONSTRAINT_NAMES = ("deadline_miss", "energy_overshoot", "queue_overload")


class CostRiskCritic(nn.Module):
    """Small state-value model for expected tail cost."""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        """Initialize the risk critic."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Predict a scalar risk value for each state."""
        return self.net(states).squeeze(-1)


class CLPPOAgent(PPOAgent):
    """PPO with Lyapunov-style dual constraints, risk critic, and safety projection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        gae_lambda: float = 0.95,
        num_epochs: int = 10,
        clip_grad: float = 0.5,
        value_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        device: str = "cuda",
        discrete: bool = False,
        constraints: dict[str, Any] | None = None,
        risk: dict[str, Any] | None = None,
        safety_layer: dict[str, Any] | None = None,
        **extra_kwargs: Any,
    ) -> None:
        """Initialize CL-PPO while preserving PPO constructor compatibility."""
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            eps_clip=eps_clip,
            gae_lambda=gae_lambda,
            num_epochs=num_epochs,
            clip_grad=clip_grad,
            value_coeff=value_coeff,
            ent_coeff=ent_coeff,
            device=device,
            discrete=discrete,
            **extra_kwargs,
        )
        self.constraints_cfg = dict(constraints or {})
        self.risk_cfg = dict(risk or {})
        self.safety_cfg = dict(safety_layer or {})
        self.constraints_enabled = bool(self.constraints_cfg.get("enabled", True))
        self.risk_enabled = bool(self.risk_cfg.get("enabled", True))
        self.safety_enabled = bool(self.safety_cfg.get("enabled", True))
        self.dual_lr = float(self.constraints_cfg.get("dual_lr", 0.01))
        self.constraint_budgets = {
            "deadline_miss": float(self.constraints_cfg.get("deadline_miss_budget", 0.0)),
            "energy_overshoot": float(self.constraints_cfg.get("energy_overshoot_budget", 0.0)),
            "queue_overload": float(self.constraints_cfg.get("queue_overload_budget", 0.0)),
        }
        self.dual_variables = {name: torch.tensor(0.0, device=self.device) for name in CONSTRAINT_NAMES}
        self.cvar_alpha = float(self.risk_cfg.get("cvar_alpha", 0.9))
        self.risk_coeff = float(self.risk_cfg.get("risk_coeff", 0.1))
        self.risk_critic = CostRiskCritic(state_dim=state_dim, hidden_dim=hidden_dim).to(self.device)
        self.risk_optimizer = optim.Adam(self.risk_critic.parameters(), lr=lr)
        self.compatible_env_types = ["continuous"]
        self.action_type = "continuous"

    def _to_tensor(self, value: Any, batch_size: int) -> torch.Tensor:
        """Convert an optional batch value into a one-dimensional tensor."""
        if value is None:
            return torch.zeros(batch_size, device=self.device)
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=tuple(range(1, tensor.ndim)))
        tensor = tensor.reshape(-1)
        if tensor.numel() == batch_size:
            return tensor
        if tensor.numel() == 1:
            return tensor.expand(batch_size)
        if tensor.numel() > batch_size:
            return tensor[:batch_size]
        repeats = int(np.ceil(batch_size / max(tensor.numel(), 1)))
        return tensor.repeat(repeats)[:batch_size]

    def _extract_constraint_costs(
        self,
        batch_data: dict[str, Any],
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Extract or synthesize the supported constraint cost signals."""
        raw_costs = batch_data.get("constraint_costs", batch_data.get("costs"))
        if isinstance(raw_costs, dict):
            costs = {
                "deadline_miss": self._to_tensor(raw_costs.get("deadline_miss"), batch_size),
                "energy_overshoot": self._to_tensor(raw_costs.get("energy_overshoot"), batch_size),
                "queue_overload": self._to_tensor(raw_costs.get("queue_overload"), batch_size),
            }
        else:
            shared = self._to_tensor(raw_costs, batch_size)
            costs = {name: shared.clone() for name in CONSTRAINT_NAMES}

        if "deadline_misses" in batch_data:
            costs["deadline_miss"] = self._to_tensor(batch_data["deadline_misses"], batch_size)
        if "energy_costs" in batch_data:
            energy = self._to_tensor(batch_data["energy_costs"], batch_size)
            costs["energy_overshoot"] = torch.relu(
                energy - float(self.constraints_cfg.get("energy_reference", 0.0))
            )
        if "queue_waits" in batch_data:
            queue = self._to_tensor(batch_data["queue_waits"], batch_size)
            costs["queue_overload"] = torch.relu(
                queue - float(self.constraints_cfg.get("queue_reference", 0.0))
            )
        if "latencies" in batch_data and "deadline_misses" not in batch_data:
            latencies = self._to_tensor(batch_data["latencies"], batch_size)
            budget = float(self.constraints_cfg.get("latency_budget", self.constraint_budgets["deadline_miss"]))
            costs["deadline_miss"] = torch.relu(latencies - budget)
        return costs

    def _aggregate_cost_target(
        self,
        costs: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Build a scalar target for the risk critic from available costs."""
        if not costs:
            return torch.zeros(batch_size, device=self.device)
        stacked = torch.stack([costs[name] for name in CONSTRAINT_NAMES], dim=0)
        aggregate = stacked.mean(dim=0)
        if aggregate.numel() <= 1:
            return aggregate
        threshold = torch.quantile(aggregate.detach(), self.cvar_alpha)
        return torch.where(aggregate >= threshold, aggregate, torch.zeros_like(aggregate))

    def update_dual_variables(self, costs: dict[str, torch.Tensor]) -> dict[str, float]:
        """Apply non-negative virtual-queue updates to the dual variables."""
        metrics: dict[str, float] = {}
        for name in CONSTRAINT_NAMES:
            cost = costs.get(name)
            mean_cost = torch.as_tensor(0.0, device=self.device) if cost is None else cost.detach().mean()
            budget = self.constraint_budgets[name]
            updated = torch.clamp(self.dual_variables[name] + self.dual_lr * (mean_cost - budget), min=0.0)
            self.dual_variables[name] = updated.detach()
            metrics[f"dual_{name}"] = float(self.dual_variables[name].item())
        return metrics

    def _constraint_penalty(self, costs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the current dual-weighted constraint violation penalty."""
        penalty = torch.tensor(0.0, device=self.device)
        if not self.constraints_enabled:
            return penalty
        for name in CONSTRAINT_NAMES:
            violation = torch.relu(costs[name].mean() - self.constraint_budgets[name])
            penalty = penalty + self.dual_variables[name].detach() * violation
        return penalty

    def project_action(self, action: Any, state: Any | None = None) -> np.ndarray:
        """Clamp and conservatively scale a continuous action when risk is high."""
        arr = np.asarray(action, dtype=np.float32).copy()
        low = float(self.safety_cfg.get("action_low", -1.0))
        high = float(self.safety_cfg.get("action_high", 1.0))
        arr = np.clip(arr, low, high)
        if state is not None:
            threshold = float(self.safety_cfg.get("risk_threshold", 0.75))
            scale = float(self.safety_cfg.get("conservative_scale", 0.8))
            if isinstance(state, dict):
                indicators = [
                    float(state.get("deadline_miss_risk", 0.0)),
                    float(state.get("energy_overshoot", 0.0)),
                    float(state.get("queue_overload", 0.0)),
                ]
                high_risk = max(indicators) > threshold
            else:
                state_arr = np.asarray(state, dtype=np.float32)
                high_risk = bool(state_arr.size and float(np.nanmax(np.abs(state_arr))) > threshold)
            if high_risk and arr.size:
                arr[-min(2, arr.size) :] *= scale
        return np.clip(arr, low, high).astype(np.float32)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, dict[str, float]]:
        """Select an action with optional continuous safety projection."""
        action, info = super().select_action(state, deterministic=deterministic)
        if self.safety_enabled and not self.discrete:
            action = self.project_action(action, state=state)
        return action, info

    def update(self, batch_data: dict[str, Any]) -> dict[str, float]:
        """Update PPO, the risk critic, and Lyapunov dual variables from one batch."""
        states = torch.as_tensor(batch_data["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch_data["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch_data["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch_data["dones"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch_data["log_probs"], dtype=torch.float32, device=self.device)
        if old_log_probs.ndim > 1:
            old_log_probs = old_log_probs.squeeze(-1)

        batch_size = states.shape[0]
        costs = self._extract_constraint_costs(batch_data, batch_size)
        risk_target = self._aggregate_cost_target(costs, batch_size)
        risk_prediction = self.risk_critic(states)
        risk_loss_tensor = F.mse_loss(risk_prediction, risk_target.detach()) if self.risk_enabled else torch.tensor(0.0, device=self.device)
        if self.risk_enabled:
            self.risk_optimizer.zero_grad()
            risk_loss_tensor.backward()
            nn.utils.clip_grad_norm_(self.risk_critic.parameters(), self.clip_grad)
            self.risk_optimizer.step()

        if batch_data.get("values") is not None:
            old_values = torch.as_tensor(batch_data["values"], dtype=torch.float32, device=self.device).detach()
        else:
            old_values = self.critic(states).squeeze(-1).detach()
        old_values = old_values.reshape(-1)

        advantages, returns = self._compute_gae(rewards, old_values, dones)
        advantages = advantages.unsqueeze(-1)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        total_constraint = 0.0
        total_risk = 0.0

        constraint_loss = self._constraint_penalty(costs)
        predicted_tail_cost = self.risk_critic(states).detach()
        risk_penalty = (
            self.risk_coeff * predicted_tail_cost.mean()
            if self.risk_enabled
            else torch.tensor(0.0, device=self.device)
        )

        for _ in range(self.num_epochs):
            dist = self.actor.get_distribution(states)
            if self.discrete:
                new_log_probs = dist.log_prob(actions.squeeze(-1).long()).unsqueeze(-1)
                entropy = dist.entropy()
            else:
                new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1)
            new_values = self.critic(states).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs.unsqueeze(-1))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            clip_frac = ((ratio - 1).abs() > self.eps_clip).float().mean()
            value_loss = F.mse_loss(new_values, returns)
            loss = (
                policy_loss
                + self.value_coeff * value_loss
                - self.ent_coeff * entropy.mean()
                + constraint_loss
                + risk_penalty
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.clip_grad
            )
            self.optimizer.step()

            approx_kl = (old_log_probs - new_log_probs.squeeze(-1)).mean().item()
            total_loss += float(loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.mean().item())
            total_kl += float(approx_kl)
            total_clip_frac += float(clip_frac.item())
            total_constraint += float(constraint_loss.item())
            total_risk += float(risk_penalty.item())

        self.update_count += 1
        dual_metrics = self.update_dual_variables(costs)
        n = max(1, self.num_epochs)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
            "clip_fraction": total_clip_frac / n,
            "constraint_loss": total_constraint / n,
            "risk_loss": float(risk_loss_tensor.detach().item()),
            "risk_penalty": total_risk / n,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
            **dual_metrics,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return PPO state plus risk critic and dual variables."""
        state = super().state_dict()
        state.update(
            {
                "risk_critic": self.risk_critic.state_dict(),
                "risk_optimizer": self.risk_optimizer.state_dict(),
                "dual_variables": {
                    name: float(value.detach().cpu().item()) for name, value in self.dual_variables.items()
                },
            }
        )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load PPO state plus risk critic and dual variables."""
        super().load_state_dict(state_dict)
        if "risk_critic" in state_dict:
            self.risk_critic.load_state_dict(state_dict["risk_critic"])
        if "risk_optimizer" in state_dict:
            self.risk_optimizer.load_state_dict(state_dict["risk_optimizer"])
        for name, value in state_dict.get("dual_variables", {}).items():
            if name in self.dual_variables:
                self.dual_variables[name] = torch.tensor(float(value), device=self.device)
