"""Graph-attention masked COMA for Stage-2 multi-agent innovation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from .coma import COMAAgent
from .game_theory_utils import (
    apply_shapley_reward_scaling,
    discrete_imitation_loss,
    eq_actions_to_discrete_indices,
)
from .utils.networks import ActorDiscreteNetwork


MASKED_LOGIT_VALUE = -1.0e9


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Apply a broadcast-compatible feasible-action mask to discrete logits."""
    if mask is None:
        return logits
    mask_tensor = mask.to(device=logits.device, dtype=torch.bool)
    while mask_tensor.ndim < logits.ndim:
        mask_tensor = mask_tensor.unsqueeze(0)
    mask_tensor = torch.broadcast_to(mask_tensor, logits.shape)
    return logits.masked_fill(~mask_tensor, MASKED_LOGIT_VALUE)


class GraphAttentionCOMACritic(nn.Module):
    """COMA-compatible centralized critic with agent-token graph attention."""

    def __init__(
        self,
        global_state_dim: int,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the graph-attention critic."""
        super().__init__()
        self.global_state_dim = int(global_state_dim)
        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if hidden_dim % max(1, num_heads) != 0:
            num_heads = 1
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=max(1, int(num_heads)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _reshape_global_state(self, global_state: torch.Tensor) -> torch.Tensor:
        """Pad or truncate global state into per-agent state tokens."""
        batch_size = global_state.shape[0]
        flat = global_state.reshape(batch_size, -1)
        target = self.num_agents * self.state_dim
        if flat.shape[-1] < target:
            pad = torch.zeros(batch_size, target - flat.shape[-1], device=flat.device, dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[-1] > target:
            flat = flat[:, :target]
        return flat.reshape(batch_size, self.num_agents, self.state_dim)

    def forward(self, global_state: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """Return Q values shaped ``[batch, num_agents, action_dim]``."""
        per_agent_state = self._reshape_global_state(global_state.float())
        action_idx = all_actions.long().reshape(global_state.shape[0], self.num_agents)
        action_idx = torch.clamp(action_idx, min=0, max=self.action_dim - 1)
        tokens = self.state_proj(per_agent_state) + self.action_embed(action_idx)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        return self.out(tokens)


class GAMCOMAAgent(COMAAgent):
    """COMA with graph-attention critic and optional feasible-action masking."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        num_agents: int = 3,
        num_epochs: int = 10,
        discrete: bool = True,
        device: str = "cuda",
        use_game_theory: bool = True,
        use_shapley_credit: bool = True,
        ctde_with_hints: bool = True,
        warm_start_steps: int = 1000,
        warm_start_lr_scale: float = 0.5,
        policy_clip_low: float = 0.8,
        policy_clip_high: float = 1.2,
        grad_clip: float = 0.5,
        critic_loss_coeff: float = 0.5,
        entropy_coeff: float = 0.0,
        graph_attention: dict[str, Any] | None = None,
        action_masking: dict[str, Any] | None = None,
        social_influence: dict[str, Any] | None = None,
    ) -> None:
        """Initialize GAM-COMA while preserving COMA-compatible knobs."""
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            num_agents=num_agents,
            num_epochs=num_epochs,
            discrete=discrete,
            device=device,
            use_game_theory=use_game_theory,
            use_shapley_credit=use_shapley_credit,
            ctde_with_hints=ctde_with_hints,
            warm_start_steps=warm_start_steps,
            warm_start_lr_scale=warm_start_lr_scale,
            policy_clip_low=policy_clip_low,
            policy_clip_high=policy_clip_high,
            grad_clip=grad_clip,
            critic_loss_coeff=critic_loss_coeff,
            entropy_coeff=entropy_coeff,
        )
        if not self.discrete:
            raise ValueError("GAM-COMA currently requires a discrete action space.")
        self.graph_attention_cfg = dict(graph_attention or {})
        self.action_masking_cfg = dict(action_masking or {})
        self.social_influence_cfg = dict(social_influence or {})
        self.graph_attention_enabled = bool(self.graph_attention_cfg.get("enabled", True))
        self.action_masking_enabled = bool(self.action_masking_cfg.get("enabled", True))
        self.social_influence_enabled = bool(self.social_influence_cfg.get("enabled", False))
        self.social_influence_coeff = float(self.social_influence_cfg.get("coeff", 0.0))
        self.critic = GraphAttentionCOMACritic(
            global_state_dim=self.global_state_dim,
            num_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_heads=int(self.graph_attention_cfg.get("num_heads", 4)),
            dropout=float(self.graph_attention_cfg.get("dropout", 0.0)),
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.multi_agent_mode = "joint"
        self.compatible_env_types = ["multi_agent"]

    def _mask_tensor(self, batch_data: dict[str, Any]) -> torch.Tensor | None:
        """Load an optional feasible-action mask from a batch."""
        raw_mask = batch_data.get("action_masks", batch_data.get("feasible_action_mask"))
        if raw_mask is None or not self.action_masking_enabled:
            return None
        return torch.as_tensor(raw_mask, dtype=torch.bool, device=self.device)

    def _masked_distribution(
        self,
        states: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[Categorical, torch.Tensor, float]:
        """Build a categorical policy distribution with optional action masking."""
        logits = self.actor(states.reshape(-1, self.state_dim))
        if mask is None:
            return Categorical(logits=logits), logits, 0.0
        mask_flat = mask.reshape(-1, self.action_dim)
        masked_logits = apply_action_mask(logits, mask_flat)
        invalid = (~mask_flat).float().mean().item()
        return Categorical(logits=masked_logits), masked_logits, float(invalid)

    def update(self, batch_data: dict[str, Any]) -> dict[str, float]:
        """Run one COMA-compatible graph-attention update with optional masks."""
        states = torch.as_tensor(batch_data["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch_data["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch_data["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch_data["dones"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch_data["log_probs"], dtype=torch.float32, device=self.device)
        if states.ndim == 2:
            states = states.reshape(-1, self.num_agents, self.state_dim)
        if actions.ndim == 1:
            actions = actions.reshape(-1, self.num_agents)
        if old_log_probs.ndim == 1:
            old_log_probs = old_log_probs.reshape(-1, self.num_agents)

        if self.use_shapley_credit and "shapley_values" in batch_data:
            rewards = apply_shapley_reward_scaling(
                rewards=rewards,
                shapley_values=batch_data["shapley_values"],
                enabled=True,
            )
        if batch_data.get("global_states") is not None:
            global_states = torch.as_tensor(batch_data["global_states"], dtype=torch.float32, device=self.device)
        else:
            global_states = states.reshape(states.shape[0], -1)
        if rewards.ndim > 1 and rewards.shape[-1] == self.num_agents:
            rewards = rewards.mean(dim=-1)
        if dones.ndim > 1 and dones.shape[-1] == self.num_agents:
            dones = dones.any(dim=-1).float()

        mask = self._mask_tensor(batch_data)
        rewards_expanded = rewards.unsqueeze(-1).expand(-1, self.num_agents)

        with torch.no_grad():
            q_values = self.critic(global_states, actions)
            chosen_q = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
            dist, _, _ = self._masked_distribution(states, mask)
            probs = dist.probs.reshape(-1, self.num_agents, self.action_dim)
            counterfactual_baseline = (probs * q_values).sum(dim=-1)
            advantages = rewards_expanded + self.gamma * (1.0 - dones.unsqueeze(-1)) * counterfactual_baseline
            advantages = advantages - chosen_q
            if advantages.numel() > 1 and advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_imitation = 0.0
        mask_fraction = 0.0

        for _ in range(max(1, self.num_epochs)):
            dist, masked_logits, mask_fraction = self._masked_distribution(states, mask)
            new_log_probs = dist.log_prob(actions.reshape(-1)).reshape(-1, self.num_agents)
            entropy = dist.entropy().reshape(-1, self.num_agents).mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, self.policy_clip_low, self.policy_clip_high) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            q_values = self.critic(global_states, actions)
            chosen_q = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
            value_loss = F.mse_loss(chosen_q, rewards_expanded)
            loss = policy_loss + self.critic_loss_coeff * value_loss - self.entropy_coeff * entropy

            imitation_loss = torch.tensor(0.0, device=self.device)
            if self.update_count < self.warm_start_steps and "eq_actions" in batch_data:
                eq_idx = eq_actions_to_discrete_indices(
                    eq_actions=batch_data["eq_actions"],
                    batch_size=states.shape[0],
                    n_agents=self.num_agents,
                    action_dim=self.action_dim,
                    device=self.device,
                )
                if eq_idx is not None:
                    imitation_loss = discrete_imitation_loss(masked_logits, eq_idx)
                    loss = loss + self.warm_start_lr_scale * imitation_loss
            if self.social_influence_enabled and "social_influence" in batch_data:
                influence = torch.as_tensor(batch_data["social_influence"], dtype=torch.float32, device=self.device)
                loss = loss - self.social_influence_coeff * influence.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), self.grad_clip
            )
            self.optimizer.step()

            total_loss += float(loss.item())
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_imitation += float(imitation_loss.detach().item())

        self.update_count += 1
        n = max(1, self.num_epochs)
        return {
            "loss": total_loss / n,
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": 0.0,
            "imitation_loss": total_imitation / n,
            "reward_mean": float(rewards.mean().item()),
            "graph_attention_enabled": 1.0 if self.graph_attention_enabled else 0.0,
            "action_masking_enabled": 1.0 if self.action_masking_enabled else 0.0,
            "mask_fraction": float(mask_fraction),
            "social_influence_enabled": 1.0 if self.social_influence_enabled else 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return actor, graph critic, optimizer, and update counter state."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Load actor, graph critic, optimizer, and update counter state."""
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.update_count = int(sd.get("update_count", 0))
