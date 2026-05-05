"""Utility helpers for game-theory-aware RL algorithm integrations."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F


_QUANTIZED_VALUES = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=torch.float32)
_RATIO_COMBOS = 5 ** 3


def to_float_tensor(data: Any, device: torch.device | str) -> Optional[torch.Tensor]:
    """Convert arbitrary numeric input to float tensor on target device."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def align_first_dim(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Repeat/cycle tensor rows so first dimension matches target_rows."""
    if tensor.shape[0] == target_rows:
        return tensor
    if tensor.shape[0] == 0:
        return torch.zeros((target_rows,) + tuple(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    row_idx = torch.arange(target_rows, device=tensor.device) % tensor.shape[0]
    return tensor.index_select(0, row_idx)


def apply_shapley_credit_assignment(
    team_reward: torch.Tensor | float,
    shapley_values: Any,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Redistribute a team reward with the paper-form Shapley credit formula.

    The contract is r_i = phi_i * R_team / sum_j phi_j. When the denominator
    is numerically zero, the team reward is split uniformly across agents.
    """
    if isinstance(team_reward, torch.Tensor):
        device = team_reward.device
        team = team_reward.to(dtype=torch.float32)
    else:
        device = torch.device("cpu")
        team = torch.as_tensor(team_reward, dtype=torch.float32)

    shapley = to_float_tensor(shapley_values, device)
    if shapley is None:
        return team.clone()
    if shapley.numel() == 0:
        return torch.zeros_like(shapley)

    team = team.to(device=device, dtype=shapley.dtype)
    if team.ndim > 0 and shapley.ndim > 1 and team.numel() == shapley.shape[0]:
        team_by_row = team.reshape(shapley.shape[0], *([1] * (shapley.ndim - 1)))
        denom = shapley.reshape(shapley.shape[0], -1).sum(dim=-1)
        denom = denom.reshape(shapley.shape[0], *([1] * (shapley.ndim - 1)))
        uniform = team_by_row / max(float(shapley[0].numel()), 1.0)
        uniform_assignment = torch.ones_like(shapley) * uniform
        use_uniform = torch.abs(denom) <= eps
        safe_denom = torch.where(use_uniform, torch.ones_like(denom), denom)
        proportional_assignment = shapley * team_by_row / safe_denom
        return torch.where(use_uniform, uniform_assignment, proportional_assignment)

    team_scalar = team.sum()
    denom_scalar = shapley.sum()
    if torch.abs(denom_scalar) <= eps:
        return torch.full_like(shapley, team_scalar / float(shapley.numel()))
    return shapley * team_scalar / denom_scalar


def apply_shapley_reward_scaling(
    rewards: torch.Tensor,
    shapley_values: Any,
    enabled: bool,
) -> torch.Tensor:
    """Legacy multiplicative Shapley scaling for historical ablations."""
    if not enabled:
        return rewards
    shapley = to_float_tensor(shapley_values, rewards.device)
    if shapley is None:
        return rewards

    if rewards.ndim == 1:
        if shapley.ndim == 1:
            shapley_vec = align_first_dim(shapley.view(-1, 1), rewards.shape[0]).view(-1)
        else:
            shapley_vec = shapley.view(shapley.shape[0], -1).mean(dim=-1)
            shapley_vec = align_first_dim(shapley_vec.unsqueeze(-1), rewards.shape[0]).view(-1)
        coeff = 0.5 + torch.clamp(shapley_vec, 0.0, 1.0)
        return rewards * coeff

    # Multi-dimensional rewards (typically [batch, n_agents])
    batch_size = rewards.shape[0]
    n_agents = rewards.shape[1]
    if shapley.ndim == 1:
        if shapley.shape[0] == n_agents:
            shapley_mat = shapley.unsqueeze(0).expand(batch_size, n_agents)
        else:
            shapley_mat = align_first_dim(shapley.view(-1, 1), batch_size).expand(batch_size, n_agents)
    else:
        shapley_mat = shapley.view(shapley.shape[0], -1)
        shapley_mat = align_first_dim(shapley_mat, batch_size)
        if shapley_mat.shape[1] < n_agents:
            repeats = int(math.ceil(float(n_agents) / float(max(shapley_mat.shape[1], 1))))
            shapley_mat = shapley_mat.repeat(1, repeats)[:, :n_agents]
        elif shapley_mat.shape[1] > n_agents:
            shapley_mat = shapley_mat[:, :n_agents]
    coeff = 0.5 + torch.clamp(shapley_mat, 0.0, 1.0)
    while coeff.ndim < rewards.ndim:
        coeff = coeff.unsqueeze(-1)
    return rewards * coeff


def apply_legacy_shapley_reward_scaling(
    rewards: torch.Tensor,
    shapley_values: Any,
    enabled: bool,
) -> torch.Tensor:
    """Compatibility alias for the legacy multiplicative Shapley scaling."""
    return apply_shapley_reward_scaling(rewards, shapley_values, enabled)


def inject_game_hints(
    states: torch.Tensor,
    game_hints: Any,
    enabled: bool,
    scale: float = 0.01,
) -> torch.Tensor:
    """Inject scalarized game hints into states/global states."""
    if not enabled:
        return states
    hints = to_float_tensor(game_hints, states.device)
    if hints is None:
        return states
    if hints.ndim == 1:
        hints = hints.unsqueeze(0)
    hints = hints.view(hints.shape[0], -1)
    hint_scalar = hints.mean(dim=-1, keepdim=True)
    hint_scalar = align_first_dim(hint_scalar, states.shape[0])
    while hint_scalar.ndim < states.ndim:
        hint_scalar = hint_scalar.unsqueeze(-1)
    return states + scale * hint_scalar


def prepare_eq_actions_continuous(
    eq_actions: Any,
    batch_size: int,
    action_dim: int,
    device: torch.device | str,
) -> Optional[torch.Tensor]:
    """Prepare equilibrium actions for continuous single-agent imitation loss."""
    eq = to_float_tensor(eq_actions, device)
    if eq is None:
        return None
    if eq.ndim == 1:
        eq = eq.unsqueeze(0)
    eq = eq.view(eq.shape[0], -1)
    if eq.shape[1] < action_dim:
        pad = torch.zeros((eq.shape[0], action_dim - eq.shape[1]), dtype=eq.dtype, device=eq.device)
        eq = torch.cat([eq, pad], dim=-1)
    elif eq.shape[1] > action_dim:
        eq = eq[:, :action_dim]
    return align_first_dim(eq, batch_size)


def prepare_eq_actions_multi_continuous(
    eq_actions: Any,
    batch_size: int,
    n_agents: int,
    action_dim: int,
    device: torch.device | str,
) -> Optional[torch.Tensor]:
    """Prepare equilibrium actions for continuous multi-agent imitation loss."""
    eq = to_float_tensor(eq_actions, device)
    if eq is None:
        return None
    total_dim = n_agents * action_dim

    if eq.ndim == 1:
        eq_flat = eq.unsqueeze(0)
    elif eq.ndim == 2 and eq.shape[0] == n_agents:
        eq_flat = eq.reshape(1, -1)
    elif eq.ndim >= 3:
        eq_flat = eq.reshape(eq.shape[0], -1)
    else:
        eq_flat = eq.view(eq.shape[0], -1)

    if eq_flat.shape[1] < total_dim:
        pad = torch.zeros((eq_flat.shape[0], total_dim - eq_flat.shape[1]), dtype=eq_flat.dtype, device=eq_flat.device)
        eq_flat = torch.cat([eq_flat, pad], dim=-1)
    elif eq_flat.shape[1] > total_dim:
        eq_flat = eq_flat[:, :total_dim]

    eq_flat = align_first_dim(eq_flat, batch_size)
    return eq_flat.view(batch_size, n_agents, action_dim)


def eq_actions_to_discrete_indices(
    eq_actions: Any,
    batch_size: int,
    n_agents: int,
    action_dim: int,
    device: torch.device | str,
) -> Optional[torch.Tensor]:
    """Map equilibrium continuous vectors to discrete action indices."""
    eq = to_float_tensor(eq_actions, device)
    if eq is None:
        return None

    if eq.ndim == 1:
        eq = eq.view(1, 1, -1)
    elif eq.ndim == 2:
        if n_agents > 1 and eq.shape[0] == n_agents:
            eq = eq.unsqueeze(0)
        elif n_agents == 1:
            eq = eq.unsqueeze(1)
        else:
            eq = eq.view(eq.shape[0], 1, -1)
    else:
        eq = eq.view(eq.shape[0], -1, eq.shape[-1])

    if eq.shape[1] != n_agents or eq.shape[2] < 4:
        flat = eq.view(eq.shape[0], -1)
        need = n_agents * 4
        if flat.shape[1] < need:
            pad = torch.zeros((flat.shape[0], need - flat.shape[1]), dtype=flat.dtype, device=flat.device)
            flat = torch.cat([flat, pad], dim=-1)
        elif flat.shape[1] > need:
            flat = flat[:, :need]
        eq = flat.view(flat.shape[0], n_agents, 4)
    else:
        eq = eq[:, :, :4]

    eq = align_first_dim(eq, batch_size)

    n_targets = max(1, action_dim // _RATIO_COMBOS)
    if n_targets <= 1:
        n_targets = max(1, min(action_dim, int(math.sqrt(max(action_dim, 1)))))
    target_max = max(n_targets - 1, 0)

    target = torch.round((torch.clamp(eq[..., 0], -1.0, 1.0) + 1.0) * 0.5 * target_max).long()
    target = torch.clamp(target, 0, max(n_targets - 1, 0))

    ratios = torch.clamp(eq[..., 1:4], -1.0, 1.0)
    quant_vals = _QUANTIZED_VALUES.to(device=ratios.device)
    dist = torch.abs(ratios.unsqueeze(-1) - quant_vals.view(1, 1, 1, -1))
    bins = torch.argmin(dist, dim=-1)
    ratio_idx = bins[..., 0] * 25 + bins[..., 1] * 5 + bins[..., 2]

    idx = target * _RATIO_COMBOS + ratio_idx
    if action_dim > 0:
        idx = torch.remainder(idx, action_dim)
    return idx.long()


def discrete_imitation_loss(
    q_values: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy imitation loss on Q logits."""
    if q_values.ndim == 3:
        return F.cross_entropy(q_values.view(-1, q_values.shape[-1]), target_indices.view(-1))
    return F.cross_entropy(q_values, target_indices.view(-1))
