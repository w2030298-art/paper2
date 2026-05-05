"""Deterministic contracts for non-simulation paper/code alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_algorithms.game_theory_utils import (  # noqa: E402
    apply_shapley_credit_assignment,
    apply_shapley_reward_scaling,
)
from src.environments.mec_v3.game_theory_env import (  # noqa: E402
    Channel3GPP,
    DVFSEnergyModel,
    EFXFairAllocation,
    QueueingDelayModel,
)


def test_shapley_credit_assignment_satisfies_efficiency() -> None:
    """Paper-form Shapley credit must exactly redistribute the team reward."""
    shapley_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    assigned = apply_shapley_credit_assignment(
        team_reward=torch.tensor(12.0),
        shapley_values=shapley_values,
    )

    assert torch.allclose(assigned, torch.tensor([2.0, 4.0, 6.0]))
    assert assigned.sum().item() == pytest.approx(12.0)


def test_shapley_credit_assignment_near_zero_fallback_is_uniform() -> None:
    """Degenerate Shapley denominators fall back to equal team-reward sharing."""
    shapley_values = torch.tensor([1e-12, -1e-12, 0.0], dtype=torch.float32)
    assigned = apply_shapley_credit_assignment(
        team_reward=torch.tensor(9.0),
        shapley_values=shapley_values,
    )

    assert torch.allclose(assigned, torch.tensor([3.0, 3.0, 3.0]))
    assert assigned.sum().item() == pytest.approx(9.0)


def test_shapley_credit_assignment_supports_batched_team_rewards() -> None:
    """Batched team rewards are redistributed row by row."""
    team_rewards = torch.tensor([6.0, 12.0], dtype=torch.float32)
    shapley_values = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float32)

    assigned = apply_shapley_credit_assignment(team_rewards, shapley_values)

    assert torch.allclose(assigned, torch.tensor([[2.0, 4.0], [9.0, 3.0]]))
    assert torch.allclose(assigned.sum(dim=1), team_rewards)


def test_legacy_shapley_reward_scaling_keeps_old_coefficients() -> None:
    """The old multiplicative scaling remains available for legacy ablations."""
    rewards = torch.tensor([10.0, 10.0], dtype=torch.float32)
    shapley_values = torch.tensor([0.2, 1.5], dtype=torch.float32)

    scaled = apply_shapley_reward_scaling(rewards, shapley_values, enabled=True)

    assert torch.allclose(scaled, torch.tensor([7.0, 15.0]))


def test_efx_repair_transfers_are_capped() -> None:
    """EFX transfer payments returned by repair_allocation are capped to [-0.05, 0.05]."""
    allocation = [set(), {"a", "b"}]

    def agent_zero_value(bundle: set[str]) -> float:
        return 20.0 if len(bundle) == 2 else 10.0 if len(bundle) == 1 else 0.0

    valuations = [agent_zero_value, lambda bundle: 0.0]
    allocator = EFXFairAllocation(n_agents=2, transfer_rate=1.0, max_iters=4)

    result = allocator.repair_allocation(allocation, valuations)

    assert np.max(np.abs(result.transfers)) <= 0.05 + 1e-8
    assert np.allclose(result.transfers, np.clip(result.transfers, -0.05, 0.05))


def test_queue_channel_dvfs_basic_unit_contracts() -> None:
    """Queue, channel, and DVFS contracts use bits/s, W, dB path loss, and linear SINR."""
    queue = QueueingDelayModel(stability_margin=0.99)
    arrival_bits_per_s = np.array([1.0e6, 1.8e6], dtype=np.float32)
    service_bits_per_s = np.array([2.0e6, 2.0e6], dtype=np.float32)
    queue_delay, system_delay, rho = queue.compute_delay(arrival_bits_per_s, service_bits_per_s)

    assert np.all(rho < 1.0)
    assert rho[1] > rho[0]
    assert queue_delay[1] > queue_delay[0]
    assert system_delay[1] > system_delay[0]

    channel = Channel3GPP(fc_ghz=3.5, bandwidth=20e6)
    assert channel.composite_path_loss(100.0) > channel.composite_path_loss(20.0)
    no_interference = channel.compute_sinr(30.0, np.array([], dtype=np.float32), 0.2, np.array([], dtype=np.float32))
    with_interference = channel.compute_sinr(
        30.0,
        np.array([30.0], dtype=np.float32),
        0.2,
        np.array([0.2], dtype=np.float32),
    )
    assert no_interference > with_interference > 0.0
    assert channel.rate(with_interference) > 0.0

    dvfs = DVFSEnergyModel(f_min=0.5e9, f_max=3.0e9)
    f_low = dvfs.optimal_frequency(delay_weight=0.0)
    f_high = dvfs.optimal_frequency(delay_weight=10.0)
    assert dvfs.f_min <= f_low <= dvfs.f_max
    assert dvfs.f_min <= f_high <= dvfs.f_max
    assert f_high >= f_low
