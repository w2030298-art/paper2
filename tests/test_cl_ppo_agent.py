"""Focused smoke tests for the CL-PPO Stage-2 agent."""

from __future__ import annotations

import numpy as np
import torch

from rl_algorithms.cl_ppo import CLPPOAgent


def _synthetic_batch(batch_size: int = 8, state_dim: int = 6, action_dim: int = 4) -> dict[str, np.ndarray]:
    """Build a minimal continuous PPO batch with constraint cost signals."""
    rng = np.random.default_rng(7)
    return {
        "states": rng.normal(size=(batch_size, state_dim)).astype(np.float32),
        "actions": rng.uniform(-0.5, 0.5, size=(batch_size, action_dim)).astype(np.float32),
        "rewards": rng.normal(size=(batch_size,)).astype(np.float32),
        "next_states": rng.normal(size=(batch_size, state_dim)).astype(np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
        "log_probs": np.zeros(batch_size, dtype=np.float32),
        "values": np.zeros(batch_size, dtype=np.float32),
        "deadline_misses": np.linspace(0.0, 0.4, batch_size, dtype=np.float32),
        "energy_costs": np.linspace(0.1, 0.8, batch_size, dtype=np.float32),
        "queue_waits": np.linspace(0.0, 0.6, batch_size, dtype=np.float32),
    }


def test_cl_ppo_select_action_projects_continuous_action() -> None:
    """The safety layer should keep continuous actions inside the configured bounds."""
    agent = CLPPOAgent(
        state_dim=6,
        action_dim=4,
        hidden_dim=16,
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        safety_layer={"enabled": True, "action_low": -1.0, "action_high": 1.0},
    )

    action, info = agent.select_action(np.ones(6, dtype=np.float32), deterministic=False)

    assert action.shape == (4,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
    assert "log_prob" in info
    assert np.all(agent.project_action(np.array([3.0, -2.0, 0.25, 4.0], dtype=np.float32)) <= 1.0)


def test_cl_ppo_update_reports_constraint_risk_and_duals() -> None:
    """One synthetic update should train the risk path and expose dual metrics."""
    agent = CLPPOAgent(
        state_dim=6,
        action_dim=4,
        hidden_dim=16,
        lr=1e-3,
        num_epochs=1,
        device="cpu",
        constraints={
            "enabled": True,
            "dual_lr": 0.05,
            "deadline_miss_budget": 0.05,
            "energy_overshoot_budget": 0.2,
            "queue_overload_budget": 0.1,
        },
        risk={"enabled": True, "cvar_alpha": 0.9, "risk_coeff": 0.2},
    )

    metrics = agent.update(_synthetic_batch())

    for key in (
        "constraint_loss",
        "risk_loss",
        "dual_deadline_miss",
        "dual_energy_overshoot",
        "dual_queue_overload",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["dual_deadline_miss"] >= 0.0


def test_cl_ppo_state_dict_round_trips_risk_and_dual_state() -> None:
    """CL-PPO checkpoints should include the added critic and dual variables."""
    agent = CLPPOAgent(state_dim=6, action_dim=4, hidden_dim=16, num_epochs=1, device="cpu")
    agent.update_dual_variables(
        {
            "deadline_miss": torch.tensor([1.0]),
            "energy_overshoot": torch.tensor([0.5]),
            "queue_overload": torch.tensor([0.25]),
        }
    )

    state = agent.state_dict()
    restored = CLPPOAgent(state_dim=6, action_dim=4, hidden_dim=16, num_epochs=1, device="cpu")
    restored.load_state_dict(state)

    assert "risk_critic" in state
    assert "risk_optimizer" in state
    assert "dual_variables" in state
    assert restored.state_dict()["dual_variables"] == state["dual_variables"]
