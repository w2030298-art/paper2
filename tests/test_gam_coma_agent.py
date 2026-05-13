"""Focused smoke tests for the GAM-COMA Stage-2 agent."""

from __future__ import annotations

import numpy as np
import torch

from rl_algorithms.gam_coma import GAMCOMAAgent, GraphAttentionCOMACritic, apply_action_mask


def test_graph_attention_coma_critic_outputs_per_agent_q_values() -> None:
    """The graph critic should return COMA-compatible per-agent Q values."""
    critic = GraphAttentionCOMACritic(
        global_state_dim=18,
        num_agents=3,
        state_dim=6,
        action_dim=5,
        hidden_dim=16,
        num_heads=2,
        dropout=0.0,
    )

    q_values = critic(
        torch.randn(4, 18),
        torch.tensor([[0, 1, 2], [2, 3, 4], [1, 1, 1], [4, 0, 2]], dtype=torch.long),
    )

    assert q_values.shape == (4, 3, 5)


def test_apply_action_mask_suppresses_invalid_actions_without_nans() -> None:
    """Invalid discrete actions should receive a large negative logit."""
    logits = torch.zeros(2, 3, 4)
    mask = torch.tensor(
        [
            [[1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 1]],
            [[1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 1]],
        ],
        dtype=torch.bool,
    )

    masked = apply_action_mask(logits, mask)

    assert torch.isfinite(masked).all()
    assert masked[0, 0, 1] < -1.0e8
    assert masked[1, 1, 0] == 0.0


def test_gam_coma_update_accepts_optional_action_masks() -> None:
    """GAM-COMA should run one masked synthetic CTDE update."""
    batch_size = 5
    num_agents = 3
    state_dim = 6
    action_dim = 5
    rng = np.random.default_rng(11)
    actions = rng.integers(0, action_dim, size=(batch_size, num_agents), dtype=np.int64)
    masks = np.ones((batch_size, num_agents, action_dim), dtype=bool)
    masks[:, :, -1] = False
    masks[np.arange(batch_size)[:, None], np.arange(num_agents)[None, :], actions] = True

    agent = GAMCOMAAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=16,
        lr=1e-3,
        gamma=0.99,
        num_agents=num_agents,
        num_epochs=1,
        discrete=True,
        device="cpu",
        graph_attention={"enabled": True, "num_heads": 2, "dropout": 0.0},
        action_masking={"enabled": True},
    )
    metrics = agent.update(
        {
            "states": rng.normal(size=(batch_size, num_agents, state_dim)).astype(np.float32),
            "global_states": rng.normal(size=(batch_size, num_agents * state_dim)).astype(np.float32),
            "actions": actions,
            "rewards": rng.normal(size=(batch_size,)).astype(np.float32),
            "dones": np.zeros(batch_size, dtype=np.float32),
            "log_probs": np.zeros((batch_size, num_agents), dtype=np.float32),
            "action_masks": masks,
        }
    )

    for key in ("policy_loss", "value_loss", "entropy", "graph_attention_enabled", "mask_fraction"):
        assert key in metrics
    assert metrics["graph_attention_enabled"] == 1.0
    assert 0.0 < metrics["mask_fraction"] < 1.0
