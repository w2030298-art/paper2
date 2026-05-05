"""Tests for optional mainline-A environment state."""

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv


def test_mainline_a_enabled_adds_state_and_metrics() -> None:
    legacy = GameTheoryMECEnv(num_agents=1, num_edge_servers=2, max_steps=1)
    mainline = GameTheoryMECEnv(
        num_agents=1,
        num_edge_servers=2,
        max_steps=1,
        enable_mainline_a=True,
    )

    obs, info = mainline.reset(seed=1)

    assert mainline.observation_space.shape[0] > legacy.observation_space.shape[0]
    assert len(obs[0]) == mainline.observation_space.shape[0]
    assert info["mainline_a_enabled"] is True
    assert "mainline_a_reward_components" in info

