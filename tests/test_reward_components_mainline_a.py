"""Tests for mainline-A reward components."""

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv


def test_mainline_reward_components_are_interpretable() -> None:
    env = GameTheoryMECEnv(num_agents=1, num_edge_servers=2, max_steps=1, enable_mainline_a=True)
    env.reset(seed=2)

    components = env._compute_mainline_a_metrics([])

    assert "delay_cost" in components
    assert "provider_revenue" in components
    assert "constraint_penalty" in components

