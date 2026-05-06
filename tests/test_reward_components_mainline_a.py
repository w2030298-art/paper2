"""Tests for mainline-A reward components."""

import numpy as np

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv


def test_mainline_reward_components_reflect_step_behavior() -> None:
    env = GameTheoryMECEnv(num_agents=1, num_edge_servers=2, max_steps=1, enable_mainline_a=True)
    env.reset(seed=2)
    _, _, _, _, info = env.step([{"target": 1, "ratio": np.array([1.0, 1.0, 0.0], dtype=np.float32)}])

    components = info["mainline_a_reward_components"]

    assert "delay_cost" in components
    assert "provider_revenue" in components
    assert "constraint_penalty" in components
    assert components["price_payment"] > 0.0
    assert components["delay_cost"] >= 0.0
    assert info["reward_terms"][0]["mainline_a_price_cost"] > 0.0
