"""Tests for legacy environment compatibility."""

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv


def test_legacy_default_keeps_mainline_disabled() -> None:
    env = GameTheoryMECEnv(num_agents=1, num_edge_servers=2, max_steps=1)
    obs, info = env.reset(seed=1)

    assert info["mainline_a_enabled"] is False
    assert len(obs[0]) == env.observation_space.shape[0]

