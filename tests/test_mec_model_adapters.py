"""Tests for legacy-to-mainline-A adapters."""

from src.environments.mec_v3.game_theory_env import GameTheoryMECEnv
from src.mec_model.adapters import (
    apply_system_decision_to_legacy_env,
    build_system_state_from_legacy_env,
    extract_reward_components,
)


def test_adapter_builds_system_state_from_env() -> None:
    env = GameTheoryMECEnv(num_agents=1, num_edge_servers=2, max_steps=1)
    env.reset(seed=1)

    state = build_system_state_from_legacy_env(env)
    components = extract_reward_components(env, state)
    apply_system_decision_to_legacy_env(env, {"enabled": True})

    assert len(state.edge_nodes) == 2
    assert "delay_cost" in components
    assert env.last_mainline_a_decision["enabled"] is True

