"""Tests for the single-policy / multi-user shared-control adapter."""

from __future__ import annotations

import pytest

from scripts.benchmark import ALGO_ENV_MAP, create_agent, load_config, make_env, resolve_game_theory_config
from src.experiment.single_policy_multi_user import (
    SinglePolicyMultiUserRunner,
    SinglePolicyMultiUserSettings,
    expand_transitions,
    normalize_reward_vector,
)


def test_reward_vector_global_and_mean_modes() -> None:
    """Shared reward expansion should be deterministic and per-user shaped."""
    assert normalize_reward_vector(1.5, 3, "mean").tolist() == [1.5, 1.5, 1.5]
    assert normalize_reward_vector([1.0, 2.0, 3.0], 3, "mean").tolist() == [1.0, 2.0, 3.0]
    assert normalize_reward_vector([1.0, 2.0, 3.0], 3, "global").tolist() == [2.0, 2.0, 2.0]


def test_expand_transitions_returns_one_sample_per_user() -> None:
    """One shared env step should expand into three single-policy samples."""
    transitions = expand_transitions(
        observations=[[1.0], [2.0], [3.0]],
        actions=[0, 1, 2],
        rewards=[0.1, 0.2, 0.3],
        next_observations=[[1.1], [2.1], [3.1]],
        done=False,
        num_users=3,
    )
    assert [item["user_index"] for item in transitions] == [0, 1, 2]
    assert [item["reward"] for item in transitions] == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize("algorithm", ["PPO", "DDPG", "DDQN"])
def test_runner_smoke_steps_single_policy_in_three_user_env(algorithm: str) -> None:
    """PPO, DDPG, and DDQN should all produce 3-user joint actions."""
    cfg = load_config(f"configs/algorithms/{algorithm.lower()}.yaml")
    gt_cfg = resolve_game_theory_config(algorithm, cfg)
    env = make_env(
        ALGO_ENV_MAP[algorithm],
        seed=123,
        num_agents=3,
        game_theory_config=gt_cfg,
        env_overrides={"max_steps": 2},
    )
    try:
        agent = create_agent(algorithm, env, cfg, "cpu")
        settings = SinglePolicyMultiUserSettings(enabled=True, num_users=3)
        runner = SinglePolicyMultiUserRunner(env, agent, settings)
        obs, _ = env.reset(seed=123)
        record = runner.collect_step(obs, deterministic=True)
    finally:
        env.close()

    assert len(record.actions) == 3
    assert record.rewards.shape == (3,)
    assert len(record.next_observations) == 3
    assert len(record.transitions) == 3
    assert "individual_latencies" in record.info
    assert "individual_energies" in record.info
