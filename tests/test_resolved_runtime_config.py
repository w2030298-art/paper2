"""Tests for resolved runtime config serialization."""

from pathlib import Path

from src.experiment.runtime_config import (
    build_resolved_runtime_config,
    serialize_space,
    sha256_file,
)


class DummyBox:
    """Minimal Box-like space for serialization tests."""

    shape = (4,)
    dtype = "float32"
    low = [-1.0, -1.0, -1.0, -1.0]
    high = [1.0, 1.0, 1.0, 1.0]


class DummyDiscrete:
    """Minimal Discrete-like space for serialization tests."""

    n = 7
    dtype = "int64"


class DummyEnv:
    """Minimal environment carrying spaces."""

    observation_space = DummyBox()
    action_space = DummyDiscrete()
    num_agents = 3


class DummyAgent:
    """Minimal agent carrying runtime attributes."""

    state_dim = 4
    action_dim = 7
    action_type = "discrete"
    discrete = True
    num_agents = 3


def test_sha256_file_returns_digest_and_none_for_missing(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_bytes(b"algorithm: PPO\n")

    assert sha256_file(config) == "f8a6c0c936f4ba5b37eb5c556664dace23f9203d36221da0efa67dc91b957947"
    assert sha256_file(tmp_path / "missing.yaml") is None


def test_serialize_space_captures_discrete_and_box_like_metadata() -> None:
    assert serialize_space(DummyDiscrete())["n"] == 7

    box = serialize_space(DummyBox())
    assert box["shape"] == [4]
    assert box["low"] == [-1.0, -1.0, -1.0, -1.0]
    assert box["high"] == [1.0, 1.0, 1.0, 1.0]


def test_build_resolved_runtime_config_excludes_live_objects(tmp_path: Path) -> None:
    config = tmp_path / "ppo.yaml"
    config.write_text("algorithm:\n  name: PPO\n", encoding="utf-8")
    env = DummyEnv()
    agent = DummyAgent()

    payload = build_resolved_runtime_config(
        algorithm="PPO",
        config_path=config,
        base_algorithm_config={"algorithm": {"name": "PPO"}},
        cli_overrides={
            "timesteps": 32,
            "seed": 42,
            "device": "cpu",
            "eval_episodes": 1,
            "env": "auto",
            "environment_profile": "mainline-a",
        },
        environment="MEC-v1-game-theory-continuous-ma",
        environment_profile="mainline-a",
        env_overrides={"enable_mainline_a": True},
        game_theory_config={"enabled": True, "reward_weights": (0.8, 0.1, 0.1)},
        trainer_kwargs={"env": env, "agent": agent, "total_timesteps": 32, "device": "cpu"},
        agent=agent,
        env=env,
        train_timesteps=32,
        eval_episodes=1,
    )

    assert payload["algorithm"] == "PPO"
    assert payload["config_sha256"] == sha256_file(config)
    assert payload["cli_overrides"]["timesteps"] == 32
    assert payload["trainer_kwargs"] == {"total_timesteps": 32, "device": "cpu"}
    assert payload["agent_runtime"]["class_name"] == "DummyAgent"
    assert payload["agent_runtime"]["num_agents"] == 3
    assert payload["action_space"]["n"] == 7
    assert payload["observation_space"]["shape"] == [4]
