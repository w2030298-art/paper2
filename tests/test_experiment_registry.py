"""Tests for experiment algorithm registry."""

import pytest

from pathlib import Path

from src.experiment.models import AlgorithmSpec
from src.experiment.registry import AlgorithmRegistry, TrainCommandBuilder


def test_canonicalize_algorithm_names() -> None:
    registry = AlgorithmRegistry()
    assert registry.canonicalize("simpo") == "SimPO"
    assert registry.canonicalize("mappo") == "MAPPO"
    assert registry.canonicalize("grpo") == "GRPO"


def test_validate_rejects_unknown_algorithm() -> None:
    registry = AlgorithmRegistry()
    with pytest.raises(ValueError, match=r"Unsupported algorithm: unknown"):
        registry.validate(["unknown"])


def test_config_path_for_known_algorithms() -> None:
    registry = AlgorithmRegistry()
    assert registry.config_path_for("GRPO") == "configs/algorithms/grpo.yaml"
    assert registry.config_path_for("SimPO") == "configs/algorithms/simpo.yaml"
    assert registry.config_path_for("MADDPG") == "configs/algorithms/maddpg.yaml"


def test_build_specs_preserves_order() -> None:
    registry = AlgorithmRegistry()
    specs = registry.build_specs(
        ["ppo", "grpo", "sac"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )
    assert [spec.name for spec in specs] == ["PPO", "GRPO", "SAC"]


def test_train_command_builder_contains_result_json() -> None:
    builder = TrainCommandBuilder(project_root=".")
    spec = AlgorithmSpec(
        name="GRPO",
        config_path="configs/algorithms/grpo.yaml",
        timesteps=5000,
        seed=42,
    )
    command = builder.build(run_id="demo", spec=spec, experiment_dir=Path("experiments/demo"))

    assert "--result-json" in command
    result_json = command[command.index("--result-json") + 1].replace("\\", "/")
    assert result_json.endswith("experiments/demo/artifacts/GRPO/result.json")
