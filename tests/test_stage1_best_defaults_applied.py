"""Regression tests for Stage-1 best artifacts becoming default configs."""

from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML mapping from the repository."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_ppo_default_config_matches_stage1_best_artifact() -> None:
    """PPO defaults should stay canonically equal to the saved Stage-1 best config."""
    best = _load_yaml("outputs/stage1/ppo_best_config.yaml")
    default = _load_yaml("configs/algorithms/ppo.yaml")

    assert default == best
    assert default["training"]["total_timesteps"] == 100000
    assert default["stage1_tuning"]["trial_id"] == "0001"
    assert default["stage1_tuning"]["trial_name"] == "sample-0001"


def test_coma_default_config_matches_stage1_best_artifact() -> None:
    """COMA defaults should stay canonically equal to the saved Stage-1 best config."""
    best = _load_yaml("outputs/stage1/coma_best_config.yaml")
    default = _load_yaml("configs/algorithms/coma.yaml")

    assert default == best
    assert default["training"]["total_timesteps"] == 100000
    assert default["stage1_tuning"]["trial_id"] == "0003"
    assert default["stage1_tuning"]["trial_name"] == "sample-0003"
