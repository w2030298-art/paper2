"""Tests for experiment preset constants."""

from __future__ import annotations

import json

from src.experiment.presets import (
    FULL_17_ALGORITHMS,
    PRESETS,
    QUICK_ALGORITHMS,
    SINGLE_POLICY_3USER_FULL17_ALGORITHMS,
)
from src.experiment.registry import AlgorithmRegistry


def test_full17_preset_has_exactly_registered_algorithms() -> None:
    assert PRESETS["full17"]["algorithms"] == FULL_17_ALGORITHMS
    assert AlgorithmRegistry.SUPPORTED_ALGORITHMS == FULL_17_ALGORITHMS
    assert len(FULL_17_ALGORITHMS) == 17


def test_quick_preset_is_subset_of_full17() -> None:
    assert PRESETS["quick"]["algorithms"] == QUICK_ALGORITHMS
    assert set(QUICK_ALGORITHMS) < set(FULL_17_ALGORITHMS)


def test_single_policy_3user_preset_is_old_single_agent_subset() -> None:
    assert PRESETS["single_policy_3user_full17"]["algorithms"] == SINGLE_POLICY_3USER_FULL17_ALGORITHMS
    assert SINGLE_POLICY_3USER_FULL17_ALGORITHMS == [
        "GRPO",
        "PPO",
        "SAC",
        "DDQN",
        "DDPG",
        "TD3",
        "A3C",
        "TRPO",
        "SimPO",
    ]
    assert "CL-PPO" not in SINGLE_POLICY_3USER_FULL17_ALGORITHMS


def test_preset_values_are_json_serializable() -> None:
    encoded = json.dumps(PRESETS)
    decoded = json.loads(encoded)
    assert decoded["full17"]["run_id"] == "paper2_full_17_mainline_a"
    assert decoded["full17"]["environment_profile"] == "mainline-a"
    assert decoded["quick"]["environment_profile"] == "mainline-a"
    assert decoded["single_policy_3user_full17"]["interface"] == "single_policy_multi_user"
