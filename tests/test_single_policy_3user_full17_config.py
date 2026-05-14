"""Config and dry-run tests for the v5.0 single-policy 3-user full17 rerun."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from src.experiment.presets import PRESETS, SINGLE_POLICY_3USER_FULL17_ALGORITHMS

CONFIG_PATH = Path("configs/benchmark_mainline_a_single_policy_3user_full17.yaml")
EXPECTED_ALGORITHMS = ["GRPO", "PPO", "SAC", "DDQN", "DDPG", "TD3", "A3C", "TRPO", "SimPO"]


def test_config_fixes_single_policy_3user_full17_conditions() -> None:
    """The v5.0 config should preserve full17 conditions under the new interface."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["algorithms"] == EXPECTED_ALGORITHMS
    assert payload["benchmark"]["algorithms"] == EXPECTED_ALGORITHMS
    assert "CL-PPO" not in payload["algorithms"]
    assert payload["benchmark"]["steps"] == 100000
    assert payload["benchmark"]["seeds"] == [42]
    assert payload["benchmark"]["eval_episodes"] == 10
    assert payload["benchmark"]["device"] == "auto"
    assert payload["benchmark"]["environment_profile"] == "mainline-a"
    assert payload["single_policy_multi_user"] == {
        "enabled": True,
        "interface": "single_policy_multi_user",
        "num_users": 3,
        "shared_reward": "mean",
        "description": payload["single_policy_multi_user"]["description"],
    }


def test_preset_exposes_single_policy_3user_full17() -> None:
    """The shared preset should expose the same 9 algorithms and run id."""
    preset = PRESETS["single_policy_3user_full17"]
    assert SINGLE_POLICY_3USER_FULL17_ALGORITHMS == EXPECTED_ALGORITHMS
    assert preset["run_id"] == "paper2_single_policy_3user_full17_mainline_a"
    assert preset["algorithms"] == EXPECTED_ALGORITHMS
    assert preset["timesteps"] == 100000
    assert preset["seed"] == 42
    assert preset["eval_episodes"] == 10
    assert preset["environment_profile"] == "mainline-a"
    assert preset["interface"] == "single_policy_multi_user"
    assert preset["num_users"] == 3


def test_dry_run_lists_nine_three_user_single_policy_runs() -> None:
    """benchmark.py dry-run should make the corrected interface explicit."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--no-latest-alias",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = yaml.safe_load(completed.stdout.split("DRY RUN benchmark", 1)[1])
    assert payload["interface"] == "single_policy_multi_user"
    assert payload["algorithms"] == EXPECTED_ALGORITHMS
    assert payload["timesteps"] == 100000
    assert payload["eval_episodes"] == 10
    assert payload["single_policy_multi_user"]["num_users"] == 3
    assert len(payload["runs"]) == 9
    assert {run["interface"] for run in payload["runs"]} == {"single_policy_multi_user"}
    assert {run["num_agents"] for run in payload["runs"]} == {3}
