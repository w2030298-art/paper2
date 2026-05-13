"""Tests for the slim Mainline-A VSCode launch surface."""

import json
from pathlib import Path


EXPECTED_ENTRY_NAMES = [
    "Mainline-A Full 17 Fresh",
    "Resume",
    "Status",
    "Backup Full17 Mainline-A",
    "Stage-1 Tune PPO Starter",
    "Stage-1 Tune COMA Starter",
    "Direct Benchmark Dry Run",
    "Plot Latest",
    "Legacy Full 17 Fresh (Explicit Fallback)",
    "Experiment List",
]


def _launch_configurations() -> list[dict]:
    payload = json.loads(Path(".vscode/launch.json").read_text(encoding="utf-8"))
    assert payload["version"] == "0.2.0"
    return payload["configurations"]


def _by_name(name: str) -> dict:
    return next(item for item in _launch_configurations() if item["name"] == name)


def test_launch_json_has_exact_review_repair_entries() -> None:
    """VSCode launch entries should match the v4.5 manual launch surface."""
    configurations = _launch_configurations()
    assert [item["name"] for item in configurations] == EXPECTED_ENTRY_NAMES
    assert len(configurations) == len(EXPECTED_ENTRY_NAMES)
    assert "Export Results" not in [item["name"] for item in configurations]


def test_mainline_a_full17_entry_is_default_profile() -> None:
    """Full17 fresh launch should use the Mainline-A profile explicitly."""
    configuration = _by_name("Mainline-A Full 17 Fresh")
    assert configuration["program"] == "${workspaceFolder}/scripts/experiment_manager.py"
    assert configuration["args"] == [
        "start",
        "--preset",
        "full17",
        "--environment-profile",
        "mainline-a",
        "--fresh",
    ]


def test_direct_benchmark_dry_run_uses_mainline_a_profile() -> None:
    """Direct benchmark entry should be a dry-run preflight."""
    configuration = _by_name("Direct Benchmark Dry Run")
    assert configuration["program"] == "${workspaceFolder}/scripts/benchmark.py"
    assert configuration["args"] == ["--all", "--environment-profile", "mainline-a", "--dry-run"]


def test_backup_full17_mainline_a_uses_backup_script() -> None:
    """Manual backup entry should call backup_experiment.py for the Mainline-A run."""
    configuration = _by_name("Backup Full17 Mainline-A")
    assert configuration["program"] == "${workspaceFolder}/scripts/backup_experiment.py"
    assert configuration["args"] == [
        "--run-id",
        "paper2_full_17_mainline_a",
        "--suffix",
        "backup",
        "--require-existing",
    ]


def test_stage1_ppo_starter_launch_invokes_tuner() -> None:
    """PPO starter launch should run the Stage-1 tuner with the PPO search config."""
    configuration = _by_name("Stage-1 Tune PPO Starter")
    assert configuration["program"] == "${workspaceFolder}/scripts/tune_mainline_a_stage1.py"
    assert configuration["args"] == [
        "--algorithm",
        "PPO",
        "--search-config",
        "configs/tuning/stage1_ppo_mainline_a.yaml",
        "--mode",
        "starter",
        "--trials",
        "4",
        "--timesteps",
        "10000",
        "--seeds",
        "42",
        "--environment-profile",
        "mainline-a",
        "--device",
        "auto",
        "--output-dir",
        "outputs/stage1",
    ]


def test_stage1_coma_starter_launch_invokes_tuner() -> None:
    """COMA starter launch should run the Stage-1 tuner with the COMA search config."""
    configuration = _by_name("Stage-1 Tune COMA Starter")
    assert configuration["program"] == "${workspaceFolder}/scripts/tune_mainline_a_stage1.py"
    assert configuration["args"] == [
        "--algorithm",
        "COMA",
        "--search-config",
        "configs/tuning/stage1_coma_mainline_a.yaml",
        "--mode",
        "starter",
        "--trials",
        "4",
        "--timesteps",
        "10000",
        "--seeds",
        "42",
        "--environment-profile",
        "mainline-a",
        "--device",
        "auto",
        "--output-dir",
        "outputs/stage1",
    ]


def test_legacy_entry_is_explicit_fallback() -> None:
    """Legacy launch should require an explicit profile and fallback run id."""
    configuration = _by_name("Legacy Full 17 Fresh (Explicit Fallback)")
    assert configuration["args"] == [
        "start",
        "--preset",
        "full17",
        "--environment-profile",
        "legacy",
        "--run-id",
        "paper2_full_17_legacy_fallback",
        "--fresh",
    ]


def test_launch_json_has_no_per_algorithm_reset_entries() -> None:
    """Per-algorithm reset entries should stay out of launch.json."""
    names = [item["name"] for item in _launch_configurations()]
    assert all("Reset" not in name for name in names)
