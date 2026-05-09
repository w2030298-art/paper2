"""Tests for the slim Mainline-A VSCode launch surface."""

import json
from pathlib import Path


EXPECTED_ENTRY_NAMES = [
    "Mainline-A Full 17 Fresh",
    "Resume",
    "Status",
    "Export Results",
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


def test_launch_json_has_no_more_than_eight_entries() -> None:
    """VSCode launch entries should stay slim after the v4.3 migration."""
    configurations = _launch_configurations()
    assert [item["name"] for item in configurations] == EXPECTED_ENTRY_NAMES
    assert len(configurations) <= 8


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
