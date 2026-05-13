"""Compatibility tests for the slim VSCode launch contract."""

from tests.test_vscode_launch_mainline_a import (
    EXPECTED_ENTRY_NAMES,
    _launch_configurations,
)


def test_launch_json_is_valid_json() -> None:
    """launch.json should remain parseable and schema-versioned."""
    configurations = _launch_configurations()
    assert isinstance(configurations, list)


def test_launch_json_contains_only_reviewed_mainline_a_and_paper2_entries() -> None:
    """The launch surface should stay exact and reviewed."""
    configurations = _launch_configurations()
    assert [item["name"] for item in configurations] == EXPECTED_ENTRY_NAMES


def test_launch_json_omits_per_algorithm_reset_entries() -> None:
    """Per-algorithm reset actions belong in CLI, not launch.json."""
    names = [item["name"] for item in _launch_configurations()]
    assert all("Reset" not in name for name in names)
