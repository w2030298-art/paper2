"""Tests for portable Stage-1 search audit provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


AUDIT_PATH = Path("outputs/stage1/search_audit.json")
LOCAL_ABSOLUTE_PREFIXES = ("/mnt/", "/home/")
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _audit() -> dict[str, Any]:
    """Load the committed Stage-1 search audit artifact."""
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_search_audit_has_runtime_evidence_for_ppo_and_coma() -> None:
    """PPO/COMA audit entries should preserve runtime evidence from starter pilots."""
    payload = _audit()

    for algorithm in ("PPO", "COMA"):
        assert payload[algorithm]["action_space"]
        assert payload[algorithm]["agent_runtime"]


def test_search_audit_commands_are_portable() -> None:
    """Committed audit commands must not include workstation-specific interpreter paths."""
    payload = _audit()

    for algorithm in ("PPO", "COMA"):
        commands = payload[algorithm]["commands"]
        assert commands
        for command_record in commands:
            command = command_record["command"]
            assert command[0] == "python"
            assert all(not token.startswith(LOCAL_ABSOLUTE_PREFIXES) for token in command)
            assert all(WINDOWS_DRIVE_PATH.match(token) is None for token in command)


def test_search_audit_describes_starter_sampling_without_adaptive_overclaim() -> None:
    """Starter audit metadata should not claim adaptive BO/TPE or pruning feedback."""
    payload = _audit()

    for algorithm in ("PPO", "COMA"):
        audit = payload[algorithm]
        assert audit["sampling_mode"] == ["deterministic_random", "recommended_start"]
        assert audit["sampling_semantics"] == {
            "mode": "starter",
            "adaptive_feedback": False,
            "description": "recommended_start plus deterministic_random samples; no objective-feedback BO/TPE or pruning loop",
        }
