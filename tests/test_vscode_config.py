"""Structure tests for VSCode experiment entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FULL_17_ALGORITHMS = [
    "GRPO",
    "PPO",
    "SAC",
    "DDQN",
    "DDPG",
    "TD3",
    "A3C",
    "TRPO",
    "SimPO",
    "MAPPO",
    "QMIX",
    "COMA",
    "IPPO",
    "VDN",
    "MADDPG",
    "IQL",
    "MATD3",
]

REQUIRED_DEBUG_ENTRIES = {
    "🧪 Experiment Quick Start/Resume",
    "🧪 Experiment Quick Fresh Clean Run",
    "🏁 Experiment Full 17 Start/Resume",
    "▶️ Experiment Full 17 Resume",
    "⏹ Experiment Full 17 Stop",
    "📊 Experiment Full 17 Status",
    "📋 Experiment List All",
    "📦 Experiment Full 17 Export Results",
    "♻️ Experiment Quick Reset GRPO",
    "🧱 Experiment Rebuild Index",
    "🏁 Benchmark Direct All 17",
}

REQUIRED_TASKS = {
    "experiment: full17 start-resume",
    "experiment: full17 status",
    "experiment: full17 stop",
    "experiment: full17 export",
    "experiment: quick start-resume",
    "experiment: quick fresh",
    "experiment: quick status",
    "experiment: quick stop",
    "experiment: quick export",
    "experiment: list",
    "experiment: rebuild-index",
    "benchmark: direct all17",
}


def _load_launch_json() -> dict[str, Any]:
    return json.loads(Path(".vscode/launch.json").read_text(encoding="utf-8"))


def _configuration_by_name(name: str) -> dict[str, Any]:
    launch_json = _load_launch_json()
    configurations = launch_json["configurations"]
    return next(item for item in configurations if item["name"] == name)


def _load_tasks_json_if_present() -> dict[str, Any] | None:
    path = Path(".vscode/tasks.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_launch_json_is_valid_json() -> None:
    launch_json = _load_launch_json()
    assert launch_json["version"] == "0.2.0"
    assert isinstance(launch_json["configurations"], list)


def test_launch_json_contains_required_debug_entries() -> None:
    launch_json = _load_launch_json()
    names = {item["name"] for item in launch_json["configurations"]}
    assert REQUIRED_DEBUG_ENTRIES <= names
    for algorithm in FULL_17_ALGORITHMS:
        assert f"♻️ Experiment Full 17 Reset {algorithm}" in names


def test_quick_entry_uses_experiment_manager_start() -> None:
    configuration = _configuration_by_name("🧪 Experiment Quick Start/Resume")
    assert configuration["program"] == "${workspaceFolder}/scripts/experiment_manager.py"
    assert configuration["args"] == ["start", "--preset", "quick"]


def test_full_17_entry_contains_all_algorithms() -> None:
    configuration = _configuration_by_name("🏁 Experiment Full 17 Start/Resume")
    assert configuration["program"] == "${workspaceFolder}/scripts/experiment_manager.py"
    assert configuration["args"] == ["start", "--preset", "full17"]


def test_tasks_json_is_valid_json_if_present() -> None:
    tasks_json = _load_tasks_json_if_present()
    if tasks_json is None:
        return
    assert tasks_json["version"] == "2.0.0"
    assert isinstance(tasks_json["tasks"], list)


def test_tasks_json_contains_required_tasks_if_present() -> None:
    tasks_json = _load_tasks_json_if_present()
    if tasks_json is None:
        return
    labels = {item["label"] for item in tasks_json["tasks"]}
    assert REQUIRED_TASKS <= labels
