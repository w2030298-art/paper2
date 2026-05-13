"""Tests for the paper2 main matrix runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RUNNER = Path("scripts/run_paper2_main_matrix.py")
MATRIX = Path("configs/paper2_main_experiment_matrix.yaml")
SCENARIOS = Path("configs/paper2_scenario_matrix.yaml")


def _run_runner(output: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    """Run the matrix runner with a small dry-run slice."""
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--scenario-matrix",
            str(SCENARIOS),
            "--level",
            "L1_screening",
            "--scenarios",
            "ID-mainline-a,N2-cl-ppo-ablation",
            "--algorithms",
            "CL-PPO,PPO",
            "--dry-run",
            "--output",
            str(output),
            *extra_args,
        ],
        check=True,
        text=True,
        capture_output=True,
    )


def test_dry_run_writes_deterministic_manifest(tmp_path: Path) -> None:
    """Dry-run should expand the grid without launching training."""
    output = tmp_path / "manifest.json"

    result = _run_runner(output)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert result.stdout.startswith("DRY RUN paper2 main matrix")
    assert manifest["schema_version"] == "paper2-main-matrix-manifest/v1"
    assert manifest["matrix_version"] == "paper2-main-experiment-matrix/v1"
    assert manifest["scenario_version"] == "paper2-scenario-matrix/v1"
    assert manifest["level"] == "L1_screening"
    assert manifest["dry_run"] is True
    assert manifest["claim_level"] == "screening_only"
    assert [run["run_id"] for run in manifest["runs"]] == [
        "L1_screening__ID-mainline-a__CL-PPO__seed42",
        "L1_screening__ID-mainline-a__PPO__seed42",
        "L1_screening__N2-cl-ppo-ablation__CL-PPO__seed42",
    ]


def test_manifest_runs_contain_required_contract_fields(tmp_path: Path) -> None:
    """Each manifest run should contain the dispatch contract fields."""
    output = tmp_path / "manifest.json"

    _run_runner(output)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    for run in manifest["runs"]:
        assert {
            "matrix_version",
            "scenario_version",
            "level",
            "algorithm",
            "scenario",
            "seed",
            "steps",
            "command",
            "output_path",
            "claim_level",
        } <= set(run)
        assert run["steps"] == 50000
        assert run["seed"] == 42
        assert "--dry-run" not in run["command"]
        assert run["command"][:2] == [sys.executable, "scripts/benchmark.py"]


def test_dry_run_is_stable_across_repeated_invocations(tmp_path: Path) -> None:
    """Repeated dry-runs should produce identical manifests apart from output path."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    _run_runner(first)
    _run_runner(second)

    first_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    for payload in (first_payload, second_payload):
        for run in payload["runs"]:
            run["output_path"] = "<normalized>"
            run["command"] = ["<normalized>" if part.endswith(".json") else part for part in run["command"]]

    assert first_payload == second_payload
