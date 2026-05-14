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
        "L1_screening__ID-mainline-a__CL-PPO__full__seed42",
        "L1_screening__ID-mainline-a__PPO__full__seed42",
        "L1_screening__N2-cl-ppo-ablation__CL-PPO__full__seed42",
        "L1_screening__N2-cl-ppo-ablation__CL-PPO__no_constraint_signal__seed42",
        "L1_screening__N2-cl-ppo-ablation__CL-PPO__no_risk_critic__seed42",
        "L1_screening__N2-cl-ppo-ablation__CL-PPO__no_safety_layer__seed42",
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
            "stage",
            "algorithm",
            "scenario",
            "seed",
            "steps",
            "ablation",
            "source_config",
            "profile",
            "result_kind",
            "command",
            "output_path",
            "claim_level",
            "generated_config_path",
        } <= set(run)
        assert run["steps"] == 50000
        assert run["seed"] == 42
        assert "--dry-run" not in run["command"]
        assert run["command"][:2] == [sys.executable, "scripts/benchmark.py"]
        assert "--configs-dir" in run["command"]


def test_full_l1_dry_run_expands_n2_ablation_dimensions(tmp_path: Path) -> None:
    """The default L1 matrix should expand all v4.9 scenario dimensions."""
    output = tmp_path / "full_manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--scenario-matrix",
            str(SCENARIOS),
            "--level",
            "L1_screening",
            "--dry-run",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    n2_runs = [run for run in manifest["runs"] if run["stage"] == "N2"]
    n1_runs = [run for run in manifest["runs"] if run["stage"] == "N1"]

    assert manifest["run_count"] == 72
    assert {(run["scenario"], run["ablation"]) for run in n2_runs} == {
        ("N2-cl-ppo-ablation", "full"),
        ("N2-cl-ppo-ablation", "no_constraint_signal"),
        ("N2-cl-ppo-ablation", "no_risk_critic"),
        ("N2-cl-ppo-ablation", "no_safety_layer"),
        ("N2-gam-coma-ablation", "full"),
        ("N2-gam-coma-ablation", "no_graph_attention"),
        ("N2-gam-coma-ablation", "no_feasible_action_masking"),
        ("N2-gam-coma-ablation", "no_warm_start"),
        ("N2-gam-coma-ablation", "no_shapley_credit"),
    }
    assert {run["result_kind"] for run in n1_runs} == {"benchmark_diagnostic"}


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
