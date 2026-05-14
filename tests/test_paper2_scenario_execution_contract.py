"""Scenario execution contract tests for the paper2 v4.9 matrix runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RUNNER = Path("scripts/run_paper2_main_matrix.py")
MATRIX = Path("configs/paper2_main_experiment_matrix.yaml")
SCENARIOS = Path("configs/paper2_scenario_matrix.yaml")

REQUIRED_RUN_FIELDS = {
    "stage",
    "scenario",
    "algorithm",
    "seed",
    "steps",
    "ablation",
    "source_config",
    "profile",
    "result_kind",
    "command",
    "output_path",
    "claim_level",
}


def test_l1_contract_manifest_has_execution_ready_metadata(tmp_path: Path) -> None:
    """Every run should carry enough metadata for strict collection and review."""
    output = tmp_path / "v49_l1_contract_manifest.json"

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

    assert manifest["schema_version"] == "paper2-main-matrix-manifest/v1"
    assert manifest["run_count"] == 72
    for run in manifest["runs"]:
        assert REQUIRED_RUN_FIELDS <= set(run)
        assert Path(run["generated_config_path"]).name == "algorithm.yaml"
        assert Path(run["output_path"]).suffix == ".json"


def test_n1_oracle_boundary_is_explicitly_not_a_local_oracle_claim(tmp_path: Path) -> None:
    """N1 records should not masquerade as oracle outputs without an oracle wrapper."""
    output = tmp_path / "manifest.json"

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
            "--scenarios",
            "N1-oracle-small",
            "--dry-run",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert manifest["claim_boundaries"]["n1_oracle_claim"] == "blocked_without_oracle_wrapper"
    assert all(run["result_kind"] == "benchmark_diagnostic" for run in manifest["runs"])
    assert all("oracle" not in run["result_kind"] for run in manifest["runs"])
    assert Path("ref/paper2_n1_oracle_gap.md").is_file()
