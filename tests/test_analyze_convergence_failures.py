"""Tests for convergence failure analysis."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analyze_convergence_failures import build_failure_matrix, write_failure_outputs  # noqa: E402


def _mock_results() -> list[dict]:
    """Return benchmark-like results covering failure categories."""
    return [
        {
            "algorithm": "GRPO",
            "convergence_by_seed": {
                "42": {
                    "seed": 42,
                    "run_status": "success",
                    "eval/reward_mean": [-5.0, -28.0, -31.0, -31.1, -31.0, -31.0],
                    "eval/latency_mean": [0.8, 0.7, 0.68, 0.67, 0.67, 0.67],
                    "eval/energy_mean": [1.2, 1.1, 1.05, 1.04, 1.04, 1.04],
                    "eval/comm_score": [9.0, 8.0, 7.8, 7.8, 7.8, 7.8],
                    "eval_interval": 1000,
                }
            },
        },
        {
            "algorithm": "A3C",
            "convergence_by_seed": {
                "42": {
                    "seed": 42,
                    "run_status": "success",
                    "eval/reward_mean": [-8.0, -4.0, -12.0, -3.0, -11.0, -2.5],
                    "eval/latency_mean": [0.9, 0.5, 0.95, 0.45, 1.0, 0.42],
                    "eval/energy_mean": [1.3, 0.8, 1.4, 0.78, 1.45, 0.75],
                    "eval/comm_score": [4.0, 8.0, 3.0, 9.0, 2.0, 9.5],
                    "eval_interval": 1000,
                }
            },
        },
        {
            "algorithm": "VDN",
            "convergence_by_seed": {
                "42": {
                    "seed": 42,
                    "run_status": "success",
                    "eval/reward_mean": [-4.0, -5.0, -7.0, -10.0, -14.0, -20.0],
                    "eval/latency_mean": [0.5, 0.55, 0.6, 0.7, 0.8, 0.95],
                    "eval/energy_mean": [0.8, 0.85, 0.9, 1.0, 1.1, 1.3],
                    "eval/comm_score": [9.0, 8.5, 8.0, 7.0, 6.0, 5.0],
                    "eval_interval": 1000,
                }
            },
        },
        {
            "algorithm": "IQL",
            "convergence_by_seed": {
                "42": {
                    "seed": 42,
                    "run_status": "success",
                    "eval/reward_mean": [-8.0, -7.0, -65000.0, -6.2, -6.1, -6.0],
                    "eval/latency_mean": [0.9, 0.85, 0.8, 0.78, 0.77, 0.76],
                    "eval/energy_mean": [1.5, 1.4, 1.3, 1.25, 1.24, 1.23],
                    "eval/comm_score": [4.0, 4.5, 5.0, 5.2, 5.3, 5.4],
                    "eval_interval": 1000,
                },
                "43": {
                    "seed": 43,
                    "run_status": "failed",
                    "failure_reason": "mock episode failure",
                    "eval/reward_mean": [-9.0, -8.0, -7.0],
                    "eval/latency_mean": [1.0, 0.9, 0.8],
                    "eval/energy_mean": [1.6, 1.5, 1.4],
                    "eval/comm_score": [3.0, 3.5, 4.0],
                    "eval_interval": 1000,
                },
            },
        },
    ]


def _mock_quality_records() -> list[dict]:
    """Return quality records matching the mock IQL catastrophic outlier."""
    return [
        {
            "algorithm": "IQL",
            "seed": 42,
            "metric": "reward",
            "run_status": "success",
            "severe_outlier": True,
            "outlier_count": 1,
            "outlier_ratio": 0.17,
        },
        {
            "algorithm": "IQL",
            "seed": 43,
            "metric": "reward",
            "run_status": "failed",
            "severe_outlier": False,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
        },
    ]


def test_build_failure_matrix_classifies_new_statuses() -> None:
    """Failure matrix should expose module 12 convergence categories."""
    matrix = build_failure_matrix(_mock_results(), quality_records=_mock_quality_records())
    records = {record["algorithm"]: record for record in matrix["records"]}

    assert records["GRPO"]["metrics"]["reward"]["status"] == "bad_plateau"
    assert records["A3C"]["metrics"]["reward"]["status"] == "oscillating"
    assert records["VDN"]["metrics"]["reward"]["status"] == "diverging"
    assert records["IQL"]["metrics"]["reward"]["status"] == "catastrophic_outlier"
    assert records["IQL"]["failed_seed_count"] == 1
    assert records["IQL"]["requires_event_audit"] is True


def test_write_failure_outputs_creates_json_and_markdown(tmp_path: Path) -> None:
    """Analyzer should write a failure matrix JSON and decision report."""
    matrix = build_failure_matrix(_mock_results(), quality_records=_mock_quality_records())
    json_path = tmp_path / "results" / "convergence_failure_matrix.json"
    md_path = tmp_path / "docs" / "convergence_failure_analysis.md"

    write_failure_outputs(matrix, json_path=json_path, markdown_path=md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert payload["records"]
    assert "IQL" in markdown
    assert "bad_plateau" in markdown
    assert "catastrophic_outlier" in markdown


def test_cli_help() -> None:
    """The analyzer CLI should expose help without reading real experiment data."""
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_convergence_failures.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--quality-report" in completed.stdout
    assert "convergence_failure_matrix" in completed.stdout


def test_stability_overrides_yaml_is_candidate_only() -> None:
    """The stability override file should be parseable and disabled by default."""
    payload = yaml.safe_load(Path("configs/stability_overrides.yaml").read_text(encoding="utf-8"))

    assert payload["enabled"] is False
    assert "value_decomposition" in payload["algorithm_families"]
