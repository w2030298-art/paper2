"""Tests for the small-scale oracle."""

import json

import pytest

from src.analysis.small_scale_oracle import (
    compare_policy_to_oracle,
    enumerate_offloading_assignments,
    export_oracle_gap_report,
    solve_small_scale_optimum,
)


def test_small_scale_oracle_enumerates_and_compares() -> None:
    instance = {"num_users": 2, "num_edges": 2}

    assignments = list(enumerate_offloading_assignments(instance))
    result = solve_small_scale_optimum(instance, objective=lambda a: -sum(a))
    gap = compare_policy_to_oracle({"objective_value": result.objective_value - 1}, result)

    assert len(assignments) == 9
    assert gap["optimality_gap"] == 1.0


def test_small_scale_oracle_rejects_large_cases() -> None:
    with pytest.raises(ValueError):
        list(enumerate_offloading_assignments({"num_users": 5, "num_edges": 2}))


def test_oracle_gap_report_schema_is_stable(tmp_path) -> None:
    """Oracle-gap reports should keep a machine-readable envelope."""
    report_path = tmp_path / "oracle_gap_report.json"
    export_oracle_gap_report(
        [
            {
                "stage": "N1",
                "seed": 42,
                "num_users": 2,
                "num_edges": 2,
                "policy_label": "baseline_static_stackelberg",
                "oracle_gap": 0.25,
                "constraint_violation": 0.0,
                "runtime_s": 0.01,
            }
        ],
        report_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["record_count"] == 1
    assert payload["records"][0]["stage"] == "N1"
    assert payload["records"][0]["oracle_gap"] == 0.25
