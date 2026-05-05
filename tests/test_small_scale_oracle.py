"""Tests for the small-scale oracle."""

import pytest

from src.analysis.small_scale_oracle import (
    compare_policy_to_oracle,
    enumerate_offloading_assignments,
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

