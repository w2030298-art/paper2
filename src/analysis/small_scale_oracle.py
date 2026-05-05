"""Small-scale oracle for mainline-A offloading experiments."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import time
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class OracleResult:
    """Optimal small-scale oracle result."""

    assignment: tuple[int, ...]
    objective_value: float
    constraint_violation: float
    runtime_s: float


def enumerate_offloading_assignments(instance: dict[str, int]) -> Iterable[tuple[int, ...]]:
    """Enumerate discrete offloading assignments for small cases."""
    num_users = int(instance.get("num_users", 0))
    num_edges = int(instance.get("num_edges", 0))
    if num_users > 4 or num_edges > 3:
        raise ValueError("small-scale oracle supports num_users <= 4 and num_edges <= 3")
    return itertools.product(range(num_edges + 1), repeat=num_users)


def solve_small_scale_optimum(
    instance: dict[str, int | float], objective: Callable[[tuple[int, ...]], float] | None = None
) -> OracleResult:
    """Solve a small discrete oracle by enumeration."""
    start = time.time()
    objective_fn = objective or (lambda assignment: -float(sum(assignment)))
    best_assignment: tuple[int, ...] | None = None
    best_value = float("-inf")
    for assignment in enumerate_offloading_assignments(instance):
        value = float(objective_fn(tuple(assignment)))
        if value > best_value:
            best_assignment = tuple(assignment)
            best_value = value
    if best_assignment is None:
        best_assignment = ()
        best_value = 0.0
    return OracleResult(best_assignment, best_value, constraint_violation=0.0, runtime_s=time.time() - start)


def compare_policy_to_oracle(policy_result: dict[str, float], oracle_result: OracleResult) -> dict[str, float]:
    """Compute optimality gap and constraint-violation difference."""
    policy_value = float(policy_result.get("objective_value", 0.0))
    gap = float(oracle_result.objective_value - policy_value)
    violation = float(policy_result.get("constraint_violation", 0.0))
    return {
        "optimality_gap": gap,
        "constraint_violation": violation,
        "oracle_runtime_s": oracle_result.runtime_s,
    }


def export_oracle_gap_report(records: Sequence[dict[str, float]], output_path: str | Path) -> None:
    """Export oracle comparison records."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), indent=2), encoding="utf-8")

