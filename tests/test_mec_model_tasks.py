"""Tests for mainline-A task DAG helpers."""

import pytest

from src.mec_model.tasks import (
    TaskDAGSpec,
    TaskSegment,
    generate_dag_task_batch,
    generate_independent_task_batch,
    topological_task_order,
    validate_task_dag,
)
from src.mec_model.types import TaskId, UserId


def test_independent_task_is_single_node_dag() -> None:
    tasks = generate_independent_task_batch(num_users=2, tasks_per_user=1, seed=1)

    assert len(tasks) == 2
    assert list(tasks[0].segments) == ["root"]
    validate_task_dag(tasks[0])


def test_topological_order_rejects_cycle() -> None:
    spec = TaskDAGSpec(
        task_id=TaskId("cyclic"),
        user_id=UserId("u0"),
        segments={
            "a": TaskSegment("a", 1.0, 1.0, 1.0, ("b",), 1.0),
            "b": TaskSegment("b", 1.0, 1.0, 1.0, ("a",), 1.0),
        },
    )

    with pytest.raises(ValueError, match="cycle"):
        topological_task_order(spec)


def test_generate_dag_from_template() -> None:
    tasks = generate_dag_task_batch(
        num_users=1,
        dag_template=[
            {"segment_id": "a", "data_size_bits": 1, "cpu_cycles": 1, "deadline_s": 1},
            {
                "segment_id": "b",
                "data_size_bits": 1,
                "cpu_cycles": 1,
                "deadline_s": 1,
                "predecessors": ["a"],
            },
        ],
        seed=7,
    )

    assert topological_task_order(tasks[0]) == ["a", "b"]

