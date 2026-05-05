"""Tests for cooperative edge topology helpers."""

from src.mec_model.edge_topology import (
    EdgeNodeSpec,
    compute_cooperation_gain,
    compute_migration_cost,
)
from src.mec_model.tasks import TaskDAGSpec, TaskSegment
from src.mec_model.types import NodeId, TaskId, UserId


def test_migration_cost_has_transfer_components() -> None:
    task = TaskDAGSpec(
        task_id=TaskId("t"),
        user_id=UserId("u"),
        segments={"s": TaskSegment("s", 1000.0, 1000.0, 1.0, (), 1.0)},
    )

    cost = compute_migration_cost(task, NodeId("a"), NodeId("b"))

    assert cost.total_cost_s > cost.data_transfer_s


def test_cooperation_gain_positive_for_faster_target() -> None:
    task = TaskDAGSpec(
        task_id=TaskId("t"),
        user_id=UserId("u"),
        segments={"s": TaskSegment("s", 1000.0, 10_000.0, 1.0, (), 1.0)},
    )

    gain = compute_cooperation_gain(
        EdgeNodeSpec(NodeId("slow"), cpu_frequency_hz=1.0),
        EdgeNodeSpec(NodeId("fast"), cpu_frequency_hz=10.0),
        task,
    )

    assert gain > 0.0

