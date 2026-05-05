"""Task and DAG helpers for the mainline-A MEC model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Any, Iterable

from .types import TaskId, UserId


@dataclass(frozen=True)
class TaskSegment:
    """One segment in a task DAG."""

    segment_id: str
    data_size_bits: float
    cpu_cycles: float
    deadline_s: float
    predecessors: tuple[str, ...] = ()
    semantic_weight: float = 1.0


@dataclass(frozen=True)
class TaskDAGSpec:
    """A task DAG; a scalar task is represented by one segment."""

    task_id: TaskId
    user_id: UserId
    segments: dict[str, TaskSegment]
    arrival_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskArrivalProcess:
    """Simple seeded arrival-process descriptor."""

    rate_per_user: float = 1.0
    seed: int | None = None

    def sample_count(self, horizon_s: float) -> int:
        """Return a deterministic rounded arrival count for a horizon."""
        return max(0, int(round(float(self.rate_per_user) * float(horizon_s))))


def _single_segment_task(user_id: int, task_index: int, rng: random.Random) -> TaskDAGSpec:
    """Create a scalar task as a one-node DAG."""
    data_size_bits = rng.uniform(0.5e6, 2.0e6) * 8.0
    cpu_cycles = rng.uniform(0.5e9, 2.0e9)
    segment = TaskSegment(
        segment_id="root",
        data_size_bits=data_size_bits,
        cpu_cycles=cpu_cycles,
        deadline_s=rng.uniform(0.5, 2.5),
        predecessors=(),
        semantic_weight=rng.uniform(0.5, 1.0),
    )
    return TaskDAGSpec(
        task_id=TaskId(f"u{user_id}-t{task_index}"),
        user_id=UserId(f"user-{user_id}"),
        segments={"root": segment},
    )


def generate_independent_task_batch(
    num_users: int, tasks_per_user: int, seed: int | None = None
) -> list[TaskDAGSpec]:
    """Generate scalar independent tasks as single-node DAGs."""
    rng = random.Random(seed)
    tasks: list[TaskDAGSpec] = []
    for user_id in range(int(num_users)):
        for task_index in range(int(tasks_per_user)):
            tasks.append(_single_segment_task(user_id, task_index, rng))
    return tasks


def _segment_from_template(raw: dict[str, Any]) -> TaskSegment:
    """Build a segment from a template dictionary."""
    return TaskSegment(
        segment_id=str(raw["segment_id"]),
        data_size_bits=float(raw["data_size_bits"]),
        cpu_cycles=float(raw["cpu_cycles"]),
        deadline_s=float(raw["deadline_s"]),
        predecessors=tuple(str(item) for item in raw.get("predecessors", ())),
        semantic_weight=float(raw.get("semantic_weight", 1.0)),
    )


def generate_dag_task_batch(
    num_users: int, dag_template: Iterable[dict[str, Any]], seed: int | None = None
) -> list[TaskDAGSpec]:
    """Generate one DAG task per user from a segment template."""
    random.Random(seed)  # Keeps the interface explicitly seeded for future variants.
    template_segments = [_segment_from_template(item) for item in dag_template]
    task_specs: list[TaskDAGSpec] = []
    for user_id in range(int(num_users)):
        segments = {
            segment.segment_id: TaskSegment(
                segment_id=segment.segment_id,
                data_size_bits=segment.data_size_bits,
                cpu_cycles=segment.cpu_cycles,
                deadline_s=segment.deadline_s,
                predecessors=segment.predecessors,
                semantic_weight=segment.semantic_weight,
            )
            for segment in template_segments
        }
        spec = TaskDAGSpec(
            task_id=TaskId(f"dag-u{user_id}"),
            user_id=UserId(f"user-{user_id}"),
            segments=segments,
        )
        validate_task_dag(spec)
        task_specs.append(spec)
    return task_specs


def validate_task_dag(task_spec: TaskDAGSpec) -> None:
    """Validate segment fields and acyclicity."""
    if not task_spec.segments:
        raise ValueError("TaskDAGSpec must contain at least one segment.")
    for segment_id, segment in task_spec.segments.items():
        if segment.segment_id != segment_id:
            raise ValueError(f"Segment key mismatch for {segment_id}.")
        if segment.data_size_bits <= 0 or segment.cpu_cycles <= 0 or segment.deadline_s <= 0:
            raise ValueError(f"Segment {segment_id} has non-positive task fields.")
        if segment.semantic_weight <= 0:
            raise ValueError(f"Segment {segment_id} has non-positive semantic_weight.")
        for predecessor in segment.predecessors:
            if predecessor not in task_spec.segments:
                raise ValueError(f"Unknown predecessor {predecessor} for {segment_id}.")
    topological_task_order(task_spec)


def topological_task_order(task_spec: TaskDAGSpec) -> list[str]:
    """Return a topological order or raise on cycles."""
    indegree = {segment_id: 0 for segment_id in task_spec.segments}
    children: dict[str, list[str]] = {segment_id: [] for segment_id in task_spec.segments}
    for segment_id, segment in task_spec.segments.items():
        for predecessor in segment.predecessors:
            children[predecessor].append(segment_id)
            indegree[segment_id] += 1

    ready = deque(sorted(segment_id for segment_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while ready:
        current = ready.popleft()
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(order) != len(task_spec.segments):
        raise ValueError("TaskDAGSpec contains a cycle.")
    return order

