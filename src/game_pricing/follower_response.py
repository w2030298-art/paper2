"""Follower demand response helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import PriceVector


@dataclass(frozen=True)
class FollowerConstraints:
    """Follower feasibility constraints."""

    budget: float = 10.0
    deadline_s: float = 1.0
    local_cpu_hz: float = 1.0e9


@dataclass(frozen=True)
class FollowerResponse:
    """Projected user response to prices."""

    demand: tuple[float, ...]
    selected_edge: int
    feasible: bool
    projected: bool = False


def compute_demand_elasticity(user_state: Any, price_vector: PriceVector) -> tuple[float, ...]:
    """Compute monotone non-increasing demand with respect to price."""
    base = float(getattr(user_state, "budget", 1.0))
    values = tuple(max(0.0, base / (1.0 + price)) for price in price_vector.values)
    return values


def project_response_to_constraints(
    response: FollowerResponse, constraints: FollowerConstraints | dict[str, float]
) -> FollowerResponse:
    """Project demand to budget, deadline, and local-CPU constraints."""
    if isinstance(constraints, dict):
        constraints = FollowerConstraints(
            budget=float(constraints.get("budget", 10.0)),
            deadline_s=float(constraints.get("deadline_s", 1.0)),
            local_cpu_hz=float(constraints.get("local_cpu_hz", 1.0e9)),
        )
    max_total = max(0.0, min(constraints.budget, constraints.local_cpu_hz / 1e9, constraints.deadline_s))
    total = sum(response.demand)
    if total <= max_total or total <= 0.0:
        return response
    scale = max_total / total
    return FollowerResponse(
        demand=tuple(value * scale for value in response.demand),
        selected_edge=response.selected_edge,
        feasible=True,
        projected=True,
    )


def compute_best_response(user_state: Any, price_vector: PriceVector, system_state: Any) -> FollowerResponse:
    """Compute a projected best response for one user."""
    base_demand = compute_demand_elasticity(user_state, price_vector)
    edge_nodes = list(getattr(system_state, "edge_nodes", {}).values())
    adjusted: list[float] = []
    for idx, value in enumerate(base_demand):
        node = edge_nodes[idx] if idx < len(edge_nodes) else None
        node_id = getattr(node, "node_id", None)
        channel = getattr(user_state, "channel_by_node", {}).get(node_id) if node_id is not None else None
        channel_quality = float(getattr(channel, "quality", 1.0))
        queue = getattr(node, "queue", None)
        queue_length = float(getattr(queue, "queue_length", 0.0))
        service_rate = max(float(getattr(queue, "service_rate", 1.0)), 1e-9)
        queue_discount = 1.0 / (1.0 + queue_length / service_rate)
        channel_factor = max(0.05, min(1.0, channel_quality / 20.0 if channel_quality > 1.0 else channel_quality))
        adjusted.append(float(value) * queue_discount * channel_factor)
    demand = tuple(adjusted)
    selected = int(max(range(len(demand)), key=lambda idx: demand[idx])) if demand else -1
    metadata = getattr(system_state, "metadata", {}) or {}
    constraints = FollowerConstraints(
        budget=float(getattr(user_state, "budget", 10.0)),
        deadline_s=float(metadata.get("deadline_s", getattr(user_state, "deadline_s", 1.0))),
        local_cpu_hz=float(getattr(user_state, "local_cpu_hz", 1.0e9)),
    )
    return project_response_to_constraints(
        FollowerResponse(demand=demand, selected_edge=selected, feasible=True),
        constraints,
    )
