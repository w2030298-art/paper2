"""Adapters between the legacy environment and the mainline-A model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .state import (
    ChannelSnapshot,
    EdgeNodeState,
    MobilitySnapshot,
    QueueSnapshot,
    SystemState,
    UserState,
)
from .types import NodeId, UserId


@dataclass(frozen=True)
class LegacyEnvSnapshot:
    """Serializable view of the legacy environment."""

    step: int
    queue_lengths: tuple[float, ...]
    channel_qualities: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemModelAdapter:
    """Map legacy env fields to mainline-A state containers."""

    def build(self, env: Any) -> SystemState:
        """Build a SystemState from a legacy env object."""
        return build_system_state_from_legacy_env(env)

    def apply_decision(self, env: Any, decision: dict[str, Any]) -> None:
        """Apply a mainline-A decision to legacy env metadata."""
        apply_system_decision_to_legacy_env(env, decision)

    def reward_components(self, env: Any, system_state: SystemState) -> dict[str, float]:
        """Extract interpretable reward components."""
        return extract_reward_components(env, system_state)


def build_system_state_from_legacy_env(env: Any) -> SystemState:
    """Build a mainline-A SystemState without mutating the legacy env."""
    num_edges = int(getattr(env, "_num_edge_servers", getattr(env, "num_edge_servers", 1)))
    num_users = int(getattr(env, "num_agents", 1))
    queues = list(getattr(env, "queue_lengths", [0.0] * num_edges))
    arrivals = list(getattr(env, "arrival_rates", [0.0] * num_edges))
    services = list(getattr(env, "service_rates", [1.0] * num_edges))
    prices = list(getattr(env, "latest_equilibrium_prices", [1.0] * num_edges))
    edge_nodes: dict[NodeId, EdgeNodeState] = {}
    for idx in range(num_edges):
        node_id = NodeId(f"edge-{idx}")
        edge_nodes[node_id] = EdgeNodeState(
            node_id=node_id,
            node_type="BS",
            queue=QueueSnapshot(
                arrival_rate=float(arrivals[idx] if idx < len(arrivals) else 0.0),
                service_rate=float(services[idx] if idx < len(services) else 1.0),
                queue_length=float(queues[idx] if idx < len(queues) else 0.0),
            ),
            price=float(prices[idx] if idx < len(prices) else 1.0),
        )

    channels = getattr(env, "channel_qualities", None)
    users: dict[UserId, UserState] = {}
    for user_idx in range(num_users):
        user_id = UserId(f"user-{user_idx}")
        channel_by_node: dict[NodeId, ChannelSnapshot] = {}
        for edge_idx in range(num_edges):
            quality = 1.0
            if channels is not None:
                quality = float(channels[user_idx][edge_idx])
            node_id = NodeId(f"edge-{edge_idx}")
            channel_by_node[node_id] = ChannelSnapshot(
                quality=quality,
                sinr_linear=max(10.0 ** (quality / 10.0), 1e-9),
                rate_bps=max(quality, 0.0),
            )
        users[user_id] = UserState(
            user_id=user_id,
            active_task_ids=(f"user-{user_idx}-active",),
            mobility=MobilitySnapshot(cell_id=f"cell-{user_idx % max(num_edges, 1)}"),
            channel_by_node=channel_by_node,
        )

    return SystemState(
        time_step=int(getattr(env, "current_step", 0)),
        users=users,
        edge_nodes=edge_nodes,
        metadata={
            "legacy_env": type(env).__name__,
            "deadline_s": float(getattr(env, "latency_budget", 1.0)),
            "energy_budget": float(getattr(env, "energy_budget", 10.0)),
        },
    )


def apply_system_decision_to_legacy_env(env: Any, decision: dict[str, Any]) -> None:
    """Apply a mainline-A decision to fields consumed by the next env step."""
    normalized = dict(decision)
    setattr(env, "last_mainline_a_decision", normalized)
    setattr(env, "mainline_a_pending_decision", normalized)

    price_values = (
        normalized.get("price_vector")
        or normalized.get("dynamic_prices")
        or normalized.get("prices")
    )
    if price_values is not None:
        prices = np.asarray(tuple(price_values), dtype=np.float32)
        num_edges = int(getattr(env, "_num_edge_servers", prices.size))
        if prices.size != num_edges:
            raise ValueError(f"price decision length {prices.size} does not match {num_edges} edges")
        setattr(env, "latest_equilibrium_prices", prices.copy())
        setattr(env, "mainline_a_applied_price_vector", tuple(float(value) for value in prices))


def extract_reward_components(env: Any, system_state: SystemState) -> dict[str, float]:
    """Extract interpretable reward components from legacy metrics."""
    avg_latency = float(getattr(env, "total_latency", 0.0)) / max(1, int(getattr(env, "task_completed", 0)))
    avg_energy = float(getattr(env, "total_energy", 0.0)) / max(1, int(getattr(env, "task_completed", 0)))
    queue_penalty = sum(node.queue.queue_length for node in system_state.edge_nodes.values())
    prices = getattr(env, "latest_equilibrium_prices", None)
    price_payment = float(np.sum(prices)) if prices is not None else sum(node.price for node in system_state.edge_nodes.values())
    latency_components = list(getattr(env, "last_latency_components", []) or [])
    deadline_penalty = float(
        np.mean([float(item.get("deadline_miss", 0.0)) for item in latency_components])
    ) if latency_components else 0.0
    migration_penalty = float(
        np.mean([1.0 - float(item.get("nearest_server_selected", 1.0)) for item in latency_components])
    ) if latency_components else 0.0
    cooperation_gain = float(getattr(env, "last_cooperation_gain", 0.0))
    constraint_penalty = float(
        max(0.0, deadline_penalty)
        + max(0.0, float(np.mean(getattr(env, "rho", [0.0]))) - 1.0)
    )
    return {
        "delay_cost": avg_latency,
        "energy_cost": avg_energy,
        "queue_penalty": float(queue_penalty),
        "migration_penalty": migration_penalty,
        "deadline_violation_penalty": deadline_penalty,
        "cooperation_gain": cooperation_gain,
        "price_payment": float(price_payment),
        "provider_revenue": float(price_payment),
        "constraint_penalty": constraint_penalty,
    }
