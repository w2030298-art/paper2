"""State containers for the mainline-A MEC model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import ChannelModelType, MigrationPolicyType, NodeId, NodeType, UserId


@dataclass(frozen=True)
class QueueSnapshot:
    """Queue state for one edge node."""

    arrival_rate: float = 0.0
    service_rate: float = 1.0
    queue_length: float = 0.0
    servers: int = 1
    capacity: int | None = None


@dataclass(frozen=True)
class MobilitySnapshot:
    """User mobility and handover-risk state."""

    cell_id: str = "cell-0"
    position_xy: tuple[float, float] = (0.0, 0.0)
    velocity_mps: float = 0.0
    handover_risk: float = 0.0
    migration_policy: MigrationPolicyType = "no_migration"


@dataclass(frozen=True)
class ChannelSnapshot:
    """Channel quality toward one node."""

    model_type: ChannelModelType = "analytic"
    quality: float = 1.0
    sinr_linear: float = 1.0
    rate_bps: float = 1.0


@dataclass(frozen=True)
class MigrationSnapshot:
    """Migration state between two edge nodes."""

    source_node: NodeId
    target_node: NodeId
    data_bits: float = 0.0
    state_bits: float = 0.0
    result_bits: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class UserState:
    """Per-user mainline-A state."""

    user_id: UserId
    active_task_ids: tuple[str, ...] = ()
    battery_level: float = 1.0
    local_cpu_hz: float = 1.0e9
    mobility: MobilitySnapshot = field(default_factory=MobilitySnapshot)
    channel_by_node: dict[NodeId, ChannelSnapshot] = field(default_factory=dict)
    budget: float = 1.0


@dataclass(frozen=True)
class EdgeNodeState:
    """Per-edge-node mainline-A state."""

    node_id: NodeId
    node_type: NodeType = "BS"
    cpu_frequency_hz: float = 5.0e9
    queue: QueueSnapshot = field(default_factory=QueueSnapshot)
    price: float = 1.0
    cooperation_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemState:
    """Complete MEC system snapshot used by adapters, pricing, and critics."""

    time_step: int = 0
    users: dict[UserId, UserState] = field(default_factory=dict)
    edge_nodes: dict[NodeId, EdgeNodeState] = field(default_factory=dict)
    migrations: tuple[MigrationSnapshot, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

