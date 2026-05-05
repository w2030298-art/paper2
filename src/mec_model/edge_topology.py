"""Heterogeneous cooperative edge topology helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import EdgeNodeState, SystemState
from .tasks import TaskDAGSpec
from .types import NodeId, NodeType, UserId


@dataclass(frozen=True)
class EdgeNodeSpec:
    """Static edge-node capabilities."""

    node_id: NodeId
    node_type: NodeType = "BS"
    cpu_frequency_hz: float = 5.0e9
    cooperation_enabled: bool = False
    position_xy: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class CooperationLink:
    """Cooperation link between two edge nodes."""

    source_node: NodeId
    target_node: NodeId
    bandwidth_bps: float
    latency_s: float
    enabled: bool = True


@dataclass(frozen=True)
class CooperativeEdgeGraph:
    """Collection of nodes and optional cooperation links."""

    nodes: dict[NodeId, EdgeNodeSpec] = field(default_factory=dict)
    links: tuple[CooperationLink, ...] = ()

    def enabled_neighbors(self, node_id: NodeId) -> list[NodeId]:
        """Return enabled neighbors for a node."""
        neighbors: list[NodeId] = []
        for link in self.links:
            if not link.enabled:
                continue
            if link.source_node == node_id:
                neighbors.append(link.target_node)
            elif link.target_node == node_id:
                neighbors.append(link.source_node)
        return neighbors


@dataclass(frozen=True)
class MigrationCost:
    """Migration cost components."""

    data_transfer_s: float
    state_transfer_s: float
    result_return_s: float
    total_cost_s: float


def _as_node_id(value: EdgeNodeSpec | EdgeNodeState | NodeId | str) -> NodeId:
    """Extract a node id from supported inputs."""
    if isinstance(value, EdgeNodeSpec | EdgeNodeState):
        return value.node_id
    return NodeId(str(value))


def select_candidate_edges(system_state: SystemState, user_id: UserId | str) -> list[NodeId]:
    """Select candidate edges; BS nodes are the legacy default."""
    _ = UserId(str(user_id))
    bs_nodes = [
        node_id
        for node_id, node in system_state.edge_nodes.items()
        if node.node_type == "BS"
    ]
    cooperative = [
        node_id
        for node_id, node in system_state.edge_nodes.items()
        if node.cooperation_enabled and node_id not in bs_nodes
    ]
    return bs_nodes + cooperative


def compute_migration_cost(
    task_spec: TaskDAGSpec,
    source_node: EdgeNodeSpec | EdgeNodeState | NodeId | str,
    target_node: EdgeNodeSpec | EdgeNodeState | NodeId | str,
) -> MigrationCost:
    """Compute migration cost from data, state, and result transfer."""
    _ = (_as_node_id(source_node), _as_node_id(target_node))
    total_data_bits = sum(segment.data_size_bits for segment in task_spec.segments.values())
    total_cpu = sum(segment.cpu_cycles for segment in task_spec.segments.values())
    state_bits = max(total_data_bits * 0.05, 1_024.0)
    result_bits = max(total_data_bits * 0.02, 1_024.0)
    link_rate_bps = 100e6
    data_transfer_s = total_data_bits / link_rate_bps
    state_transfer_s = state_bits / link_rate_bps
    result_return_s = result_bits / link_rate_bps
    compute_penalty_s = total_cpu / 1e12
    total = data_transfer_s + state_transfer_s + result_return_s + compute_penalty_s
    return MigrationCost(data_transfer_s, state_transfer_s, result_return_s, total)


def compute_cooperation_gain(
    source_node: EdgeNodeSpec | EdgeNodeState,
    target_node: EdgeNodeSpec | EdgeNodeState,
    task_spec: TaskDAGSpec,
) -> float:
    """Estimate cooperation gain from target/source compute ratio."""
    source_cpu = max(float(source_node.cpu_frequency_hz), 1.0)
    target_cpu = max(float(target_node.cpu_frequency_hz), 1.0)
    total_cpu = sum(segment.cpu_cycles for segment in task_spec.segments.values())
    local_time = total_cpu / source_cpu
    target_time = total_cpu / target_cpu
    return float(max(0.0, local_time - target_time))


def graph_from_config(nodes: list[dict[str, Any]], links: list[dict[str, Any]] | None = None) -> CooperativeEdgeGraph:
    """Build a cooperative graph from serializable config."""
    specs = {
        NodeId(str(item["node_id"])): EdgeNodeSpec(
            node_id=NodeId(str(item["node_id"])),
            node_type=item.get("node_type", "BS"),
            cpu_frequency_hz=float(item.get("cpu_frequency_hz", 5.0e9)),
            cooperation_enabled=bool(item.get("cooperation_enabled", False)),
        )
        for item in nodes
    }
    coop_links = tuple(
        CooperationLink(
            source_node=NodeId(str(item["source_node"])),
            target_node=NodeId(str(item["target_node"])),
            bandwidth_bps=float(item.get("bandwidth_bps", 100e6)),
            latency_s=float(item.get("latency_s", 0.001)),
            enabled=bool(item.get("enabled", True)),
        )
        for item in (links or [])
    )
    return CooperativeEdgeGraph(nodes=specs, links=coop_links)

