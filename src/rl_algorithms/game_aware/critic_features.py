"""Feature construction for game-aware critics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.game_pricing.types import PriceVector
from src.mec_model.queues import compute_queue_pressure
from src.mec_model.state import SystemState


@dataclass(frozen=True)
class GameAwareCriticFeatures:
    """Structured critic features."""

    queue_pressure: tuple[float, ...]
    channel_quality: tuple[float, ...]
    migration_risk: tuple[float, ...]
    price_vector: tuple[float, ...]
    follower_demand_elasticity: tuple[float, ...]
    constraint_residuals: tuple[float, ...]

    def as_vector(self) -> tuple[float, ...]:
        """Flatten features into a deterministic vector."""
        return (
            self.queue_pressure
            + self.channel_quality
            + self.migration_risk
            + self.price_vector
            + self.follower_demand_elasticity
            + self.constraint_residuals
        )


def build_critic_features(
    system_state: SystemState,
    price_vector: PriceVector,
    reward_components: Mapping[str, float],
) -> GameAwareCriticFeatures:
    """Build critic features from system, pricing, and reward state."""
    queue = tuple(compute_queue_pressure(node.queue) for node in system_state.edge_nodes.values())
    channel_values: list[float] = []
    migration_values: list[float] = []
    for user in system_state.users.values():
        if user.channel_by_node:
            channel_values.append(
                sum(snapshot.quality for snapshot in user.channel_by_node.values())
                / max(len(user.channel_by_node), 1)
            )
        migration_values.append(user.mobility.handover_risk)
    elasticity = tuple(1.0 / (1.0 + max(price, 0.0)) for price in price_vector.values)
    residuals = tuple(
        float(reward_components.get(key, 0.0))
        for key in (
            "deadline_violation_penalty",
            "energy_cost",
            "queue_penalty",
            "migration_penalty",
            "constraint_penalty",
        )
    )
    return GameAwareCriticFeatures(
        queue_pressure=queue,
        channel_quality=tuple(channel_values),
        migration_risk=tuple(migration_values),
        price_vector=price_vector.values,
        follower_demand_elasticity=elasticity,
        constraint_residuals=residuals,
    )


def normalize_pricing_features(
    features: GameAwareCriticFeatures, running_stats: Mapping[str, tuple[float, float]]
) -> GameAwareCriticFeatures:
    """Normalize price features with mean/std statistics."""
    mean, std = running_stats.get("price_vector", (0.0, 1.0))
    denom = max(float(std), 1e-9)
    prices = tuple((value - float(mean)) / denom for value in features.price_vector)
    return GameAwareCriticFeatures(
        features.queue_pressure,
        features.channel_quality,
        features.migration_risk,
        prices,
        features.follower_demand_elasticity,
        features.constraint_residuals,
    )


def mask_unavailable_edges(
    features: GameAwareCriticFeatures, edge_mask: Sequence[bool]
) -> GameAwareCriticFeatures:
    """Mask edge-indexed features for unavailable edges."""
    mask = tuple(bool(value) for value in edge_mask)

    def apply(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(value if idx < len(mask) and mask[idx] else 0.0 for idx, value in enumerate(values))

    return GameAwareCriticFeatures(
        queue_pressure=apply(features.queue_pressure),
        channel_quality=features.channel_quality,
        migration_risk=features.migration_risk,
        price_vector=apply(features.price_vector),
        follower_demand_elasticity=apply(features.follower_demand_elasticity),
        constraint_residuals=features.constraint_residuals,
    )

