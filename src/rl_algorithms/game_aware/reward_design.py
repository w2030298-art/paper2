"""Interpretable reward design helpers for mainline-A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .primal_dual import PrimalDualState


@dataclass(frozen=True)
class RewardComponent:
    """Named reward component."""

    name: str
    value: float
    weight: float = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    """Computed reward and component contributions."""

    total_reward: float
    components: tuple[RewardComponent, ...]
    dual_penalty: float = 0.0
    ablation: str = "full_model"

    def means(self) -> dict[str, float]:
        """Return component values keyed by name."""
        return {component.name: component.value for component in self.components}


ABLATION_DISABLED_COMPONENTS = {
    "no_price": {"price_payment", "provider_revenue"},
    "no_queue": {"queue_penalty"},
    "no_migration": {"migration_penalty"},
    "no_dual": {"constraint_penalty"},
    "no_cooperation": {"cooperation_gain"},
}


def compute_interpretable_reward(
    reward_components: Mapping[str, float],
    dual_state: PrimalDualState | None,
    weights: Mapping[str, float],
    ablation: str = "full_model",
) -> RewardBreakdown:
    """Compute weighted reward with optional ablation switches."""
    disabled = ABLATION_DISABLED_COMPONENTS.get(ablation, set())
    components: list[RewardComponent] = []
    total = 0.0
    for name, value in reward_components.items():
        if name in disabled:
            continue
        weight = float(weights.get(name, 1.0))
        contribution = weight * float(value)
        components.append(RewardComponent(name=name, value=float(value), weight=weight))
        total += contribution
    dual_penalty = 0.0
    if dual_state is not None and ablation != "no_dual":
        dual_penalty = sum(float(value) for value in dual_state.dual_variables.values()) * float(
            reward_components.get("constraint_penalty", 0.0)
        )
        total -= dual_penalty
    return RewardBreakdown(float(total), tuple(components), float(dual_penalty), ablation)


def export_reward_explanation(record: RewardBreakdown | Mapping[str, float]) -> dict[str, float | str]:
    """Return a serializable reward explanation record."""
    if isinstance(record, RewardBreakdown):
        payload: dict[str, float | str] = {
            "total_reward": record.total_reward,
            "dual_penalty": record.dual_penalty,
            "ablation": record.ablation,
        }
        payload.update({component.name: component.value for component in record.components})
        return payload
    return {str(key): float(value) for key, value in record.items()}

