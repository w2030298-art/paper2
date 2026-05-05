"""Leader-side objective helpers for dynamic pricing."""

from __future__ import annotations

from typing import Iterable

from .types import FollowerDemand, PriceVector, ProviderCost


def compute_provider_revenue(price_vector: PriceVector, demand: FollowerDemand | Iterable[float]) -> float:
    """Compute provider revenue."""
    demand_values = demand.values if isinstance(demand, FollowerDemand) else tuple(float(v) for v in demand)
    return float(sum(price * qty for price, qty in zip(price_vector.values, demand_values)))


def compute_provider_cost(system_state: object, demand: FollowerDemand | Iterable[float]) -> float:
    """Compute provider cost from marginal node costs."""
    demand_values = demand.values if isinstance(demand, FollowerDemand) else tuple(float(v) for v in demand)
    marginal_costs = []
    for node in getattr(system_state, "edge_nodes", {}).values():
        marginal_costs.append(float(getattr(node, "metadata", {}).get("marginal_cost", 0.1)))
    if not marginal_costs:
        marginal_costs = [0.1] * len(demand_values)
    return float(sum(cost * qty for cost, qty in zip(marginal_costs, demand_values)))


def compute_leader_utility(price_vector: PriceVector, demand: FollowerDemand, system_state: object) -> float:
    """Compute revenue minus operating cost."""
    return compute_provider_revenue(price_vector, demand) - compute_provider_cost(system_state, demand)


def compute_social_welfare(user_utilities: Iterable[float], provider_utility: float) -> float:
    """Compute social welfare."""
    return float(sum(float(value) for value in user_utilities) + float(provider_utility))


__all__ = [
    "ProviderCost",
    "compute_leader_utility",
    "compute_provider_cost",
    "compute_provider_revenue",
    "compute_social_welfare",
]

