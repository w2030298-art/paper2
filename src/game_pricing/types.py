"""Shared pricing data containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PricingState:
    """State inputs for dynamic Stackelberg pricing."""

    queue_pressure: tuple[float, ...]
    channel_quality: tuple[float, ...]
    migration_risk: tuple[float, ...]
    base_prices: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PriceVector:
    """Provider prices by edge node."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class PricingBounds:
    """Lower and upper price bounds."""

    min_price: float = 0.05
    max_price: float = 10.0


@dataclass(frozen=True)
class PricingParameters:
    """Dynamic pricing coefficients."""

    base_price: float = 1.0
    alpha_queue: float = 0.4
    alpha_channel: float = 0.2
    alpha_migration: float = 0.3


@dataclass(frozen=True)
class FollowerDemand:
    """Demand by provider."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class ProviderCost:
    """Provider cost model."""

    fixed_cost: float = 0.0
    marginal_costs: tuple[float, ...] = ()

