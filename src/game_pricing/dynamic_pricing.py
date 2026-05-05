"""State-dependent Stackelberg pricing policy."""

from __future__ import annotations

from .types import PriceVector, PricingBounds, PricingParameters, PricingState


def compute_queue_price_component(queue_pressure: float, params: PricingParameters) -> float:
    """Return the nonnegative queue-pressure price component."""
    return float(params.alpha_queue * max(0.0, float(queue_pressure)))


def compute_channel_price_component(channel_quality: float, params: PricingParameters) -> float:
    """Return the channel-quality component; higher quality lowers risk price."""
    return float(-params.alpha_channel * max(0.0, float(channel_quality)))


def compute_migration_price_component(migration_risk: float, params: PricingParameters) -> float:
    """Return the migration-risk price component."""
    return float(params.alpha_migration * max(0.0, float(migration_risk)))


def clip_price_to_bounds(price: float, bounds: PricingBounds) -> float:
    """Clip a price to configured bounds."""
    return float(min(max(float(price), float(bounds.min_price)), float(bounds.max_price)))


def compute_state_dependent_price(
    pricing_state: PricingState,
    bounds: PricingBounds,
    params: PricingParameters | None = None,
) -> PriceVector:
    """Compute p_i(t)=clip(p0 + alpha_q Q - alpha_h H + alpha_m M)."""
    coefficients = params or PricingParameters()
    n_prices = max(
        len(pricing_state.queue_pressure),
        len(pricing_state.channel_quality),
        len(pricing_state.migration_risk),
    )
    base_prices = pricing_state.base_prices or tuple(coefficients.base_price for _ in range(n_prices))
    values: list[float] = []
    for idx in range(n_prices):
        base = float(base_prices[idx] if idx < len(base_prices) else coefficients.base_price)
        queue = float(pricing_state.queue_pressure[idx] if idx < len(pricing_state.queue_pressure) else 0.0)
        channel = float(pricing_state.channel_quality[idx] if idx < len(pricing_state.channel_quality) else 0.0)
        migration = float(pricing_state.migration_risk[idx] if idx < len(pricing_state.migration_risk) else 0.0)
        price = (
            base
            + compute_queue_price_component(queue, coefficients)
            + compute_channel_price_component(channel, coefficients)
            + compute_migration_price_component(migration, coefficients)
        )
        values.append(clip_price_to_bounds(price, bounds))
    return PriceVector(tuple(values))


__all__ = [
    "PriceVector",
    "PricingBounds",
    "PricingParameters",
    "PricingState",
    "clip_price_to_bounds",
    "compute_channel_price_component",
    "compute_migration_price_component",
    "compute_queue_price_component",
    "compute_state_dependent_price",
]

