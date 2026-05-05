"""Tests for dynamic pricing policy."""

from src.game_pricing.dynamic_pricing import (
    PricingBounds,
    PricingParameters,
    PricingState,
    compute_state_dependent_price,
)


def test_price_increases_with_queue_and_migration() -> None:
    low = compute_state_dependent_price(
        PricingState((0.0,), (1.0,), (0.0,)), PricingBounds(), PricingParameters()
    )
    high = compute_state_dependent_price(
        PricingState((1.0,), (1.0,), (1.0,)), PricingBounds(), PricingParameters()
    )

    assert high.values[0] > low.values[0]

