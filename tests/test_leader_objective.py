"""Tests for leader objective helpers."""

from src.game_pricing.leader_objective import (
    compute_leader_utility,
    compute_provider_revenue,
    compute_social_welfare,
)
from src.game_pricing.types import FollowerDemand, PriceVector
from src.mec_model.state import SystemState


def test_leader_utility_is_revenue_minus_cost() -> None:
    prices = PriceVector((2.0,))
    demand = FollowerDemand((3.0,))

    revenue = compute_provider_revenue(prices, demand)
    utility = compute_leader_utility(prices, demand, SystemState())

    assert revenue == 6.0
    assert utility < revenue
    assert compute_social_welfare([1.0], utility) == utility + 1.0

