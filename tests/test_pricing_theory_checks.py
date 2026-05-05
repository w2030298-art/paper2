"""Tests for pricing theory checks."""

from src.game_pricing.theory_checks import (
    check_demand_price_monotonicity,
    check_price_lipschitz_bound,
    check_strong_concavity_proxy,
    check_unique_best_response_grid,
)


def test_pricing_theory_checks_pass_simple_cases() -> None:
    assert check_demand_price_monotonicity([(1, 3), (2, 2)]).passed
    assert check_strong_concavity_proxy([[-1, 0], [0, -2]]).passed
    assert check_unique_best_response_grid(None, [[1, 2], [2, 3]]).passed
    assert check_price_lipschitz_bound(lambda x: [x], [1, 2, 3]).passed

