"""Tests for follower response."""

from src.game_pricing.follower_response import compute_demand_elasticity
from src.game_pricing.types import PriceVector
from src.mec_model.state import UserState
from src.mec_model.types import UserId


def test_demand_is_non_increasing_in_price() -> None:
    user = UserState(user_id=UserId("u"), budget=10.0)
    low, high = compute_demand_elasticity(user, PriceVector((1.0, 2.0)))

    assert low > high

