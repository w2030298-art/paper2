"""Tests for interpretable reward design."""

from src.rl_algorithms.game_aware.reward_design import (
    compute_interpretable_reward,
    export_reward_explanation,
)


def test_reward_design_supports_price_ablation() -> None:
    breakdown = compute_interpretable_reward(
        {"delay_cost": -1.0, "price_payment": -2.0},
        dual_state=None,
        weights={"delay_cost": 1.0, "price_payment": 1.0},
        ablation="no_price",
    )

    exported = export_reward_explanation(breakdown)

    assert breakdown.total_reward == -1.0
    assert "price_payment" not in exported

