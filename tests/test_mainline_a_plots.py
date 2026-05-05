"""Tests for mainline-A plot data helpers."""

from scripts.plot_results import (
    build_dual_variable_series,
    build_ood_generalization_table,
    build_oracle_gap_table,
    build_price_state_series,
    build_reward_breakdown_series,
)


def test_mainline_a_plot_helpers_extract_tables() -> None:
    records = [
        {
            "algorithm": "game_aware_pd_marl",
            "reward_breakdown": {"delay_cost": -1.0},
            "price": 1.0,
            "queue_pressure": 0.2,
            "channel_quality": 0.8,
            "migration_risk": 0.1,
            "dual_variables": {"latency_deadline": 0.5},
            "optimality_gap": 0.1,
            "stage": "N3",
            "social_welfare": 1.0,
        }
    ]

    assert build_reward_breakdown_series(records)["delay_cost"] == [-1.0]
    assert build_price_state_series(records)["price"] == [1.0]
    assert build_dual_variable_series(records)["latency_deadline"] == [0.5]
    assert build_oracle_gap_table(records)[0]["optimality_gap"] == 0.1
    assert build_ood_generalization_table(records)[0]["social_welfare"] == 1.0

