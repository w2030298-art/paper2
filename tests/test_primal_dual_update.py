"""Tests for primal-dual updates."""

from src.rl_algorithms.game_aware.primal_dual import (
    ConstraintResiduals,
    compute_lagrangian_reward,
    update_dual_variables,
)


def test_dual_variables_increase_on_positive_residuals() -> None:
    residuals = ConstraintResiduals(latency_deadline=2.0)
    state = update_dual_variables(residuals, dual_lr=0.5)

    assert state.dual_variables["latency_deadline"] == 1.0
    assert compute_lagrangian_reward(10.0, residuals, state) == 8.0

