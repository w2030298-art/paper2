"""Tests for mainline-A channel models."""

from src.mec_model.channel import (
    AnalyticRateModel,
    ThreeGppLiteRateModel,
    compute_shannon_rate_bps,
    compute_sinr_linear,
)


def test_shannon_rate_and_sinr_are_positive() -> None:
    sinr = compute_sinr_linear(1.0, 2.0, 0.1, 0.1)
    rate = compute_shannon_rate_bps(1e6, sinr)

    assert sinr > 0.0
    assert rate > 0.0


def test_rate_models_return_positive_rates() -> None:
    assert AnalyticRateModel().rate_bps(1.0, 10.0) > 0.0
    assert ThreeGppLiteRateModel(rng_seed=1).rate_bps(1.0, 10.0) > 0.0

