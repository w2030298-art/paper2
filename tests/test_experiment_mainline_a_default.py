"""Tests for Mainline-A as the default experiment profile."""

from src.experiment.presets import FULL_17_ALGORITHMS, PRESETS
from src.experiment.registry import AlgorithmRegistry


def test_registry_injects_mainline_a_profile_by_default() -> None:
    """Registry-built train specs should default to Mainline-A."""
    spec = AlgorithmRegistry().build_specs(
        ["GRPO"],
        timesteps=1,
        seed=42,
        device="cpu",
        eval_episodes=1,
    )[0]
    assert spec.extra_args == ["--environment-profile", "mainline-a"]


def test_registry_can_build_explicit_legacy_fallback_specs() -> None:
    """Legacy fallback should be available only through explicit profile selection."""
    spec = AlgorithmRegistry().build_specs(
        ["GRPO"],
        timesteps=1,
        seed=42,
        device="cpu",
        eval_episodes=1,
        environment_profile="legacy",
    )[0]
    assert spec.extra_args == ["--environment-profile", "legacy"]


def test_full17_preset_keeps_17_algorithms_and_mainline_a_profile() -> None:
    """The full17 preset should remain one preset with a Mainline-A profile."""
    assert len(FULL_17_ALGORITHMS) == 17
    assert PRESETS["full17"]["environment_profile"] == "mainline-a"
