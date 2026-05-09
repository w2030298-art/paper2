"""Tests for shared environment profile resolution."""

import pytest

from src.experiment.environment_profiles import (
    DEFAULT_ENVIRONMENT_PROFILE,
    LEGACY_ENVIRONMENT_PROFILE,
    profile_to_env_overrides,
    profile_to_train_extra_args,
    resolve_environment_profile,
)


def test_default_environment_profile_is_mainline_a() -> None:
    """The implicit profile should resolve to Mainline-A."""
    profile = resolve_environment_profile()
    assert profile.name == DEFAULT_ENVIRONMENT_PROFILE
    assert profile.enable_mainline_a is True
    assert profile.system_model_config == "configs/system_model_mainline_a.yaml"
    assert profile.dynamic_pricing_config == "configs/pricing_dynamic_mainline_a.yaml"


def test_legacy_profile_disables_mainline_a_injection() -> None:
    """Legacy profile should be explicit and should not inject Mainline-A configs."""
    profile = resolve_environment_profile(LEGACY_ENVIRONMENT_PROFILE)
    overrides = profile_to_env_overrides(profile)
    assert profile.enable_mainline_a is False
    assert overrides == {"enable_mainline_a": False}


def test_profile_to_train_extra_args_preserves_profile_name() -> None:
    """Train commands should carry the selected profile explicitly."""
    assert profile_to_train_extra_args("mainline-a") == [
        "--environment-profile",
        "mainline-a",
    ]
    assert profile_to_train_extra_args("legacy") == ["--environment-profile", "legacy"]


def test_legacy_profile_rejects_mainline_a_compat_args() -> None:
    """Legacy fallback should fail fast when Mainline-A knobs are mixed in."""
    with pytest.raises(ValueError, match="incompatible"):
        profile_to_env_overrides("legacy", enable_mainline_a=True)
    with pytest.raises(ValueError, match="Mainline-A config"):
        profile_to_env_overrides("legacy", system_model_config="configs/system_model_mainline_a.yaml")
