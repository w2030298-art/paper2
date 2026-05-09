"""Environment profile definitions for experiment entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_ENVIRONMENT_PROFILE = "mainline-a"
LEGACY_ENVIRONMENT_PROFILE = "legacy"
MAINLINE_A_SYSTEM_MODEL_CONFIG = "configs/system_model_mainline_a.yaml"
MAINLINE_A_DYNAMIC_PRICING_CONFIG = "configs/pricing_dynamic_mainline_a.yaml"


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """Resolved environment profile used by train and benchmark commands."""

    name: str
    enable_mainline_a: bool
    system_model_config: str | None
    dynamic_pricing_config: str | None
    description: str


MAINLINE_A_PROFILE = EnvironmentProfile(
    name=DEFAULT_ENVIRONMENT_PROFILE,
    enable_mainline_a=True,
    system_model_config=MAINLINE_A_SYSTEM_MODEL_CONFIG,
    dynamic_pricing_config=MAINLINE_A_DYNAMIC_PRICING_CONFIG,
    description="Default Mainline-A system model and dynamic-pricing environment.",
)
LEGACY_PROFILE = EnvironmentProfile(
    name=LEGACY_ENVIRONMENT_PROFILE,
    enable_mainline_a=False,
    system_model_config=None,
    dynamic_pricing_config=None,
    description="Explicit fallback for historical legacy environment reproduction.",
)
_PROFILES = {
    MAINLINE_A_PROFILE.name: MAINLINE_A_PROFILE,
    LEGACY_PROFILE.name: LEGACY_PROFILE,
}


def available_environment_profile_names() -> tuple[str, ...]:
    """Return CLI choices for supported environment profiles."""
    return tuple(_PROFILES)


def resolve_environment_profile(
    profile: str | EnvironmentProfile | None = None,
) -> EnvironmentProfile:
    """Resolve a profile name to an immutable profile definition."""
    if isinstance(profile, EnvironmentProfile):
        return profile
    name = DEFAULT_ENVIRONMENT_PROFILE if profile is None else str(profile).strip().lower()
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(available_environment_profile_names())
        raise ValueError(f"Unsupported environment profile: {profile}. Expected one of: {choices}") from exc


def profile_to_train_extra_args(profile: str | EnvironmentProfile | None = None) -> list[str]:
    """Return train.py CLI arguments needed to preserve the chosen profile."""
    resolved = resolve_environment_profile(profile)
    return ["--environment-profile", resolved.name]


def profile_to_env_overrides(
    profile: str | EnvironmentProfile | None = None,
    *,
    system_model_config: str | None = None,
    dynamic_pricing_config: str | None = None,
    enable_mainline_a: bool | None = None,
) -> dict[str, Any]:
    """Translate a profile plus explicit overrides into environment kwargs."""
    resolved = resolve_environment_profile(profile)
    if not resolved.enable_mainline_a:
        if enable_mainline_a:
            raise ValueError("--enable-mainline-a is incompatible with --environment-profile legacy")
        if system_model_config is not None or dynamic_pricing_config is not None:
            raise ValueError("Mainline-A config arguments require --environment-profile mainline-a")
        return {"enable_mainline_a": False}

    overrides: dict[str, Any] = {
        "enable_mainline_a": True if enable_mainline_a is None else bool(enable_mainline_a),
        "system_model_config": system_model_config or resolved.system_model_config,
        "dynamic_pricing_config": dynamic_pricing_config or resolved.dynamic_pricing_config,
    }
    return {key: value for key, value in overrides.items() if value is not None}
