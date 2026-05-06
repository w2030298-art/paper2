"""Strict schema helpers for Mainline-A configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LEGACY_SYSTEM_MODEL_KEYS = ("queue", "channel")


def validate_mainline_a_system_model_config(
    config: Mapping[str, Any],
    source: str = "Mainline-A config",
) -> dict[str, Any]:
    """Validate and return the canonical ``system_model`` mapping."""
    system_model = config.get("system_model", {})
    if system_model is None:
        return {}
    if not isinstance(system_model, Mapping):
        raise ValueError(f"{source}: system_model must be a mapping")
    legacy_keys = sorted(key for key in LEGACY_SYSTEM_MODEL_KEYS if key in system_model)
    if legacy_keys:
        joined = ", ".join(legacy_keys)
        raise ValueError(
            f"{source}: legacy system_model field(s) {joined} are not allowed; "
            "use queue_model and channel_model.{theory,simulation}"
        )
    channel_model = system_model.get("channel_model")
    if channel_model is not None and not isinstance(channel_model, Mapping):
        raise ValueError(f"{source}: channel_model must be a mapping")
    return dict(system_model)


def resolve_queue_model(config: Mapping[str, Any], default: str | None = None) -> str | None:
    """Resolve the canonical queue model value."""
    system_model = validate_mainline_a_system_model_config(config)
    value = system_model.get("queue_model", default)
    return None if value is None else str(value)


def resolve_channel_model(
    config: Mapping[str, Any],
    default: str | None = None,
    preference: str = "simulation",
) -> str | None:
    """Resolve a canonical channel model value from ``channel_model``."""
    system_model = validate_mainline_a_system_model_config(config)
    channel_model = system_model.get("channel_model", {})
    if not isinstance(channel_model, Mapping):
        return default
    value = channel_model.get(preference)
    if value is None and preference != "theory":
        value = channel_model.get("theory")
    if value is None:
        value = default
    return None if value is None else str(value)
