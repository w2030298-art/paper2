"""Resolved runtime configuration capture for experiment result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert common scientific/runtime values into JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def sha256_file(path: str | Path) -> str | None:
    """Return the SHA256 digest for a file, or None when it does not exist."""
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serialize_space(space: Any) -> dict[str, Any]:
    """Serialize a Gym-like space without depending on Gym at import time."""
    if space is None:
        return {"class_name": None}

    payload: dict[str, Any] = {
        "class_name": space.__class__.__name__,
        "repr": repr(space),
    }
    for attr in ("shape", "dtype", "n", "nvec"):
        if hasattr(space, attr):
            payload[attr] = _jsonable(getattr(space, attr))
    for attr in ("low", "high"):
        if hasattr(space, attr):
            value = getattr(space, attr)
            payload[attr] = _jsonable(value)
    if hasattr(space, "spaces"):
        payload["spaces"] = _jsonable(
            {
                str(key): serialize_space(value)
                for key, value in getattr(space, "spaces").items()
            }
            if isinstance(getattr(space, "spaces"), dict)
            else [serialize_space(value) for value in getattr(space, "spaces")]
        )
    return payload


def _agent_runtime(agent: Any, env: Any | None = None) -> dict[str, Any]:
    """Extract stable agent metadata useful for post-hoc run audits."""
    action_space = getattr(env, "action_space", None)
    observation_space = getattr(env, "observation_space", None)
    return _jsonable(
        {
            "class_name": agent.__class__.__name__ if agent is not None else None,
            "action_type": getattr(agent, "action_type", None),
            "discrete": getattr(agent, "discrete", None),
            "num_agents": getattr(agent, "num_agents", getattr(env, "num_agents", 1)),
            "state_dim": getattr(
                agent,
                "state_dim",
                None if observation_space is None else getattr(observation_space, "shape", [None])[0],
            ),
            "action_dim": getattr(
                agent,
                "action_dim",
                getattr(action_space, "n", None)
                if action_space is not None
                else None,
            ),
        }
    )


def build_resolved_runtime_config(
    *,
    algorithm: str,
    config_path: str | Path | None,
    base_algorithm_config: dict[str, Any],
    cli_overrides: dict[str, Any],
    environment: str,
    environment_profile: str,
    env_overrides: dict[str, Any],
    game_theory_config: dict[str, Any],
    trainer_kwargs: dict[str, Any],
    agent: Any,
    env: Any,
    train_timesteps: int,
    eval_episodes: int,
) -> dict[str, Any]:
    """Build the resolved runtime metadata stored beside training results."""
    config_path_str = None if config_path is None else str(config_path)
    safe_trainer_kwargs = {
        key: value
        for key, value in trainer_kwargs.items()
        if key not in {"env", "agent"}
    }
    return _jsonable(
        {
            "algorithm": algorithm,
            "config_path": config_path_str,
            "config_sha256": sha256_file(config_path) if config_path is not None else None,
            "base_algorithm_config": base_algorithm_config,
            "cli_overrides": cli_overrides,
            "environment": environment,
            "environment_profile": environment_profile,
            "env_overrides": env_overrides,
            "game_theory_config": game_theory_config,
            "trainer_kwargs": safe_trainer_kwargs,
            "agent_runtime": _agent_runtime(agent, env),
            "observation_space": serialize_space(getattr(env, "observation_space", None)),
            "action_space": serialize_space(getattr(env, "action_space", None)),
            "train_timesteps": int(train_timesteps),
            "eval_episodes": int(eval_episodes),
        }
    )


def write_resolved_runtime_config(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a resolved runtime config JSON atomically."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
