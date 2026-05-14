"""Single-policy shared-control helpers for multi-user MEC baselines.

This interface evaluates one single-agent policy in a multi-user environment:
the same policy object selects one action per user observation, the environment
receives a joint action for one shared step, and transitions are expanded back
into per-user samples for the existing single-agent update code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

INTERFACE_NAME = "single_policy_multi_user"
DEFAULT_SINGLE_POLICY_USERS = 3
DEFAULT_SHARED_REWARD = "mean"
VALID_SHARED_REWARDS = {"mean", "global"}
SINGLE_POLICY_FULL17_ALGORITHMS = [
    "GRPO",
    "PPO",
    "SAC",
    "DDQN",
    "DDPG",
    "TD3",
    "A3C",
    "TRPO",
    "SimPO",
]


@dataclass(frozen=True)
class SinglePolicyMultiUserSettings:
    """Resolved single-policy multi-user benchmark settings."""

    enabled: bool = False
    num_users: int = DEFAULT_SINGLE_POLICY_USERS
    shared_reward: str = DEFAULT_SHARED_REWARD
    interface: str = INTERFACE_NAME

    def validate(self) -> "SinglePolicyMultiUserSettings":
        """Return a validated copy of the settings."""
        if self.num_users <= 0:
            raise ValueError("single_policy_num_users must be positive")
        if self.shared_reward not in VALID_SHARED_REWARDS:
            allowed = ", ".join(sorted(VALID_SHARED_REWARDS))
            raise ValueError(f"single_policy_shared_reward must be one of: {allowed}")
        return self


@dataclass(frozen=True)
class SinglePolicyStepRecord:
    """One shared environment step and its per-user transition expansion."""

    actions: list[Any]
    rewards: np.ndarray
    next_observations: list[np.ndarray]
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    transitions: list[dict[str, Any]]


def resolve_settings(
    file_cfg: dict[str, Any] | None = None,
    *,
    cli_enabled: bool = False,
    cli_num_users: int | None = None,
    cli_shared_reward: str | None = None,
) -> SinglePolicyMultiUserSettings:
    """Resolve settings from benchmark config plus CLI overrides."""
    cfg = dict((file_cfg or {}).get("single_policy_multi_user", {}) or {})
    enabled = bool(cli_enabled or cfg.get("enabled", False))
    num_users = int(
        cli_num_users
        if cli_num_users is not None
        else cfg.get("num_users", cfg.get("num_agents", DEFAULT_SINGLE_POLICY_USERS))
    )
    shared_reward = str(
        cli_shared_reward
        if cli_shared_reward is not None
        else cfg.get("shared_reward", DEFAULT_SHARED_REWARD)
    )
    return SinglePolicyMultiUserSettings(
        enabled=enabled,
        num_users=num_users,
        shared_reward=shared_reward,
    ).validate()


def apply_env_overrides(
    env_overrides: dict[str, Any] | None,
    settings: SinglePolicyMultiUserSettings,
) -> dict[str, Any]:
    """Force the benchmark environment to the configured user count when enabled."""
    resolved = dict(env_overrides or {})
    if settings.enabled:
        resolved["num_agents_multi"] = int(settings.num_users)
        resolved["force_num_agents"] = int(settings.num_users)
        resolved["single_policy_interface"] = settings.interface
    return resolved


def validate_algorithm_set(
    algorithms: Iterable[str],
    settings: SinglePolicyMultiUserSettings,
) -> None:
    """Reject algorithms outside the v5.0 single-policy baseline set."""
    if not settings.enabled:
        return
    allowed = set(SINGLE_POLICY_FULL17_ALGORITHMS)
    disallowed = [str(algo) for algo in algorithms if str(algo) not in allowed]
    if disallowed:
        joined = ", ".join(disallowed)
        raise ValueError(f"{INTERFACE_NAME} excludes unsupported algorithms: {joined}")


def build_dry_run_runs(
    algorithms: Iterable[str],
    settings: SinglePolicyMultiUserSettings,
) -> list[dict[str, Any]]:
    """Build explicit dry-run run records for preflight display and tests."""
    interface = settings.interface if settings.enabled else "standard"
    num_agents = settings.num_users if settings.enabled else None
    return [
        {
            "algorithm": str(algo),
            "interface": interface,
            "num_agents": num_agents,
            "shared_reward": settings.shared_reward if settings.enabled else None,
        }
        for algo in algorithms
    ]


def normalize_reward_vector(
    reward: Any,
    num_users: int,
    shared_reward: str = DEFAULT_SHARED_REWARD,
) -> np.ndarray:
    """Return one reward value per user for transition expansion."""
    if shared_reward not in VALID_SHARED_REWARDS:
        allowed = ", ".join(sorted(VALID_SHARED_REWARDS))
        raise ValueError(f"shared_reward must be one of: {allowed}")
    try:
        arr = np.asarray(reward, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        arr = np.zeros(0, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float32)
    if arr.size == 1:
        return np.full(num_users, float(arr[0]), dtype=np.float32)
    if shared_reward == "global":
        return np.full(num_users, float(np.mean(arr)), dtype=np.float32)
    if arr.size < num_users:
        arr = np.pad(arr, (0, num_users - arr.size), mode="edge")
    return arr[:num_users].astype(np.float32, copy=False)


def _normalize_observations(observations: Any, num_users: int) -> list[np.ndarray]:
    """Normalize environment observations to a per-user list."""
    if isinstance(observations, list):
        items = observations
    else:
        arr = np.asarray(observations, dtype=np.float32)
        if arr.ndim >= 2 and arr.shape[0] >= num_users:
            items = [arr[i] for i in range(num_users)]
        else:
            items = [arr for _ in range(num_users)]
    if len(items) < num_users:
        items = list(items) + [items[-1] if items else np.zeros(0, dtype=np.float32)] * (num_users - len(items))
    return [np.asarray(item, dtype=np.float32) for item in items[:num_users]]


def _normalize_action(action: Any, action_space: Any) -> Any:
    """Match an agent action to the adapter action-space shape."""
    if isinstance(action, np.ndarray) and action.ndim > 1:
        action = action[0]
    if hasattr(action_space, "n"):
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return int(action.reshape(-1)[0])
            return int(np.argmax(action.reshape(-1)))
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)
    if isinstance(action, np.ndarray):
        arr = action.reshape(-1).astype(np.float32, copy=False)
    else:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


def select_user_actions(
    agent: Any,
    observations: Any,
    action_space: Any,
    *,
    num_users: int,
    deterministic: bool = False,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Use one policy object to select one action for each user observation."""
    actions: list[Any] = []
    infos: list[dict[str, Any]] = []
    for obs in _normalize_observations(observations, num_users):
        action, info = agent.select_action(obs, deterministic=deterministic)
        actions.append(_normalize_action(action, action_space))
        infos.append(dict(info or {}))
    return actions, infos


def expand_transitions(
    observations: Any,
    actions: list[Any],
    rewards: Any,
    next_observations: Any,
    done: bool,
    *,
    num_users: int,
    shared_reward: str = DEFAULT_SHARED_REWARD,
) -> list[dict[str, Any]]:
    """Expand one shared multi-user step into per-user single-policy samples."""
    obs_list = _normalize_observations(observations, num_users)
    next_list = _normalize_observations(next_observations, num_users)
    reward_vec = normalize_reward_vector(rewards, num_users, shared_reward)
    return [
        {
            "state": obs_list[i],
            "action": actions[i],
            "reward": float(reward_vec[i]),
            "next_state": next_list[i],
            "done": bool(done),
            "user_index": i,
        }
        for i in range(num_users)
    ]


class SinglePolicyMultiUserRunner:
    """Collect shared-control steps for a single policy in a multi-user env."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        settings: SinglePolicyMultiUserSettings | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.settings = (settings or SinglePolicyMultiUserSettings(enabled=True)).validate()
        if not self.settings.enabled:
            raise ValueError("SinglePolicyMultiUserRunner requires enabled settings")
        env_agents = int(getattr(env, "num_agents", self.settings.num_users))
        if env_agents != self.settings.num_users:
            raise ValueError(f"Expected {self.settings.num_users} users, got env.num_agents={env_agents}")

    def collect_step(self, observations: Any, deterministic: bool = False) -> SinglePolicyStepRecord:
        """Select per-user actions, step the env once, and return transitions."""
        actions, _infos = select_user_actions(
            self.agent,
            observations,
            self.env.action_space,
            num_users=self.settings.num_users,
            deterministic=deterministic,
        )
        next_obs, rewards, terminated, truncated, info = self.env.step(actions)
        done = bool(terminated) or bool(truncated)
        reward_vec = normalize_reward_vector(
            rewards,
            self.settings.num_users,
            self.settings.shared_reward,
        )
        transitions = expand_transitions(
            observations,
            actions,
            rewards,
            next_obs,
            done,
            num_users=self.settings.num_users,
            shared_reward=self.settings.shared_reward,
        )
        return SinglePolicyStepRecord(
            actions=actions,
            rewards=reward_vec,
            next_observations=_normalize_observations(next_obs, self.settings.num_users),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
            transitions=transitions,
        )


def annotate_result(
    result: dict[str, Any],
    settings: SinglePolicyMultiUserSettings,
    *,
    num_agents: int,
) -> dict[str, Any]:
    """Attach interface metadata to a benchmark result record."""
    if not settings.enabled:
        return result
    result["interface"] = settings.interface
    result["num_agents"] = int(num_agents)
    result["single_policy_shared_reward"] = settings.shared_reward
    result["single_policy_num_users"] = int(settings.num_users)
    return result
