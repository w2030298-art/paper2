#!/usr/bin/env python3
"""
Benchmark — 多算法对比评测入口 (GameTheory 环境唯一)

公平性设计:
- 所有算法统一在 GameTheory 适配环境上评测:
    离散动作 → MEC-v1-game-theory-discrete-ma
    连续动作 → MEC-v1-game-theory-continuous-ma
- 使用 --env 可强制覆盖环境

用法:
    python scripts/benchmark.py --all              # 全部 17 算法
    python scripts/benchmark.py --algorithms ppo sac --episodes 3 --timesteps 10000
    python scripts/benchmark.py --all --include-heuristics --scale medium
"""

import argparse
import importlib
import logging
import os
import sys
import json
import time
import yaml
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.composite_score import CompositeScorer  # noqa: E402
from src.experiment.environment_profiles import (  # noqa: E402
    DEFAULT_ENVIRONMENT_PROFILE,
    available_environment_profile_names,
    profile_to_env_overrides,
    resolve_environment_profile,
)
from src.experiment.runtime_config import build_resolved_runtime_config  # noqa: E402
from src.mec_model.config_schema import (  # noqa: E402
    resolve_channel_model,
    resolve_queue_model,
    validate_mainline_a_system_model_config,
)

ALGORITHM_CLASSES = {
    "GRPO": ("rl_algorithms.grpo", "GRPOAgent"),
    "PPO": ("rl_algorithms.ppo", "PPOAgent"),
    "CL-PPO": ("rl_algorithms.cl_ppo", "CLPPOAgent"),
    "SAC": ("rl_algorithms.sac", "SACAgent"),
    "DDQN": ("rl_algorithms.ddqn", "DDQNAgent"),
    "DDPG": ("rl_algorithms.ddpg", "DDPGAgent"),
    "TD3": ("rl_algorithms.td3", "TD3Agent"),
    "A3C": ("rl_algorithms.a3c", "A3CAgent"),
    "TRPO": ("rl_algorithms.trpo", "TRPOAgent"),
    "SimPO": ("rl_algorithms.simpo", "SimPOAgent"),
    "MAPPO": ("rl_algorithms.mappo", "MAPPOAgent"),
    "QMIX": ("rl_algorithms.qmix", "QMIXAgent"),
    "COMA": ("rl_algorithms.coma", "COMAAgent"),
    "GAM-COMA": ("rl_algorithms.gam_coma", "GAMCOMAAgent"),
    "IPPO": ("rl_algorithms.ippo", "IPPOAgent"),
    "VDN": ("rl_algorithms.vdn", "VDNAgent"),
    "MADDPG": ("rl_algorithms.maddpg", "MADDPGAgent"),
    "IQL": ("rl_algorithms.iql", "IQLAgent"),
    "MATD3": ("rl_algorithms.matd3", "MATD3Agent"),
}
STAGE2_ALGOS = {"CL-PPO", "GAM-COMA"}
ON_POLICY = {"GRPO", "PPO", "CL-PPO", "A3C", "TRPO", "SimPO", "MAPPO", "COMA", "GAM-COMA", "IPPO"}
OFF_POLICY = {"SAC", "DDQN", "DDPG", "TD3", "QMIX", "VDN", "MADDPG", "IQL", "MATD3"}
ALL_ALGOS = [name for name in ALGORITHM_CLASSES if name not in STAGE2_ALGOS]
ALGORITHM_ALIASES = {
    "game_aware_pd_marl": "MAPPO",
}
ALGO_ENV_MAP = {
    "A3C": "MEC-v1-game-theory-discrete-ma",
    "SimPO": "MEC-v1-game-theory-discrete-ma",
    "DDQN": "MEC-v1-game-theory-discrete-ma",
    "QMIX": "MEC-v1-game-theory-discrete-ma",
    "COMA": "MEC-v1-game-theory-discrete-ma",
    "GAM-COMA": "MEC-v1-game-theory-discrete-ma",
    "IPPO": "MEC-v1-game-theory-discrete-ma",
    "VDN": "MEC-v1-game-theory-discrete-ma",
    "IQL": "MEC-v1-game-theory-discrete-ma",
    "GRPO": "MEC-v1-game-theory-continuous-ma",
    "PPO": "MEC-v1-game-theory-continuous-ma",
    "CL-PPO": "MEC-v1-game-theory-continuous-ma",
    "TRPO": "MEC-v1-game-theory-continuous-ma",
    "SAC": "MEC-v1-game-theory-continuous-ma",
    "DDPG": "MEC-v1-game-theory-continuous-ma",
    "TD3": "MEC-v1-game-theory-continuous-ma",
    "MADDPG": "MEC-v1-game-theory-continuous-ma",
    "MATD3": "MEC-v1-game-theory-continuous-ma",
    "MAPPO": "MEC-v1-game-theory-discrete-ma",
    "Greedy": "MEC-v1-game-theory-continuous-ma",
    "Random": "MEC-v1-game-theory-continuous-ma",
    "Local-only": "MEC-v1-game-theory-continuous-ma",
    "Full-offload": "MEC-v1-game-theory-continuous-ma",
}

CONTINUOUS_ALGOS = {"CL-PPO", "SAC", "DDPG", "TD3", "MADDPG", "MATD3"}
MULTI_AGENT_ALGOS = {"MAPPO", "QMIX", "COMA", "GAM-COMA", "IPPO", "VDN", "MADDPG", "IQL", "MATD3"}
DEEP_FUSION_ALGOS = {"GRPO", "MAPPO", "QMIX", "COMA", "GAM-COMA", "IPPO", "VDN", "MADDPG", "IQL", "MATD3"}
HEURISTIC_ALGOS = ["Greedy", "Random", "Local-only", "Full-offload"]

SCALE_PRESETS = {
    "small": {"num_edge_servers": 3, "num_agents_multi": 3, "max_steps": 100},
    "medium": {"num_edge_servers": 5, "num_agents_multi": 5, "max_steps": 120},
    "large": {"num_edge_servers": 10, "num_agents_multi": 10, "max_steps": 150},
}


def _canonical_algorithm_name(name: str) -> str:
    """Return the registered algorithm name, accepting case-insensitive input."""
    lookup = {algo.lower(): algo for algo in ALGORITHM_CLASSES}
    lookup.update({algo.lower(): algo for algo in HEURISTIC_ALGOS})
    lookup.update(ALGORITHM_ALIASES)
    return lookup.get(str(name).lower(), name)


def _normalize_reward_weights(value):
    if value is None:
        return (0.5, 0.3, 0.2)
    if isinstance(value, (int, float)):
        return (float(value), 0.3, 0.2)
    vals = list(value)
    if len(vals) < 3:
        vals = vals + [0.0] * (3 - len(vals))
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def resolve_game_theory_config(name: str, cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    deep = name in DEEP_FUSION_ALGOS
    gt_cfg = dict(cfg.get("game_theory", {}) or {})
    defaults = {
        "enabled": True,
        "use_shapley_credit": deep,
        "ctde_with_hints": deep,
        "warm_start_steps": 1000 if deep else 0,
        "warm_start_lr_scale": 0.5,
        "shapley_samples": 128,
        "reward_weights": (0.8, 0.1, 0.1),
        "efx_enabled": True,
        "cpnet_enabled": True,
        "efx_transfer_rate": 0.5,
    }
    resolved = {**defaults, **gt_cfg}
    resolved["reward_weights"] = _normalize_reward_weights(resolved.get("reward_weights"))
    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            resolved[k] = v
    resolved["reward_weights"] = _normalize_reward_weights(resolved.get("reward_weights"))
    return resolved


def load_algo_class(name):
    mpath, cname = ALGORITHM_CLASSES[name]
    return getattr(importlib.import_module(mpath), cname)


def _read_text_with_encoding_fallback(path, encodings=("utf-8-sig", "utf-8", "gb18030", "cp936")):
    path = Path(path)
    errors = []
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: byte {exc.start} ({exc.reason})")
    joined = "; ".join(errors)
    raise UnicodeDecodeError(
        "utf-8",
        path.read_bytes(),
        0,
        1,
        f"Unable to decode {path} with supported encodings: {joined}",
    )


def load_config(path):
    text = _read_text_with_encoding_fallback(path)
    return yaml.safe_load(text) or {}


def _cfg_value(
    cfg: Dict[str, Any],
    key: str,
    default: Any = None,
    sections: Tuple[str, ...] = ("algorithm", "training", "exploration", "network"),
) -> Any:
    """Read an algorithm parameter from canonical and legacy config sections."""
    for section_name in sections:
        section = cfg.get(section_name, {}) if isinstance(cfg, dict) else {}
        if isinstance(section, dict) and key in section and section[key] is not None:
            return section[key]
    return default


class _TeeWriter:
    """Mirror stream writes to terminal and file."""

    def __init__(self, stream, mirror_file):
        self._stream = stream
        self._mirror_file = mirror_file

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        written = self._stream.write(data)
        self._mirror_file.write(data)
        return written

    def flush(self):
        self._stream.flush()
        self._mirror_file.flush()

    def isatty(self):
        isatty = getattr(self._stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self):
        fileno = getattr(self._stream, "fileno", None)
        if callable(fileno):
            return fileno()
        raise OSError("Underlying stream does not expose fileno()")

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


@contextmanager
def _tee_streams(stdout_log_path: Path, stderr_log_path: Path):
    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    with open(stdout_log_path, "w", encoding="utf-8", buffering=1) as stdout_log, open(
        stderr_log_path, "w", encoding="utf-8", buffering=1
    ) as stderr_log:
        sys.stdout = _TeeWriter(orig_stdout, stdout_log)
        sys.stderr = _TeeWriter(orig_stderr, stderr_log)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr


def _write_results_json(path: Path, payload: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _sync_latest_results_alias(output_path: Path, payload: list[dict]) -> None:
    latest_path = project_root / "results" / "benchmark.json"
    try:
        same_target = output_path.resolve() == latest_path.resolve()
    except OSError:
        same_target = output_path == latest_path
    if not same_target:
        _write_results_json(latest_path, payload)


def make_env(
    name,
    seed=None,
    num_agents=3,
    game_theory_config: Optional[Dict[str, Any]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
):
    from src.environments.mec_v3 import (
        GameTheoryMECEnv,
        GameTheoryDiscreteMAEnv,
        GameTheoryContinuousMAEnv,
    )
    mec_v3_map = {
        "MEC-v1-game-theory": GameTheoryMECEnv,
        "MEC-v1-game-theory-discrete-ma": GameTheoryDiscreteMAEnv,
        "MEC-v1-game-theory-continuous-ma": GameTheoryContinuousMAEnv,
    }
    env_cls = mec_v3_map.get(name, GameTheoryDiscreteMAEnv)
    env_kwargs: Dict[str, Any] = {"num_agents": num_agents, "num_edge_servers": 3, "max_steps": 100}
    if env_overrides:
        for k in ("num_edge_servers", "max_steps"):
            if env_overrides.get(k) is not None:
                env_kwargs[k] = int(env_overrides[k])
        for k in (
            "system_model_config",
            "dynamic_pricing_config",
            "channel_model",
            "queue_model",
            "mobility_intensity",
        ):
            if env_overrides.get(k) is not None:
                env_kwargs[k] = env_overrides[k]
        if env_overrides.get("enable_mainline_a") is not None:
            env_kwargs["enable_mainline_a"] = bool(env_overrides["enable_mainline_a"])
    gt_cfg = game_theory_config or {}
    if gt_cfg:
        env_kwargs["shapley_samples"] = int(gt_cfg.get("shapley_samples", 128))
        env_kwargs["reward_weights"] = _normalize_reward_weights(gt_cfg.get("reward_weights"))
        env_kwargs["enable_efx"] = bool(gt_cfg.get("efx_enabled", True))
        env_kwargs["enable_cp_nets"] = bool(gt_cfg.get("cpnet_enabled", True))
        env_kwargs["efx_transfer_rate"] = float(gt_cfg.get("efx_transfer_rate", 0.5))
        if "enabled" in gt_cfg and not bool(gt_cfg.get("enabled")):
            env_kwargs["enable_game_init"] = False
            env_kwargs["enable_shapley"] = False
    env = env_cls(**env_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def create_agent(name, env, cfg, device, game_theory_overrides: Optional[Dict[str, Any]] = None):
    from gymnasium import spaces

    cls = load_algo_class(name)
    ac = cfg.get("algorithm", {}) or {}
    nc = cfg.get("network", {}) or {}
    tc = cfg.get("training", {}) or {}
    is_multi = hasattr(env, "num_agents")
    obs = env.observation_space.shape[0]
    # 动作维度: Discrete用n, Continuous Box用shape[0]
    if hasattr(env.action_space, "n"):
        ad = env.action_space.n
    elif hasattr(env.action_space, "shape"):
        ad = int(env.action_space.shape[0])
    else:
        ad = int(env.action_space.high[0]) + 1
    n_agents = getattr(env, "num_agents", 1) if is_multi else 1
    gt_cfg = resolve_game_theory_config(name, cfg, game_theory_overrides)

    def pget(key: str, default: Any = None, sections: Tuple[str, ...] = ("algorithm", "training", "exploration", "network")) -> Any:
        return _cfg_value(cfg, key, default, sections)

    lr_default = pget("learning_rate", 3e-4, ("training", "algorithm", "network"))
    kw = dict(
        state_dim=obs,
        action_dim=ad,
        hidden_dim=int(pget("hidden_dim", nc.get("hidden_dim", 256), ("network", "algorithm"))),
        lr=float(pget("lr", lr_default, ("training", "algorithm", "network"))),
        gamma=float(pget("gamma", ac.get("gamma", 0.99), ("algorithm", "training"))),
        device=device,
        use_game_theory=bool(gt_cfg.get("enabled", True)),
        use_shapley_credit=bool(gt_cfg.get("use_shapley_credit", name in DEEP_FUSION_ALGOS)),
        ctde_with_hints=bool(gt_cfg.get("ctde_with_hints", name in DEEP_FUSION_ALGOS)),
        warm_start_steps=int(
            gt_cfg.get("warm_start_steps", 1000 if name in DEEP_FUSION_ALGOS else 0)
        ),
        warm_start_lr_scale=float(gt_cfg.get("warm_start_lr_scale", 0.5)),
    )
    if name == "GRPO":
        return cls(**kw, eps_clip=float(pget("eps_clip", 0.2)), num_epochs=int(pget("num_epochs", 10)),
                   group_size=int(pget("group_size", 64)))
    if name == "PPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, eps_clip=float(pget("eps_clip", 0.2)),
                   gae_lambda=float(pget("gae_lambda", 0.95)),
                   clip_grad=float(pget("clip_grad", 0.5)),
                   value_coeff=float(pget("value_coeff", 0.5)),
                   ent_coeff=float(pget("ent_coeff", 0.01)),
                   num_epochs=int(pget("num_epochs", 10)), discrete=is_discrete_env)
    if name == "CL-PPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, eps_clip=float(pget("eps_clip", 0.2)),
                   gae_lambda=float(pget("gae_lambda", 0.95)),
                   clip_grad=float(pget("clip_grad", 0.5)),
                   value_coeff=float(pget("value_coeff", 0.5)),
                   ent_coeff=float(pget("ent_coeff", 0.01)),
                   num_epochs=int(pget("num_epochs", 10)), discrete=is_discrete_env,
                   constraints=dict(cfg.get("constraints", {}) or {}),
                   risk=dict(cfg.get("risk", {}) or {}),
                   safety_layer=dict(cfg.get("safety_layer", {}) or {}))
    if name == "SAC":
        return cls(**kw, tau=float(pget("tau", 0.005)), alpha=float(pget("alpha", 0.2)),
                   automatic_entropy_tuning=bool(pget("automatic_entropy_tuning", True)),
                   batch_size=int(pget("batch_size", 256)), buffer_size=int(pget("buffer_size", 1000000)))
    if name == "DDPG":
        explore_steps = int(pget("explore_steps", pget("start_steps", 10000, ("training", "exploration"))))
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)),
                   action_scale=float(pget("action_scale", 1.0)), action_bias=float(pget("action_bias", 0.0)),
                   explore_steps=explore_steps,
                   ou_mu=float(pget("ou_mu", 0.0)),
                   ou_theta=float(pget("ou_theta", 0.15)),
                   ou_sigma=float(pget("ou_sigma", 0.2)))
    if name == "TD3":
        explore_steps = int(pget("explore_steps", pget("start_steps", 10000, ("training", "exploration"))))
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)),
                   noise_std=float(pget("noise_std", 0.2)), noise_clip=float(pget("noise_clip", 0.5)),
                   exploration_noise_std=float(pget("exploration_noise_std", pget("noise_std", 0.2))),
                   policy_delay=int(pget("policy_delay", 2)), explore_steps=explore_steps,
                   action_scale=float(pget("action_scale", 1.0)), action_bias=float(pget("action_bias", 0.0)))
    if name == "DDQN":
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)),
                   epsilon_start=float(pget("epsilon_start", 1.0)),
                   epsilon_end=float(pget("epsilon_end", 0.05)),
                   epsilon_decay=int(pget("epsilon_decay", 50000)))
    if name == "QMIX":
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)), n_agents=n_agents,
                   epsilon_start=float(pget("epsilon_start", 1.0)),
                   epsilon_end=float(pget("epsilon_end", 0.05)),
                   epsilon_decay=int(pget("epsilon_decay", 50000)))
    if name == "A3C":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, num_steps=int(pget("num_steps", 20)), discrete=is_discrete_env)
    if name == "TRPO":
        cg_iters = pget("cg_iters", pget("cg_iterations", 10))
        return cls(**kw, max_kl=float(pget("max_kl", 0.01)), cg_iters=int(cg_iters))
    if name == "SimPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, beta=float(pget("beta", 0.1)), ref_coeff=float(pget("ref_coeff", 0.2)),
                   discrete=is_discrete_env)
    if name == "MAPPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, gae_lambda=float(pget("gae_lambda", 0.95)), eps_clip=float(pget("eps_clip", 0.2)),
                   num_agents=n_agents, num_epochs=int(pget("num_epochs", 10)),
                   discrete=is_discrete_env)
    if name == "COMA":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, num_agents=n_agents, num_epochs=int(pget("num_epochs", 10)),
                   policy_clip_low=float(pget("policy_clip_low", 0.8)),
                   policy_clip_high=float(pget("policy_clip_high", 1.2)),
                   grad_clip=float(pget("grad_clip", 0.5)),
                   critic_loss_coeff=float(pget("critic_loss_coeff", 0.5)),
                   entropy_coeff=float(pget("entropy_coeff", 0.0)),
                   discrete=is_discrete_env)
    if name == "GAM-COMA":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, num_agents=n_agents, num_epochs=int(pget("num_epochs", 10)),
                   policy_clip_low=float(pget("policy_clip_low", 0.8)),
                   policy_clip_high=float(pget("policy_clip_high", 1.2)),
                   grad_clip=float(pget("grad_clip", 0.5)),
                   critic_loss_coeff=float(pget("critic_loss_coeff", 0.5)),
                   entropy_coeff=float(pget("entropy_coeff", 0.0)),
                   discrete=is_discrete_env,
                   graph_attention=dict(cfg.get("graph_attention", {}) or {}),
                   action_masking=dict(cfg.get("action_masking", {}) or {}),
                   social_influence=dict(cfg.get("social_influence", {}) or {}))
    if name == "IPPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, eps_clip=float(pget("eps_clip", 0.2)), num_epochs=int(pget("num_epochs", 10)),
                   discrete=is_discrete_env, num_agents=n_agents)
    if name == "VDN":
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)), n_agents=n_agents,
                   epsilon_start=float(pget("epsilon_start", 1.0)),
                   epsilon_end=float(pget("epsilon_end", 0.05)),
                   epsilon_decay=int(pget("epsilon_decay", 50000)))
    if name == "MADDPG":
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)), n_agents=n_agents)
    if name == "IQL":
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)),
                   epsilon_start=float(pget("epsilon_start", 1.0)),
                   epsilon_end=float(pget("epsilon_end", 0.05)),
                   epsilon_decay=int(pget("epsilon_decay", 50000)))
    if name == "MATD3":
        return cls(**kw, tau=float(pget("tau", 0.005)), batch_size=int(pget("batch_size", 256)),
                   buffer_size=int(pget("buffer_size", 1000000)), n_agents=n_agents,
                   noise_std=float(pget("noise_std", 0.2)), policy_delay=int(pget("policy_delay", 2)))
    raise ValueError(name)

def _resolve_num_agents(algo_name: str, env_overrides: Optional[Dict[str, Any]] = None) -> int:
    if algo_name in MULTI_AGENT_ALGOS or algo_name in HEURISTIC_ALGOS:
        if env_overrides and env_overrides.get("num_agents_multi") is not None:
            return int(env_overrides["num_agents_multi"])
        return 3
    return 1


def _parse_obs_core(obs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    queue = obs[:k]
    snr = obs[k : 2 * k]
    return queue, snr


def _nearest_bin(value: float) -> int:
    bins = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    idx = int(np.argmin(np.abs(bins - float(np.clip(value, -1.0, 1.0)))))
    return int(np.clip(idx, 0, len(bins) - 1))


def _encode_discrete_action(target: int, ratio: np.ndarray, num_edge_servers: int) -> int:
    target = int(np.clip(target, 0, num_edge_servers))
    ratio = np.asarray(ratio, dtype=np.float32).reshape(-1)
    if ratio.size < 3:
        ratio = np.pad(ratio, (0, 3 - ratio.size), constant_values=0.0)
    r0, r1, r2 = _nearest_bin(float(ratio[0])), _nearest_bin(float(ratio[1])), _nearest_bin(float(ratio[2]))
    ratio_idx = r0 * 25 + r1 * 5 + r2
    return target * 125 + ratio_idx


def _build_heuristic_action(
    policy_name: str,
    obs: np.ndarray,
    num_edge_servers: int,
    action_space,
    rng: np.random.Generator,
):
    queue, snr = _parse_obs_core(obs, num_edge_servers)
    q_norm = queue / max(float(np.max(queue) + 1e-6), 1.0)
    snr_norm = np.clip(snr, 0.0, 1.0)

    if policy_name == "Random":
        target = int(rng.integers(0, num_edge_servers + 1))
    elif policy_name == "Local-only":
        target = 0
    elif policy_name == "Full-offload":
        target = int(np.argmax(snr_norm)) + 1
    else:
        # Greedy: prefer higher SNR and lower queue; fallback to local if score weak.
        server_scores = snr_norm - 0.6 * q_norm
        best_idx = int(np.argmax(server_scores))
        target = best_idx + 1 if float(server_scores[best_idx]) > 0.10 else 0

    if policy_name == "Local-only":
        ratio = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    elif policy_name == "Full-offload":
        ratio = np.array([1.0, 0.0, 0.2], dtype=np.float32)
    elif policy_name == "Random":
        ratio = rng.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32)
    else:
        # Greedy ratio: more offload with better channel and lighter queue.
        if target == 0:
            ratio = np.array([-0.6, -0.2, -0.6], dtype=np.float32)
        else:
            idx = target - 1
            off = np.clip(0.2 + 0.8 * snr_norm[idx] * (1.0 - q_norm[idx]), 0.1, 0.95)
            ratio = np.array([2.0 * off - 1.0, 0.0, -0.2], dtype=np.float32)

    if hasattr(action_space, "n"):
        return _encode_discrete_action(target=target, ratio=ratio, num_edge_servers=num_edge_servers)

    target_selector = -1.0 + 2.0 * target / max(1, num_edge_servers)
    action = np.array([target_selector, ratio[0], ratio[1], ratio[2]], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def _extract_step_metrics(info: Dict[str, Any]) -> Dict[str, Any]:
    lat = 0.0
    eng = 0.0
    task_count = 0
    deadline_misses = 0
    latency_samples = np.asarray([], dtype=np.float64)
    energy_samples = np.asarray([], dtype=np.float64)
    if "latency_components" in info:
        components = info.get("latency_components") or []
        latency_samples = np.asarray([float(c.get("e2e_latency", 0.0)) for c in components], dtype=np.float64)
        deadline_misses = int(
            sum(float(c.get("e2e_latency", 0.0)) > float(c.get("deadline", np.inf)) for c in components)
        )
        lat = float(np.sum(latency_samples))
        task_count = max(task_count, int(latency_samples.size))
    elif "individual_latencies" in info:
        latency_samples = np.asarray(info["individual_latencies"], dtype=np.float64).reshape(-1)
        lat = float(np.sum(latency_samples))
        task_count = max(task_count, int(latency_samples.size))
    elif "latency" in info:
        lat = float(info["latency"])
        latency_samples = np.asarray([lat], dtype=np.float64)
        task_count = max(task_count, 1)
    if "individual_energies" in info:
        energy_samples = np.asarray(info["individual_energies"], dtype=np.float64).reshape(-1)
        eng = float(np.sum(energy_samples))
        task_count = max(task_count, int(energy_samples.size))
    elif "energy" in info:
        eng = float(info["energy"])
        energy_samples = np.asarray([eng], dtype=np.float64)
        task_count = max(task_count, 1)
    return {
        "latency_total": lat,
        "energy_total": eng,
        "latency_samples": latency_samples,
        "energy_samples": energy_samples,
        "deadline_misses": deadline_misses,
        "task_count": task_count,
    }


def benchmark_heuristic(
    name: str,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
    override_ep: Optional[int] = None,
    env_name: Optional[str] = None,
    game_theory_overrides: Optional[Dict[str, Any]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    correct_env_name = env_name if env_name is not None else ALGO_ENV_MAP.get(name, "MEC-v1-game-theory-continuous-ma")
    num_agents = _resolve_num_agents(name, env_overrides)
    gt_cfg = resolve_game_theory_config("GRPO", {"game_theory": {}}, game_theory_overrides)
    env = make_env(
        correct_env_name,
        seed=seed,
        num_agents=num_agents,
        game_theory_config=gt_cfg,
        env_overrides=env_overrides,
    )

    episodes = int(override_ep if override_ep is not None else 10)
    ep_rewards: List[float] = []
    ep_latencies: List[float] = []
    ep_energies: List[float] = []
    e2e_latency_samples: List[float] = []
    e2e_energy_samples: List[float] = []
    deadline_misses = 0
    total_tasks = 0
    total_steps = 0

    start = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        r_total = 0.0
        l_total = 0.0
        e_total = 0.0
        while not done:
            if isinstance(obs, list):
                actions = [
                    _build_heuristic_action(name, o, getattr(env, "num_edge_servers", 3), env.action_space, rng)
                    for o in obs
                ]
                obs, reward, terminated, truncated, info = env.step(actions)
                if isinstance(reward, (list, np.ndarray)):
                    r_total += float(np.sum(np.asarray(reward, dtype=np.float64)))
                else:
                    r_total += float(reward)
            else:
                action = _build_heuristic_action(name, obs, getattr(env, "num_edge_servers", 3), env.action_space, rng)
                obs, reward, terminated, truncated, info = env.step(action)
                r_total += float(reward)
            metrics = _extract_step_metrics(info)
            l_total += metrics["latency_total"]
            e_total += metrics["energy_total"]
            e2e_latency_samples.extend(metrics["latency_samples"].tolist())
            e2e_energy_samples.extend(metrics["energy_samples"].tolist())
            deadline_misses += int(metrics["deadline_misses"])
            total_tasks += int(metrics["task_count"])
            total_steps += 1
            done = bool(terminated) or bool(truncated)
        ep_rewards.append(r_total)
        ep_latencies.append(l_total)
        ep_energies.append(e_total)
    elapsed = time.time() - start
    e2e_arr = np.asarray(e2e_latency_samples, dtype=np.float64)
    energy_arr = np.asarray(e2e_energy_samples, dtype=np.float64)
    e2e_mean = float(np.mean(e2e_arr)) if e2e_arr.size else float(np.mean(ep_latencies))
    e2e_p95 = float(np.percentile(e2e_arr, 95)) if e2e_arr.size else e2e_mean
    energy_per_task = float(np.mean(energy_arr)) if energy_arr.size else float(np.mean(ep_energies))
    deadline_miss_rate = float(deadline_misses / max(total_tasks, 1))
    throughput = float(total_tasks / max(total_steps, 1))
    comm_score = float(100.0 * throughput * max(0.0, 1.0 - deadline_miss_rate) / (1.0 + e2e_p95 + 0.3 * energy_per_task))
    avg_steps_per_episode = float(total_steps / max(episodes, 1))

    result = {
        "algorithm": name,
        "environment": correct_env_name,
        "seed": seed,
        "device": device,
        "train_timesteps": 0,
        "train_time_seconds": round(elapsed, 2),
        "final_reward_mean": round(float(np.mean(ep_rewards)), 4),
        "final_reward_std": round(float(np.std(ep_rewards)), 4),
        "final_latency_mean": round(e2e_mean, 4),
        "final_e2e_latency_mean": round(e2e_mean, 4),
        "final_e2e_latency_p95": round(e2e_p95, 4),
        "final_deadline_miss_rate": round(deadline_miss_rate, 4),
        "final_throughput_tasks_per_step": round(throughput, 4),
        "final_comm_score": round(comm_score, 4),
        "final_energy_mean": round(float(np.mean(ep_energies)), 4),
        "final_latency_total_mean": round(float(np.mean(ep_latencies)), 4),
        "final_energy_total_mean": round(float(np.mean(ep_energies)), 4),
        "final_latency_per_step_mean": round(float(np.mean(ep_latencies)) / max(avg_steps_per_episode, 1.0), 4),
        "final_energy_per_step_mean": round(float(np.mean(ep_energies)) / max(avg_steps_per_episode, 1.0), 4),
        "final_latency_per_task_mean": round(e2e_mean, 4),
        "final_energy_per_task_mean": round(energy_per_task, 4),
        "total_episodes": episodes,
        "total_updates": 0,
    }
    if verbose:
        print(f"[{name}] reward={result['final_reward_mean']:.4f}+/-{result['final_reward_std']:.4f}  eval_time={elapsed:.1f}s")
    return result


def benchmark_single(
    name,
    cfg,
    seed=42,
    device="auto",
    verbose=True,
    override_ts=None,
    override_ep=None,
    env_name=None,
    game_theory_overrides: Optional[Dict[str, Any]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    eval_interval_override: Optional[int] = None,
    min_eval_points: Optional[int] = None,
    eval_at_start: Optional[bool] = None,
    environment_profile: str = DEFAULT_ENVIRONMENT_PROFILE,
    config_path=None,
):
    """
    运行单算法评测。

    环境自动选择: ALGO_ENV_MAP 保证了每个算法在正确的环境类型上评测，
    确保离散/连续/多智能体算法各在其适对应的环境。
    """
    import torch
    from src.trainer.on_policy_trainer import OnPolicyTrainer
    from src.trainer.off_policy_trainer import OffPolicyTrainer
    from src.utils.helpers import set_seed

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    # 自动选择正确环境 (修复: 每个算法在其适配的环境上评测)
    correct_env_name = env_name if env_name is not None else ALGO_ENV_MAP.get(name, "MEC-v1-game-theory-discrete-ma")
    num_agents = _resolve_num_agents(name, env_overrides)
    gt_cfg = resolve_game_theory_config(name, cfg, game_theory_overrides)
    env = make_env(
        correct_env_name,
        seed=seed,
        num_agents=num_agents,
        game_theory_config=gt_cfg,
        env_overrides=env_overrides,
    )
    agent = create_agent(name, env, cfg, device, game_theory_overrides=game_theory_overrides)
    if game_theory_overrides and game_theory_overrides.get("game_aware_enabled") is not None:
        setattr(agent, "game_aware_enabled", bool(game_theory_overrides["game_aware_enabled"]))
    TrainerCls = OnPolicyTrainer if name in ON_POLICY else OffPolicyTrainer
    tc = cfg.get("training", {})
    ec = cfg.get("evaluation", {})
    lc = cfg.get("logging", {})
    ac = cfg.get("algorithm", {})
    total_ts = int(override_ts if override_ts is not None else tc.get("total_timesteps", 100000))
    rollout_steps = min(int(tc.get("rollout_steps", 2048)), int(total_ts))
    eval_ep = int(override_ep if override_ep is not None else ec.get("num_episodes", 10))
    eval_interval = int(eval_interval_override if eval_interval_override is not None else lc.get("eval_interval", 5000))
    min_eval_points_used = int(min_eval_points if min_eval_points is not None else lc.get("min_eval_points", 0) or 0)
    eval_at_start_used = bool(lc.get("eval_at_start", False) if eval_at_start is None else eval_at_start)
    trainer_kwargs = dict(
        env=env, agent=agent, total_timesteps=total_ts,
        rollout_steps=rollout_steps,
        eval_interval=eval_interval, eval_episodes=eval_ep,
        min_eval_points=min_eval_points_used, eval_at_start=eval_at_start_used,
        save_interval=lc.get("save_interval", 50000),
        save_dir=lc.get("checkpoint_dir", "checkpoints/" + name.lower()),
        log_interval=lc.get("log_interval", 10), device=device, seed=seed,
    )
    if name in ON_POLICY:
        trainer_kwargs["num_epochs"] = ac.get("num_epochs", 10)
    else:
        trainer_kwargs["warmup_steps"] = tc.get("start_steps", 10000)
        trainer_kwargs["update_interval"] = ac.get("update_interval", 1)
    trainer = TrainerCls(**trainer_kwargs)
    if verbose:
        print("=" * 60)
        print(f"Algorithm: {name}  |  Env: {correct_env_name}  |  Device: {device}")
        print("=" * 60)
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    # Module 8: collect convergence data from training eval logs
    convergence_data = {}
    for key, values in trainer.eval_logs.items():
        convergence_data[key] = [round(v, 6) for v in values]
    convergence_data["schema_version"] = 2
    convergence_data["metric_schema_version"] = 3
    convergence_data["seed"] = seed
    convergence_data["algorithm"] = name
    convergence_data["run_status"] = "success"
    convergence_data["failure_reason"] = None
    convergence_data["eval_interval"] = trainer.eval_interval
    convergence_data["effective_eval_interval"] = trainer._effective_eval_interval()
    convergence_data["min_eval_points"] = trainer.min_eval_points
    convergence_data["eval_at_start"] = trainer.eval_at_start
    convergence_data["eval_steps"] = list(trainer.eval_steps)
    convergence_data["eval_sample_count"] = len(trainer.eval_steps)
    convergence_data["total_timesteps"] = trainer.total_steps
    fe = trainer.evaluate()
    latency_e2e = fe.get("eval/e2e_latency_mean", fe.get("eval/latency_per_task_mean", fe.get("eval/latency_mean", 0.0)))
    latency_p95 = fe.get("eval/e2e_latency_p95", latency_e2e)
    deadline_miss_rate = fe.get("eval/deadline_miss_rate", None)
    throughput = fe.get("eval/throughput_tasks_per_step", None)
    comm_score = fe.get("eval/comm_score", None)
    latency_total = fe.get("eval/latency_total_mean", 0.0)
    energy_total = fe.get("eval/energy_total_mean", fe.get("eval/energy_mean", 0.0))
    latency_per_step = fe.get("eval/latency_per_step_mean", None)
    energy_per_step = fe.get("eval/energy_per_step_mean", None)
    latency_per_task = fe.get("eval/latency_per_task_mean", None)
    energy_per_task = fe.get("eval/energy_per_task_mean", None)
    result = {
        "algorithm": name, "environment": correct_env_name, "seed": seed, "device": device,
        "train_timesteps": trainer.total_steps, "train_time_seconds": round(elapsed, 2),
        "final_reward_mean": round(fe["eval/reward_mean"], 4),
        "final_reward_std": round(fe["eval/reward_std"], 4),
        "final_latency_mean": round(latency_e2e, 4),
        "final_e2e_latency_mean": round(latency_e2e, 4),
        "final_e2e_latency_p95": round(latency_p95, 4),
        "final_deadline_miss_rate": None if deadline_miss_rate is None else round(deadline_miss_rate, 4),
        "final_throughput_tasks_per_step": None if throughput is None else round(throughput, 4),
        "final_comm_score": None if comm_score is None else round(comm_score, 4),
        "final_energy_mean": round(energy_total, 4),
        "final_latency_total_mean": round(latency_total, 4),
        "final_energy_total_mean": round(energy_total, 4),
        "final_latency_per_step_mean": None if latency_per_step is None else round(latency_per_step, 4),
        "final_energy_per_step_mean": None if energy_per_step is None else round(energy_per_step, 4),
        "final_latency_per_task_mean": None if latency_per_task is None else round(latency_per_task, 4),
        "final_energy_per_task_mean": None if energy_per_task is None else round(energy_per_task, 4),
        "final_social_welfare_mean": round(fe.get("eval/social_welfare_mean", fe["eval/reward_mean"]), 4),
        "final_social_welfare_per_step_mean": round(fe.get("eval/social_welfare_per_step_mean", 0.0), 4),
        "final_agent_reward_jain_mean": round(fe.get("eval/agent_reward_jain_mean", 1.0), 4),
        "final_constraint_violation_rate": round(fe.get("eval/constraint/any_violation_mean", 0.0), 4),
        "final_constraint_penalty_mean": round(fe.get("eval/constraint/penalty_mean_mean", 0.0), 4),
        "final_queue_wait_mean": round(fe.get("eval/queue_wait_mean_mean", 0.0), 4),
        "final_offload_ratio_mean": round(fe.get("eval/offload_ratio_mean_mean", 0.0), 4),
        "final_price_mean": round(fe.get("eval/pricing/price_mean", 0.0), 4),
        "final_provider_revenue_proxy_mean": round(fe.get("eval/pricing/provider_revenue_proxy_mean", 0.0), 4),
        "final_efx_satisfaction_rate": round(fe.get("eval/fairness/efx_satisfied_mean", 0.0), 4),
        "total_episodes": trainer.episode_count, "total_updates": trainer.update_count,
        "convergence": convergence_data,
    }
    result["resolved_runtime_config"] = build_resolved_runtime_config(
        algorithm=name,
        config_path=config_path,
        base_algorithm_config=cfg,
        cli_overrides={
            "timesteps": override_ts,
            "seed": seed,
            "device": device,
            "eval_episodes": override_ep,
            "env": env_name if env_name is not None else "auto",
            "environment_profile": environment_profile,
        },
        environment=correct_env_name,
        environment_profile=environment_profile,
        env_overrides=env_overrides or {},
        game_theory_config=gt_cfg,
        trainer_kwargs=trainer_kwargs,
        agent=agent,
        env=env,
        train_timesteps=trainer.total_steps,
        eval_episodes=eval_ep,
    )
    if verbose:
        print(f"[{name}] reward={result['final_reward_mean']:.4f}+/-{result['final_reward_std']:.4f}  time={result['train_time_seconds']:.1f}s")
    return result


def _to_float(value):
    """Best-effort numeric conversion for aggregation."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_first_numeric(record: Dict[str, Any], keys: List[str]) -> float | None:
    """Pick the first finite numeric value from a benchmark record."""
    for key in keys:
        value = _to_float(record.get(key))
        if value is not None and np.isfinite(value):
            return value
    return None


def _pick_latency_metric(record: Dict[str, Any]) -> float | None:
    """Return the preferred latency metric for scoring/reporting."""
    return _pick_first_numeric(
        record,
        [
            "final_latency_per_task_mean_mean",
            "final_latency_per_task_mean",
            "final_latency_total_mean_mean",
            "final_latency_mean_mean",
            "final_latency_total_mean",
            "final_latency_mean",
        ],
    )


def _pick_energy_metric(record: Dict[str, Any]) -> float | None:
    """Return the preferred energy metric for scoring/reporting."""
    return _pick_first_numeric(
        record,
        [
            "final_energy_per_task_mean_mean",
            "final_energy_per_task_mean",
            "final_energy_total_mean_mean",
            "final_energy_mean_mean",
            "final_energy_total_mean",
            "final_energy_mean",
        ],
    )


def _fmt_metric(value, width, precision):
    """Format metric value for table output, with NA fallback."""
    numeric = _to_float(value)
    if numeric is None or not np.isfinite(numeric):
        return f"{'NA':>{width}}"
    return f"{numeric:>{width}.{precision}f}"


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _resolve_env_overrides(
    scale: Optional[str],
    num_edge_servers: Optional[int],
    multi_agent_count: Optional[int],
    max_steps: Optional[int],
    system_model_config: Optional[str] = None,
    dynamic_pricing_config: Optional[str] = None,
    enable_mainline_a: Optional[bool] = None,
    channel_model: Optional[str] = None,
    queue_model: Optional[str] = None,
    mobility_intensity: Optional[str] = None,
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    if scale:
        preset = SCALE_PRESETS[str(scale).lower()]
        resolved["num_edge_servers"] = int(preset["num_edge_servers"])
        resolved["num_agents_multi"] = int(preset["num_agents_multi"])
        resolved["max_steps"] = int(preset["max_steps"])
    if num_edge_servers is not None:
        resolved["num_edge_servers"] = int(num_edge_servers)
    if multi_agent_count is not None:
        resolved["num_agents_multi"] = int(multi_agent_count)
    if max_steps is not None:
        resolved["max_steps"] = int(max_steps)
    if system_model_config is not None:
        resolved["system_model_config"] = system_model_config
    if dynamic_pricing_config is not None:
        resolved["dynamic_pricing_config"] = dynamic_pricing_config
    if enable_mainline_a is not None:
        resolved["enable_mainline_a"] = bool(enable_mainline_a)
    if channel_model is not None:
        resolved["channel_model"] = channel_model
    if queue_model is not None:
        resolved["queue_model"] = queue_model
    if mobility_intensity is not None:
        resolved["mobility_intensity"] = mobility_intensity
    return resolved


def _coerce_list(value: Any, default: Optional[List[Any]] = None) -> List[Any]:
    """Coerce config scalar/list values to a list."""
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _load_benchmark_cli_config(path: Optional[str]) -> Dict[str, Any]:
    """Load optional benchmark config."""
    if path is None:
        return {}
    config = load_config(path)
    validate_mainline_a_system_model_config(config, path)
    return config


def _apply_benchmark_cli_config(args, file_cfg: Dict[str, Any]) -> None:
    """Apply experiment config defaults to benchmark CLI arguments."""
    if not file_cfg:
        return
    benchmark_cfg = file_cfg.get("benchmark", {})
    if args.timesteps is None and (file_cfg.get("steps") or benchmark_cfg.get("steps")):
        args.timesteps = int(file_cfg.get("steps") or benchmark_cfg.get("steps"))
    if args.seeds == [42] and (file_cfg.get("seeds") or benchmark_cfg.get("seeds")):
        args.seeds = [int(seed) for seed in _coerce_list(file_cfg.get("seeds") or benchmark_cfg.get("seeds"))]
    if args.num_edge_servers is None and file_cfg.get("edges") is not None and not isinstance(file_cfg.get("edges"), list):
        args.num_edge_servers = int(file_cfg["edges"])
    if args.multi_agent_count is None and file_cfg.get("users") is not None and not isinstance(file_cfg.get("users"), list):
        args.multi_agent_count = int(file_cfg["users"])
    if file_cfg.get("system_model", {}).get("enabled"):
        args.enable_mainline_a = True
        if args.system_model_config is None:
            args.system_model_config = args.config
    if file_cfg.get("dynamic_pricing") and args.dynamic_pricing_config is None:
        args.dynamic_pricing_config = args.config
    if args.queue_model is None:
        args.queue_model = resolve_queue_model(file_cfg)
    if args.channel_model is None:
        args.channel_model = resolve_channel_model(file_cfg)
    if getattr(args, "eval_interval", None) is None and benchmark_cfg.get("eval_interval") is not None:
        args.eval_interval = int(benchmark_cfg["eval_interval"])
    if getattr(args, "min_eval_points", None) is None and benchmark_cfg.get("min_eval_points") is not None:
        args.min_eval_points = int(benchmark_cfg["min_eval_points"])
    if getattr(args, "eval_at_start", None) is None and benchmark_cfg.get("eval_at_start") is not None:
        args.eval_at_start = bool(benchmark_cfg["eval_at_start"])


def _resolve_profile_env_overrides(args) -> Dict[str, Any]:
    """Resolve benchmark profile settings without starting training."""
    profile = resolve_environment_profile(args.environment_profile)
    return profile_to_env_overrides(
        profile,
        system_model_config=args.system_model_config,
        dynamic_pricing_config=args.dynamic_pricing_config,
        enable_mainline_a=True if args.enable_mainline_a else None,
    )


def _dry_run_payload(args, file_cfg: Dict[str, Any], algorithms: List[str]) -> Dict[str, Any]:
    """Build a dry-run payload without starting training."""
    benchmark_cfg = file_cfg.get("benchmark", {})
    system_model_cfg = validate_mainline_a_system_model_config(file_cfg, args.config or "benchmark config")
    profile = resolve_environment_profile(args.environment_profile)
    profile_env_overrides = _resolve_profile_env_overrides(args)
    channel_model = args.channel_model or resolve_channel_model(file_cfg)
    queue_model = args.queue_model or resolve_queue_model(file_cfg)
    eval_interval = getattr(args, "eval_interval", None)
    min_eval_points = getattr(args, "min_eval_points", None)
    eval_at_start = getattr(args, "eval_at_start", None)
    return {
        "dry_run": True,
        "environment_profile": profile.name,
        "algorithms": algorithms,
        "seeds": args.seeds or _coerce_list(file_cfg.get("seeds"), [42]),
        "timesteps": args.timesteps or file_cfg.get("steps") or benchmark_cfg.get("steps"),
        "eval_interval": eval_interval or benchmark_cfg.get("eval_interval"),
        "min_eval_points": min_eval_points if min_eval_points is not None else benchmark_cfg.get("min_eval_points"),
        "eval_at_start": bool(eval_at_start if eval_at_start is not None else benchmark_cfg.get("eval_at_start", False)),
        "enable_mainline_a": bool(
            profile_env_overrides.get("enable_mainline_a", False)
            or system_model_cfg.get("enabled", False)
        ),
        "system_model_config": profile_env_overrides.get("system_model_config"),
        "dynamic_pricing_config": profile_env_overrides.get("dynamic_pricing_config"),
        "channel_model": channel_model,
        "queue_model": queue_model,
        "resolved_system_model": system_model_cfg,
        "mobility_intensity": args.mobility_intensity,
        "config": args.config,
    }


def run_benchmark(algorithms, env_name=None, configs_dir=None, seeds=None,
                 timesteps=None, episodes=None, device="auto",
                 output_file=None, verbose=True, game_theory_overrides: Optional[Dict[str, Any]] = None,
                 env_overrides: Optional[Dict[str, Any]] = None,
                 sync_latest_alias: bool = True,
                 eval_interval: Optional[int] = None,
                 min_eval_points: Optional[int] = None,
                 eval_at_start: Optional[bool] = None,
                 environment_profile: str = DEFAULT_ENVIRONMENT_PROFILE):
    """
    运行多算法对比评测 (GameTheory 环境唯一)。

    环境公平性: 所有算法统一在 GameTheory 适配环境上评测:
    - 离散动作 → MEC-v1-game-theory-discrete-ma
    - 连续动作 → MEC-v1-game-theory-continuous-ma

    当 env_name 被显式指定时，仅在该环境下评测；
    当 env_name 为 None 或 "auto" 时，每个算法使用其对应的正确环境。
    """
    if seeds is None:
        seeds = [42]
    if configs_dir is None:
        configs_dir = project_root / "configs" / "algorithms"
    if env_name is None:
        env_name = "auto"  # 自动选择正确环境
    algorithms = [_canonical_algorithm_name(a) for a in algorithms]

    # 按环境分组，相同环境的算法一起评测
    if env_name == "auto":
        # 分组: 每个算法在其正确环境上
        env_groups = {}  # env_name -> list of algos
        for algo in algorithms:
            correct_env = ALGO_ENV_MAP.get(algo, "MEC-v1-game-theory-discrete-ma")
            env_groups.setdefault(correct_env, []).append(algo)
        run_all_groups = True
    else:
        # 用户指定了单一环境，所有算法在同一环境
        env_groups = {env_name: algorithms}
        run_all_groups = False

    all_results = []
    for group_env, group_algos in env_groups.items():
        group_results = []
        for algo in group_algos:
            algo_results = []
            algo_errors = []
            for seed in seeds:
                try:
                    if algo in HEURISTIC_ALGOS:
                        r = benchmark_heuristic(
                            name=algo,
                            seed=seed,
                            device=device,
                            verbose=verbose,
                            override_ep=episodes,
                            env_name=group_env,
                            game_theory_overrides=game_theory_overrides,
                            env_overrides=env_overrides,
                        )
                    else:
                        cp = Path(configs_dir) / (algo.lower() + ".yaml")
                        cfg = load_config(str(cp)) if cp.exists() else {
                            "algorithm": {}, "network": {}, "training": {},
                            "evaluation": {}, "logging": {}
                        }
                        r = benchmark_single(
                            algo,
                            cfg,
                            seed,
                            device,
                            verbose,
                            timesteps,
                            episodes,
                            env_name=group_env,
                            game_theory_overrides=game_theory_overrides,
                            env_overrides=env_overrides,
                            eval_interval_override=eval_interval,
                            min_eval_points=min_eval_points,
                            eval_at_start=eval_at_start,
                            environment_profile=environment_profile,
                            config_path=cp,
                        )
                    algo_results.append(r)
                except Exception as e:
                    failure_reason = str(e)
                    algo_errors.append(
                        {
                            "seed": seed,
                            "error": failure_reason,
                            "convergence": {
                                "schema_version": 2,
                                "seed": seed,
                                "algorithm": algo,
                                "run_status": "failed",
                                "failure_reason": failure_reason,
                            },
                        }
                    )
                    logging.getLogger(__name__).exception("Error %s seed=%s", algo, seed)
                    if verbose:
                        print(f"Error {algo} seed={seed}: {e}")
            if algo_results:
                avg = {
                    "algorithm": algo,
                    "n_seeds": len(algo_results),
                    "environment_profile": environment_profile,
                }
                for key in algo_results[0].keys():
                    if key in ("algorithm", "environment", "seed", "device", "n_seeds"):
                        avg[key] = algo_results[0][key]
                        continue
                    if key == "resolved_runtime_config":
                        avg[key] = algo_results[0][key]
                        continue
                    vals = []
                    for r in algo_results:
                        val = _to_float(r.get(key))
                        if val is not None:
                            vals.append(val)
                    if vals:
                        avg[key + "_mean"] = round(np.mean(vals), 4)
                        if len(vals) > 1:
                            avg[key + "_std"] = round(np.std(vals), 4)
                avg.setdefault("final_reward_mean_mean", None)
                avg.setdefault("final_latency_mean_mean", None)
                avg.setdefault("final_e2e_latency_mean_mean", avg.get("final_latency_mean_mean"))
                avg.setdefault("final_e2e_latency_p95_mean", None)
                avg.setdefault("final_deadline_miss_rate_mean", None)
                avg.setdefault("final_throughput_tasks_per_step_mean", None)
                avg.setdefault("final_comm_score_mean", None)
                avg.setdefault("final_energy_mean_mean", None)
                avg.setdefault("final_latency_total_mean_mean", None)
                avg.setdefault("final_energy_total_mean_mean", None)
                avg.setdefault("final_latency_per_step_mean_mean", None)
                avg.setdefault("final_energy_per_step_mean_mean", None)
                avg.setdefault("final_latency_per_task_mean_mean", None)
                avg.setdefault("final_energy_per_task_mean_mean", None)
                avg.setdefault("final_social_welfare_mean_mean", None)
                avg.setdefault("final_agent_reward_jain_mean_mean", None)
                avg.setdefault("final_constraint_violation_rate_mean", None)
                avg.setdefault("final_queue_wait_mean_mean", None)
                avg.setdefault("final_offload_ratio_mean_mean", None)
                avg.setdefault("train_time_seconds_mean", None)
                # Module 8: collect convergence data per seed (no cross-seed averaging)
                convergence_by_seed = {}
                for r in algo_results:
                    seed_val = r.get("seed")
                    conv = r.get("convergence")
                    if conv is not None and seed_val is not None:
                        convergence_by_seed[str(seed_val)] = conv
                for error in algo_errors:
                    conv = error.get("convergence")
                    seed_val = error.get("seed")
                    if conv is not None and seed_val is not None:
                        convergence_by_seed[str(seed_val)] = conv
                if convergence_by_seed:
                    avg["convergence_by_seed"] = convergence_by_seed
                avg["status"] = "ok" if not algo_errors else "partial"
                if algo_errors:
                    avg["failed_seeds"] = len(algo_errors)
                    avg["errors"] = algo_errors
                group_results.append(avg)
                all_results.append(avg)
            else:
                failed = {
                    "algorithm": algo,
                    "environment": group_env,
                    "environment_profile": environment_profile,
                    "device": device,
                    "n_seeds": 0,
                    "status": "failed",
                    "final_reward_mean_mean": None,
                    "final_latency_mean_mean": None,
                    "final_e2e_latency_mean_mean": None,
                    "final_e2e_latency_p95_mean": None,
                    "final_deadline_miss_rate_mean": None,
                    "final_throughput_tasks_per_step_mean": None,
                    "final_comm_score_mean": None,
                    "final_energy_mean_mean": None,
                    "final_latency_total_mean_mean": None,
                    "final_energy_total_mean_mean": None,
                    "final_latency_per_step_mean_mean": None,
                    "final_energy_per_step_mean_mean": None,
                    "final_latency_per_task_mean_mean": None,
                    "final_energy_per_task_mean_mean": None,
                    "final_social_welfare_mean_mean": None,
                    "final_agent_reward_jain_mean_mean": None,
                    "final_constraint_violation_rate_mean": None,
                    "final_queue_wait_mean_mean": None,
                    "final_offload_ratio_mean_mean": None,
                    "train_time_seconds_mean": None,
                    "errors": algo_errors or [{"seed": None, "error": "Unknown benchmark failure"}],
                }
                convergence_by_seed = {}
                for error in algo_errors:
                    conv = error.get("convergence")
                    seed_val = error.get("seed")
                    if conv is not None and seed_val is not None:
                        convergence_by_seed[str(seed_val)] = conv
                if convergence_by_seed:
                    failed["convergence_by_seed"] = convergence_by_seed
                group_results.append(failed)
                all_results.append(failed)

        if group_results:
            print("=" * 80)
            env_label = group_env if run_all_groups else env_name
            print(f"BENCHMARK  --  {env_label}  ({len(group_algos)} algorithms)")
            print("=" * 80)
            header = f"{'Algorithm':<12} {'Reward Mean':>12} {'Reward Std':>12} {'Time (s)':>10}"
            print(header)
            print("-" * 80)
            for r in group_results:
                rm = _fmt_metric(r.get("final_reward_mean_mean"), 12, 4)
                rs = _fmt_metric(r.get("final_reward_std_mean"), 12, 4)
                tm = _fmt_metric(r.get("train_time_seconds_mean"), 10, 1)
                print(f"{r['algorithm']:<12} {rm} {rs} {tm}")
            print("=" * 80)

    # Module 9: Composite Score calculation
    scoring_profiles_path = Path(__file__).parent.parent / "configs" / "scoring_profiles.yaml"
    if scoring_profiles_path.exists() and all_results:
        try:
            with open(scoring_profiles_path, encoding="utf-8") as f:
                profiles_cfg = yaml.safe_load(f)
            profiles = profiles_cfg.get("profiles", {})
            if profiles:
                scorer = CompositeScorer(profiles)
                # Prepare results for scoring: extract required fields
                scoring_input = []
                for r in all_results:
                    entry = {
                        "algorithm": r.get("algorithm", "unknown"),
                        "reward_mean": _to_float(r.get("final_reward_mean_mean", r.get("final_reward_mean", 0))) or 0.0,
                        "reward_std": _to_float(r.get("final_reward_std_mean", r.get("final_reward_std", 0))) or 0.0,
                        "latency_mean": _pick_latency_metric(r) or 0.0,
                        "energy_mean": _pick_energy_metric(r) or 0.0,
                    }
                    scoring_input.append(entry)
                # Score all profiles
                all_profile_scores = scorer.score_all_profiles(scoring_input)
                robustness = scorer.robustness_summary(scoring_input)
                # Attach scores back to all_results
                robustness_map = {item["algorithm"]: item for item in robustness}
                for r in all_results:
                    algo = r.get("algorithm", "unknown")
                    r["composite_scores"] = {}
                    for profile_name, scored_list in all_profile_scores.items():
                        for scored in scored_list:
                            if scored["algorithm"] == algo:
                                r["composite_scores"][profile_name] = {
                                    "score": scored["composite_score"],
                                    "rank": scored["rank"],
                                    "breakdown": scored.get("breakdown", {}),
                                }
                                break
                    if algo in robustness_map:
                        r["robustness"] = robustness_map[algo]
                if verbose:
                    print(f"Composite scores computed for {len(all_results)} algorithms across {len(profiles)} profiles")
        except Exception as e:
            logging.getLogger(__name__).warning("Composite scoring failed: %s", e)

    if output_file:
        op = Path(output_file)
        _write_results_json(op, all_results)
        if sync_latest_alias:
            _sync_latest_results_alias(op, all_results)
        if verbose:
            print(f"\nResults saved to: {op}")
            if sync_latest_alias:
                print(f"Latest alias updated: {project_root / 'results' / 'benchmark.json'}")
            else:
                print("Latest alias not updated")
    print("Benchmark finished")
    return all_results


def main():
    p = argparse.ArgumentParser(description="GRPO_MEC Benchmark")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--algorithms", type=str, nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--include-heuristics", action="store_true")
    p.add_argument("--env", type=str, default="auto",
                   help="环境 (auto=每算法自动选择正确环境, 或指定 MEC-v1-game-theory-* )")
    p.add_argument(
        "--environment-profile",
        choices=available_environment_profile_names(),
        default=DEFAULT_ENVIRONMENT_PROFILE,
        help="Environment profile. Defaults to mainline-a; legacy is explicit fallback only.",
    )
    p.add_argument("--configs-dir", type=str, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None, help="Override checkpoint/evaluation interval in timesteps")
    p.add_argument("--min-eval-points", type=int, default=None, help="Minimum requested evaluation checkpoints per training run")
    p.add_argument("--eval-at-start", action="store_true", default=None, help="Record an evaluation checkpoint at timestep 0")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default=None)  # 自动生成带时间戳的文件名
    p.add_argument("--no-latest-alias", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--scale", type=str, choices=["small", "medium", "large"], default=None)
    p.add_argument("--num-edge-servers", type=int, default=None)
    p.add_argument("--multi-agent-count", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warm-start-steps", type=int, default=None)
    p.add_argument("--warm-start-lr-scale", type=float, default=None)
    p.add_argument("--shapley-samples", type=int, default=None)
    p.add_argument("--ctde-with-hints", type=str, default=None, choices=["true", "false"])
    p.add_argument("--game-theory-enabled", type=str, default=None, choices=["true", "false"])
    p.add_argument("--reward-weights", type=float, nargs=3, default=None)
    p.add_argument("--efx-enabled", type=str, default=None, choices=["true", "false"])
    p.add_argument("--cpnet-enabled", type=str, default=None, choices=["true", "false"])
    p.add_argument("--efx-transfer-rate", type=float, default=None)
    p.add_argument("--system-model-config", type=str, default=None)
    p.add_argument("--dynamic-pricing-config", type=str, default=None)
    p.add_argument("--enable-mainline-a", action="store_true")
    p.add_argument(
        "--channel-model",
        type=str,
        choices=["analytic", "3gpp_lite", "rayleigh", "pathloss_only"],
        default=None,
    )
    p.add_argument(
        "--queue-model",
        type=str,
        choices=["mm1", "mmc", "parallel", "finite_capacity"],
        default=None,
    )
    p.add_argument(
        "--mobility-intensity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
    )
    args = p.parse_args()

    file_cfg = _load_benchmark_cli_config(args.config)
    cfg_algorithms = _coerce_list(file_cfg.get("algorithms") or file_cfg.get("benchmark", {}).get("algorithms"))
    if args.all:
        algorithms = list(ALL_ALGOS)
    elif args.algorithms:
        algorithms = [_canonical_algorithm_name(a) for a in args.algorithms]
    elif cfg_algorithms:
        algorithms = [_canonical_algorithm_name(str(a)) for a in cfg_algorithms]
    elif args.dry_run:
        algorithms = ["GRPO"]
    else:
        algorithms = []

    if args.config and file_cfg:
        _apply_benchmark_cli_config(args, file_cfg)

    if args.dry_run:
        print("DRY RUN benchmark")
        print(yaml.safe_dump(_dry_run_payload(args, file_cfg, algorithms), sort_keys=True))
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 自动生成输出文件名（带时间戳）
    if args.output is None:
        args.output = f"results/benchmark_{run_id}.json"

    # 创建日志目录与分离日志文件
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log_file = log_dir / f"benchmark_{run_id}.log"
    stderr_log_file = log_dir / f"benchmark_{run_id}.err.log"

    with _tee_streams(stdout_log_file, stderr_log_file):
        # 配置日志：只输出到 stdout，由 tee 负责同步到 log 文件
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True,
        )
        logger = logging.getLogger(__name__)

        logger.info("Benchmark started")
        logger.info("Run id: %s", run_id)
        logger.info("Output file: %s", args.output)
        logger.info("Stdout log file: %s", stdout_log_file)
        logger.info("Stderr log file: %s", stderr_log_file)

        if args.all:
            logger.info("Running all %s algorithms", len(algorithms))
        elif algorithms:
            logger.info("Running algorithms: %s", algorithms)
        else:
            p.print_help()
            print("\nSpecify --algorithms or --all")
            sys.exit(1)
        if args.include_heuristics:
            for h in HEURISTIC_ALGOS:
                if h not in algorithms:
                    algorithms.append(h)
        valid_algos = set(ALGORITHM_CLASSES) | set(HEURISTIC_ALGOS)
        for a in algorithms:
            if a not in valid_algos:
                logger.error("Unknown algorithm: %s", a)
                print(f"Unknown: {a}")
                sys.exit(1)

        env_to_use = None if args.env == "auto" else args.env
        logger.info("Environment: %s", env_to_use or "auto")
        profile = resolve_environment_profile(args.environment_profile)
        logger.info("Environment profile: %s", profile.name)

        gt_overrides = {
            "warm_start_steps": args.warm_start_steps,
            "warm_start_lr_scale": args.warm_start_lr_scale,
            "shapley_samples": args.shapley_samples,
            "ctde_with_hints": _parse_optional_bool(args.ctde_with_hints),
            "enabled": _parse_optional_bool(args.game_theory_enabled),
            "reward_weights": tuple(args.reward_weights) if args.reward_weights else None,
            "efx_enabled": _parse_optional_bool(args.efx_enabled),
            "cpnet_enabled": _parse_optional_bool(args.cpnet_enabled),
            "efx_transfer_rate": args.efx_transfer_rate,
            "game_aware_enabled": file_cfg.get("game_aware", {}).get("enabled") if file_cfg else None,
        }
        logger.info("Game theory config: %s", gt_overrides)

        profile_env_overrides = _resolve_profile_env_overrides(args)
        env_overrides = _resolve_env_overrides(
            scale=args.scale,
            num_edge_servers=args.num_edge_servers,
            multi_agent_count=args.multi_agent_count,
            max_steps=args.max_steps,
            system_model_config=profile_env_overrides.get("system_model_config"),
            dynamic_pricing_config=profile_env_overrides.get("dynamic_pricing_config"),
            enable_mainline_a=profile_env_overrides.get("enable_mainline_a"),
            channel_model=args.channel_model,
            queue_model=args.queue_model,
            mobility_intensity=args.mobility_intensity,
        )

        try:
            run_benchmark(
                algorithms,
                env_to_use,
                args.configs_dir,
                args.seeds,
                args.timesteps,
                args.episodes,
                args.device,
                args.output,
                not args.quiet,
                game_theory_overrides=gt_overrides,
                env_overrides=env_overrides,
                sync_latest_alias=not args.no_latest_alias,
                eval_interval=args.eval_interval,
                min_eval_points=args.min_eval_points,
                eval_at_start=args.eval_at_start,
                environment_profile=profile.name,
            )
            logger.info("Benchmark finished")
            logger.info("Benchmark completed successfully. Results saved to: %s", args.output)
            logger.info("Log files: stdout=%s stderr=%s", stdout_log_file, stderr_log_file)
        except Exception as e:
            logger.exception("Benchmark failed: %s", e)
            logger.info("Benchmark finished")
            raise

if __name__ == "__main__":
    main()
