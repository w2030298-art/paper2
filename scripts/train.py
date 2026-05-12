#!/usr/bin/env python3
"""Train one RL algorithm with the current MEC trainer stack."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

project_root = Path(__file__).resolve().parents[1]
for path in [
    project_root,
    project_root / "scripts",
    project_root / "src",
    project_root / "rl_algorithms",
]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.experiment.environment_profiles import (  # noqa: E402
    DEFAULT_ENVIRONMENT_PROFILE,
    available_environment_profile_names,
    profile_to_env_overrides,
    resolve_environment_profile,
)
from src.experiment.presets import FULL_17_ALGORITHMS  # noqa: E402
from src.experiment.runtime_config import build_resolved_runtime_config  # noqa: E402

CANONICAL_ALGORITHM_NAMES = {name.upper(): name for name in FULL_17_ALGORITHMS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one RL algorithm on MEC")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/algorithms/grpo.yaml",
        help="Path to algorithm config yaml",
    )
    parser.add_argument("--algorithm", type=str, default=None, help="Algorithm name, e.g. GRPO")
    parser.add_argument(
        "--env",
        type=str,
        default="auto",
        help="Environment name or 'auto' to use algorithm default",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override training timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--save-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episode count",
    )
    parser.add_argument("--warm-start-steps", type=int, default=None)
    parser.add_argument("--warm-start-lr-scale", type=float, default=None)
    parser.add_argument("--shapley-samples", type=int, default=None)
    parser.add_argument("--ctde-with-hints", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--game-theory-enabled", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--reward-weights", type=float, nargs=3, default=None)
    parser.add_argument("--efx-enabled", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--cpnet-enabled", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--efx-transfer-rate", type=float, default=None)
    parser.add_argument("--scale", type=str, choices=["small", "medium", "large"], default=None)
    parser.add_argument("--num-edge-servers", type=int, default=None)
    parser.add_argument("--multi-agent-count", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--environment-profile",
        choices=available_environment_profile_names(),
        default=DEFAULT_ENVIRONMENT_PROFILE,
        help="Environment profile. Defaults to mainline-a; legacy is explicit fallback only.",
    )
    parser.add_argument(
        "--enable-mainline-a",
        action="store_true",
        help="Compatibility flag; mainline-a profile enables this automatically.",
    )
    parser.add_argument("--system-model-config", type=str, default=None)
    parser.add_argument("--dynamic-pricing-config", type=str, default=None)
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Optional path to write machine-readable final training result JSON",
    )
    return parser.parse_args()


def _canonical_algorithm_name(name: str) -> str:
    normalized = str(name).strip().upper()
    return CANONICAL_ALGORITHM_NAMES.get(normalized, normalized)


def _resolve_algorithm(args: argparse.Namespace, cfg: dict) -> str:
    if args.algorithm:
        return _canonical_algorithm_name(args.algorithm)
    return _canonical_algorithm_name(cfg.get("algorithm", {}).get("name", "GRPO"))


def _resolve_env_name(algo: str, env_arg: str, algo_env_map: dict[str, str]) -> str:
    if env_arg != "auto":
        return env_arg
    return algo_env_map.get(algo, "MEC-v1-game-theory-discrete-ma")


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _resolve_profile_env_overrides(args: argparse.Namespace) -> dict[str, Any]:
    profile = resolve_environment_profile(args.environment_profile)
    return profile_to_env_overrides(
        profile,
        system_model_config=args.system_model_config,
        dynamic_pricing_config=args.dynamic_pricing_config,
        enable_mainline_a=True if args.enable_mainline_a else None,
    )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _write_result_json(path: str | os.PathLike[str], payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def _load_training_dependencies() -> dict[str, Any]:
    import torch
    from benchmark import (
        ALGO_ENV_MAP,
        MULTI_AGENT_ALGOS,
        ON_POLICY,
        _resolve_env_overrides,
        create_agent,
        load_config,
        make_env,
        resolve_game_theory_config,
    )

    from src.trainer.off_policy_trainer import OffPolicyTrainer
    from src.trainer.on_policy_trainer import OnPolicyTrainer
    from src.utils.helpers import set_seed

    return {
        "torch": torch,
        "OnPolicyTrainer": OnPolicyTrainer,
        "OffPolicyTrainer": OffPolicyTrainer,
        "set_seed": set_seed,
        "ALGO_ENV_MAP": ALGO_ENV_MAP,
        "ON_POLICY": ON_POLICY,
        "MULTI_AGENT_ALGOS": MULTI_AGENT_ALGOS,
        "create_agent": create_agent,
        "make_env": make_env,
        "load_config": load_config,
        "resolve_game_theory_config": resolve_game_theory_config,
        "_resolve_env_overrides": _resolve_env_overrides,
    }


def main() -> None:
    args = parse_args()
    deps = _load_training_dependencies()

    torch = deps["torch"]
    on_policy_trainer_cls = deps["OnPolicyTrainer"]
    off_policy_trainer_cls = deps["OffPolicyTrainer"]
    set_seed = deps["set_seed"]
    algo_env_map = deps["ALGO_ENV_MAP"]
    on_policy = deps["ON_POLICY"]
    multi_agent_algos = deps["MULTI_AGENT_ALGOS"]
    create_agent = deps["create_agent"]
    make_env = deps["make_env"]
    load_config = deps["load_config"]
    resolve_game_theory_config = deps["resolve_game_theory_config"]
    _resolve_env_overrides = deps["_resolve_env_overrides"]

    cfg = load_config(args.config)

    algo = _resolve_algorithm(args, cfg)
    if algo not in algo_env_map:
        raise ValueError(f"Unknown algorithm: {algo}")

    env_name = _resolve_env_name(algo, args.env, algo_env_map)

    seed = args.seed
    if seed is None:
        seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    env_overrides = _resolve_env_overrides(
        scale=args.scale,
        num_edge_servers=args.num_edge_servers,
        multi_agent_count=args.multi_agent_count,
        max_steps=args.max_steps,
    )
    profile = resolve_environment_profile(args.environment_profile)
    env_overrides.update(_resolve_profile_env_overrides(args))
    num_agents = int(env_overrides.get("num_agents_multi", 3)) if algo in multi_agent_algos else 1
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
    }
    gt_cfg = resolve_game_theory_config(algo, cfg, gt_overrides)
    env = make_env(
        env_name,
        seed=seed,
        num_agents=num_agents,
        game_theory_config=gt_cfg,
        env_overrides=env_overrides,
    )
    agent = create_agent(algo, env, cfg, device, game_theory_overrides=gt_overrides)

    tc = cfg.get("training", {})
    ec = cfg.get("evaluation", {})
    lc = cfg.get("logging", {})
    ac = cfg.get("algorithm", {})

    total_timesteps = (
        args.timesteps if args.timesteps is not None else int(tc.get("total_timesteps", 100000))
    )
    eval_episodes = (
        args.eval_episodes if args.eval_episodes is not None else int(ec.get("num_episodes", 10))
    )
    save_dir = args.save_dir or lc.get("checkpoint_dir", os.path.join("checkpoints", algo.lower()))

    trainer_kwargs = {
        "env": env,
        "agent": agent,
        "total_timesteps": total_timesteps,
        "rollout_steps": int(tc.get("rollout_steps", 2048)),
        "eval_interval": int(lc.get("eval_interval", 5000)),
        "eval_episodes": eval_episodes,
        "save_interval": int(lc.get("save_interval", 50000)),
        "save_dir": save_dir,
        "log_interval": int(lc.get("log_interval", 10)),
        "device": device,
        "seed": seed,
    }

    if algo in on_policy:
        trainer_kwargs["num_epochs"] = int(ac.get("num_epochs", 10))
        trainer = on_policy_trainer_cls(**trainer_kwargs)
    else:
        trainer_kwargs["warmup_steps"] = int(tc.get("start_steps", 1000))
        trainer_kwargs["update_interval"] = int(ac.get("update_interval", 1))
        trainer = off_policy_trainer_cls(**trainer_kwargs)

    print(f"Algorithm: {algo}")
    print(f"Environment: {env_name}")
    print(f"Environment profile: {profile.name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps}")

    trainer.train()
    final_eval = trainer.evaluate()
    resolved_runtime_config = build_resolved_runtime_config(
        algorithm=algo,
        config_path=args.config,
        base_algorithm_config=cfg,
        cli_overrides={
            "timesteps": args.timesteps,
            "seed": args.seed,
            "device": args.device,
            "eval_episodes": args.eval_episodes,
            "env": args.env,
            "environment_profile": args.environment_profile,
        },
        environment=env_name,
        environment_profile=profile.name,
        env_overrides=env_overrides,
        game_theory_config=gt_cfg,
        trainer_kwargs=trainer_kwargs,
        agent=agent,
        env=env,
        train_timesteps=total_timesteps,
        eval_episodes=eval_episodes,
    )

    result_payload = {
        "algorithm": algo,
        "environment": env_name,
        "environment_profile": profile.name,
        "seed": seed,
        "device": str(device),
        "train_timesteps": total_timesteps,
        "checkpoint_dir": save_dir,
        "final_eval": final_eval,
        "resolved_runtime_config": resolved_runtime_config,
        "status": "success",
    }
    if args.result_json is not None:
        _write_result_json(args.result_json, result_payload)

    print("Final evaluation:")
    for k, v in final_eval.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
