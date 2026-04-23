#!/usr/bin/env python3
"""Train one RL algorithm with the current MEC trainer stack."""

import argparse
import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.trainer.on_policy_trainer import OnPolicyTrainer
from src.trainer.off_policy_trainer import OffPolicyTrainer
from src.utils.helpers import set_seed

# Reuse benchmark helpers to keep agent/env creation logic in one place.
from benchmark import (
    ALGO_ENV_MAP,
    ON_POLICY,
    MULTI_AGENT_ALGOS,
    create_agent,
    make_env,
    load_config,
    resolve_game_theory_config,
)


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
    return parser.parse_args()


def _resolve_algorithm(args: argparse.Namespace, cfg: dict) -> str:
    if args.algorithm:
        return args.algorithm.upper()
    return str(cfg.get("algorithm", {}).get("name", "GRPO")).upper()


def _resolve_env_name(algo: str, env_arg: str) -> str:
    if env_arg != "auto":
        return env_arg
    return ALGO_ENV_MAP.get(algo, "MEC-v1-game-theory-discrete-ma")


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    algo = _resolve_algorithm(args, cfg)
    if algo not in ALGO_ENV_MAP:
        raise ValueError(f"Unknown algorithm: {algo}")

    env_name = _resolve_env_name(algo, args.env)

    seed = args.seed
    if seed is None:
        seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    num_agents = 3 if algo in MULTI_AGENT_ALGOS else 1
    gt_overrides = {
        "warm_start_steps": args.warm_start_steps,
        "warm_start_lr_scale": args.warm_start_lr_scale,
        "shapley_samples": args.shapley_samples,
        "ctde_with_hints": _parse_optional_bool(args.ctde_with_hints),
        "enabled": _parse_optional_bool(args.game_theory_enabled),
        "reward_weights": tuple(args.reward_weights) if args.reward_weights else None,
    }
    gt_cfg = resolve_game_theory_config(algo, cfg, gt_overrides)
    env = make_env(env_name, seed=seed, num_agents=num_agents, game_theory_config=gt_cfg)
    agent = create_agent(algo, env, cfg, device, game_theory_overrides=gt_overrides)

    tc = cfg.get("training", {})
    ec = cfg.get("evaluation", {})
    lc = cfg.get("logging", {})
    ac = cfg.get("algorithm", {})

    total_timesteps = args.timesteps if args.timesteps is not None else int(tc.get("total_timesteps", 100000))
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else int(ec.get("num_episodes", 10))
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

    if algo in ON_POLICY:
        trainer_kwargs["num_epochs"] = int(ac.get("num_epochs", 10))
        trainer = OnPolicyTrainer(**trainer_kwargs)
    else:
        trainer_kwargs["warmup_steps"] = int(tc.get("start_steps", 1000))
        trainer_kwargs["update_interval"] = int(ac.get("update_interval", 1))
        trainer = OffPolicyTrainer(**trainer_kwargs)

    print(f"Algorithm: {algo}")
    print(f"Environment: {env_name}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Timesteps: {total_timesteps}")

    trainer.train()
    final_eval = trainer.evaluate()

    print("Final evaluation:")
    for k, v in final_eval.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
