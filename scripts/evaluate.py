#!/usr/bin/env python3
"""Evaluate one trained RL algorithm checkpoint on MEC environments."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from gymnasium import spaces

from src.utils.helpers import is_done, set_seed

# Reuse benchmark helpers to keep agent/env creation logic in one place.
from benchmark import (
    ALGO_ENV_MAP,
    MULTI_AGENT_ALGOS,
    create_agent,
    make_env,
    load_config,
    resolve_game_theory_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one MEC RL checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
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
    parser.add_argument("--num-episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Optional output JSON path for evaluation metrics",
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


def _format_action(action, action_space):
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    if isinstance(action, np.ndarray) and action.ndim > 1:
        action = action[0]

    if isinstance(action_space, spaces.Discrete):
        if isinstance(action, np.ndarray) and action.ndim >= 1:
            return int(np.argmax(action, axis=-1).flatten()[0])
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)

    if isinstance(action, np.ndarray):
        action = action.flatten()
    else:
        action = np.atleast_1d(np.asarray(action, dtype=np.float32)).flatten()

    if not np.isfinite(action).all():
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)

    return action


def _load_agent_checkpoint(agent, checkpoint_path: str, device: str) -> None:
    if hasattr(agent, "load"):
        agent.load(checkpoint_path)
        return
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(state_dict)


def _extract_step_metrics(info: dict, prev_task_completed: int) -> tuple[float, float, int, int]:
    """Extract latency/energy from one env step and estimate task count."""
    step_latency = 0.0
    step_energy = 0.0
    task_count = 0

    if "individual_latencies" in info:
        latencies = np.asarray(info["individual_latencies"], dtype=np.float64).reshape(-1)
        step_latency = float(np.sum(latencies))
        task_count = max(task_count, int(latencies.size))
    elif "latency" in info:
        step_latency = float(info["latency"])
        task_count = max(task_count, 1)

    if "individual_energies" in info:
        energies = np.asarray(info["individual_energies"], dtype=np.float64).reshape(-1)
        step_energy = float(np.sum(energies))
        task_count = max(task_count, int(energies.size))
    elif "energy" in info:
        step_energy = float(info["energy"])
        task_count = max(task_count, 1)

    current_task_completed = info.get("task_completed", prev_task_completed)
    if isinstance(current_task_completed, (int, float, np.integer, np.floating)):
        current_task_completed = int(current_task_completed)
        task_delta = max(0, current_task_completed - prev_task_completed)
        if task_delta > 0:
            task_count = task_delta
        prev_task_completed = current_task_completed

    return step_latency, step_energy, task_count, prev_task_completed


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    algo = _resolve_algorithm(args, cfg)
    if algo not in ALGO_ENV_MAP:
        raise ValueError(f"Unknown algorithm: {algo}")

    env_name = _resolve_env_name(algo, args.env)
    set_seed(args.seed)

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
    env = make_env(env_name, seed=args.seed, num_agents=num_agents, game_theory_config=gt_cfg)
    agent = create_agent(algo, env, cfg, device, game_theory_overrides=gt_overrides)
    _load_agent_checkpoint(agent, args.checkpoint, device)

    rewards = []
    latency_totals = []
    energy_totals = []
    latency_per_step = []
    energy_per_step = []
    latency_per_task = []
    energy_per_task = []
    episode_steps = []
    episode_tasks = []

    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0
        ep_latency_total = 0.0
        ep_energy_total = 0.0
        ep_steps = 0
        ep_tasks = 0
        prev_task_completed = 0

        while not done:
            if hasattr(env, "num_agents") and isinstance(obs, list):
                actions = []
                for agent_obs in obs:
                    action, _ = agent.select_action(agent_obs, deterministic=True)
                    actions.append(_format_action(action, env.action_space))
                obs, reward, terminated, truncated, info = env.step(actions)
                ep_reward += float(sum(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward)
            else:
                action, _ = agent.select_action(obs, deterministic=True)
                env_action = _format_action(action, env.action_space)
                obs, reward, terminated, truncated, info = env.step(env_action)
                ep_reward += float(reward)

            done = is_done(terminated, truncated)
            step_latency, step_energy, step_tasks, prev_task_completed = _extract_step_metrics(
                info, prev_task_completed
            )
            ep_latency_total += step_latency
            ep_energy_total += step_energy
            ep_steps += 1
            ep_tasks += step_tasks

        rewards.append(ep_reward)
        latency_totals.append(ep_latency_total)
        energy_totals.append(ep_energy_total)
        episode_steps.append(ep_steps)
        episode_tasks.append(ep_tasks)

        safe_steps = max(1, ep_steps)
        safe_tasks = max(1, ep_tasks)
        latency_per_step.append(ep_latency_total / safe_steps)
        energy_per_step.append(ep_energy_total / safe_steps)
        latency_per_task.append(ep_latency_total / safe_tasks)
        energy_per_task.append(ep_energy_total / safe_tasks)

    latency_total_mean = float(np.mean(latency_totals))
    energy_total_mean = float(np.mean(energy_totals))

    summary = {
        "algorithm": algo,
        "environment": env_name,
        "episodes": args.num_episodes,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        # Backward-compatible aliases (historically episode totals).
        "latency_mean": latency_total_mean,
        "energy_mean": energy_total_mean,
        # Explicit latency/energy definitions.
        "latency_total_mean": latency_total_mean,
        "energy_total_mean": energy_total_mean,
        "latency_per_step_mean": float(np.mean(latency_per_step)),
        "energy_per_step_mean": float(np.mean(energy_per_step)),
        "latency_per_task_mean": float(np.mean(latency_per_task)),
        "energy_per_task_mean": float(np.mean(energy_per_task)),
        "episode_steps_mean": float(np.mean(episode_steps)),
        "episode_tasks_mean": float(np.mean(episode_tasks)),
    }

    print("Evaluation summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **summary,
            "episode_rewards": rewards,
            "episode_latencies": latency_totals,
            "episode_energies": energy_totals,
            "episode_latency_totals": latency_totals,
            "episode_energy_totals": energy_totals,
            "episode_latency_per_step": latency_per_step,
            "episode_energy_per_step": energy_per_step,
            "episode_latency_per_task": latency_per_task,
            "episode_energy_per_task": energy_per_task,
            "episode_steps": episode_steps,
            "episode_tasks": episode_tasks,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
