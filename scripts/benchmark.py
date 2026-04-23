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
"""

import argparse
import importlib
import os
import sys
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from gymnasium import spaces

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trainer.on_policy_trainer import OnPolicyTrainer
from src.trainer.off_policy_trainer import OffPolicyTrainer
from src.utils.helpers import set_seed

ALGORITHM_CLASSES = {
    "GRPO": ("rl_algorithms.grpo", "GRPOAgent"),
    "PPO": ("rl_algorithms.ppo", "PPOAgent"),
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
    "IPPO": ("rl_algorithms.ippo", "IPPOAgent"),
    "VDN": ("rl_algorithms.vdn", "VDNAgent"),
    "MADDPG": ("rl_algorithms.maddpg", "MADDPGAgent"),
    "IQL": ("rl_algorithms.iql", "IQLAgent"),
    "MATD3": ("rl_algorithms.matd3", "MATD3Agent"),
}
ON_POLICY = {"GRPO", "PPO", "A3C", "TRPO", "SimPO", "MAPPO", "COMA", "IPPO"}
OFF_POLICY = {"SAC", "DDQN", "DDPG", "TD3", "QMIX", "VDN", "MADDPG", "IQL", "MATD3"}
ALL_ALGOS = list(ALGORITHM_CLASSES.keys())
ALGO_ENV_MAP = {
    "A3C": "MEC-v1-game-theory-discrete-ma",
    "SimPO": "MEC-v1-game-theory-discrete-ma",
    "DDQN": "MEC-v1-game-theory-discrete-ma",
    "QMIX": "MEC-v1-game-theory-discrete-ma",
    "COMA": "MEC-v1-game-theory-discrete-ma",
    "IPPO": "MEC-v1-game-theory-discrete-ma",
    "VDN": "MEC-v1-game-theory-discrete-ma",
    "IQL": "MEC-v1-game-theory-discrete-ma",
    "GRPO": "MEC-v1-game-theory-continuous-ma",
    "PPO": "MEC-v1-game-theory-continuous-ma",
    "TRPO": "MEC-v1-game-theory-continuous-ma",
    "SAC": "MEC-v1-game-theory-continuous-ma",
    "DDPG": "MEC-v1-game-theory-continuous-ma",
    "TD3": "MEC-v1-game-theory-continuous-ma",
    "MADDPG": "MEC-v1-game-theory-continuous-ma",
    "MATD3": "MEC-v1-game-theory-continuous-ma",
    "MAPPO": "MEC-v1-game-theory-discrete-ma",
}

CONTINUOUS_ALGOS = {"SAC", "DDPG", "TD3", "MADDPG", "MATD3"}
MULTI_AGENT_ALGOS = {"MAPPO", "QMIX", "COMA", "IPPO", "VDN", "MADDPG", "IQL", "MATD3"}
DEEP_FUSION_ALGOS = {"GRPO", "MAPPO", "QMIX", "COMA", "IPPO", "VDN", "MADDPG", "IQL", "MATD3"}


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
        "reward_weights": (0.5, 0.3, 0.2),
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


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(name, seed=None, num_agents=3, game_theory_config: Optional[Dict[str, Any]] = None):
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
    gt_cfg = game_theory_config or {}
    if gt_cfg:
        env_kwargs["shapley_samples"] = int(gt_cfg.get("shapley_samples", 128))
        env_kwargs["reward_weights"] = _normalize_reward_weights(gt_cfg.get("reward_weights"))
        if "enabled" in gt_cfg and not bool(gt_cfg.get("enabled")):
            env_kwargs["enable_game_init"] = False
            env_kwargs["enable_shapley"] = False
    env = env_cls(**env_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


def create_agent(name, env, cfg, device, game_theory_overrides: Optional[Dict[str, Any]] = None):
    cls = load_algo_class(name)
    ac = cfg.get("algorithm", {})
    nc = cfg.get("network", {})
    tc = cfg.get("training", {})
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
    kw = dict(
        state_dim=obs,
        action_dim=ad,
        hidden_dim=nc.get("hidden_dim", 256),
        lr=tc.get("lr", 3e-4),
        gamma=ac.get("gamma", 0.99),
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
        return cls(**kw, eps_clip=ac.get("eps_clip", 0.2), num_epochs=ac.get("num_epochs", 10),
                   group_size=ac.get("group_size", 64))
    if name == "PPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, eps_clip=ac.get("eps_clip", 0.2), num_epochs=ac.get("num_epochs", 10),
                   discrete=is_discrete_env)
    if name == "SAC":
        return cls(**kw, tau=ac.get("tau", 0.005), alpha=ac.get("alpha", 0.2),
                   automatic_entropy_tuning=ac.get("automatic_entropy_tuning", True),
                   batch_size=ac.get("batch_size", 256), buffer_size=ac.get("buffer_size", 1000000))
    if name == "DDPG":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000),
                   action_scale=ac.get("action_scale", 1.0), action_bias=ac.get("action_bias", 0.0))
    if name == "TD3":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000),
                   noise_std=ac.get("noise_std", 0.2), policy_delay=ac.get("policy_delay", 2),
                   action_scale=ac.get("action_scale", 1.0))
    if name == "DDQN":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000),
                   epsilon_start=ac.get("epsilon_start", 1.0),
                   epsilon_end=ac.get("epsilon_end", 0.05),
                   epsilon_decay=ac.get("epsilon_decay", 50000))
    if name == "QMIX":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000), n_agents=n_agents,
                   epsilon_start=ac.get("epsilon_start", 1.0),
                   epsilon_end=ac.get("epsilon_end", 0.05),
                   epsilon_decay=ac.get("epsilon_decay", 50000))
    if name == "A3C":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, num_steps=ac.get("num_steps", 20), discrete=is_discrete_env)
    if name == "TRPO":
        return cls(**kw, max_kl=ac.get("max_kl", 0.01), cg_iters=ac.get("cg_iters", 10))
    if name == "SimPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, beta=ac.get("beta", 0.1), ref_coeff=ac.get("ref_coeff", 0.2),
                   discrete=is_discrete_env)
    if name == "MAPPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, gae_lambda=ac.get("gae_lambda", 0.95), eps_clip=ac.get("eps_clip", 0.2),
                   num_agents=n_agents, num_epochs=ac.get("num_epochs", 10),
                   discrete=is_discrete_env)
    if name == "COMA":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, num_agents=n_agents, num_epochs=ac.get("num_epochs", 10),
                   discrete=is_discrete_env)
    if name == "IPPO":
        is_discrete_env = isinstance(env.action_space, spaces.Discrete)
        return cls(**kw, eps_clip=ac.get("eps_clip", 0.2), num_epochs=ac.get("num_epochs", 10),
                   discrete=is_discrete_env, num_agents=n_agents)
    if name == "VDN":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000), n_agents=n_agents,
                   epsilon_start=ac.get("epsilon_start", 1.0),
                   epsilon_end=ac.get("epsilon_end", 0.05),
                   epsilon_decay=ac.get("epsilon_decay", 50000))
    if name == "MADDPG":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000), n_agents=n_agents)
    if name == "IQL":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000),
                   epsilon_start=ac.get("epsilon_start", 1.0),
                   epsilon_end=ac.get("epsilon_end", 0.05),
                   epsilon_decay=ac.get("epsilon_decay", 50000))
    if name == "MATD3":
        return cls(**kw, tau=ac.get("tau", 0.005), batch_size=ac.get("batch_size", 256),
                   buffer_size=ac.get("buffer_size", 1000000), n_agents=n_agents,
                   noise_std=ac.get("noise_std", 0.2), policy_delay=ac.get("policy_delay", 2))
    raise ValueError(name)


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
):
    """
    运行单算法评测。

    环境自动选择: ALGO_ENV_MAP 保证了每个算法在正确的环境类型上评测，
    确保离散/连续/多智能体算法各在其适对应的环境。
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    # 自动选择正确环境 (修复: 每个算法在其适配的环境上评测)
    correct_env_name = env_name if env_name is not None else ALGO_ENV_MAP.get(name, "MEC-v1-game-theory-discrete-ma")
    num_agents = 3 if name in MULTI_AGENT_ALGOS else 1
    gt_cfg = resolve_game_theory_config(name, cfg, game_theory_overrides)
    env = make_env(
        correct_env_name,
        seed=seed,
        num_agents=num_agents,
        game_theory_config=gt_cfg,
    )
    agent = create_agent(name, env, cfg, device, game_theory_overrides=game_theory_overrides)
    TrainerCls = OnPolicyTrainer if name in ON_POLICY else OffPolicyTrainer
    tc = cfg.get("training", {})
    ec = cfg.get("evaluation", {})
    lc = cfg.get("logging", {})
    ac = cfg.get("algorithm", {})
    total_ts = override_ts if override_ts is not None else tc.get("total_timesteps", 100000)
    eval_ep = override_ep if override_ep is not None else ec.get("num_episodes", 10)
    trainer_kwargs = dict(
        env=env, agent=agent, total_timesteps=total_ts,
        rollout_steps=tc.get("rollout_steps", 2048),
        eval_interval=lc.get("eval_interval", 5000), eval_episodes=eval_ep,
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
    fe = trainer.evaluate()
    latency_total = fe.get("eval/latency_total_mean", fe.get("eval/latency_mean", 0.0))
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
        "final_latency_mean": round(latency_total, 4),
        "final_energy_mean": round(energy_total, 4),
        "final_latency_total_mean": round(latency_total, 4),
        "final_energy_total_mean": round(energy_total, 4),
        "final_latency_per_step_mean": None if latency_per_step is None else round(latency_per_step, 4),
        "final_energy_per_step_mean": None if energy_per_step is None else round(energy_per_step, 4),
        "final_latency_per_task_mean": None if latency_per_task is None else round(latency_per_task, 4),
        "final_energy_per_task_mean": None if energy_per_task is None else round(energy_per_task, 4),
        "total_episodes": trainer.episode_count, "total_updates": trainer.update_count,
    }
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


def run_benchmark(algorithms, env_name=None, configs_dir=None, seeds=None,
                 timesteps=None, episodes=None, device="auto",
                 output_file=None, verbose=True, game_theory_overrides: Optional[Dict[str, Any]] = None):
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

    # 按环境分组，相同环境的算法一起评测
    if env_name == "auto":
        # 分组: 每个算法在其正确环境上
        env_groups = {}  # env_name -> list of algos
        for algo in algorithms:
            correct_env = ALGO_ENV_MAP.get(algo, "MEC-v0-discrete")
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
                cp = Path(configs_dir) / (algo.lower() + ".yaml")
                cfg = load_config(str(cp)) if cp.exists() else {
                    "algorithm": {}, "network": {}, "training": {},
                    "evaluation": {}, "logging": {}
                }
                try:
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
                    )
                    algo_results.append(r)
                except Exception as e:
                    algo_errors.append({"seed": seed, "error": str(e)})
                    if verbose:
                        print(f"Error {algo} seed={seed}: {e}")
            if algo_results:
                avg = {"algorithm": algo, "n_seeds": len(algo_results)}
                for key in algo_results[0].keys():
                    if key in ("algorithm", "environment", "seed", "device", "n_seeds"):
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
                avg.setdefault("final_energy_mean_mean", None)
                avg.setdefault("final_latency_total_mean_mean", None)
                avg.setdefault("final_energy_total_mean_mean", None)
                avg.setdefault("final_latency_per_step_mean_mean", None)
                avg.setdefault("final_energy_per_step_mean_mean", None)
                avg.setdefault("final_latency_per_task_mean_mean", None)
                avg.setdefault("final_energy_per_task_mean_mean", None)
                avg.setdefault("train_time_seconds_mean", None)
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
                    "device": device,
                    "n_seeds": 0,
                    "status": "failed",
                    "final_reward_mean_mean": None,
                    "final_latency_mean_mean": None,
                    "final_energy_mean_mean": None,
                    "final_latency_total_mean_mean": None,
                    "final_energy_total_mean_mean": None,
                    "final_latency_per_step_mean_mean": None,
                    "final_energy_per_step_mean_mean": None,
                    "final_latency_per_task_mean_mean": None,
                    "final_energy_per_task_mean_mean": None,
                    "train_time_seconds_mean": None,
                    "errors": algo_errors or [{"seed": None, "error": "Unknown benchmark failure"}],
                }
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

    if output_file:
        op = Path(output_file)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(all_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {op}")
    return all_results


def main():
    p = argparse.ArgumentParser(description="GRPO_MEC Benchmark")
    p.add_argument("--algorithms", type=str, nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--env", type=str, default="auto",
                   help="环境 (auto=每算法自动选择正确环境, 或指定 MEC-v0-discrete/continuous/multi-agent)")
    p.add_argument("--configs-dir", type=str, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=str, default="results/benchmark.json")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--warm-start-steps", type=int, default=None)
    p.add_argument("--warm-start-lr-scale", type=float, default=None)
    p.add_argument("--shapley-samples", type=int, default=None)
    p.add_argument("--ctde-with-hints", type=str, default=None, choices=["true", "false"])
    p.add_argument("--game-theory-enabled", type=str, default=None, choices=["true", "false"])
    p.add_argument("--reward-weights", type=float, nargs=3, default=None)
    args = p.parse_args()
    if args.all:
        algorithms = ALL_ALGOS
    elif args.algorithms:
        algorithms = args.algorithms
    else:
        p.print_help()
        print("\nSpecify --algorithms or --all")
        sys.exit(1)
    for a in algorithms:
        if a not in ALL_ALGOS:
            print(f"Unknown: {a}")
            sys.exit(1)
    env_to_use = None if args.env == "auto" else args.env
    gt_overrides = {
        "warm_start_steps": args.warm_start_steps,
        "warm_start_lr_scale": args.warm_start_lr_scale,
        "shapley_samples": args.shapley_samples,
        "ctde_with_hints": _parse_optional_bool(args.ctde_with_hints),
        "enabled": _parse_optional_bool(args.game_theory_enabled),
        "reward_weights": tuple(args.reward_weights) if args.reward_weights else None,
    }
    run_benchmark(algorithms, env_to_use, args.configs_dir, args.seeds, args.timesteps,
                  args.episodes, args.device, args.output, not args.quiet,
                  game_theory_overrides=gt_overrides)


if __name__ == "__main__":
    main()
