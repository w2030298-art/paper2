"""
Benchmark — 多算法对比评测

用法:
    python -m src.trainer.benchmark --env discrete --algorithms ppo sac ddqn \\
        --timesteps 100000 --episodes 10
"""

import os
import sys
import argparse
import json
import importlib
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "rl_algorithms"))


# 算法分类
ON_POLICY_ALGOS = ["grpo", "ppo", "a3c", "trpo", "simpo", "mappo"]
OFF_POLICY_ALGOS = ["sac", "ddpg", "td3", "ddqn", "qmix"]
ALGO_MAP = {
    "grpo": "grpo.GRPOAgent",
    "ppo": "ppo.PPOAgent",
    "sac": "sac.SACAgent",
    "ddpg": "ddpg.DDPGAgent",
    "td3": "td3.TD3Agent",
    "ddqn": "ddqn.DDQNAgent",
    "a3c": "a3c.A3CAgent",
    "trpo": "trpo.TRPOAgent",
    "simpo": "simpo.SimPOAgent",
    "mappo": "mappo.MAPPOAgent",
    "qmix": "qmix.QMIXAgent",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark RL algorithms on MEC environment")
    parser.add_argument("--env", type=str, default="discrete",
                        choices=["discrete", "continuous", "multi_agent"],
                        help="Environment type")
    parser.add_argument("--algorithms", type=str, nargs="+",
                        default=["ppo", "sac", "ddqn"],
                        help="Algorithms to benchmark")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total training timesteps per algorithm")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                        help="Rollout steps per update")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Evaluation episodes")
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="Evaluation interval")
    parser.add_argument("--save-dir", type=str, default="./benchmark_results",
                        help="Results save directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto/cuda/cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-agents", type=int, default=2,
                        help="Number of agents (for multi_agent env)")
    parser.add_argument("--hide-progress", action="store_true",
                        help="Hide progress bars")
    return parser.parse_args()


def create_env(env_type: str, num_agents: int = 2, seed: int = 42):
    """创建MEC环境"""
    if env_type == "discrete":
        from src.environments.mec_v2 import MECEnvDiscrete
        env = MECEnvDiscrete(
            num_edge_servers=3,
            num_tasks=5,
            max_steps=100,
            task_arrival_prob=0.5,
            normalize_obs=True,
            fading_type="rayleigh",
        )
        env.reset(seed=seed)
        return env
    elif env_type == "continuous":
        from src.environments.mec_v2 import MECEnvContinuous
        env = MECEnvContinuous(
            num_edge_servers=3,
            num_tasks=5,
            max_steps=100,
            task_arrival_prob=0.5,
            normalize_obs=True,
            fading_type="rayleigh",
        )
        env.reset(seed=seed)
        return env
    elif env_type == "multi_agent":
        from src.environments.mec_v2 import MECEnvMultiAgent
        env = MECEnvMultiAgent(
            num_agents=num_agents,
            num_edge_servers=3,
            num_tasks=5,
            max_steps=100,
            task_arrival_prob=0.5,
            normalize_obs=True,
            fading_type="rayleigh",
        )
        env.reset(seed=seed)
        return env
    else:
        raise ValueError(f"Unknown env type: {env_type}")


def create_agent(algo_name: str, env, device: str = "cuda") -> Any:
    """创建agent"""
    algo_path = ALGO_MAP.get(algo_name.lower())
    if algo_path is None:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    mod_name, cls_name = algo_path.split(".")
    module = importlib.import_module(f"rl_algorithms.{mod_name}")
    AgentClass = getattr(module, cls_name)

    # 从环境获取维度
    if hasattr(env, "observation_space"):
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = env.observation_space[0].shape[0]  # multi-agent

    if hasattr(env, "action_space"):
        if isinstance(env.action_space, importlib.import_module("gymnasium").spaces.Discrete):
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space[0].n  # multi-agent
    num_agents = getattr(env, "num_agents", 1)

    # 根据算法类型创建
    common_kwargs = {
        "state_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": 128,
        "lr": 3e-4,
        "gamma": 0.99,
        "device": device,
    }

    if algo_name.lower() == "grpo":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            group_size=64,
            num_epochs=10,
            device=device,
        )
    elif algo_name.lower() == "ppo":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            gae_lambda=0.95,
            num_epochs=10,
            device=device,
        )
    elif algo_name.lower() == "sac":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            device=device,
        )
    elif algo_name.lower() == "ddpg":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            device=device,
        )
    elif algo_name.lower() == "td3":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            device=device,
        )
    elif algo_name.lower() == "ddqn":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            device=device,
        )
    elif algo_name.lower() == "a3c":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            num_steps=20,
            device=device,
        )
    elif algo_name.lower() == "trpo":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            device=device,
        )
    elif algo_name.lower() == "simpo":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            beta=0.1,
            ref_coeff=0.2,
            num_epochs=10,
            discrete=(isinstance(env.action_space, importlib.import_module("gymnasium").spaces.Discrete)),
            device=device,
        )
    elif algo_name.lower() == "mappo":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            gae_lambda=0.95,
            num_epochs=10,
            num_agents=num_agents,
            discrete=(isinstance(env.action_space, importlib.import_module("gymnasium").spaces.Discrete)),
            device=device,
        )
    elif algo_name.lower() == "qmix":
        return AgentClass(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=64,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            n_agents=num_agents,
            global_state_dim=obs_dim * num_agents,
            device=device,
        )
    else:
        return AgentClass(**common_kwargs)


def get_trainer_class(algo_name: str):
    """根据算法类型选择训练器"""
    from src.trainer import OnPolicyTrainer, OffPolicyTrainer

    if algo_name.lower() in ON_POLICY_ALGOS:
        return OnPolicyTrainer
    else:
        return OffPolicyTrainer


def run_benchmark(args):
    """运行benchmark"""
    print("=" * 60)
    print(f"  GRPO_MEC Benchmark")
    print(f"  Env: {args.env}")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Timesteps: {args.timesteps}")
    print("=" * 60)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    # 创建环境
    env = create_env(args.env, args.num_agents, args.seed)

    results = {}

    for algo_name in args.algorithms:
        print(f"\n{'='*40}")
        print(f"  Training: {algo_name.upper()}")
        print(f"{'='*40}")

        try:
            # 创建agent
            agent = create_agent(algo_name, env, device=str(device))
            print(f"  Agent created: {type(agent).__name__}")
            print(f"  Device: {device}")
            print(f"  Obs dim: {env.observation_space.shape}")
            print(f"  Action space: {env.action_space}")

            # 选择训练器
            TrainerClass = get_trainer_class(algo_name)

            # 创建训练器
            trainer = TrainerClass(
                env=env,
                agent=agent,
                total_timesteps=args.timesteps,
                rollout_steps=args.rollout_steps,
                eval_interval=args.eval_interval,
                eval_episodes=args.episodes,
                save_interval=args.timesteps,
                save_dir=os.path.join(args.save_dir, algo_name),
                log_interval=10,
                device=str(device),
                seed=args.seed,
            )

            # 训练
            train_logs = trainer.train()

            # 最终评估
            final_eval = trainer.evaluate()

            results[algo_name] = {
                "train_logs": train_logs,
                "final_eval": final_eval,
                "status": "success",
            }

            print(f"\n  Final Evaluation:")
            for k, v in final_eval.items():
                print(f"    {k}: {v:.4f}")

            # 保存
            agent_path = os.path.join(args.save_dir, algo_name, "final.pt")
            os.makedirs(os.path.dirname(agent_path), exist_ok=True)
            trainer.save(agent_path)
            print(f"  Model saved: {agent_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[algo_name] = {
                "status": "failed",
                "error": str(e),
            }

    # 汇总结果
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Status':<10} {'Mean Reward':>15} {'Std':>10}")
    print("-" * 60)

    for algo_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            final_eval = result.get("final_eval", {})
            reward = final_eval.get("eval/reward_mean", 0)
            std = final_eval.get("eval/reward_std", 0)
            print(f"{algo_name:<12} {'success':<10} {reward:>15.2f} {std:>10.2f}")
        else:
            print(f"{algo_name:<12} {'failed':<10} {'N/A':>15} {'N/A':>10}")

    # 保存汇总
    summary_path = os.path.join(args.save_dir, "summary.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {summary_path}")

    return results


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
