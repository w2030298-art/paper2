"""Experiment presets shared by CLI and VSCode entrypoints."""

from __future__ import annotations

from typing import Any

from .environment_profiles import DEFAULT_ENVIRONMENT_PROFILE

QUICK_RUN_ID = "vscode_quick"
QUICK_NAME = "VSCode Quick Benchmark"
QUICK_ALGORITHMS = ["GRPO", "PPO", "SAC"]
QUICK_TIMESTEPS = 5000
QUICK_SEED = 42
QUICK_DEVICE = "auto"
QUICK_EVAL_EPISODES = 3

FULL_17_RUN_ID = "paper2_full_17_mainline_a"
FULL_17_NAME = "Paper2 Full 17 Algorithms Mainline-A Benchmark"
FULL_17_ALGORITHMS = [
    "GRPO",
    "PPO",
    "SAC",
    "DDQN",
    "DDPG",
    "TD3",
    "A3C",
    "TRPO",
    "SimPO",
    "MAPPO",
    "QMIX",
    "COMA",
    "IPPO",
    "VDN",
    "MADDPG",
    "IQL",
    "MATD3",
]
FULL_17_TIMESTEPS = 100000
FULL_17_SEED = 42
FULL_17_DEVICE = "auto"
FULL_17_EVAL_EPISODES = 10

SINGLE_POLICY_3USER_FULL17_RUN_ID = "paper2_single_policy_3user_full17_mainline_a"
SINGLE_POLICY_3USER_FULL17_NAME = "Single-Policy 3-User Full17 Mainline-A Benchmark"
SINGLE_POLICY_3USER_FULL17_ALGORITHMS = [
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
SINGLE_POLICY_3USER_FULL17_TIMESTEPS = 100000
SINGLE_POLICY_3USER_FULL17_SEED = 42
SINGLE_POLICY_3USER_FULL17_DEVICE = "auto"
SINGLE_POLICY_3USER_FULL17_EVAL_EPISODES = 10
SINGLE_POLICY_3USER_FULL17_NUM_USERS = 3

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "run_id": QUICK_RUN_ID,
        "name": QUICK_NAME,
        "algorithms": list(QUICK_ALGORITHMS),
        "timesteps": QUICK_TIMESTEPS,
        "seed": QUICK_SEED,
        "device": QUICK_DEVICE,
        "eval_episodes": QUICK_EVAL_EPISODES,
        "env": "auto",
        "environment_profile": DEFAULT_ENVIRONMENT_PROFILE,
        "output_dir": "results",
    },
    "full17": {
        "run_id": FULL_17_RUN_ID,
        "name": FULL_17_NAME,
        "algorithms": list(FULL_17_ALGORITHMS),
        "timesteps": FULL_17_TIMESTEPS,
        "seed": FULL_17_SEED,
        "device": FULL_17_DEVICE,
        "eval_episodes": FULL_17_EVAL_EPISODES,
        "env": "auto",
        "environment_profile": DEFAULT_ENVIRONMENT_PROFILE,
        "output_dir": "results",
    },
    "single_policy_3user_full17": {
        "run_id": SINGLE_POLICY_3USER_FULL17_RUN_ID,
        "name": SINGLE_POLICY_3USER_FULL17_NAME,
        "algorithms": list(SINGLE_POLICY_3USER_FULL17_ALGORITHMS),
        "timesteps": SINGLE_POLICY_3USER_FULL17_TIMESTEPS,
        "seed": SINGLE_POLICY_3USER_FULL17_SEED,
        "device": SINGLE_POLICY_3USER_FULL17_DEVICE,
        "eval_episodes": SINGLE_POLICY_3USER_FULL17_EVAL_EPISODES,
        "env": "auto",
        "environment_profile": DEFAULT_ENVIRONMENT_PROFILE,
        "output_dir": "results",
        "interface": "single_policy_multi_user",
        "num_users": SINGLE_POLICY_3USER_FULL17_NUM_USERS,
        "shared_reward": "mean",
    },
}
