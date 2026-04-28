"""Experiment presets shared by CLI and VSCode entrypoints."""

from __future__ import annotations

from typing import Any


QUICK_RUN_ID = "vscode_quick"
QUICK_NAME = "VSCode Quick Benchmark"
QUICK_ALGORITHMS = ["GRPO", "PPO", "SAC"]
QUICK_TIMESTEPS = 5000
QUICK_SEED = 42
QUICK_DEVICE = "auto"
QUICK_EVAL_EPISODES = 3

FULL_17_RUN_ID = "paper2_full_17_vscode"
FULL_17_NAME = "Paper2 Full 17 Algorithms VSCode Benchmark"
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
        "output_dir": "results",
    },
}
