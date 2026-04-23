"""
Integration tests: all 17 algorithms on GameTheory environments.

Verifies each algorithm can:
1. Be instantiated
2. Select actions on the environment
3. Complete a short episode without crashing
"""

import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "src"))

import pytest
import numpy as np
import torch
import yaml
from pathlib import Path

# Import all algorithm classes
from rl_algorithms.grpo import GRPOAgent as GRPO
from rl_algorithms.ppo import PPOAgent as PPO
from rl_algorithms.sac import SACAgent as SAC
from rl_algorithms.ddqn import DDQNAgent as DDQN
from rl_algorithms.ddpg import DDPGAgent as DDPG
from rl_algorithms.td3 import TD3Agent as TD3
from rl_algorithms.a3c import A3CAgent as A3C
from rl_algorithms.trpo import TRPOAgent as TRPO
from rl_algorithms.simpo import SimPOAgent as SimPO
from rl_algorithms.mappo import MAPPOAgent as MAPPO
from rl_algorithms.qmix import QMIXAgent as QMIX
from rl_algorithms.coma import COMAAgent as COMA
from rl_algorithms.ippo import IPPOAgent as IPPO
from rl_algorithms.vdn import VDNAgent as VDN
from rl_algorithms.maddpg import MADDPGAgent as MADDPG
from rl_algorithms.iql import IQLAgent as IQL
from rl_algorithms.matd3 import MATD3Agent as MATD3

from src.environments.mec_v3 import GameTheoryDiscreteMAEnv, GameTheoryContinuousMAEnv


def _to_scalar_action(action, is_discrete):
    """Convert agent action output to env-compatible scalar."""
    if hasattr(action, "numpy"):
        action = action.numpy()
    if isinstance(action, np.ndarray):
        if action.ndim > 1:
            action = action.flatten()
        action = action.item() if action.size == 1 else action[0]
    if isinstance(action, (int, float)):
        return int(action) if is_discrete else float(action)
    return int(action) if is_discrete else float(action)


def _is_done(terminated, truncated):
    t = terminated
    tr = truncated
    if hasattr(t, "item"):
        t = t.item()
    if hasattr(tr, "item"):
        tr = tr.item()
    return bool(t) or bool(tr)


def _build_agent_kwargs(algo_class, name, obs_dim, action_dim, num_agents):
    """Build constructor kwargs for each algorithm."""
    kw = dict(state_dim=obs_dim, action_dim=action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, device="cpu")
    if name in ("COMA", "IPPO", "MAPPO"):
        kw["num_agents"] = num_agents
    if name in ("QMIX", "VDN", "IQL", "MADDPG", "MATD3"):
        kw["n_agents"] = num_agents
    if name in ("COMA", "IPPO", "MAPPO"):
        kw["discrete"] = True
    return kw


# ---- Discrete algorithms on GameTheory discrete adapter ----

@pytest.mark.parametrize("algo_class,name,agents", [
    # single-agent discrete
    (A3C, "A3C", 1),
    (SimPO, "SimPO", 1),
    (DDQN, "DDQN", 1),
    # multi-agent discrete
    (QMIX, "QMIX", 3),
    (MAPPO, "MAPPO", 3),
    (COMA, "COMA", 3),
    (IPPO, "IPPO", 3),
    (VDN, "VDN", 3),
    (IQL, "IQL", 3),
])
def test_game_theory_discrete(algo_class, name, agents):
    """All discrete-action algorithms on GameTheory discrete adapter."""
    env = GameTheoryDiscreteMAEnv(num_agents=agents, num_edge_servers=3, max_steps=50)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    kw = _build_agent_kwargs(algo_class, name, obs_dim, action_dim, agents)
    agent = algo_class(**kw)
    obs, _ = env.reset(seed=42)

    steps = 3 if agents > 1 else 5
    for _ in range(steps):
        if agents > 1:
            actions = []
            for a_obs in obs:
                a, _ = agent.select_action(a_obs)
                actions.append(_to_scalar_action(a, is_discrete=True))
            obs, reward, terminated, truncated, info = env.step(actions)
        else:
            a_obs = obs[0] if isinstance(obs, list) else obs
            a, _ = agent.select_action(a_obs)
            scalar = _to_scalar_action(a, is_discrete=True)
            obs, reward, terminated, truncated, info = env.step([scalar])
        if _is_done(terminated, truncated):
            obs, _ = env.reset()

    sd = agent.state_dict()
    assert isinstance(sd, dict)


# ---- Continuous algorithms on GameTheory continuous adapter ----

@pytest.mark.parametrize("algo_class,name,agents", [
    # single-agent continuous
    (GRPO, "GRPO", 1),
    (PPO, "PPO", 1),
    (TRPO, "TRPO", 1),
    (SAC, "SAC", 1),
    (DDPG, "DDPG", 1),
    (TD3, "TD3", 1),
    # multi-agent continuous
    (MADDPG, "MADDPG", 3),
    (MATD3, "MATD3", 3),
])
def test_game_theory_continuous(algo_class, name, agents):
    """All continuous-action algorithms on GameTheory continuous adapter."""
    env = GameTheoryContinuousMAEnv(num_agents=agents, num_edge_servers=3, max_steps=50)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    kw = _build_agent_kwargs(algo_class, name, obs_dim, action_dim, agents)
    agent = algo_class(**kw)
    obs, _ = env.reset(seed=42)

    steps = 3 if agents > 1 else 5
    for _ in range(steps):
        if agents > 1:
            actions = []
            for a_obs in obs:
                a, _ = agent.select_action(a_obs)
                actions.append(a if isinstance(a, np.ndarray) else np.array(a))
            obs, reward, terminated, truncated, info = env.step(actions)
        else:
            a_obs = obs[0] if isinstance(obs, list) else obs
            a, _ = agent.select_action(a_obs)
            vec = a if isinstance(a, np.ndarray) else np.array(a)
            if not isinstance(vec, np.ndarray) or vec.shape[0] != action_dim:
                vec = np.zeros(action_dim)
            obs, reward, terminated, truncated, info = env.step([vec])
        if _is_done(terminated, truncated):
            obs, _ = env.reset()

    sd = agent.state_dict()
    assert isinstance(sd, dict)


# ---- Config loading test ----

def test_load_all_configs():
    """Verify all 17 algorithm configs can be loaded."""
    config_dir = Path("configs/algorithms")
    algo_names = [
        "grpo", "ppo", "sac", "ddqn", "ddpg", "td3", "a3c", "trpo", "simpo", "mappo", "qmix",
        "coma", "ippo", "vdn", "maddpg", "iql", "matd3",
    ]

    for name in algo_names:
        config_path = config_dir / f"{name}.yaml"
        assert config_path.exists(), f"Missing config: {config_path}"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None
        assert "algorithm" in cfg or "training" in cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
