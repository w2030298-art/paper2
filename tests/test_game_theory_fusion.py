"""
Focused tests for game-theory MEC fusion components and deep-fusion algorithms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark import (  # noqa: E402
    ALGO_ENV_MAP,
    DEEP_FUSION_ALGOS,
    MULTI_AGENT_ALGOS,
    OFF_POLICY,
    ON_POLICY,
    create_agent,
    make_env,
    resolve_game_theory_config,
)
from src.environments.mec_v3.game_theory_env import (  # noqa: E402
    Channel3GPP,
    ConstraintProjection,
    DVFSEnergyModel,
    MonteCarloShapley,
    OptimalPricingMechanism,
    QueueingDelayModel,
)
from src.trainer.off_policy_trainer import OffPolicyTrainer  # noqa: E402
from src.trainer.on_policy_trainer import OnPolicyTrainer  # noqa: E402


def test_pricing_solver_respects_bounds():
    mech = OptimalPricingMechanism(
        eta=np.array([0.3, 0.4, 0.5], dtype=np.float32),
        c=np.array([0.1, 0.12, 0.11], dtype=np.float32),
        d0=np.array([1.0, 1.2, 0.9], dtype=np.float32),
        p_min=0.1,
        p_max=2.0,
    )
    prices = mech.solve_optimal_price()
    assert prices.shape == (3,)
    assert np.all(prices >= 0.1 - 1e-6)
    assert np.all(prices <= 2.0 + 1e-6)


def test_shapley_antithetic_symmetry():
    def coalition_value(s: frozenset[int]) -> float:
        return float(len(s))

    estimator = MonteCarloShapley(
        n_agents=3,
        coalition_value_fn=coalition_value,
        n_samples=64,
        use_antithetic=True,
    )
    shapley = estimator.compute()
    assert shapley.shape == (3,)
    assert np.allclose(shapley, np.array([1.0, 1.0, 1.0], dtype=np.float32), atol=0.15)


def test_queue_delay_and_dvfs_models():
    q = QueueingDelayModel(stability_margin=0.95)
    arrival = np.array([0.5, 0.8], dtype=np.float32)
    service = np.array([2.0, 2.0], dtype=np.float32)
    t_queue, t_system, rho = q.compute_delay(arrival, service)
    assert t_queue.shape == (2,)
    assert t_system.shape == (2,)
    assert rho.shape == (2,)
    assert np.all(rho < 1.0)
    assert t_queue[1] > t_queue[0]

    dvfs = DVFSEnergyModel()
    e_low = dvfs.compute_energy(freq=1.0e9, cpu_cycles=1.0e9)
    e_high = dvfs.compute_energy(freq=2.0e9, cpu_cycles=1.0e9)
    assert e_low > 0.0
    assert e_high > 0.0
    assert not np.isclose(e_low, e_high)


def test_channel_and_constraint_projection():
    channel = Channel3GPP(fc_ghz=3.5, bandwidth=20e6)
    sinr = channel.compute_sinr(
        d_target=50.0,
        d_interferers=np.array([80.0, 120.0], dtype=np.float32),
        p_target_w=0.2,
        p_interferers_w=np.array([0.1, 0.05], dtype=np.float32),
    )
    rate = channel.rate(sinr)
    assert sinr > 0.0
    assert rate > 0.0

    projector = ConstraintProjection(e_max=1.0, t_max=1.0, p_total_max=0.5, penalty_coeff=0.1, barrier_coeff=1e-4)
    projected = projector.project(
        action={"target": 1, "offload_ratio": 0.8, "cpu_freq": 2.5e9, "tx_power": 0.4},
        est_energy=2.0,
        est_latency=2.0,
        total_power=0.8,
    )
    assert projected.penalty >= 0.0
    assert np.isfinite(projected.barrier)


def test_env_info_has_required_game_theory_keys():
    env = make_env(
        "MEC-v1-game-theory-discrete-ma",
        seed=7,
        num_agents=3,
        game_theory_config={"enabled": True, "shapley_samples": 16, "reward_weights": (0.5, 0.3, 0.2)},
    )
    obs, info = env.reset(seed=7)
    actions = [env.action_space.sample() for _ in range(3)]
    _, _, _, _, info = env.step(actions)

    for key in ("game_hints", "shapley_allocation", "reward_terms", "queue_metrics", "constraint_metrics"):
        assert key in info
    assert "equilibrium_prices" in info["game_hints"]
    assert "equilibrium_actions" in info["game_hints"]


def _build_test_cfg(algo_name: str) -> dict:
    deep = algo_name in DEEP_FUSION_ALGOS
    return {
        "algorithm": {
            "name": algo_name,
            "gamma": 0.99,
            "num_epochs": 1,
            "batch_size": 8,
            "buffer_size": 256,
            "tau": 0.005,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 200,
            "update_interval": 1,
        },
        "network": {"hidden_dim": 64},
        "training": {"lr": 1e-3, "rollout_steps": 8, "total_timesteps": 8},
        "evaluation": {"num_episodes": 1},
        "logging": {"eval_interval": 999999, "save_interval": 999999, "log_interval": 999999},
        "game_theory": {
            "enabled": True,
            "use_shapley_credit": deep,
            "ctde_with_hints": deep,
            "warm_start_steps": 16 if deep else 0,
            "warm_start_lr_scale": 0.2,
            "shapley_samples": 16,
            "reward_weights": [0.5, 0.3, 0.2],
        },
    }


@pytest.mark.parametrize(
    "algo_name",
    ["GRPO", "MAPPO", "QMIX", "COMA", "IPPO", "VDN", "MADDPG", "IQL", "MATD3"],
)
def test_deep_fusion_algorithms_consume_game_theory_batches(algo_name: str):
    cfg = _build_test_cfg(algo_name)
    gt_cfg = resolve_game_theory_config(algo_name, cfg, overrides=None)
    num_agents = 3 if algo_name in MULTI_AGENT_ALGOS else 1
    env = make_env(ALGO_ENV_MAP[algo_name], seed=11, num_agents=num_agents, game_theory_config=gt_cfg)
    agent = create_agent(algo_name, env, cfg, device="cpu")

    if algo_name in ON_POLICY:
        trainer = OnPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=8,
            rollout_steps=8,
            eval_interval=999999,
            eval_episodes=1,
            save_interval=999999,
            save_dir=str(ROOT / "tmp_test_ckpt"),
            log_interval=999999,
            device="cpu",
            seed=11,
            num_epochs=1,
        )
        batch = trainer._collect_rollout()
        assert "game_hints" in batch
        assert "shapley_values" in batch
        info = agent.update(batch)
    else:
        trainer = OffPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=8,
            rollout_steps=8,
            eval_interval=999999,
            eval_episodes=1,
            save_interval=999999,
            save_dir=str(ROOT / "tmp_test_ckpt"),
            log_interval=999999,
            device="cpu",
            seed=11,
            warmup_steps=0,
            update_interval=1,
        )
        info = {}
        for _ in range(4):
            batch = trainer._collect_rollout()
            assert "game_hints" in batch
            assert "shapley_values" in batch
            info = agent.update(batch)
            if info.get("loss", 0.0) != 0.0:
                break

    assert isinstance(info, dict)
    assert "imitation_loss" in info
