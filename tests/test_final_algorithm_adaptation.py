import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from scripts.benchmark import create_agent, load_config, make_env
from scripts.plot_results import _build_timestep_axis
from src.environments.mec_v3 import GameTheoryContinuousMAEnv, GameTheoryDiscreteMAEnv


def test_continuous_adapter_accepts_batched_multi_agent_actions():
    env = GameTheoryContinuousMAEnv(num_agents=3, num_edge_servers=3, max_steps=2)
    try:
        obs, _ = env.reset(seed=123)
        assert isinstance(obs, list)
        batched_actions = np.array(
            [
                [np.nan, 2.0, -3.0, 0.5],
                [-1.0, 0.0, 0.0, np.inf],
                [1.0, -np.inf, 0.25, -0.25],
            ],
            dtype=np.float32,
        )
        next_obs, reward, terminated, truncated, info = env.step(batched_actions)
        assert isinstance(next_obs, list)
        assert len(next_obs) == 3
        assert isinstance(info, dict)
        assert isinstance(reward, (list, np.ndarray))
        assert not (bool(terminated) and bool(truncated))
    finally:
        env.close()


def test_discrete_adapter_sanitizes_logits_and_out_of_range_actions():
    env = GameTheoryDiscreteMAEnv(num_agents=2, num_edge_servers=3, max_steps=2)
    try:
        env.reset(seed=321)
        # Row vectors are treated as logits/probabilities; scalar overflow is clipped.
        actions = np.array([[0.0, 3.0, np.nan], [999999.0, -1.0, 0.0]], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        assert isinstance(next_obs, list)
        assert len(next_obs) == 2
        assert isinstance(info, dict)
        assert isinstance(reward, (list, np.ndarray))
    finally:
        env.close()


def test_create_agent_reads_exploration_section_for_noise_parameters():
    ddpg_cfg = load_config("configs/algorithms/ddpg.yaml")
    env = make_env("MEC-v1-game-theory-continuous-ma", seed=7, num_agents=1)
    try:
        agent = create_agent("DDPG", env, ddpg_cfg, device="cpu")
        assert np.isclose(agent.ou_sigma, ddpg_cfg["exploration"]["ou_sigma"])
        assert np.isclose(agent.ou_theta, ddpg_cfg["exploration"]["ou_theta"])
    finally:
        env.close()

    td3_cfg = load_config("configs/algorithms/td3.yaml")
    env = make_env("MEC-v1-game-theory-continuous-ma", seed=8, num_agents=1)
    try:
        agent = create_agent("TD3", env, td3_cfg, device="cpu")
        assert np.isclose(agent.exploration_noise_std, td3_cfg["algorithm"]["exploration_noise_std"])
    finally:
        env.close()


def test_plot_axis_prefers_explicit_eval_steps():
    seed_payload = {"eval_steps": [0, 2500, 5000], "eval_interval": 10000}
    x = _build_timestep_axis(seed_payload, 3)
    np.testing.assert_allclose(x, np.array([0.0, 2500.0, 5000.0]))


def test_final_screening_config_dry_run_exposes_sampling_fields():
    config_path = Path("configs/benchmark_mainline_a_final_screening.yaml")
    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["screening"]["diagnostic_only_or_frozen"] == ["IQL", "VDN", "QMIX", "IPPO", "SAC", "A3C"]

    completed = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--config", str(config_path), "--dry-run"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    dry_run_yaml = completed.stdout.split("DRY RUN benchmark", 1)[1]
    payload = yaml.safe_load(dry_run_yaml)
    assert payload["eval_interval"] == 2500
    assert payload["min_eval_points"] == 32
    assert payload["eval_at_start"] is True
    assert payload["enable_mainline_a"] is True
