"""Tests for Stage-1 PPO/COMA tuning search configs."""

from pathlib import Path

import yaml


def test_stage1_search_configs_encode_recommended_trial_zero() -> None:
    cases = [
        ("configs/tuning/stage1_ppo_mainline_a.yaml", "PPO", "PPO-B"),
        ("configs/tuning/stage1_coma_mainline_a.yaml", "COMA", "COMA-L"),
    ]

    for path, algorithm, start in cases:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert cfg["algorithm"] == algorithm
        assert cfg["environment_profile"] == "mainline-a"
        assert cfg["recommended_start"]["name"] == start
        assert cfg["base_config"] == f"configs/algorithms/{algorithm.lower()}.yaml"
        assert "forbidden_changes" in cfg
        assert cfg["search_space"]


def test_stage1_recommended_starts_include_required_parameters() -> None:
    ppo = yaml.safe_load(Path("configs/tuning/stage1_ppo_mainline_a.yaml").read_text(encoding="utf-8"))
    coma = yaml.safe_load(Path("configs/tuning/stage1_coma_mainline_a.yaml").read_text(encoding="utf-8"))

    assert ppo["recommended_start"]["params"]["training.lr"] == 0.00015
    assert ppo["recommended_start"]["params"]["algorithm.eps_clip"] == 0.12
    assert ppo["recommended_start"]["params"]["game_theory.reward_weights"] == [0.8, 0.1, 0.1]
    assert coma["recommended_start"]["params"]["algorithm.policy_clip_low"] == 0.85
    assert coma["recommended_start"]["params"]["algorithm.policy_clip_high"] == 1.15
    assert coma["recommended_start"]["params"]["algorithm.entropy_coeff"] == 0.0005
