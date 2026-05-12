"""Tests for Stage-1 tuning config generation."""

from pathlib import Path

import yaml

from scripts.tune_mainline_a_stage1 import (
    generate_trial_configs,
    load_search_config,
    materialize_trial_config,
)


def test_generate_trial_configs_uses_recommended_start_as_trial_zero() -> None:
    cfg = load_search_config("configs/tuning/stage1_ppo_mainline_a.yaml")

    trials = generate_trial_configs(cfg, trials=2, seed=42)

    assert trials[0]["trial_id"] == "0000"
    assert trials[0]["name"] == "PPO-B"
    assert trials[0]["params"]["training.lr"] == 0.00015
    assert trials[1]["trial_id"] == "0001"


def test_materialize_trial_config_applies_dotted_overrides(tmp_path: Path) -> None:
    search_cfg = load_search_config("configs/tuning/stage1_coma_mainline_a.yaml")
    trial = generate_trial_configs(search_cfg, trials=1, seed=42)[0]

    output = materialize_trial_config(search_cfg, trial, tmp_path)

    cfg = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert cfg["algorithm"]["policy_clip_low"] == 0.85
    assert cfg["algorithm"]["policy_clip_high"] == 1.15
    assert cfg["game_theory"]["warm_start_steps"] == 4000
