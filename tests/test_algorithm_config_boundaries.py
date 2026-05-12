"""Regression tests for Mainline-A runtime boundary defaults."""

from pathlib import Path

import yaml

from src.experiment.presets import FULL_17_EVAL_EPISODES, FULL_17_TIMESTEPS


def test_ppo_and_coma_static_timesteps_match_full17_boundary() -> None:
    assert FULL_17_TIMESTEPS == 100000
    assert FULL_17_EVAL_EPISODES == 10

    for path in ("configs/algorithms/ppo.yaml", "configs/algorithms/coma.yaml"):
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        assert cfg["training"]["total_timesteps"] == 100000
