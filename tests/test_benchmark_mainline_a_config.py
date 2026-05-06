"""Tests for Mainline-A benchmark config binding."""

from argparse import Namespace
from pathlib import Path
import subprocess
import sys

from scripts.benchmark import _apply_benchmark_cli_config
from scripts.benchmark import _canonical_algorithm_name
from scripts.benchmark import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_mainline_a_n0_config_applies_training_defaults() -> None:
    """N0 smoke config should drive benchmark execution without CLI overrides."""
    config_path = "configs/experiments/mainline_a_n0_smoke.yaml"
    args = Namespace(
        config=config_path,
        timesteps=None,
        seeds=[42],
        num_edge_servers=None,
        multi_agent_count=None,
        system_model_config="configs/system_model_mainline_a.yaml",
        dynamic_pricing_config="configs/pricing_dynamic_mainline_a.yaml",
        enable_mainline_a=False,
        queue_model=None,
        channel_model=None,
    )

    _apply_benchmark_cli_config(args, load_config(config_path))

    assert args.timesteps == 1000
    assert args.seeds == [42]
    assert args.num_edge_servers == 2
    assert args.multi_agent_count == 4
    assert args.enable_mainline_a is True
    assert args.system_model_config == config_path
    assert args.dynamic_pricing_config == config_path
    assert args.queue_model == "mm1"
    assert args.channel_model == "analytic"


def test_game_aware_pd_marl_alias_uses_registered_mappo() -> None:
    """The N0 label maps to the existing game-aware MAPPO training path."""
    assert _canonical_algorithm_name("game_aware_pd_marl") == "MAPPO"


def test_mainline_a_n1_config_benchmark_dry_run_is_parse_only() -> None:
    """N1 config should be parseable by benchmark dry-run without training."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark.py"),
            "--config",
            "configs/experiments/mainline_a_n1_oracle.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "baseline_static_stackelberg" in result.stdout
    assert "queue_model: mm1" in result.stdout
    assert "DRY RUN benchmark" in result.stdout
