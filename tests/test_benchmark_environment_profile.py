"""Tests for direct benchmark environment profile handling."""

from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.benchmark import _dry_run_payload, _resolve_profile_env_overrides


ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides):
    data = {
        "environment_profile": "mainline-a",
        "enable_mainline_a": False,
        "system_model_config": None,
        "dynamic_pricing_config": None,
        "channel_model": None,
        "queue_model": None,
        "mobility_intensity": None,
        "config": None,
        "seeds": [42],
        "timesteps": None,
    }
    data.update(overrides)
    return Namespace(**data)


def test_benchmark_profile_overrides_default_to_mainline_a() -> None:
    """Direct benchmark should inject Mainline-A configs by default."""
    overrides = _resolve_profile_env_overrides(_args())
    assert overrides["enable_mainline_a"] is True
    assert overrides["system_model_config"] == "configs/system_model_mainline_a.yaml"
    assert overrides["dynamic_pricing_config"] == "configs/pricing_dynamic_mainline_a.yaml"


def test_benchmark_legacy_profile_dry_run_disables_mainline_a() -> None:
    """Legacy direct benchmark should be explicit and should disable Mainline-A."""
    payload = _dry_run_payload(_args(environment_profile="legacy"), {}, ["GRPO"])
    assert payload["environment_profile"] == "legacy"
    assert payload["enable_mainline_a"] is False
    assert payload["system_model_config"] is None


def test_benchmark_all_mainline_a_dry_run_is_parse_only() -> None:
    """The full17 direct benchmark dry-run should not start training."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark.py"),
            "--all",
            "--environment-profile",
            "mainline-a",
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
    payload = yaml.safe_load(result.stdout.split("DRY RUN benchmark", maxsplit=1)[1])
    assert payload["environment_profile"] == "mainline-a"
    assert payload["enable_mainline_a"] is True
    assert len(payload["algorithms"]) == 17
