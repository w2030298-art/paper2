"""Tests for train.py environment profile arguments."""

from argparse import Namespace
from importlib import util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    train_path = ROOT / "scripts" / "train.py"
    spec = util.spec_from_file_location("scripts_train_profile_module", train_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_help_exposes_environment_profile() -> None:
    """train.py should advertise the shared profile argument."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--environment-profile" in result.stdout


def test_train_mainline_a_profile_injects_configs() -> None:
    """The default train profile should enable Mainline-A without extra CLI flags."""
    train_module = _load_train_module()
    args = Namespace(
        environment_profile="mainline-a",
        enable_mainline_a=False,
        system_model_config=None,
        dynamic_pricing_config=None,
    )
    overrides = train_module._resolve_profile_env_overrides(args)
    assert overrides["enable_mainline_a"] is True
    assert overrides["system_model_config"] == "configs/system_model_mainline_a.yaml"
    assert overrides["dynamic_pricing_config"] == "configs/pricing_dynamic_mainline_a.yaml"


def test_train_legacy_profile_rejects_silent_fallback_mix() -> None:
    """Legacy fallback must not be combined with Mainline-A compatibility flags."""
    train_module = _load_train_module()
    args = Namespace(
        environment_profile="legacy",
        enable_mainline_a=True,
        system_model_config=None,
        dynamic_pricing_config=None,
    )
    with pytest.raises(ValueError, match="incompatible"):
        train_module._resolve_profile_env_overrides(args)
