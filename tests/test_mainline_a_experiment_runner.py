"""Tests for the mainline-A experiment runner."""

from argparse import Namespace
from pathlib import Path
import subprocess
import sys

from scripts.run_mainline_a_experiments import resolve_plans
from scripts.run_mainline_a_experiments import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_runner_resolves_all_stage_plans() -> None:
    args = Namespace(
        config=None,
        dry_run=True,
        stage="all",
        resume=False,
        output_root="experiments/mainline_a",
        results_root="results/mainline_a",
    )

    plans = resolve_plans(args)

    assert [plan["stage"] for plan in plans] == ["N0", "N1", "N2", "N3"]
    assert all(plan["system_model"]["queue_model"] for plan in plans)


def test_default_stage_configs_are_tracked_files() -> None:
    """Default stage configs must be present in a clean checkout."""
    for name in [
        "mainline_a_n0_smoke.yaml",
        "mainline_a_n1_oracle.yaml",
        "mainline_a_n2_ablation.yaml",
        "mainline_a_n3_ood.yaml",
    ]:
        assert (ROOT / "configs" / "experiments" / name).is_file()


def test_runner_stage_all_dry_run_cli() -> None:
    """The public dry-run CLI should resolve all stages without training."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_mainline_a_experiments.py"), "--stage", "all", "--dry-run"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"stage": "N0"' in result.stdout
    assert '"stage": "N3"' in result.stdout


def test_runner_rejects_legacy_queue_channel_fields(tmp_path) -> None:
    """Mainline-A configs should not silently accept legacy queue/channel fields."""
    config = tmp_path / "legacy.yaml"
    config.write_text(
        """
name: bad_mainline_a
stage: N0
queue: mm1
channel: analytic
""",
        encoding="utf-8",
    )

    try:
        load_experiment_config(config)
    except ValueError as exc:
        assert "legacy top-level field" in str(exc)
    else:
        raise AssertionError("legacy queue/channel fields should be rejected")
