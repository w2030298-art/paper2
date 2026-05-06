"""Smoke tests for active command-line entrypoints."""

from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "scripts/benchmark.py",
        "scripts/train.py",
        "scripts/experiment_manager.py",
        "scripts/plot_results.py",
        "scripts/run_mainline_a_experiments.py",
        "scripts/run_formal_convergence_protocol.py",
    ],
)
def test_main_entrypoint_help_runs(script: str) -> None:
    """Active CLIs should expose --help without importing deleted modules."""
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_formal_convergence_l2_dry_run_without_runtime_results() -> None:
    """Formal dry-run should not require ignored results artifacts."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_formal_convergence_protocol.py"),
            "--phase",
            "L2",
            "--run-id",
            "pytest_l2_dry_run",
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
    assert "phase: l2" in result.stdout
