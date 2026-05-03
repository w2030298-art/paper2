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
