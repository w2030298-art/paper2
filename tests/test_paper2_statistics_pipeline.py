"""Tests for the paper2 v4.8 statistics pipeline."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ANALYZER = Path("scripts/analyze_paper2_statistics.py")


def _write_manifest(path: Path) -> None:
    """Write a minimal dry-run manifest fixture."""
    payload = {
        "schema_version": "paper2-main-matrix-manifest/v1",
        "matrix_version": "paper2-main-experiment-matrix/v1",
        "scenario_version": "paper2-scenario-matrix/v1",
        "level": "L1_screening",
        "claim_level": "screening_only",
        "dry_run": True,
        "runs": [
            {
                "algorithm": "CL-PPO",
                "scenario": "ID-mainline-a",
                "seed": 42,
                "steps": 50000,
                "output_path": "results/example.json",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_results(path: Path, complete: bool) -> None:
    """Write a synthetic results fixture with optional incomplete seed grid."""
    records = [
        ("ID-mainline-a", "PPO", 42, 1.0),
        ("ID-mainline-a", "PPO", 43, 1.2),
        ("ID-mainline-a", "CL-PPO", 42, 1.4),
    ]
    if complete:
        records.extend(
            [
                ("ID-mainline-a", "CL-PPO", 43, 1.5),
                ("ID-mainline-a", "PPO", 44, 1.1),
                ("ID-mainline-a", "CL-PPO", 44, 1.7),
            ]
        )
    payload = {
        "schema_version": "paper2-results/v1",
        "results": [
            {
                "scenario": scenario,
                "algorithm": algorithm,
                "seed": seed,
                "metrics": {"social_welfare_mean": value, "e2e_latency_mean": 2.0 - value / 10.0},
            }
            for scenario, algorithm, seed, value in records
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dry_run_manifest_writes_stats_plan_without_fake_statistics(tmp_path: Path) -> None:
    """Dry-run should validate a manifest without fabricating statistical values."""
    manifest = tmp_path / "manifest.json"
    output_dir = tmp_path / "stats"
    _write_manifest(manifest)

    subprocess.run(
        [sys.executable, str(ANALYZER), "--input", str(manifest), "--output-dir", str(output_dir), "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads((output_dir / "stats_plan.json").read_text(encoding="utf-8"))
    assert plan["mode"] == "manifest-only/dry-run"
    assert plan["statistics_generated"] is False
    assert "p_value" not in json.dumps(plan)
    assert "effect_size" not in json.dumps(plan)
    assert not (output_dir / "summary.csv").exists()
    assert not (output_dir / "statistics_report.json").exists()


def test_results_mode_rejects_incomplete_seed_grid_without_allow_partial(tmp_path: Path) -> None:
    """Results-present mode should reject partial seed grids by default."""
    results = tmp_path / "results.json"
    output_dir = tmp_path / "stats"
    _write_results(results, complete=False)

    result = subprocess.run(
        [sys.executable, str(ANALYZER), "--input", str(results), "--output-dir", str(output_dir)],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Incomplete seed grid" in result.stderr


def test_results_mode_writes_statistics_exports_for_complete_seed_grid(tmp_path: Path) -> None:
    """Complete multi-seed data should produce deterministic statistics exports."""
    results = tmp_path / "results.json"
    output_dir = tmp_path / "stats"
    _write_results(results, complete=True)

    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--input",
            str(results),
            "--output-dir",
            str(output_dir),
            "--primary-metric",
            "social_welfare_mean",
            "--baseline",
            "PPO",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (output_dir / "summary.csv").is_file()
    assert (output_dir / "pairwise_tests.csv").is_file()
    assert (output_dir / "effect_sizes.csv").is_file()
    report = json.loads((output_dir / "statistics_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "results-present"
    assert report["primary_metric"] == "social_welfare_mean"
    assert report["multiple_comparison_correction"]["method"] == "holm_bonferroni"

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["algorithm"] for row in rows} == {"PPO", "CL-PPO"}

    with (output_dir / "effect_sizes.csv").open(encoding="utf-8", newline="") as handle:
        effect_rows = list(csv.DictReader(handle))
    assert effect_rows[0]["baseline"] == "PPO"
    assert effect_rows[0]["comparison"] == "CL-PPO"
