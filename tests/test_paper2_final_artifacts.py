"""Tests for paper2 final figure and delivery artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


FIGURE_SCRIPT = Path("scripts/generate_paper2_figures.py")
EXPORT_SCRIPT = Path("scripts/export_paper2_final_package.py")
DELIVERY_MANIFEST = Path("ref/paper2_final_delivery_manifest.md")
ADAPTATION_CHECKLIST = Path("ref/paper2_final_adaptation_checklist.md")


def _write_results_manifest(path: Path) -> None:
    """Write a minimal planned-results manifest for figure dry-runs."""
    payload = {
        "schema_version": "paper2-main-matrix-manifest/v1",
        "dry_run": True,
        "level": "L1_screening",
        "runs": [
            {"scenario": "ID-mainline-a", "algorithm": "CL-PPO", "seed": 42},
            {"scenario": "N2-cl-ppo-ablation", "algorithm": "CL-PPO", "seed": 42},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_figure_dry_run_writes_plan_without_fake_figures(tmp_path: Path) -> None:
    """Figure dry-run should write a plan and avoid fake plot files."""
    results = tmp_path / "manifest.json"
    output_dir = tmp_path / "figures"
    _write_results_manifest(results)

    subprocess.run(
        [
            sys.executable,
            str(FIGURE_SCRIPT),
            "--results",
            str(results),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    plan = json.loads((output_dir / "figure_plan.json").read_text(encoding="utf-8"))
    assert plan["schema_version"] == "paper2-figure-plan/v1"
    assert plan["dry_run"] is True
    assert {group["id"] for group in plan["figure_groups"]} == {
        "convergence",
        "final_comparison",
        "n2_ablation",
        "n3_ood",
        "statistics",
    }
    assert list(output_dir.glob("*.pdf")) == []
    assert list(output_dir.glob("*.png")) == []


def test_export_dry_run_writes_package_manifest_without_zip(tmp_path: Path) -> None:
    """Export dry-run should report package contents without creating the zip."""
    output = tmp_path / "paper2_final_delivery_package.zip"

    subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--dry-run", "--output", str(output)],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((tmp_path / "paper2_final_delivery_package.zip.manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "paper2-final-package-manifest/v1"
    assert manifest["dry_run"] is True
    assert output.name in manifest["output"]
    assert output.exists() is False
    assert any(item["path"] == "ref/paper2_final_delivery_manifest.md" for item in manifest["items"])
    assert any(item["status"] == "missing_optional" for item in manifest["items"])


def test_final_reference_artifacts_declare_required_contracts() -> None:
    """Final manifest and checklist should be concrete enough for review."""
    manifest_text = DELIVERY_MANIFEST.read_text(encoding="utf-8")
    checklist_text = ADAPTATION_CHECKLIST.read_text(encoding="utf-8")

    for heading in [
        "Required Deliverables",
        "Source Path",
        "Generation Command",
        "Required Input Data",
        "Claim Status",
    ]:
        assert heading in manifest_text

    for item in [
        "Algorithm registration: PPO, COMA, CL-PPO, GAM-COMA, baselines.",
        "Config coverage: algorithm configs, main matrix, scenario matrix, formal convergence matrix.",
        "Experiment execution: L1/L2/L3 commands and outputs.",
        "Statistical verification: seed completeness, CI, p-value/effect-size outputs, multiple-comparison correction.",
        "Figure package: convergence, final comparison, ablation, OOD, statistics.",
        "State protocol: ledger/checkpoint only, inbox consumed, no README dashboard.",
    ]:
        assert item in checklist_text
