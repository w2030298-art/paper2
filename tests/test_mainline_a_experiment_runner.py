"""Tests for the mainline-A experiment runner."""

import json
import math
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_mainline_a_experiments import (
    N2_REQUIRED_ABLATIONS,
    N2_REQUIRED_METRICS,
    N3_REQUIRED_METRICS,
    build_n1_case_matrix,
    build_n2_ablation_matrix,
    load_experiment_config,
    resolve_plans,
    run_n1_oracle_validation,
    run_n2_ablation_validation,
    run_n3_ood_validation,
    validate_n1_oracle_config,
    validate_n2_ablation_config,
    validate_n3_ood_config,
)

ROOT = Path(__file__).resolve().parents[1]
N2_CONFIG = ROOT / "configs" / "experiments" / "mainline_a_n2_ablation.yaml"
N3_CONFIG = ROOT / "configs" / "experiments" / "mainline_a_n3_ood.yaml"


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
        [
            sys.executable,
            str(ROOT / "scripts" / "run_mainline_a_experiments.py"),
            "--stage",
            "all",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
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


def test_n1_config_within_oracle_limits() -> None:
    """N1 cases must stay within the small-scale oracle support limits."""
    config = load_experiment_config(ROOT / "configs" / "experiments" / "mainline_a_n1_oracle.yaml")

    validate_n1_oracle_config(config)
    matrix = build_n1_case_matrix(config)

    assert len(matrix) == 18
    assert max(case["num_users"] for case in matrix) == 4
    assert max(case["num_edges"] for case in matrix) == 3


def test_n1_oracle_runner_writes_stable_report(tmp_path) -> None:
    """N1 runner should write oracle-gap artifacts without benchmark training."""
    config = load_experiment_config(ROOT / "configs" / "experiments" / "mainline_a_n1_oracle.yaml")

    result = run_n1_oracle_validation(config, tmp_path)

    assert result["stage"] == "N1"
    assert result["case_count"] == 18
    assert result["record_count"] == 54
    assert result["report_path"].is_file()
    assert result["summary_path"].is_file()
    assert all(record["stage"] == "N1" for record in result["records"])
    assert all(record["oracle_gap"] is not None for record in result["records"])
    assert all(record["constraint_violation"] >= 0.0 for record in result["records"])


def test_n2_dry_run_plan_exposes_ablation_matrix() -> None:
    """N2 dry-run should expose every ablation and the concrete switches."""
    args = Namespace(
        config=str(N2_CONFIG),
        dry_run=True,
        stage="N2",
        resume=False,
        output_root="experiments/mainline_a",
        results_root="results/mainline_a",
    )

    plans = resolve_plans(args)

    assert len(plans) == 1
    plan = plans[0]
    assert plan["stage"] == "N2"
    assert plan["evidence_level"] == "deterministic controlled probe"
    assert plan["ablations"] == list(N2_REQUIRED_ABLATIONS)
    assert plan["planned_record_count"] == 27
    assert plan["results_path"].endswith("n2_ablation")
    assert all("switches" in item for item in plan["ablation_matrix"])
    assert all("dynamic_pricing" in item["switches"] for item in plan["ablation_matrix"])


def test_n2_config_rejects_label_without_switch_mapping() -> None:
    """N2 ablations must be backed by explicit switch mappings."""
    config = load_experiment_config(N2_CONFIG)
    config["ablations"] = [*config["ablations"], "label_only"]

    with pytest.raises(ValueError, match="unknown ablation"):
        validate_n2_ablation_config(config)

    matrix = build_n2_ablation_matrix(
        load_experiment_config(N2_CONFIG)
    )
    assert {item["ablation"] for item in matrix} == set(N2_REQUIRED_ABLATIONS)


def test_n2_config_rejects_duplicate_ablation_labels() -> None:
    """N2 ablations should not silently accept duplicate labels."""
    config = load_experiment_config(N2_CONFIG)
    config["ablations"] = [*config["ablations"], "full_model"]

    with pytest.raises(ValueError, match="duplicate ablation"):
        validate_n2_ablation_config(config)


def test_n2_preflight_writes_required_metrics_and_schema(tmp_path) -> None:
    """N2 preflight should write one seed per ablation with finite required metrics."""
    config = load_experiment_config(N2_CONFIG)

    result = run_n2_ablation_validation(config, tmp_path, preflight=True, preflight_steps=32)
    summary = result["summary"]
    records = result["records"]

    assert summary["status"] == "ok"
    assert summary["run_type"] == "preflight"
    assert summary["evidence_level"] == "deterministic controlled probe"
    assert summary["record_count"] == len(N2_REQUIRED_ABLATIONS)
    assert Path(summary["records_path"]).is_file()
    assert Path(summary["matrix_path"]).is_file()
    assert summary["benchmark_alias_overwrite"] is False
    schemas = {tuple(sorted(record["metrics"])) for record in records}
    assert len(schemas) == 1
    for record in records:
        for metric in N2_REQUIRED_METRICS:
            assert metric in record["metrics"]
            assert math.isfinite(float(record["metrics"][metric]))


def test_n2_controlled_ablation_does_not_overwrite_benchmark_alias(tmp_path) -> None:
    """N2 outputs should stay under n2_ablation and preserve benchmark alias files."""
    config = load_experiment_config(N2_CONFIG)
    sentinel = tmp_path / "benchmark.json"
    sentinel.write_text(json.dumps({"keep": True}), encoding="utf-8")

    result = run_n2_ablation_validation(config, tmp_path)

    assert sentinel.read_text(encoding="utf-8") == '{"keep": true}'
    assert result["summary"]["run_type"] == "controlled"
    assert result["summary"]["record_count"] == 27
    assert "full_model" in result["summary"]["metric_means"]
    assert "no_dynamic_price" in result["summary"]["metric_deltas_vs_full_model"]
    assert Path(result["summary"]["output_dir"]).name == "n2_ablation"


def test_n2_controlled_probe_does_not_create_missing_benchmark_alias(tmp_path) -> None:
    """N2 controlled probe must not create results/benchmark.json when absent."""
    config = load_experiment_config(N2_CONFIG)
    benchmark_alias = tmp_path / "benchmark.json"

    result = run_n2_ablation_validation(config, tmp_path)

    assert not benchmark_alias.exists()
    assert result["summary"]["benchmark_alias_overwrite"] is False


def test_n3_dry_run_plan_exposes_ood_distribution() -> None:
    """N3 dry-run should expose train/test distribution and required metrics."""
    args = Namespace(
        config=str(N3_CONFIG),
        dry_run=True,
        stage="N3",
        resume=False,
        output_root="experiments/mainline_a",
        results_root="results/mainline_a",
    )

    plans = resolve_plans(args)

    assert len(plans) == 1
    plan = plans[0]
    assert plan["stage"] == "N3"
    assert plan["evidence_level"] == "OOD formal execution"
    assert plan["required_metrics"] == list(N3_REQUIRED_METRICS)
    assert plan["distribution_shift"]["train"]["users"] == 20
    assert plan["distribution_shift"]["test"]["users"] == 40
    assert plan["distribution_shift"]["test"]["channel"] == "3gpp_lite"
    assert plan["distribution_shift"]["test"]["mobility"] == "high"
    assert plan["distribution_shift"]["test"]["queue_model"] == "parallel"
    assert plan["distribution_shift"]["test"]["cooperation_enabled"] is True
    assert plan["results_path"].endswith("n3_ood")
    assert plan["preflight_supported"] is True


def test_n3_config_requires_actual_ood_test_distribution() -> None:
    """N3 config should reject label-only OOD tests that miss required test knobs."""
    config = load_experiment_config(N3_CONFIG)
    validate_n3_ood_config(config)

    config["test"]["channel"] = "analytic"
    with pytest.raises(ValueError, match="3gpp_lite"):
        validate_n3_ood_config(config)


def test_n3_preflight_writes_required_metrics_and_audit(tmp_path) -> None:
    """N3 preflight should write finite metrics and verify the OOD test settings."""
    config = load_experiment_config(N3_CONFIG)

    result = run_n3_ood_validation(config, tmp_path, preflight=True, preflight_steps=32)
    summary = result["summary"]
    records = result["records"]

    assert summary["status"] == "ok"
    assert summary["run_type"] == "preflight"
    assert summary["record_count"] == 2
    assert Path(summary["records_path"]).is_file()
    assert Path(summary["distribution_path"]).is_file()
    assert summary["benchmark_alias_overwrite"] is False
    assert summary["audit"]["ood_test_distribution_applied"] is True
    assert not summary["audit"]["issues"]
    schemas = {tuple(sorted(record["metrics"])) for record in records}
    assert schemas == {tuple(sorted(N3_REQUIRED_METRICS))}
    for record in records:
        for metric in N3_REQUIRED_METRICS:
            assert metric in record["metrics"]
            assert math.isfinite(float(record["metrics"][metric]))


def test_n3_formal_execution_stays_under_n3_output_dir(tmp_path) -> None:
    """N3 formal execution should not create or overwrite a benchmark alias."""
    config = load_experiment_config(N3_CONFIG)
    sentinel = tmp_path / "benchmark.json"
    sentinel.write_text(json.dumps({"keep": True}), encoding="utf-8")

    result = run_n3_ood_validation(config, tmp_path)

    assert sentinel.read_text(encoding="utf-8") == '{"keep": true}'
    assert result["summary"]["run_type"] == "formal"
    assert result["summary"]["record_count"] == 6
    assert Path(result["summary"]["output_dir"]).name == "n3_ood"
    assert "social_welfare" in result["summary"]["test_metric_means"]
    assert result["summary"]["audit"]["metrics_not_all_identical"] is True
