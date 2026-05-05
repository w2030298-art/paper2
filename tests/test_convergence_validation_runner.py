"""Tests for the targeted convergence validation runner."""

from pathlib import Path

import yaml

from scripts.run_convergence_validation import (
    TARGET_ALGORITHMS,
    build_validation_plan,
    load_matrix,
    plan_to_manifest,
    write_plan_artifacts,
)


def test_convergence_validation_matrix_is_parseable() -> None:
    """The validation matrix should load and keep full-17 disabled."""
    matrix = load_matrix(Path("configs/convergence_validation_matrix.yaml"))

    assert matrix["default"]["no_full_17"] is True
    assert len(matrix["phases"]["baseline_50k"]["algorithms"]) == 11


def test_baseline_phase_does_not_enable_overrides(tmp_path: Path) -> None:
    """baseline_50k must run the target set without stability overrides."""
    matrix = load_matrix(Path("configs/convergence_validation_matrix.yaml"))
    plan = build_validation_plan(
        matrix,
        phase="baseline_50k",
        output_root=tmp_path,
        timestamp="test",
    )
    manifest = plan_to_manifest(plan, dry_run=True, no_submit=False)

    assert plan.algorithms == TARGET_ALGORITHMS
    assert plan.steps == 50000
    assert manifest["use_stability_overrides"] is False
    assert manifest["algorithms"] != sorted(
        [
            "GRPO",
            "PPO",
            "SAC",
            "DDQN",
            "DDPG",
            "TD3",
            "A3C",
            "TRPO",
            "SimPO",
            "MAPPO",
            "QMIX",
            "COMA",
            "IPPO",
            "VDN",
            "MADDPG",
            "IQL",
            "MATD3",
        ]
    )


def test_override_phase_requires_single_family_and_event_audit(tmp_path: Path) -> None:
    """override_50k should allow only one family after event audit exists."""
    matrix = load_matrix(Path("configs/convergence_validation_matrix.yaml"))
    audit = tmp_path / "convergence_event_audit.md"
    audit.write_text("decision: rerun_baseline\n", encoding="utf-8")

    plan = build_validation_plan(
        matrix,
        phase="override_50k",
        algorithms=["IQL", "VDN"],
        output_root=tmp_path,
        timestamp="test",
        event_audit_path=audit,
    )

    assert plan.family == "value_decomposition"
    assert plan.use_stability_overrides is True


def test_override_phase_rejects_missing_event_audit(tmp_path: Path) -> None:
    """override_50k should not execute before catastrophic event audit exists."""
    matrix = load_matrix(Path("configs/convergence_validation_matrix.yaml"))

    try:
        build_validation_plan(
            matrix,
            phase="override_50k",
            algorithms=["IQL"],
            output_root=tmp_path,
            timestamp="test",
            event_audit_path=tmp_path / "missing.md",
        )
    except ValueError as exc:
        assert "convergence_event_audit" in str(exc)
    else:
        raise AssertionError("override_50k should require event audit")


def test_override_phase_rejects_mixed_families(tmp_path: Path) -> None:
    """override_50k cannot mix algorithm families in one run."""
    matrix = load_matrix(Path("configs/convergence_validation_matrix.yaml"))
    audit = tmp_path / "convergence_event_audit.md"
    audit.write_text("decision: rerun_baseline\n", encoding="utf-8")

    try:
        build_validation_plan(
            matrix,
            phase="override_50k",
            algorithms=["IQL", "SAC"],
            output_root=tmp_path,
            timestamp="test",
            event_audit_path=audit,
        )
    except ValueError as exc:
        assert "one family" in str(exc)
    else:
        raise AssertionError("override_50k should reject mixed families")


def test_write_plan_artifacts_uses_ignored_experiment_root(tmp_path: Path) -> None:
    """Dry-run should write manifest and commands without launching training."""
    matrix = load_matrix(Path("configs/convergence_validation_matrix.yaml"))
    plan = build_validation_plan(
        matrix,
        phase="baseline_50k",
        algorithms=["A3C"],
        seeds=[42],
        output_root=tmp_path,
        timestamp="test",
    )

    write_plan_artifacts(plan, dry_run=True, no_submit=False)

    assert plan.manifest_path.is_file()
    assert plan.commands_path.is_file()
    assert "--timesteps 50000" in plan.commands_path.read_text(encoding="utf-8")


def test_stability_overrides_remains_disabled() -> None:
    """The candidate stability override config must stay disabled by default."""
    payload = yaml.safe_load(Path("configs/stability_overrides.yaml").read_text(encoding="utf-8"))

    assert payload["enabled"] is False
