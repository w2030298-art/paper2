"""Regression tests for repository hygiene after slimming."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_and_generated_directories_are_not_active() -> None:
    """Legacy docs archive and graphify cache should stay out of the active tree."""
    assert not (ROOT / "docs" / ".archive").exists()
    assert not (ROOT / "graphify-out" / "cache").exists()


def test_removed_entrypoints_and_old_tools_are_absent() -> None:
    """Confirmed obsolete entrypoints and tools should not come back."""
    removed_paths = [
        "src/trainer/benchmark.py",
        "scripts/evaluate.py",
        "scripts/generate_report.py",
        "src/trainer/callbacks.py",
        "src/utils/logger.py",
        "src/utils/config.py",
        "src/utils/action_utils.py",
        "rl_algorithms/utils/buffers.py",
        "docs_paper",
    ]

    for rel_path in removed_paths:
        assert not (ROOT / rel_path).exists(), f"{rel_path} should be removed"


def test_preserved_core_assets_still_exist() -> None:
    """Slimming should not remove explicitly preserved core assets."""
    preserved_paths = [
        "rl_algorithms",
        "src/comm",
        "src/experiment",
        "scripts/benchmark.py",
        "scripts/train.py",
        "scripts/experiment_manager.py",
        "scripts/plot_results.py",
        "scripts/backup_experiment.py",
        "src/utils/buffer.py",
    ]

    for rel_path in preserved_paths:
        assert (ROOT / rel_path).exists(), f"{rel_path} should be preserved"


def test_gitignore_covers_generated_artifacts() -> None:
    """Generated data directories and caches should be ignored."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in [
        "experiments/",
        "experiments/.index.sqlite3",
        "experiments/**/*.lock",
        "experiments/**/control/",
        "experiments/**/process.json",
        "experiments/**/logs/",
        "experiments/**/artifacts/**/*.pt",
        "experiments/**/artifacts/**/*.pth",
        "experiments/**/artifacts/**/*.ckpt",
        "figures/",
        "results/",
        "logs/",
        "checkpoints/",
        "graphify-out/cache/",
    ]:
        assert pattern in gitignore


def test_docs_slimming_manifest_records_dashboard_check() -> None:
    """Slim docs manifest should record the v4.3 package boundary."""
    audit = (ROOT / "docs" / "DOCS_SLIMMING_MANIFEST.md").read_text(encoding="utf-8")

    assert "paper2-docs-slim-v4.3" in audit
    assert "Long historical module-by-module plan" in audit
    assert "Do not copy long archived docs back into active `docs/`" in audit
