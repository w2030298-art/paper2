"""Tests for backup_experiment script."""

from __future__ import annotations

import pytest

from scripts.backup_experiment import BackupResult, backup_experiment


def test_backup_experiment_copies_experiment_dir_and_benchmark_files(tmp_path) -> None:
    # Setup experiment directory
    exp_dir = tmp_path / "experiments" / "paper2_full_17_vscode"
    exp_dir.mkdir(parents=True)
    (exp_dir / "run.json").write_text("{}")
    (exp_dir / "state.json").write_text("{}")

    # Setup results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "benchmark.json").write_text("{}")
    (results_dir / "benchmark_paper2_full_17_vscode.json").write_text("{}")

    # Setup old archive (should NOT be copied)
    old_archive = results_dir / "archive" / "old"
    old_archive.mkdir(parents=True)
    (old_archive / "benchmark_old.json").write_text("{}")

    result = backup_experiment(
        run_id="paper2_full_17_vscode",
        experiments_dir=tmp_path / "experiments",
        results_dir=results_dir,
        figures_dir=tmp_path / "figures",
        timestamp="20260501_150000",
    )

    assert not result.skipped
    assert result.experiment_backup_dir is not None
    assert (tmp_path / "experiments" / "paper2_full_17_vscode_backup_20260501_150000" / "run.json").exists()
    assert (tmp_path / "experiments" / "paper2_full_17_vscode_backup_20260501_150000" / "state.json").exists()
    assert (tmp_path / "results" / "archive" / "20260501_150000" / "benchmark.json").exists()
    assert (tmp_path / "results" / "archive" / "20260501_150000" / "benchmark_paper2_full_17_vscode.json").exists()
    # Old archive should NOT be copied
    assert not (tmp_path / "results" / "archive" / "20260501_150000" / "archive").exists()
    assert len(result.copied_result_files) == 2


def test_backup_experiment_include_plots_copies_top_level_figures(tmp_path) -> None:
    # Setup experiment directory
    exp_dir = tmp_path / "experiments" / "paper2_full_17_vscode"
    exp_dir.mkdir(parents=True)
    (exp_dir / "run.json").write_text("{}")

    # Setup figures directory
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "convergence_curves.png").write_text("")

    # Setup old archive (should NOT be copied)
    old_archive = figures_dir / "archive" / "old"
    old_archive.mkdir(parents=True)
    (old_archive / "old.png").write_text("")

    result = backup_experiment(
        run_id="paper2_full_17_vscode",
        experiments_dir=tmp_path / "experiments",
        results_dir=tmp_path / "results",
        figures_dir=figures_dir,
        include_plots=True,
        timestamp="20260501_150000",
    )

    assert not result.skipped
    assert result.figures_archive_dir is not None
    assert (tmp_path / "figures" / "archive" / "20260501_150000" / "convergence_curves.png").exists()
    # Old archive should NOT be copied
    assert not (tmp_path / "figures" / "archive" / "20260501_150000" / "archive").exists()
    assert len(result.copied_figure_files) == 1


def test_backup_experiment_skips_missing_when_not_required(tmp_path) -> None:
    result = backup_experiment(
        run_id="nonexistent_run",
        experiments_dir=tmp_path / "experiments",
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        timestamp="20260501_150000",
    )

    assert result.skipped
    assert result.reason == "experiment directory not found"
    assert result.experiment_backup_dir is None


def test_backup_experiment_requires_existing_when_requested(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Experiment directory not found"):
        backup_experiment(
            run_id="nonexistent_run",
            experiments_dir=tmp_path / "experiments",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            timestamp="20260501_150000",
            require_existing=True,
        )


def test_backup_experiment_refuses_running_experiment(tmp_path) -> None:
    # Setup experiment directory with process.json (running)
    exp_dir = tmp_path / "experiments" / "paper2_full_17_vscode"
    exp_dir.mkdir(parents=True)
    (exp_dir / "run.json").write_text("{}")
    (exp_dir / "process.json").write_text("{}")

    with pytest.raises(RuntimeError, match="Cannot backup running experiment"):
        backup_experiment(
            run_id="paper2_full_17_vscode",
            experiments_dir=tmp_path / "experiments",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            timestamp="20260501_150000",
        )


def test_backup_experiment_rejects_unsafe_run_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="run_id cannot be empty"):
        backup_experiment(
            run_id="",
            experiments_dir=tmp_path / "experiments",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
        )

    with pytest.raises(ValueError, match="run_id contains path traversal"):
        backup_experiment(
            run_id="../x",
            experiments_dir=tmp_path / "experiments",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
        )

    with pytest.raises(ValueError, match="run_id contains invalid characters"):
        backup_experiment(
            run_id="a/b",
            experiments_dir=tmp_path / "experiments",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
        )

    with pytest.raises(ValueError, match="run_id contains invalid characters"):
        backup_experiment(
            run_id="a\\b",
            experiments_dir=tmp_path / "experiments",
            results_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
        )


def test_backup_experiment_does_not_copy_existing_archives(tmp_path) -> None:
    # Setup experiment directory
    exp_dir = tmp_path / "experiments" / "paper2_full_17_vscode"
    exp_dir.mkdir(parents=True)
    (exp_dir / "run.json").write_text("{}")

    # Setup results with nested archive
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "benchmark.json").write_text("{}")
    old_archive = results_dir / "archive" / "old"
    old_archive.mkdir(parents=True)
    (old_archive / "benchmark_old.json").write_text("{}")

    result = backup_experiment(
        run_id="paper2_full_17_vscode",
        experiments_dir=tmp_path / "experiments",
        results_dir=results_dir,
        figures_dir=tmp_path / "figures",
        timestamp="20260501_150000",
    )

    # Verify only top-level benchmark files were copied
    assert len(result.copied_result_files) == 1
    assert "benchmark.json" in result.copied_result_files
    assert "benchmark_old.json" not in result.copied_result_files
