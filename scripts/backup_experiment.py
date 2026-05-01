#!/usr/bin/env python3
"""Backup experiment directory, benchmark result files, and optional figures."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BackupResult:
    run_id: str
    timestamp: str
    experiment_backup_dir: str | None
    results_archive_dir: str | None
    figures_archive_dir: str | None
    copied_result_files: list[str]
    copied_figure_files: list[str]
    skipped: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    """Return timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def assert_safe_run_id(run_id: str) -> None:
    """Allow only [A-Za-z0-9_.-]+ and reject path traversal."""
    if not run_id:
        raise ValueError("run_id cannot be empty")
    if ".." in run_id:
        raise ValueError(f"run_id contains path traversal: {run_id!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(f"run_id contains invalid characters: {run_id!r}")


def backup_experiment(
    *,
    run_id: str,
    experiments_dir: Path = Path("experiments"),
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
    include_plots: bool = False,
    suffix: str = "backup",
    timestamp: str | None = None,
    require_existing: bool = False,
) -> BackupResult:
    """Backup experiment directory, benchmark result json files, and optional figures."""
    assert_safe_run_id(run_id)
    if suffix not in ("backup", "auto"):
        raise ValueError(f"suffix must be 'backup' or 'auto', got {suffix!r}")

    if timestamp is None:
        timestamp = utc_timestamp()

    experiment_dir = experiments_dir / run_id
    experiment_backup_dir = experiments_dir / f"{run_id}_{suffix}_{timestamp}"

    # Check experiment directory existence
    if not experiment_dir.exists():
        if require_existing:
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        return BackupResult(
            run_id=run_id,
            timestamp=timestamp,
            experiment_backup_dir=None,
            results_archive_dir=None,
            figures_archive_dir=None,
            copied_result_files=[],
            copied_figure_files=[],
            skipped=True,
            reason="experiment directory not found",
        )

    # Refuse running experiment
    if (experiment_dir / "process.json").exists():
        raise RuntimeError(f"Cannot backup running experiment: {run_id}")

    # Check backup dir doesn't already exist
    if experiment_backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {experiment_backup_dir}")

    # Copy experiment directory
    shutil.copytree(experiment_dir, experiment_backup_dir)

    # Archive result files
    results_archive_dir: Path | None = None
    copied_result_files: list[str] = []
    result_files = list(results_dir.glob("benchmark*.json"))
    if result_files:
        results_archive_dir = results_dir / "archive" / timestamp
        results_archive_dir.mkdir(parents=True, exist_ok=True)
        for result_file in result_files:
            dest = results_archive_dir / result_file.name
            shutil.copy2(result_file, dest)
            copied_result_files.append(result_file.name)

    # Archive figures (only top-level files, not archive subdirectory)
    figures_archive_dir: Path | None = None
    copied_figure_files: list[str] = []
    if include_plots:
        figure_files = [
            f for f in figures_dir.iterdir()
            if f.is_file() and f.name != "archive"
        ]
        if figure_files:
            figures_archive_dir = figures_dir / "archive" / timestamp
            figures_archive_dir.mkdir(parents=True, exist_ok=True)
            for figure_file in figure_files:
                dest = figures_archive_dir / figure_file.name
                shutil.copy2(figure_file, dest)
                copied_figure_files.append(figure_file.name)

    return BackupResult(
        run_id=run_id,
        timestamp=timestamp,
        experiment_backup_dir=str(experiment_backup_dir),
        results_archive_dir=str(results_archive_dir) if results_archive_dir else None,
        figures_archive_dir=str(figures_archive_dir) if figures_archive_dir else None,
        copied_result_files=copied_result_files,
        copied_figure_files=copied_figure_files,
        skipped=False,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Backup experiment directory, benchmark result files, and optional figures."
    )
    parser.add_argument("--run-id", required=True, help="Experiment run ID to backup.")
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Root experiments directory (default: experiments).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory (default: results).",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Figures directory (default: figures).",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Also archive top-level figure files.",
    )
    parser.add_argument(
        "--suffix",
        choices=["backup", "auto"],
        default="backup",
        help="Suffix for backup directory name (default: backup).",
    )
    parser.add_argument(
        "--require-existing",
        action="store_true",
        help="Raise error if experiment directory does not exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run backup command and print BackupResult JSON."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = backup_experiment(
            run_id=args.run_id,
            experiments_dir=Path(args.experiments_dir),
            results_dir=Path(args.results_dir),
            figures_dir=Path(args.figures_dir),
            include_plots=args.include_plots,
            suffix=args.suffix,
            require_existing=args.require_existing,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return 0
    except FileNotFoundError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
