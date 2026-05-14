"""Tests for Full17 backup plotting helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.plot_full17_backup_results import (
    load_algorithm_curves,
    load_summary_rows,
    numeric_columns,
)


def test_load_algorithm_curves_reads_explicit_backup_dir(tmp_path: Path) -> None:
    """Convergence data should come from the requested experiment backup directory."""
    log_dir = tmp_path / "artifacts" / "GRPO" / "checkpoints"
    log_dir.mkdir(parents=True)
    (log_dir / "train_logs.json").write_text(
        json.dumps(
            {
                "eval_steps": [1000, 2000, 3000],
                "eval_eval/reward_mean": [-4.0, -3.0, -2.0],
                "eval_eval/e2e_latency_mean": [0.4, 0.3, 0.2],
                "eval_eval/comm_score": [10.0, 20.0, 30.0],
                "eval_eval/energy_per_task_mean": [0.08, 0.07, 0.06],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "GRPO" / "result.json").write_text(
        json.dumps({"algorithm": "GRPO", "final_eval": {"eval/reward_mean": -2.0}}),
        encoding="utf-8",
    )

    curves = load_algorithm_curves(tmp_path)

    assert list(curves) == ["GRPO"]
    assert curves["GRPO"].steps.tolist() == [1000.0, 2000.0, 3000.0]
    assert curves["GRPO"].metrics["reward"].tolist() == [-4.0, -3.0, -2.0]
    assert curves["GRPO"].metrics["latency"].tolist() == [0.4, 0.3, 0.2]


def test_load_summary_rows_and_numeric_columns(tmp_path: Path) -> None:
    """CSV summary rows should retain text fields and expose numeric metric columns."""
    csv_path = tmp_path / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["algorithm", "family", "reward_mean", "risk_flags"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "algorithm": "GRPO",
                "family": "single-agent",
                "reward_mean": "-5.0",
                "risk_flags": "none",
            }
        )

    rows, fieldnames = load_summary_rows(csv_path)

    assert fieldnames == ["algorithm", "family", "reward_mean", "risk_flags"]
    assert rows[0]["algorithm"] == "GRPO"
    assert rows[0]["reward_mean"] == "-5.0"
    assert numeric_columns(rows, fieldnames) == ["reward_mean"]
