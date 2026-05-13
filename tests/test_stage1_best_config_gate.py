"""Regression tests for Stage-1 best-config artifact gating."""

from pathlib import Path

import yaml

from scripts.tune_mainline_a_stage1 import _write_best_config
from src.experiment.pareto import is_pareto_efficient


def _write_trial_config(path: Path, marker: str) -> None:
    """Write a minimal trial config marker for best-config copy assertions."""
    path.write_text(
        yaml.safe_dump({"algorithm": {"name": marker}, "marker": marker}, sort_keys=False),
        encoding="utf-8",
    )


def test_all_failed_trials_write_no_success_marker(tmp_path: Path) -> None:
    """All-failed trials must not copy trial 0000 into the best-config artifact."""
    trial_config = tmp_path / "trial_0000.yaml"
    _write_trial_config(trial_config, "failed-trial-zero")
    best_config = tmp_path / "ppo_best_config.yaml"

    _write_best_config(
        best_config,
        [
            {
                "trial_id": "0000",
                "status": "failed",
                "j_phys": "inf",
                "config_path": str(trial_config),
            }
        ],
    )

    payload = yaml.safe_load(best_config.read_text(encoding="utf-8"))
    assert payload == {
        "status": "no_successful_trial",
        "reason": "all_trials_failed_or_infeasible",
    }


def test_successful_trial_with_lowest_j_phys_is_selected(tmp_path: Path) -> None:
    """Best-config artifact should copy the successful trial with lowest j_phys."""
    failed_config = tmp_path / "trial_0000.yaml"
    worse_config = tmp_path / "trial_0001.yaml"
    best_source = tmp_path / "trial_0002.yaml"
    _write_trial_config(failed_config, "failed-trial")
    _write_trial_config(worse_config, "worse-success")
    _write_trial_config(best_source, "best-success")
    best_config = tmp_path / "coma_best_config.yaml"

    _write_best_config(
        best_config,
        [
            {
                "trial_id": "0000",
                "status": "failed",
                "j_phys": "inf",
                "config_path": str(failed_config),
            },
            {
                "trial_id": "0001",
                "status": "success",
                "j_phys": "4.2",
                "config_path": str(worse_config),
            },
            {
                "trial_id": "0002",
                "status": "success",
                "j_phys": "1.5",
                "config_path": str(best_source),
            },
        ],
    )

    payload = yaml.safe_load(best_config.read_text(encoding="utf-8"))
    assert payload["marker"] == "best-success"


def test_failed_rows_are_never_pareto_eligible() -> None:
    """Pareto filtering must exclude failed rows even when objective values look strong."""
    rows = [
        {
            "trial_id": "failed-fast",
            "status": "failed",
            "feasible": True,
            "pareto_eligible": True,
            "constraint_violation": 0.0,
            "p95_latency": 0.1,
            "mean_latency": 0.1,
            "energy_per_task": 0.1,
            "tail_instability": 0.1,
        },
        {
            "trial_id": "successful",
            "status": "success",
            "feasible": True,
            "pareto_eligible": True,
            "constraint_violation": 0.0,
            "p95_latency": 1.0,
            "mean_latency": 1.0,
            "energy_per_task": 1.0,
            "tail_instability": 1.0,
        },
    ]

    pareto_rows = is_pareto_efficient(rows)

    assert [row["trial_id"] for row in pareto_rows] == ["successful"]
