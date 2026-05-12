"""Tests for Stage-1 physical objective and Pareto filtering."""

from src.experiment.pareto import is_pareto_efficient
from src.experiment.physical_objective import compute_j_phys, extract_stage1_metrics


def test_extract_stage1_metrics_prefers_physical_eval_keys() -> None:
    metrics = extract_stage1_metrics(
        {
            "eval/e2e_latency_p95": 2.0,
            "eval/e2e_latency_mean": 1.2,
            "eval/energy_per_task_mean": 0.4,
            "eval/deadline_miss_rate": 0.0005,
            "eval/constraint/any_violation_mean": 0.0,
        }
    )

    assert metrics["p95_latency"] == 2.0
    assert metrics["mean_latency"] == 1.2
    assert metrics["energy_per_task"] == 0.4
    assert metrics["deadline_miss_rate"] == 0.0005
    assert metrics["feasible"] is True
    assert metrics["pareto_eligible"] is True


def test_compute_j_phys_marks_deadline_and_constraint_failures_ineligible() -> None:
    ppo_bad_deadline = extract_stage1_metrics(
        {
            "eval/e2e_latency_p95": 2.0,
            "eval/e2e_latency_mean": 1.0,
            "eval/energy_per_task_mean": 0.5,
            "eval/deadline_miss_rate": 0.002,
            "eval/constraint/any_violation_mean": 0.0,
        }
    )
    constraint_bad = extract_stage1_metrics(
        {
            "eval/e2e_latency_p95": 2.0,
            "eval/e2e_latency_mean": 1.0,
            "eval/energy_per_task_mean": 0.5,
            "eval/deadline_miss_rate": 0.0,
            "eval/constraint/any_violation_mean": 1.0,
        }
    )

    assert compute_j_phys(ppo_bad_deadline, "PPO") == float("inf")
    assert compute_j_phys(constraint_bad, "COMA") == float("inf")


def test_pareto_filter_returns_only_feasible_non_dominated_rows() -> None:
    rows = [
        {
            "trial_id": "0000",
            "feasible": True,
            "pareto_eligible": True,
            "constraint_violation": 0.0,
            "p95_latency": 2.0,
            "mean_latency": 1.0,
            "energy_per_task": 0.5,
            "tail_instability": 1.0,
        },
        {
            "trial_id": "0001",
            "feasible": True,
            "pareto_eligible": True,
            "constraint_violation": 0.0,
            "p95_latency": 3.0,
            "mean_latency": 1.5,
            "energy_per_task": 0.7,
            "tail_instability": 1.5,
        },
        {
            "trial_id": "0002",
            "feasible": False,
            "pareto_eligible": False,
            "constraint_violation": 1.0,
            "p95_latency": 1.0,
            "mean_latency": 0.5,
            "energy_per_task": 0.2,
            "tail_instability": 0.5,
        },
    ]

    pareto = is_pareto_efficient(rows)
    assert [row["trial_id"] for row in pareto] == ["0000"]
