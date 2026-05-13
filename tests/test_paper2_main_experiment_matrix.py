"""Contract tests for the paper2 v4.8 main experiment matrix."""

from __future__ import annotations

from pathlib import Path

import yaml


MATRIX_PATH = Path("configs/paper2_main_experiment_matrix.yaml")

EXPECTED_GROUPS = {
    "proposed": ["CL-PPO", "GAM-COMA"],
    "stage1_best_defaults": ["PPO", "COMA"],
    "strong_single_agent_baselines": ["SimPO", "TRPO", "GRPO", "DDPG", "TD3", "DDQN"],
    "strong_multi_agent_baselines": ["MAPPO", "MADDPG", "MATD3"],
    "diagnostic_only_or_frozen": ["IQL", "VDN", "QMIX", "IPPO", "SAC", "A3C"],
}

MAXIMIZE_METRICS = {
    "reward_mean",
    "social_welfare_mean",
    "social_welfare_per_step_mean",
    "throughput_tasks_per_step",
    "agent_reward_jain_mean",
    "comm_score",
    "provider_revenue_proxy_mean",
    "efx_satisfaction_rate",
}

MINIMIZE_METRICS = {
    "e2e_latency_mean",
    "e2e_latency_p95",
    "deadline_miss_rate",
    "energy_total_mean",
    "energy_per_task_mean",
    "constraint_violation_rate",
    "constraint_penalty_mean",
    "queue_wait_mean",
    "price_mean",
}


def _matrix() -> dict:
    """Load the main experiment matrix."""
    return yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))


def test_main_experiment_matrix_declares_required_groups() -> None:
    """The matrix should preserve the reviewed v4.8 algorithm groups exactly."""
    matrix = _matrix()

    assert matrix["schema_version"] == "paper2-main-experiment-matrix/v1"
    assert matrix["algorithm_groups"] == EXPECTED_GROUPS
    assert matrix["default_execution_groups"] == [
        "proposed",
        "stage1_best_defaults",
        "strong_single_agent_baselines",
        "strong_multi_agent_baselines",
    ]


def test_main_experiment_matrix_declares_evidence_levels() -> None:
    """L1/L2/L3 levels should separate screening from final claim evidence."""
    levels = _matrix()["evidence_levels"]

    assert levels["L1_screening"] == {
        "steps": 50000,
        "seeds": [42],
        "purpose": "screening_only",
        "final_claim": False,
    }
    assert levels["L2_candidate_validation"] == {
        "steps": 100000,
        "seeds": [42, 43, 44],
        "purpose": "candidate_validation",
        "final_claim": False,
    }
    assert levels["L3_formal_verification"] == {
        "steps": 200000,
        "seeds": [42, 43, 44, 45, 46],
        "purpose": "formal_verification",
        "final_claim": True,
    }
    assert _matrix()["final_claim_level"] == "L3_formal_verification"


def test_main_experiment_matrix_declares_metric_directions_and_gates() -> None:
    """Metric direction and formal gates should be machine-checkable."""
    matrix = _matrix()
    metrics = matrix["metrics"]

    assert {name for name, spec in metrics.items() if spec["direction"] == "maximize"} == MAXIMIZE_METRICS
    assert {name for name, spec in metrics.items() if spec["direction"] == "minimize"} == MINIMIZE_METRICS
    assert {spec["role"] for spec in metrics.values()} <= {"primary", "secondary", "safety"}
    assert matrix["gates"] == {
        "reward_best_tail_gap_max": 0.10,
        "metric_regression_max": 0.10,
        "catastrophic_outlier_count": 0,
        "failed_seed_count": 0,
    }


def test_matrix_algorithms_are_registered_or_diagnostic_only() -> None:
    """Executable algorithms should exist in benchmark; diagnostics may remain frozen."""
    from scripts.benchmark import ALGORITHM_CLASSES

    matrix = _matrix()
    executable_groups = [
        group
        for name, group in matrix["algorithm_groups"].items()
        if name != "diagnostic_only_or_frozen"
    ]
    executable_algorithms = {algorithm for group in executable_groups for algorithm in group}

    assert executable_algorithms <= set(ALGORITHM_CLASSES)
    assert set(matrix["algorithm_groups"]["diagnostic_only_or_frozen"]) <= set(ALGORITHM_CLASSES)
    assert matrix["diagnostic_only_default_execution"] is False
