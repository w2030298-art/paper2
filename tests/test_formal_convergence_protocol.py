"""Tests for the formal convergence verification protocol helpers."""

from scripts.analyze_convergence_failures import (
    classify_evidence_level,
    classify_formal_convergence,
    compute_tail_metrics,
    detect_metric_regression,
)


def test_compute_tail_metrics_exposes_protocol_fields() -> None:
    """Tail metrics should expose the fields used by L1/L2/L3 gates."""
    metrics = compute_tail_metrics([-10.0, -7.0, -5.0, -4.8, -4.7, -4.7], True)

    assert metrics["status"] in {"converged_good", "improving", "bad_plateau"}
    assert "tail_relative_change" in metrics
    assert "tail_slope" in metrics
    assert "best_tail_gap" in metrics


def test_classify_evidence_level_from_steps_and_seeds() -> None:
    """Evidence level should depend on both training length and seed count."""
    assert classify_evidence_level({"steps": 50_000, "seeds": [42]}) == "L1"
    assert classify_evidence_level({"steps": 100_000, "seeds": [42, 43, 44]}) == "L2"
    assert classify_evidence_level({"steps": 200_000, "seeds": [42, 43, 44, 45, 46]}) == "L3"


def test_l1_never_returns_verified_protocol_status() -> None:
    """L1 can only produce a screening decision, not the L3 verdict."""
    decision = classify_formal_convergence(
        {
            "algorithm": "COMA",
            "evidence_level": "L1",
            "failed_seed_count": 0,
            "extreme_outlier_count": 0,
            "metrics": {
                "reward": {
                    "status": "converged_good",
                    "best_tail_gap": 0.05,
                    "tail_relative_change": 0.01,
                }
            },
        }
    )

    assert decision["decision"] == "l1_candidate"


def test_l3_can_return_verified_protocol_status() -> None:
    """Only an L3 record that clears protocol gates can receive the final verdict."""
    decision = classify_formal_convergence(
        {
            "algorithm": "TRPO",
            "evidence_level": "L3",
            "failed_seed_count": 0,
            "extreme_outlier_count": 0,
            "metrics": {
                "reward": {
                    "status": "converged_good",
                    "best_tail_gap": 0.05,
                    "tail_relative_change": 0.01,
                }
            },
        }
    )

    assert decision["decision"] == "verified_converged_under_protocol"


def test_detect_metric_regression_respects_metric_direction() -> None:
    """Reward/comm are higher-is-better while latency/energy/deadline miss are lower."""
    regression = detect_metric_regression(
        {
            "final_reward_mean_mean": 9.0,
            "final_latency_per_task_mean_mean": 1.2,
            "final_energy_per_task_mean_mean": 1.0,
            "final_comm_score_mean": 4.0,
            "final_deadline_miss_rate_mean": 0.03,
        },
        {
            "final_reward_mean_mean": 10.0,
            "final_latency_per_task_mean_mean": 1.0,
            "final_energy_per_task_mean_mean": 1.0,
            "final_comm_score_mean": 5.0,
            "final_deadline_miss_rate_mean": 0.02,
        },
    )

    assert regression["reward"]["regression_ratio"] == 0.1
    assert regression["latency"]["regression_ratio"] > 0.1
    assert regression["comm_score"]["regression_ratio"] == 0.2
