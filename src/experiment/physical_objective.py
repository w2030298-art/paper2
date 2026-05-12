"""Physical Stage-1 objective for Mainline-A tuning."""

from __future__ import annotations

import math
from typing import Any


_PPO_DEADLINE_THRESHOLD = 0.001
_COMA_DEADLINE_THRESHOLD = 0.003


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _pick(payload: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for key in keys:
        if key in payload:
            return _as_float(payload.get(key), default)
    return default


def extract_stage1_metrics(final_eval: dict[str, Any]) -> dict[str, Any]:
    """Extract physical metrics from train result or benchmark-style payloads."""
    p95_latency = _pick(
        final_eval,
        (
            "eval/e2e_latency_p95",
            "eval/e2e_latency_p95_mean",
            "final_e2e_latency_p95",
            "final_e2e_latency_p95_mean",
            "p95_latency",
        ),
    )
    mean_latency = _pick(
        final_eval,
        (
            "eval/e2e_latency_mean",
            "eval/latency_per_task_mean",
            "final_e2e_latency_mean",
            "final_latency_per_task_mean",
            "mean_latency",
        ),
    )
    energy_per_task = _pick(
        final_eval,
        (
            "eval/energy_per_task_mean",
            "final_energy_per_task_mean",
            "final_energy_per_task_mean_mean",
            "energy_per_task",
        ),
    )
    deadline_miss_rate = _pick(
        final_eval,
        (
            "eval/deadline_miss_rate",
            "final_deadline_miss_rate",
            "final_deadline_miss_rate_mean",
            "deadline_miss_rate",
        ),
    )
    constraint_violation = _pick(
        final_eval,
        (
            "eval/constraint/any_violation_mean",
            "final_constraint_violation_rate",
            "final_constraint_violation_rate_mean",
            "constraint_violation",
        ),
    )
    tail_instability = _pick(
        final_eval,
        (
            "eval/e2e_latency_std",
            "eval/latency_per_task_std",
            "tail_instability",
        ),
        default=max(0.0, p95_latency - mean_latency),
    )
    feasible = constraint_violation <= 0.0
    return {
        "p95_latency": p95_latency,
        "mean_latency": mean_latency,
        "energy_per_task": energy_per_task,
        "tail_instability": tail_instability,
        "deadline_miss_rate": deadline_miss_rate,
        "constraint_violation": constraint_violation,
        "feasible": feasible,
        "pareto_eligible": feasible,
    }


def _norm(value: float) -> float:
    """Map non-negative physical costs into a bounded objective component."""
    value = max(0.0, _as_float(value))
    return value / (1.0 + value)


def compute_j_phys(metrics: dict[str, Any], algorithm: str) -> float:
    """Compute the Stage-1 physical cost; lower is better."""
    if not bool(metrics.get("feasible", True)):
        metrics["pareto_eligible"] = False
        return float("inf")
    constraint_violation = _as_float(metrics.get("constraint_violation"), 0.0)
    if constraint_violation > 0.0:
        metrics["pareto_eligible"] = False
        return float("inf")

    threshold = _COMA_DEADLINE_THRESHOLD if str(algorithm).upper() == "COMA" else _PPO_DEADLINE_THRESHOLD
    deadline_miss_rate = _as_float(metrics.get("deadline_miss_rate"), 0.0)
    if deadline_miss_rate > threshold:
        metrics["pareto_eligible"] = False
        return float("inf")

    metrics["pareto_eligible"] = True
    return float(
        0.40 * _norm(_as_float(metrics.get("p95_latency")))
        + 0.30 * _norm(_as_float(metrics.get("mean_latency")))
        + 0.20 * _norm(_as_float(metrics.get("energy_per_task")))
        + 0.10 * _norm(_as_float(metrics.get("tail_instability")))
    )
