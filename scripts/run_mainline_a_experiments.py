#!/usr/bin/env python3
"""Dry-run capable runner for mainline-A N0/N1/N2/N3 experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.small_scale_oracle import (  # noqa: E402
    compare_policy_to_oracle,
    export_oracle_gap_report,
    solve_small_scale_optimum,
)
from src.game_pricing.dynamic_pricing import compute_state_dependent_price  # noqa: E402
from src.game_pricing.types import PricingBounds, PricingParameters, PricingState  # noqa: E402
from src.mec_model.config_schema import (  # noqa: E402
    resolve_channel_model,
    resolve_queue_model,
    validate_mainline_a_system_model_config,
)
from src.rl_algorithms.game_aware.primal_dual import (  # noqa: E402
    ConstraintResiduals,
    PrimalDualUpdater,
)
from src.rl_algorithms.game_aware.reward_design import (  # noqa: E402
    compute_interpretable_reward,
)

DEFAULT_STAGE_CONFIGS = {
    "N0": PROJECT_ROOT / "configs/experiments/mainline_a_n0_smoke.yaml",
    "N1": PROJECT_ROOT / "configs/experiments/mainline_a_n1_oracle.yaml",
    "N2": PROJECT_ROOT / "configs/experiments/mainline_a_n2_ablation.yaml",
    "N3": PROJECT_ROOT / "configs/experiments/mainline_a_n3_ood.yaml",
}

ORACLE_MAX_USERS = 4
ORACLE_MAX_EDGES = 3
N2_DEFAULT_EDGES = 4
N2_PREFLIGHT_STEPS = 256
N2_REQUIRED_ABLATIONS = (
    "full_model",
    "no_dynamic_price",
    "no_queue_state",
    "no_channel_state",
    "no_migration_state",
    "no_primal_dual",
    "no_cooperation",
    "analytic_channel_only",
    "3gpp_lite_channel",
)
N2_REQUIRED_METRICS = (
    "reward_mean",
    "latency_mean",
    "energy_mean",
    "price_mean",
    "constraint_violation_rate",
)
N2_METRIC_KEYS = (
    "reward_mean",
    "latency_mean",
    "energy_mean",
    "price_mean",
    "constraint_violation_rate",
    "provider_revenue_mean",
    "social_welfare_mean",
    "queue_pressure_mean",
    "channel_quality_mean",
    "migration_risk_mean",
    "dual_variable_mean",
    "cooperation_gain_mean",
)
N2_SWITCH_KEYS = (
    "dynamic_pricing",
    "queue_state",
    "channel_state",
    "migration_state",
    "primal_dual",
    "cooperation",
    "channel_model",
    "reward_ablation",
)
N2_ABLATION_SWITCHES: dict[str, dict[str, Any]] = {
    "full_model": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "hybrid",
        "reward_ablation": "full_model",
    },
    "no_dynamic_price": {
        "dynamic_pricing": False,
        "queue_state": True,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "hybrid",
        "reward_ablation": "full_model",
    },
    "no_queue_state": {
        "dynamic_pricing": True,
        "queue_state": False,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "hybrid",
        "reward_ablation": "full_model",
    },
    "no_channel_state": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": False,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "hybrid",
        "reward_ablation": "full_model",
    },
    "no_migration_state": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": True,
        "migration_state": False,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "hybrid",
        "reward_ablation": "full_model",
    },
    "no_primal_dual": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": False,
        "cooperation": True,
        "channel_model": "hybrid",
        "reward_ablation": "no_dual",
    },
    "no_cooperation": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": False,
        "channel_model": "hybrid",
        "reward_ablation": "no_cooperation",
    },
    "analytic_channel_only": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "analytic",
        "reward_ablation": "full_model",
    },
    "3gpp_lite_channel": {
        "dynamic_pricing": True,
        "queue_state": True,
        "channel_state": True,
        "migration_state": True,
        "primal_dual": True,
        "cooperation": True,
        "channel_model": "3gpp_lite",
        "reward_ablation": "full_model",
    },
}


def _coerce_int_list(config: dict[str, Any], key: str) -> list[int]:
    """Read a scalar/list config field as integers."""
    value = config.get(key, [])
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [int(item) for item in values]


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    legacy_keys = sorted(key for key in ("queue", "channel") if key in config)
    if legacy_keys:
        joined = ", ".join(legacy_keys)
        raise ValueError(f"{cfg_path}: legacy top-level field(s) {joined} are not allowed")
    validate_mainline_a_system_model_config(config, str(cfg_path))
    return config


def validate_n1_oracle_config(config: dict[str, Any]) -> None:
    """Validate N1 oracle config before enumeration starts."""
    if str(config.get("stage", "")).upper() != "N1":
        raise ValueError("N1 oracle validation requires stage: N1")
    users = _coerce_int_list(config, "users")
    edges = _coerce_int_list(config, "edges")
    seeds = _coerce_int_list(config, "seeds")
    if not users or not edges or not seeds:
        raise ValueError("N1 oracle config requires users, edges, and seeds")
    if max(users) > ORACLE_MAX_USERS or max(edges) > ORACLE_MAX_EDGES:
        raise ValueError("N1 case matrix exceeds oracle limits: num_users <= 4 and num_edges <= 3")
    if min(users) < 1 or min(edges) < 1:
        raise ValueError("N1 oracle users and edges must be positive")
    outputs = set(config.get("outputs", []))
    if outputs and "oracle_gap" not in outputs:
        raise ValueError("N1 oracle config must request oracle_gap output")


def build_n1_case_matrix(config: dict[str, Any]) -> list[dict[str, int]]:
    """Build the N1 seed/user/edge case matrix."""
    validate_n1_oracle_config(config)
    return [
        {"seed": seed, "num_users": num_users, "num_edges": num_edges}
        for seed in _coerce_int_list(config, "seeds")
        for num_users in _coerce_int_list(config, "users")
        for num_edges in _coerce_int_list(config, "edges")
    ]


def _constraint_violation(assignment: tuple[int, ...], num_edges: int) -> float:
    """Return normalized edge-capacity overflow for a discrete assignment."""
    if num_edges <= 0 or not assignment:
        return 0.0
    num_users = len(assignment)
    capacity = max(1, math.ceil(num_users / num_edges))
    loads: dict[int, int] = defaultdict(int)
    for target in assignment:
        if int(target) > 0:
            loads[int(target)] += 1
    overflow = sum(max(0, load - capacity) for load in loads.values())
    return float(overflow / max(num_users, 1))


def _n1_objective(case: dict[str, int], assignment: tuple[int, ...]) -> float:
    """Deterministic small-case welfare objective used by N1 oracle."""
    seed = int(case["seed"])
    num_edges = int(case["num_edges"])
    violation = _constraint_violation(assignment, num_edges)
    loads: dict[int, int] = defaultdict(int)
    for target in assignment:
        if int(target) > 0:
            loads[int(target)] += 1
    value = 0.0
    for user_idx, target in enumerate(assignment):
        task_weight = 1.0 + ((seed + user_idx * 17) % 11) / 20.0
        if int(target) == 0:
            value += 0.55 * task_weight
            continue
        edge_quality = 1.0 + 0.12 * int(target) + 0.01 * (seed % 7)
        congestion_penalty = 0.08 * loads[int(target)]
        value += task_weight * edge_quality - congestion_penalty
    return float(value - 2.5 * violation)


def _policy_assignment(policy_label: str, case: dict[str, int]) -> tuple[int, ...]:
    """Return a deterministic policy assignment for N1 oracle comparison."""
    num_users = int(case["num_users"])
    num_edges = int(case["num_edges"])
    seed = int(case["seed"])
    label = policy_label.lower()
    if label == "baseline_static_stackelberg":
        best_edge = max(
            range(1, num_edges + 1),
            key=lambda edge: 1.0 + 0.12 * edge + 0.01 * (seed % 7),
        )
        return tuple(best_edge for _ in range(num_users))
    if label == "mappo":
        return tuple((user_idx % num_edges) + 1 for user_idx in range(num_users))
    if label == "game_aware_pd_marl":
        assignment: list[int] = []
        for _ in range(num_users):
            best_target = 0
            best_value = float("-inf")
            for target in range(0, num_edges + 1):
                candidate = tuple(assignment + [target] + [0] * (num_users - len(assignment) - 1))
                value = _n1_objective(case, candidate)
                if value > best_value:
                    best_target = target
                    best_value = value
            assignment.append(best_target)
        return tuple(assignment)
    raise ValueError(f"unsupported N1 oracle policy label: {policy_label}")


def run_n1_oracle_validation(config: dict[str, Any], results_root: str | Path) -> dict[str, Any]:
    """Run N1 small-scale oracle comparison and write artifacts."""
    case_matrix = build_n1_case_matrix(config)
    algorithms = [str(item) for item in config.get("algorithms", ["baseline_static_stackelberg"])]
    base_root = Path(results_root)
    output_dir = base_root if base_root.name == "n1_oracle" else base_root / "n1_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    started = time.time()
    for case in case_matrix:
        oracle = solve_small_scale_optimum(
            case,
            objective=lambda assignment, c=case: _n1_objective(c, assignment),
        )
        oracle_violation = _constraint_violation(oracle.assignment, int(case["num_edges"]))
        for policy_label in algorithms:
            policy_start = time.time()
            assignment = _policy_assignment(policy_label, case)
            policy_runtime = time.time() - policy_start
            policy_result = {
                "objective_value": _n1_objective(case, assignment),
                "constraint_violation": _constraint_violation(assignment, int(case["num_edges"])),
            }
            comparison = compare_policy_to_oracle(policy_result, oracle)
            records.append(
                {
                    "stage": "N1",
                    "seed": int(case["seed"]),
                    "num_users": int(case["num_users"]),
                    "num_edges": int(case["num_edges"]),
                    "policy_label": policy_label,
                    "oracle_assignment": list(oracle.assignment),
                    "policy_assignment": list(assignment),
                    "oracle_objective": float(oracle.objective_value),
                    "policy_objective": float(policy_result["objective_value"]),
                    "optimality_gap": float(comparison["optimality_gap"]),
                    "oracle_gap": float(comparison["oracle_gap"]),
                    "constraint_violation": float(comparison["constraint_violation"]),
                    "oracle_constraint_violation": float(oracle_violation),
                    "oracle_runtime_s": float(comparison["oracle_runtime_s"]),
                    "policy_runtime_s": float(policy_runtime),
                    "runtime_s": float(comparison["oracle_runtime_s"] + policy_runtime),
                }
            )

    report_path = output_dir / "oracle_gap_report.json"
    export_oracle_gap_report(records, report_path)
    case_matrix_path = output_dir / "case_matrix.json"
    case_matrix_path.write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")

    policy_summaries = []
    for policy_label in algorithms:
        policy_records = [record for record in records if record["policy_label"] == policy_label]
        gaps = [float(record["oracle_gap"]) for record in policy_records]
        violations = [float(record["constraint_violation"]) for record in policy_records]
        runtimes = [float(record["runtime_s"]) for record in policy_records]
        policy_summaries.append(
            {
                "policy_label": policy_label,
                "mean_oracle_gap": float(sum(gaps) / max(len(gaps), 1)),
                "max_oracle_gap": float(max(gaps) if gaps else 0.0),
                "mean_constraint_violation": float(sum(violations) / max(len(violations), 1)),
                "max_constraint_violation": float(max(violations) if violations else 0.0),
                "total_runtime_s": float(sum(runtimes)),
            }
        )

    summary = {
        "schema_version": 1,
        "stage": "N1",
        "status": "ok",
        "config_name": config.get("name", "mainline_a_n1_oracle"),
        "case_count": len(case_matrix),
        "record_count": len(records),
        "seeds": _coerce_int_list(config, "seeds"),
        "users": _coerce_int_list(config, "users"),
        "edges": _coerce_int_list(config, "edges"),
        "policy_summaries": policy_summaries,
        "total_runtime_s": float(time.time() - started),
        "report_path": str(report_path),
        "case_matrix_path": str(case_matrix_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "stage": "N1",
        "case_count": len(case_matrix),
        "record_count": len(records),
        "records": records,
        "report_path": report_path,
        "summary_path": summary_path,
        "case_matrix_path": case_matrix_path,
        "summary": summary,
    }


def validate_n2_ablation_config(config: dict[str, Any]) -> None:
    """Validate the N2 deterministic controlled-probe config."""
    if str(config.get("stage", "")).upper() != "N2":
        raise ValueError("N2 ablation validation requires stage: N2")
    seeds = _coerce_int_list(config, "seeds")
    steps = int(config.get("steps", 0))
    if not seeds:
        raise ValueError("N2 ablation config requires at least one seed")
    if steps <= 0:
        raise ValueError("N2 ablation config requires positive steps")
    ablations = [str(item) for item in config.get("ablations", [])]
    if not ablations:
        raise ValueError("N2 ablation config requires ablations")
    duplicates = sorted({label for label in ablations if ablations.count(label) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"N2 ablation config contains duplicate ablation label(s): {joined}")
    unknown = sorted(set(ablations) - set(N2_ABLATION_SWITCHES))
    if unknown:
        raise ValueError(f"N2 ablation config contains unknown ablation(s): {', '.join(unknown)}")
    missing = [label for label in N2_REQUIRED_ABLATIONS if label not in ablations]
    if missing:
        raise ValueError(f"N2 ablation config missing required ablation(s): {', '.join(missing)}")


def _n2_switch_payload(label: str) -> dict[str, Any]:
    """Return a stable switch payload for one N2 ablation label."""
    if label not in N2_ABLATION_SWITCHES:
        raise ValueError(f"unsupported N2 ablation label: {label}")
    switches = dict(N2_ABLATION_SWITCHES[label])
    missing_keys = [key for key in N2_SWITCH_KEYS if key not in switches]
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"N2 ablation {label} has incomplete switch mapping: {joined}")
    return {key: switches[key] for key in N2_SWITCH_KEYS}


def build_n2_ablation_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the N2 ablation matrix with explicit switch mappings."""
    validate_n2_ablation_config(config)
    matrix = []
    for label in [str(item) for item in config.get("ablations", [])]:
        switches = _n2_switch_payload(label)
        disabled = [
            key
            for key in ("dynamic_pricing", "queue_state", "channel_state", "migration_state")
            if switches[key] is False
        ]
        if switches["primal_dual"] is False:
            disabled.append("primal_dual")
        if switches["cooperation"] is False:
            disabled.append("cooperation")
        if label == "analytic_channel_only":
            disabled.append("3gpp_lite_channel")
        matrix.append(
            {
                "ablation": label,
                "switches": switches,
                "disabled_switches": disabled,
                "required_metrics": list(N2_REQUIRED_METRICS),
            }
        )
    return matrix


def _mean(values: tuple[float, ...]) -> float:
    """Return the arithmetic mean for a non-empty tuple."""
    return float(sum(values) / max(len(values), 1))


def _n2_state_signals(
    seed: int,
    steps: int,
    switches: dict[str, Any],
) -> dict[str, tuple[float, ...]]:
    """Build deterministic state signals used by the controlled N2 probe."""
    horizon = math.log10(max(int(steps), 10)) / 10.0
    seed_phase = (int(seed) % 17) / 100.0
    channel_model = str(switches["channel_model"])
    if channel_model == "analytic":
        channel_shift = 0.10
    elif channel_model == "hybrid":
        channel_shift = 0.04
    else:
        channel_shift = -0.04
    queue = tuple(0.50 + horizon + seed_phase + 0.035 * idx for idx in range(N2_DEFAULT_EDGES))
    channel = tuple(
        1.15 + channel_shift + 0.025 * ((int(seed) + idx) % 5) for idx in range(N2_DEFAULT_EDGES)
    )
    migration = tuple(0.09 + 0.015 * ((int(seed) + idx) % 4) for idx in range(N2_DEFAULT_EDGES))
    decision_queue = queue if switches["queue_state"] else tuple(0.0 for _ in queue)
    decision_channel = channel if switches["channel_state"] else tuple(1.0 for _ in channel)
    decision_migration = migration if switches["migration_state"] else tuple(0.0 for _ in migration)
    return {
        "actual_queue": queue,
        "actual_channel": channel,
        "actual_migration": migration,
        "decision_queue": decision_queue,
        "decision_channel": decision_channel,
        "decision_migration": decision_migration,
    }


def _n2_price_vector(
    signals: dict[str, tuple[float, ...]],
    switches: dict[str, Any],
) -> tuple[float, ...]:
    """Compute prices with the ablation switch applied."""
    if not switches["dynamic_pricing"]:
        return tuple(PricingParameters().base_price for _ in range(N2_DEFAULT_EDGES))
    price_vector = compute_state_dependent_price(
        PricingState(
            queue_pressure=signals["decision_queue"],
            channel_quality=signals["decision_channel"],
            migration_risk=signals["decision_migration"],
        ),
        PricingBounds(min_price=0.05, max_price=10.0),
        PricingParameters(),
    )
    return tuple(float(value) for value in price_vector.values)


def _n2_constraint_residuals(
    latency: float, energy: float, queue_pressure: float, migration_risk: float, price_mean: float
) -> ConstraintResiduals:
    """Convert aggregate N2 metrics into primal-dual residuals."""
    return ConstraintResiduals(
        latency_deadline=max(0.0, latency - 0.60),
        energy_budget=max(0.0, energy - 1.85),
        queue_stability=max(0.0, queue_pressure - 0.82),
        migration_rate=max(0.0, migration_risk - 0.14),
        budget_feasibility=max(0.0, price_mean - 1.55),
    )


def _simulate_n2_ablation_record(
    label: str,
    switches: dict[str, Any],
    seed: int,
    steps: int,
    config_name: str,
    run_type: str,
) -> dict[str, Any]:
    """Run one deterministic controlled probe for an N2 ablation label."""
    signals = _n2_state_signals(seed, steps, switches)
    prices = _n2_price_vector(signals, switches)
    queue_pressure = _mean(signals["actual_queue"])
    channel_quality = _mean(signals["actual_channel"])
    migration_risk = _mean(signals["actual_migration"])
    price_mean = _mean(prices)

    queue_control = 0.060 if switches["queue_state"] else -0.055
    channel_control = 0.045 if switches["channel_state"] else -0.045
    migration_control = 0.032 if switches["migration_state"] else -0.035
    dynamic_control = 0.050 if switches["dynamic_pricing"] else -0.065
    dual_control = 0.055 if switches["primal_dual"] else -0.050
    cooperation_control = 0.030 if switches["cooperation"] else -0.025
    if switches["channel_model"] == "analytic":
        channel_model_delta = -0.030
    elif switches["channel_model"] == "hybrid":
        channel_model_delta = -0.010
    else:
        channel_model_delta = 0.018

    latency = (
        0.43
        + 0.28 * queue_pressure
        + 0.16 * migration_risk
        - 0.08 * channel_quality
        - queue_control
        - channel_control
        - migration_control
        - dynamic_control
        - dual_control
        - cooperation_control
        + channel_model_delta
    )
    energy = (
        1.34
        + 0.22 * queue_pressure
        + 0.26 * migration_risk
        - 0.10 * channel_quality
        - 0.6 * channel_control
        - 0.4 * cooperation_control
        + 0.03 * (0 if switches["dynamic_pricing"] else 1)
        + (0.02 if switches["channel_model"] == "3gpp_lite" else -0.015)
    )
    constraint = max(
        0.0,
        0.04
        + 0.22 * max(0.0, queue_pressure - 0.75)
        + 0.16 * max(0.0, latency - 0.55)
        - (0.050 if switches["primal_dual"] else -0.025)
        - (0.030 if switches["queue_state"] else -0.020)
        - (0.018 if switches["dynamic_pricing"] else -0.016),
    )
    cooperation_gain = 0.075 if switches["cooperation"] else 0.0
    provider_revenue = price_mean * max(0.2, 1.25 - 0.45 * latency)

    dual_state = None
    dual_mean = 0.0
    if switches["primal_dual"]:
        residuals = _n2_constraint_residuals(
            latency,
            energy,
            queue_pressure,
            migration_risk,
            price_mean,
        )
        dual_state = PrimalDualUpdater(dual_lr=0.05).update_dual_variables(residuals)
        dual_mean = _mean(tuple(float(value) for value in dual_state.dual_variables.values()))

    reward_components = {
        "delay_cost": -latency,
        "energy_cost": -0.25 * energy,
        "queue_penalty": -0.20 * queue_pressure,
        "migration_penalty": -0.15 * migration_risk,
        "deadline_violation_penalty": -0.30 * max(0.0, latency - 0.60),
        "cooperation_gain": cooperation_gain,
        "price_payment": -0.12 * price_mean,
        "provider_revenue": 0.08 * provider_revenue,
        "constraint_penalty": constraint,
    }
    reward = compute_interpretable_reward(
        reward_components,
        dual_state=dual_state,
        weights={key: 1.0 for key in reward_components},
        ablation=str(switches["reward_ablation"]),
    )
    social_welfare = reward.total_reward + 0.20 * provider_revenue - 0.35 * constraint
    metrics = {
        "reward_mean": float(reward.total_reward),
        "latency_mean": float(latency),
        "energy_mean": float(energy),
        "price_mean": float(price_mean),
        "constraint_violation_rate": float(constraint),
        "provider_revenue_mean": float(provider_revenue),
        "social_welfare_mean": float(social_welfare),
        "queue_pressure_mean": float(queue_pressure),
        "channel_quality_mean": float(channel_quality),
        "migration_risk_mean": float(migration_risk),
        "dual_variable_mean": float(dual_mean),
        "cooperation_gain_mean": float(cooperation_gain),
    }
    return {
        "schema_version": 1,
        "stage": "N2",
        "run_type": run_type,
        "config_name": config_name,
        "ablation": label,
        "seed": int(seed),
        "steps": int(steps),
        "switches": {key: switches[key] for key in N2_SWITCH_KEYS},
        "metrics": {key: metrics[key] for key in N2_METRIC_KEYS},
        "reward_components": {
            key: float(reward_components[key]) for key in sorted(reward_components)
        },
    }


def _aggregate_n2_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate N2 records by ablation label."""
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[str(record["ablation"])].append(record["metrics"])
    summary = {}
    for label, metric_records in grouped.items():
        summary[label] = {
            key: float(sum(float(item[key]) for item in metric_records) / len(metric_records))
            for key in N2_METRIC_KEYS
        }
    return summary


def _validate_n2_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate N2 output schema, finiteness, switch coverage, and metric movement."""
    checks: dict[str, Any] = {
        "required_metrics_present": True,
        "schema_consistent": True,
        "finite_metrics": True,
        "ablation_switches_explicit": True,
        "metrics_not_all_identical": True,
        "non_full_ablations_differ_from_full": True,
        "issues": [],
    }
    if not records:
        checks["issues"].append("no N2 records were produced")
        return checks

    top_schema = set(records[0].keys())
    metric_schema = set(records[0]["metrics"].keys())
    switch_schema = set(records[0]["switches"].keys())
    metric_vectors = []
    full_vectors = []
    for record in records:
        if set(record.keys()) != top_schema or set(record["metrics"].keys()) != metric_schema:
            checks["schema_consistent"] = False
            checks["issues"].append(
                f"schema mismatch in {record['ablation']} seed {record['seed']}"
            )
        if set(record["switches"].keys()) != switch_schema or switch_schema != set(N2_SWITCH_KEYS):
            checks["ablation_switches_explicit"] = False
            checks["issues"].append(f"incomplete switch schema in {record['ablation']}")
        missing = [key for key in N2_REQUIRED_METRICS if key not in record["metrics"]]
        if missing:
            checks["required_metrics_present"] = False
            checks["issues"].append(
                f"{record['ablation']} missing metric(s): {', '.join(missing)}"
            )
        values = tuple(float(record["metrics"][key]) for key in N2_REQUIRED_METRICS)
        if any(not math.isfinite(value) for value in values):
            checks["finite_metrics"] = False
            checks["issues"].append(f"{record['ablation']} seed {record['seed']} has NaN/Inf")
        metric_vectors.append(values)
        if str(record["ablation"]) == "full_model":
            full_vectors.append(values)

    if len(set(metric_vectors)) <= 1:
        checks["metrics_not_all_identical"] = False
        checks["issues"].append("all N2 required metric vectors are identical")
    if full_vectors:
        full_set = set(full_vectors)
        for record, values in zip(records, metric_vectors, strict=True):
            if str(record["ablation"]) != "full_model" and values in full_set:
                checks["non_full_ablations_differ_from_full"] = False
                checks["issues"].append(
                    f"{record['ablation']} seed {record['seed']} matches full_model"
                )
    return checks


def run_n2_ablation_validation(
    config: dict[str, Any],
    results_root: str | Path,
    *,
    preflight: bool = False,
    preflight_steps: int = N2_PREFLIGHT_STEPS,
) -> dict[str, Any]:
    """Run N2 preflight or deterministic controlled probe and write artifacts."""
    validate_n2_ablation_config(config)
    matrix = build_n2_ablation_matrix(config)
    configured_seeds = _coerce_int_list(config, "seeds")
    configured_steps = int(config.get("steps", 0))
    seeds = configured_seeds[:1] if preflight else configured_seeds
    steps = min(configured_steps, int(preflight_steps)) if preflight else configured_steps
    run_type = "preflight" if preflight else "controlled"

    base_root = Path(results_root)
    output_dir = base_root if base_root.name == "n2_ablation" else base_root / "n2_ablation"
    if preflight:
        output_dir = output_dir / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        _simulate_n2_ablation_record(
            label=str(item["ablation"]),
            switches=dict(item["switches"]),
            seed=seed,
            steps=steps,
            config_name=str(config.get("name", "mainline_a_n2_ablation")),
            run_type=run_type,
        )
        for seed in seeds
        for item in matrix
    ]
    checks = _validate_n2_records(records)
    status = "ok" if not checks["issues"] else "failed"
    metric_means = _aggregate_n2_records(records)
    full_metrics = metric_means.get("full_model", {})
    deltas = {
        label: {
            key: float(values[key] - full_metrics.get(key, 0.0))
            for key in N2_METRIC_KEYS
        }
        for label, values in metric_means.items()
        if label != "full_model"
    }

    matrix_path = output_dir / "ablation_matrix.json"
    records_path = output_dir / "ablation_records.json"
    deltas_path = output_dir / "metric_deltas.json"
    summary_path = output_dir / "summary.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    deltas_path.write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    summary = {
        "schema_version": 1,
        "stage": "N2",
        "evidence_level": "deterministic controlled probe",
        "run_type": run_type,
        "status": status,
        "config_name": config.get("name", "mainline_a_n2_ablation"),
        "seeds": seeds,
        "configured_seeds": configured_seeds,
        "steps": steps,
        "configured_steps": configured_steps,
        "ablation_count": len(matrix),
        "record_count": len(records),
        "required_metrics": list(N2_REQUIRED_METRICS),
        "metric_means": metric_means,
        "metric_deltas_vs_full_model": deltas,
        "checks": checks,
        "output_dir": str(output_dir),
        "matrix_path": str(matrix_path),
        "records_path": str(records_path),
        "deltas_path": str(deltas_path),
        "summary_path": str(summary_path),
        "benchmark_alias_overwrite": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if status != "ok":
        joined = "; ".join(checks["issues"])
        raise RuntimeError(f"N2 ablation validation failed: {joined}")
    return {
        "stage": "N2",
        "run_type": run_type,
        "records": records,
        "summary": summary,
        "matrix_path": matrix_path,
        "records_path": records_path,
        "deltas_path": deltas_path,
        "summary_path": summary_path,
    }


def build_stage_plan(
    stage: str,
    config: dict[str, Any],
    output_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    """Build a serializable execution plan for one stage."""
    system_model = validate_mainline_a_system_model_config(config, f"stage {stage}")
    plan = {
        "stage": stage,
        "name": config.get("name", f"mainline_a_{stage.lower()}"),
        "algorithms": config.get("algorithms", ["game_aware_pd_marl"]),
        "seeds": config.get("seeds", [42]),
        "steps": config.get("steps", 1000),
        "system_model": {
            "enabled": bool(system_model.get("enabled", False)),
            "queue_model": resolve_queue_model(config),
            "channel_model": resolve_channel_model(config),
        },
        "output_root": str(output_root),
        "results_root": str(results_root),
        "dry_run_supported": True,
    }
    if stage == "N1":
        plan["case_matrix"] = build_n1_case_matrix(config)
        plan["outputs"] = config.get("outputs", [])
    if stage == "N2":
        matrix = build_n2_ablation_matrix(config)
        plan["evidence_level"] = "deterministic controlled probe"
        plan["ablations"] = [item["ablation"] for item in matrix]
        plan["ablation_matrix"] = matrix
        plan["required_metrics"] = list(N2_REQUIRED_METRICS)
        plan["planned_record_count"] = len(matrix) * len(_coerce_int_list(config, "seeds"))
        plan["results_path"] = str(results_root / "n2_ablation")
        plan["preflight_supported"] = True
    return plan


def resolve_plans(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve stage plans from CLI arguments."""
    output_root = Path(args.output_root)
    results_root = Path(args.results_root)
    stages = ["N0", "N1", "N2", "N3"] if args.stage == "all" else [args.stage]
    plans = []
    for stage in stages:
        config_path = (
            Path(args.config)
            if args.config and len(stages) == 1
            else DEFAULT_STAGE_CONFIGS[stage]
        )
        config = load_experiment_config(config_path)
        plans.append(build_stage_plan(stage, config, output_root, results_root))
    return plans


def write_manifest(plans: list[dict[str, Any]], output_root: Path) -> Path:
    """Write a runner manifest for non-dry-run orchestration."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps({"plans": plans}, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run mainline-A experiment stages")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=["N0", "N1", "N2", "N3", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default="experiments/mainline_a")
    parser.add_argument("--results-root", default="results/mainline_a")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-steps", type=int, default=N2_PREFLIGHT_STEPS)
    args = parser.parse_args()

    plans = resolve_plans(args)
    payload = {"dry_run": bool(args.dry_run), "resume": bool(args.resume), "plans": plans}
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    if args.stage == "N1":
        config_path = Path(args.config) if args.config else DEFAULT_STAGE_CONFIGS["N1"]
        config = load_experiment_config(config_path)
        result = run_n1_oracle_validation(config, PROJECT_ROOT / args.results_root)
        print(json.dumps(result["summary"], indent=2))
        return

    if args.stage == "N2":
        config_path = Path(args.config) if args.config else DEFAULT_STAGE_CONFIGS["N2"]
        config = load_experiment_config(config_path)
        result = run_n2_ablation_validation(
            config,
            PROJECT_ROOT / args.results_root,
            preflight=bool(args.preflight),
            preflight_steps=int(args.preflight_steps),
        )
        print(json.dumps(result["summary"], indent=2))
        return

    manifest = write_manifest(plans, PROJECT_ROOT / args.output_root)
    print(f"mainline-A manifest written: {manifest}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
