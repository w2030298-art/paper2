#!/usr/bin/env python3
"""Dry-run capable runner for mainline-A N0/N1/N2/N3 experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mec_model.config_schema import (  # noqa: E402
    resolve_channel_model,
    resolve_queue_model,
    validate_mainline_a_system_model_config,
)
from src.analysis.small_scale_oracle import (  # noqa: E402
    compare_policy_to_oracle,
    export_oracle_gap_report,
    solve_small_scale_optimum,
)

DEFAULT_STAGE_CONFIGS = {
    "N0": PROJECT_ROOT / "configs/experiments/mainline_a_n0_smoke.yaml",
    "N1": PROJECT_ROOT / "configs/experiments/mainline_a_n1_oracle.yaml",
    "N2": PROJECT_ROOT / "configs/experiments/mainline_a_n2_ablation.yaml",
    "N3": PROJECT_ROOT / "configs/experiments/mainline_a_n3_ood.yaml",
}

ORACLE_MAX_USERS = 4
ORACLE_MAX_EDGES = 3


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
        best_edge = max(range(1, num_edges + 1), key=lambda edge: 1.0 + 0.12 * edge + 0.01 * (seed % 7))
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
        oracle = solve_small_scale_optimum(case, objective=lambda assignment, c=case: _n1_objective(c, assignment))
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


def build_stage_plan(stage: str, config: dict[str, Any], output_root: Path, results_root: Path) -> dict[str, Any]:
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
    return plan


def resolve_plans(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Resolve stage plans from CLI arguments."""
    output_root = Path(args.output_root)
    results_root = Path(args.results_root)
    stages = ["N0", "N1", "N2", "N3"] if args.stage == "all" else [args.stage]
    plans = []
    for stage in stages:
        config_path = Path(args.config) if args.config and len(stages) == 1 else DEFAULT_STAGE_CONFIGS[stage]
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

    manifest = write_manifest(plans, PROJECT_ROOT / args.output_root)
    print(f"mainline-A manifest written: {manifest}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
