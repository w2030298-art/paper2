#!/usr/bin/env python3
"""Run L2/L3 formal convergence validation without changing training semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_convergence_failures import (  # noqa: E402
    build_failure_matrix,
    build_formal_decision_report,
    load_quality_records,
    render_formal_decision_markdown,
)
from scripts.plot_results import plot_convergence_curves  # noqa: E402


L1_ASSESSMENT_PATH = PROJECT_ROOT / "results" / "l1_baseline_convergence_assessment.json"
L1_BASELINE_RESULTS_PATH = PROJECT_ROOT / "results" / "convergence_validation_baseline_50k.json"
EVENT_AUDIT_PATH = PROJECT_ROOT / "results" / "convergence_event_audit.json"
FORMAL_MATRIX_PATH = PROJECT_ROOT / "configs" / "formal_convergence_matrix.yaml"
SINGLE_VARIABLE_FIXES_PATH = PROJECT_ROOT / "configs" / "formal_single_variable_fixes.yaml"
STABILITY_OVERRIDES_PATH = PROJECT_ROOT / "configs" / "stability_overrides.yaml"
FULL_17_ALGORITHMS = {
    "GRPO",
    "PPO",
    "SAC",
    "DDQN",
    "DDPG",
    "TD3",
    "A3C",
    "TRPO",
    "SimPO",
    "MAPPO",
    "QMIX",
    "COMA",
    "IPPO",
    "VDN",
    "MADDPG",
    "IQL",
    "MATD3",
}


@dataclass(frozen=True)
class FormalRunPlan:
    """Concrete L2/L3 formal convergence run plan."""

    phase: str
    evidence_level: str
    algorithms: list[str]
    deferred_algorithms: dict[str, str]
    steps: int
    seeds: list[int]
    run_id: str
    config_hash: str
    override_id: str
    result_path: Path
    figure_dir: Path
    manifest_path: Path
    command: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> Any:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_records(payload: Any) -> list[dict[str, Any]]:
    """Return result records from supported payload shapes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "results", "benchmarks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _hash_config(paths: Sequence[Path]) -> str:
    """Create a short hash over protocol config files."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def _ensure_stability_overrides_disabled(path: Path = STABILITY_OVERRIDES_PATH) -> None:
    """Enforce that candidate stability overrides stay disabled by default."""
    payload = _load_yaml(path)
    if payload.get("enabled") is not False:
        raise ValueError("configs/stability_overrides.yaml must remain enabled: false")


def _l1_decisions(path: Path = L1_ASSESSMENT_PATH) -> dict[str, str]:
    """Load L1 decisions by algorithm."""
    if not path.exists():
        return {}
    payload = _load_json(path)
    decisions: dict[str, str] = {}
    for record in _coerce_records(payload):
        algorithm = str(record.get("algorithm", ""))
        if algorithm:
            decisions[algorithm] = str(record.get("decision", "unknown"))
    return decisions


def _event_audit_allows_l2(path: Path = EVENT_AUDIT_PATH) -> set[str]:
    """Return algorithms whose event audit may proceed to L2/L3."""
    if not path.exists():
        return set()
    payload = _load_json(path)
    allowed: set[str] = set()
    for record in _coerce_records(payload):
        if record.get("decision") == "allow_l2_l3_gate":
            allowed.add(str(record.get("algorithm")))
    return allowed


def select_l2_algorithms(
    matrix: dict[str, Any],
    *,
    requested: Sequence[str] | None = None,
    l1_decisions: Mapping[str, str] | None = None,
    event_allowed: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Select L2 algorithms from L1 and event-audit gates."""
    resolved_l1_decisions = dict(l1_decisions or {})
    resolved_event_allowed = set(event_allowed or set())
    target_groups = matrix.get("target_algorithms", {})
    selected: list[str] = []
    deferred: dict[str, str] = {}

    for algorithm in target_groups.get("l1_candidates", []):
        if resolved_l1_decisions.get(algorithm) == "l1_candidate":
            selected.append(str(algorithm))
        else:
            deferred[str(algorithm)] = "not_l1_candidate"

    for algorithm in target_groups.get("catastrophic_or_unstable", []):
        if str(algorithm) in resolved_event_allowed:
            selected.append(str(algorithm))
        else:
            deferred[str(algorithm)] = "event_audit_not_allowing_l2"

    for algorithm in target_groups.get("uncertain_or_plateau", []):
        deferred[str(algorithm)] = "requires_single_variable_fix_before_l2"

    selected = list(dict.fromkeys(selected))
    if requested:
        requested_set = {str(item) for item in requested}
        selected = [algorithm for algorithm in selected if algorithm in requested_set]
        for algorithm in requested_set - set(selected):
            deferred[algorithm] = deferred.get(algorithm, "not_allowed_by_l2_gate")
    if set(selected) == FULL_17_ALGORITHMS or len(selected) >= len(FULL_17_ALGORITHMS):
        raise ValueError("Refusing to run full 17 from formal convergence protocol")
    return selected, deferred


def select_l3_algorithms(l2_report_path: Path, requested: Sequence[str] | None = None) -> list[str]:
    """Select L3 algorithms from L2 candidate decisions."""
    if not l2_report_path.exists():
        return []
    payload = _load_json(l2_report_path)
    selected = [
        str(record.get("algorithm"))
        for record in _coerce_records(payload)
        if record.get("decision") == "candidate_converged_under_protocol"
    ]
    selected = [algorithm for algorithm in selected if algorithm and algorithm != "None"]
    if requested:
        requested_set = {str(item) for item in requested}
        selected = [algorithm for algorithm in selected if algorithm in requested_set]
    return list(dict.fromkeys(selected))


def build_formal_run_plan(
    *,
    phase: str,
    run_id: str | None = None,
    algorithms: Sequence[str] | None = None,
    l1_decisions: Mapping[str, str] | None = None,
    event_allowed: set[str] | None = None,
) -> FormalRunPlan:
    """Build a formal L2/L3 benchmark plan."""
    _ensure_stability_overrides_disabled()
    matrix = _load_yaml(FORMAL_MATRIX_PATH)
    config_hash = _hash_config(
        [FORMAL_MATRIX_PATH, SINGLE_VARIABLE_FIXES_PATH, STABILITY_OVERRIDES_PATH]
    )
    evidence_level = phase.upper()
    if evidence_level not in {"L2", "L3"}:
        raise ValueError("phase must be L2 or L3")
    level_cfg = matrix["evidence_levels"][evidence_level]
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if evidence_level == "L2":
        selected, deferred = select_l2_algorithms(
            matrix,
            requested=algorithms,
            l1_decisions=l1_decisions if l1_decisions is not None else _l1_decisions(),
            event_allowed=event_allowed if event_allowed is not None else _event_audit_allows_l2(),
        )
        result_path = PROJECT_ROOT / "results" / "l2_candidate_convergence_results.json"
        figure_dir = PROJECT_ROOT / "figures" / "l2_candidate_convergence"
        manifest_path = PROJECT_ROOT / "experiments" / "formal_convergence" / "l2" / resolved_run_id / "manifest.json"
    else:
        selected = select_l3_algorithms(
            PROJECT_ROOT / "results" / "l2_candidate_convergence_report.json",
            requested=algorithms,
        )
        deferred = {}
        result_path = PROJECT_ROOT / "results" / "l3_verified_convergence_results.json"
        figure_dir = PROJECT_ROOT / "figures" / "l3_verified_convergence"
        manifest_path = PROJECT_ROOT / "experiments" / "formal_convergence" / "l3" / resolved_run_id / "manifest.json"

    command = [
        sys.executable,
        "scripts/benchmark.py",
        "--algorithms",
        *selected,
        "--timesteps",
        str(int(level_cfg["steps"])),
        "--seeds",
        *[str(seed) for seed in level_cfg["seeds"]],
        "--output",
        str(result_path),
    ]
    return FormalRunPlan(
        phase=evidence_level.lower(),
        evidence_level=evidence_level,
        algorithms=selected,
        deferred_algorithms=deferred,
        steps=int(level_cfg["steps"]),
        seeds=[int(seed) for seed in level_cfg["seeds"]],
        run_id=resolved_run_id,
        config_hash=config_hash,
        override_id="none",
        result_path=result_path,
        figure_dir=figure_dir,
        manifest_path=manifest_path,
        command=command,
    )


def plan_to_manifest(plan: FormalRunPlan, *, dry_run: bool, no_submit: bool) -> dict[str, Any]:
    """Serialize the formal run plan."""
    return {
        "schema_version": 1,
        "phase": plan.phase,
        "evidence_level": plan.evidence_level,
        "run_id": plan.run_id,
        "config_hash": plan.config_hash,
        "override_id": plan.override_id,
        "algorithms": plan.algorithms,
        "deferred_algorithms": plan.deferred_algorithms,
        "steps": plan.steps,
        "seeds": plan.seeds,
        "result_path": str(plan.result_path.relative_to(PROJECT_ROOT)),
        "figure_dir": str(plan.figure_dir.relative_to(PROJECT_ROOT)),
        "command": plan.command,
        "dry_run": dry_run,
        "no_submit": no_submit,
    }


def write_manifest(plan: FormalRunPlan, *, dry_run: bool, no_submit: bool) -> None:
    """Write a manifest and command file for the formal run."""
    plan.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = plan_to_manifest(plan, dry_run=dry_run, no_submit=no_submit)
    plan.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    commands_path = plan.manifest_path.with_name("commands.txt")
    commands_path.write_text(" ".join(plan.command) + "\n", encoding="utf-8")


def _load_baseline_records() -> list[dict[str, Any]]:
    """Load L1 baseline benchmark records."""
    if not L1_BASELINE_RESULTS_PATH.exists():
        return []
    return _coerce_records(_load_json(L1_BASELINE_RESULTS_PATH))


def _write_not_run_report(plan: FormalRunPlan, reason: str) -> None:
    """Write an empty formal report when a phase has no runnable algorithms."""
    output_json = (
        PROJECT_ROOT / "results" / "l2_candidate_convergence_report.json"
        if plan.evidence_level == "L2"
        else PROJECT_ROOT / "results" / "l3_verified_convergence_report.json"
    )
    output_md = (
        PROJECT_ROOT / "docs" / "l2_candidate_convergence_report.md"
        if plan.evidence_level == "L2"
        else PROJECT_ROOT / "docs" / "l3_verified_convergence_report.md"
    )
    report = {
        "schema_version": 1,
        "evidence_level": plan.evidence_level,
        "records": [],
        "decision_counts": {},
        "protocol_scope": "engineering_validation",
        "status": "not_run",
        "reason": reason,
        "deferred_algorithms": plan.deferred_algorithms,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_formal_decision_markdown(report), encoding="utf-8")


def finalize_phase(plan: FormalRunPlan) -> dict[str, Any]:
    """Generate plots and formal reports after benchmark completion."""
    results = _coerce_records(_load_json(plan.result_path))
    plot_convergence_curves(
        results,
        plan.figure_dir,
        fmt="png",
        evidence_level=plan.evidence_level,
        run_id=plan.run_id,
        seed_set=plan.seeds,
        config_hash=plan.config_hash,
        override_id=plan.override_id,
    )
    quality_records = load_quality_records(plan.figure_dir / "convergence_quality_report.json")
    matrix = build_failure_matrix(results, quality_records=quality_records, algorithms=set(plan.algorithms))
    matrix["source_inputs"] = [str(plan.result_path.relative_to(PROJECT_ROOT))]
    matrix["quality_report"] = str((plan.figure_dir / "convergence_quality_report.json").relative_to(PROJECT_ROOT))
    formal_report = build_formal_decision_report(
        matrix,
        evidence_level=plan.evidence_level,
        baseline_records=_load_baseline_records(),
        protocol=_load_yaml(FORMAL_MATRIX_PATH),
    )
    formal_report["run_id"] = plan.run_id
    formal_report["config_hash"] = plan.config_hash
    formal_report["seed_set"] = plan.seeds
    formal_report["override_id"] = plan.override_id
    formal_report["deferred_algorithms"] = plan.deferred_algorithms

    if plan.evidence_level == "L2":
        output_json = PROJECT_ROOT / "results" / "l2_candidate_convergence_report.json"
        output_md = PROJECT_ROOT / "docs" / "l2_candidate_convergence_report.md"
    else:
        output_json = PROJECT_ROOT / "results" / "l3_verified_convergence_report.json"
        output_md = PROJECT_ROOT / "docs" / "l3_verified_convergence_report.md"
    output_json.write_text(json.dumps(formal_report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_formal_decision_markdown(formal_report), encoding="utf-8")
    return formal_report


def run_formal_plan(plan: FormalRunPlan, *, dry_run: bool, no_submit: bool, auto_l3: bool) -> int:
    """Run or materialize a formal convergence plan."""
    write_manifest(plan, dry_run=dry_run, no_submit=no_submit)
    print(f"phase: {plan.phase}")
    print(f"run_id: {plan.run_id}")
    print(f"algorithms: {' '.join(plan.algorithms) if plan.algorithms else 'NONE'}")
    print(f"deferred_algorithms: {plan.deferred_algorithms}")
    print(f"steps: {plan.steps}")
    print(f"seeds: {' '.join(str(seed) for seed in plan.seeds)}")
    print(f"manifest: {plan.manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"result_path: {plan.result_path.relative_to(PROJECT_ROOT)}")
    print(f"figure_dir: {plan.figure_dir.relative_to(PROJECT_ROOT)}")
    if dry_run or no_submit:
        return 0
    if not plan.algorithms:
        _write_not_run_report(plan, "no algorithms passed the previous gate")
        return 0

    completed = subprocess.run(plan.command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    report = finalize_phase(plan)
    if auto_l3 and plan.evidence_level == "L2":
        l3_algorithms = [
            record["algorithm"]
            for record in report.get("records", [])
            if record.get("decision") == "candidate_converged_under_protocol"
        ]
        if l3_algorithms:
            l3_plan = build_formal_run_plan(phase="L3", algorithms=l3_algorithms)
            return run_formal_plan(l3_plan, dry_run=False, no_submit=False, auto_l3=False)
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run formal L2/L3 convergence validation.")
    parser.add_argument("--phase", required=True, choices=["L2", "L3", "l2", "l3"])
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--auto-l3", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the formal convergence protocol CLI."""
    args = _parse_args()
    plan = build_formal_run_plan(
        phase=str(args.phase).upper(),
        run_id=args.run_id,
        algorithms=args.algorithms,
    )
    raise SystemExit(
        run_formal_plan(
            plan,
            dry_run=bool(args.dry_run),
            no_submit=bool(args.no_submit),
            auto_l3=bool(args.auto_l3),
        )
    )


if __name__ == "__main__":
    main()
