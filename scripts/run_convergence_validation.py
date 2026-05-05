#!/usr/bin/env python3
"""Run or materialize targeted convergence validation phases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_ALGORITHMS = [
    "A3C",
    "COMA",
    "GRPO",
    "IPPO",
    "IQL",
    "MADDPG",
    "MAPPO",
    "MATD3",
    "SAC",
    "TRPO",
    "VDN",
]
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
EVENT_AUDIT_PATH = PROJECT_ROOT / "docs" / "convergence_event_audit.md"


@dataclass(frozen=True)
class ValidationPlan:
    """Concrete command plan for one validation phase."""

    phase: str
    algorithms: list[str]
    seeds: list[int]
    steps: int
    eval_episodes: int
    use_stability_overrides: bool
    output_root: Path
    run_dir: Path
    result_path: Path
    commands_path: Path
    manifest_path: Path
    command: list[str]
    mode: str | None = None
    family: str | None = None


def load_matrix(path: Path) -> dict[str, Any]:
    """Load and validate a convergence validation matrix."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Matrix must be a mapping: {path}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported matrix schema_version")
    phases = payload.get("phases")
    if not isinstance(phases, dict) or not phases:
        raise ValueError("Matrix must define phases")
    return payload


def _canonical_algorithm(name: str) -> str:
    """Return the canonical target algorithm spelling."""
    lookup = {algo.lower(): algo for algo in TARGET_ALGORITHMS}
    canonical = lookup.get(str(name).lower())
    if canonical is None:
        raise ValueError(f"Unsupported target algorithm: {name}")
    return canonical


def _resolve_algorithms(
    matrix: dict[str, Any],
    phase: str,
    requested: Sequence[str] | None,
) -> list[str]:
    """Resolve phase algorithms while preventing accidental full-17 execution."""
    phase_cfg = matrix["phases"][phase]
    if requested:
        algorithms = [_canonical_algorithm(name) for name in requested]
    elif phase_cfg.get("algorithms"):
        algorithms = [_canonical_algorithm(name) for name in phase_cfg["algorithms"]]
    else:
        algorithms = TARGET_ALGORITHMS.copy()

    deduped = list(dict.fromkeys(algorithms))
    if set(deduped) == FULL_17_ALGORITHMS or len(deduped) >= len(FULL_17_ALGORITHMS):
        raise ValueError("Refusing to run full 17 from convergence validation")
    return deduped


def _family_for_algorithms(phase_cfg: dict[str, Any], algorithms: Sequence[str]) -> str | None:
    """Return the single override family for algorithms, or reject mixed families."""
    families = phase_cfg.get("families")
    if not isinstance(families, dict):
        return None

    matches: set[str] = set()
    for algorithm in algorithms:
        for family, members in families.items():
            if algorithm in set(members or []):
                matches.add(str(family))
                break
        else:
            raise ValueError(f"No override family configured for {algorithm}")

    if len(matches) > 1:
        joined = ", ".join(sorted(matches))
        raise ValueError(f"override_50k accepts one family at a time, got: {joined}")
    return next(iter(matches), None)


def _event_audit_allows_override(path: Path = EVENT_AUDIT_PATH) -> bool:
    """Return whether override execution is allowed by the event audit."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    return "needs_code_review" not in text and "needs_escalation" not in text


def build_validation_plan(
    matrix: dict[str, Any],
    *,
    phase: str,
    algorithms: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    output_root: Path | None = None,
    timestamp: str | None = None,
    event_audit_path: Path = EVENT_AUDIT_PATH,
) -> ValidationPlan:
    """Build a concrete validation plan from the matrix."""
    if phase not in matrix["phases"]:
        raise ValueError(f"Unknown phase: {phase}")

    default_cfg = matrix.get("default", {})
    if default_cfg.get("no_full_17") is not True:
        raise ValueError("Matrix must explicitly set default.no_full_17=true")

    phase_cfg = matrix["phases"][phase]
    selected_algorithms = _resolve_algorithms(matrix, phase, algorithms)
    selected_seeds = [int(seed) for seed in (seeds or default_cfg.get("seeds", []))]
    if not selected_seeds:
        raise ValueError("At least one seed is required")

    use_overrides = bool(phase_cfg.get("use_stability_overrides", False))
    family = None
    mode = phase_cfg.get("mode")
    if phase == "override_50k":
        if mode != "single_family_only":
            raise ValueError("override_50k must use mode=single_family_only")
        family = _family_for_algorithms(phase_cfg, selected_algorithms)
        if not _event_audit_allows_override(event_audit_path):
            raise ValueError("override_50k requires docs/convergence_event_audit.md without needs_code_review")

    steps = int(phase_cfg.get("steps", 0))
    if steps <= 0:
        raise ValueError(f"Phase {phase} must define a positive steps value")

    root = output_root or Path(default_cfg.get("output_root", "experiments/convergence_validation"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    run_stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / phase / run_stamp
    result_path = PROJECT_ROOT / "results" / f"convergence_validation_{phase}.json"
    command = [
        sys.executable,
        "scripts/benchmark.py",
        "--algorithms",
        *selected_algorithms,
        "--timesteps",
        str(steps),
        "--seeds",
        *[str(seed) for seed in selected_seeds],
        "--episodes",
        str(int(default_cfg.get("eval_episodes", 10))),
        "--output",
        str(result_path),
    ]
    return ValidationPlan(
        phase=phase,
        algorithms=selected_algorithms,
        seeds=selected_seeds,
        steps=steps,
        eval_episodes=int(default_cfg.get("eval_episodes", 10)),
        use_stability_overrides=use_overrides,
        output_root=root,
        run_dir=run_dir,
        result_path=result_path,
        commands_path=run_dir / "commands.txt",
        manifest_path=run_dir / "manifest.json",
        command=command,
        mode=str(mode) if mode else None,
        family=family,
    )


def plan_to_manifest(plan: ValidationPlan, *, dry_run: bool, no_submit: bool) -> dict[str, Any]:
    """Serialize a validation plan into the persisted manifest schema."""
    return {
        "schema_version": 1,
        "phase": plan.phase,
        "dry_run": dry_run,
        "no_submit": no_submit,
        "algorithms": plan.algorithms,
        "seeds": plan.seeds,
        "steps": plan.steps,
        "eval_episodes": plan.eval_episodes,
        "use_stability_overrides": plan.use_stability_overrides,
        "mode": plan.mode,
        "family": plan.family,
        "result_path": str(plan.result_path.relative_to(PROJECT_ROOT)),
        "run_dir": str(plan.run_dir.relative_to(PROJECT_ROOT)),
        "command": plan.command,
    }


def write_plan_artifacts(plan: ValidationPlan, *, dry_run: bool, no_submit: bool) -> None:
    """Write manifest.json and commands.txt for the validation plan."""
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = plan_to_manifest(plan, dry_run=dry_run, no_submit=no_submit)
    plan.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rendered_command = " ".join(plan.command)
    plan.commands_path.write_text(rendered_command + "\n", encoding="utf-8")


def print_plan(plan: ValidationPlan, *, dry_run: bool, no_submit: bool) -> None:
    """Print a concise human-readable validation plan."""
    print(f"phase: {plan.phase}")
    print(f"algorithms: {' '.join(plan.algorithms)}")
    print(f"seeds: {' '.join(str(seed) for seed in plan.seeds)}")
    print(f"steps: {plan.steps}")
    print(f"use_stability_overrides: {str(plan.use_stability_overrides).lower()}")
    if plan.family:
        print(f"family: {plan.family}")
    print(f"manifest: {plan.manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"commands: {plan.commands_path.relative_to(PROJECT_ROOT)}")
    print(f"result_path: {plan.result_path.relative_to(PROJECT_ROOT)}")
    if dry_run:
        print("mode: dry-run; training not started")
    elif no_submit:
        print("mode: no-submit; command manifest written only")
    else:
        print("mode: execute")


def run_plan(plan: ValidationPlan, *, dry_run: bool, no_submit: bool) -> int:
    """Write artifacts and optionally execute the benchmark command."""
    write_plan_artifacts(plan, dry_run=dry_run, no_submit=no_submit)
    print_plan(plan, dry_run=dry_run, no_submit=no_submit)
    if dry_run or no_submit:
        return 0
    if plan.use_stability_overrides:
        raise RuntimeError(
            "Benchmark execution does not yet support applying stability_overrides.yaml; "
            "use --no-submit for command audit until a supported override adapter exists."
        )
    completed = subprocess.run(plan.command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run targeted convergence validation phases.")
    parser.add_argument("--matrix", default="configs/convergence_validation_matrix.yaml")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["baseline_50k", "override_50k", "expansion_100k", "expansion_200k"],
    )
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the convergence validation CLI."""
    args = _parse_args()
    matrix = load_matrix(PROJECT_ROOT / args.matrix)
    plan = build_validation_plan(
        matrix,
        phase=args.phase,
        algorithms=args.algorithms,
        seeds=args.seeds,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    raise SystemExit(run_plan(plan, dry_run=bool(args.dry_run), no_submit=bool(args.no_submit)))


if __name__ == "__main__":
    main()
