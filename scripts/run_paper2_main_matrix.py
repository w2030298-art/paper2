"""Run or dry-run the paper2 v4.8 main experiment matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


MANIFEST_SCHEMA_VERSION = "paper2-main-matrix-manifest/v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _split_csv(value: str | None) -> list[str] | None:
    """Split a comma-delimited CLI value into trimmed tokens."""
    if value is None:
        return None
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    return tokens or None


def _sanitize(value: str) -> str:
    """Return a filesystem-safe identifier segment."""
    return value.lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def _default_algorithms(matrix: dict[str, Any]) -> list[str]:
    """Resolve default executable algorithms from matrix execution groups."""
    algorithms: list[str] = []
    groups = matrix["algorithm_groups"]
    for group_name in matrix["default_execution_groups"]:
        algorithms.extend(groups[group_name])
    return algorithms


def _resolve_selection(
    matrix: dict[str, Any],
    scenario_matrix: dict[str, Any],
    selected_scenarios: list[str] | None,
    selected_algorithms: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve and validate scenario and algorithm selections."""
    scenario_ids = selected_scenarios or list(scenario_matrix["scenarios"])
    unknown_scenarios = [item for item in scenario_ids if item not in scenario_matrix["scenarios"]]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario(s): {', '.join(unknown_scenarios)}")

    algorithms = selected_algorithms or _default_algorithms(matrix)
    known_algorithms = {
        algorithm
        for group in matrix["algorithm_groups"].values()
        for algorithm in group
    }
    unknown_algorithms = [item for item in algorithms if item not in known_algorithms]
    if unknown_algorithms:
        raise ValueError(f"Unknown algorithm(s): {', '.join(unknown_algorithms)}")
    return scenario_ids, algorithms


def _benchmark_command(
    algorithm: str,
    seed: int,
    steps: int,
    output_path: Path,
    device: str,
    scenario_spec: dict[str, Any],
) -> list[str]:
    """Build the benchmark command for a single run."""
    profile = scenario_spec["profile"].get("environment_profile", "mainline-a")
    command = [
        sys.executable,
        "scripts/benchmark.py",
        "--algorithms",
        algorithm,
        "--environment-profile",
        profile,
        "--timesteps",
        str(steps),
        "--seeds",
        str(seed),
        "--device",
        device,
        "--output",
        str(output_path),
        "--no-latest-alias",
    ]

    profile_values = scenario_spec.get("profile", {})
    if profile_values.get("channel_model"):
        command.extend(["--channel-model", str(profile_values["channel_model"])])
    if profile_values.get("queue_model"):
        command.extend(["--queue-model", str(profile_values["queue_model"])])
    if profile_values.get("mobility_intensity"):
        command.extend(["--mobility-intensity", str(profile_values["mobility_intensity"])])
    return command


def build_manifest(
    matrix: dict[str, Any],
    scenario_matrix: dict[str, Any],
    level: str,
    scenario_ids: list[str],
    algorithms: list[str],
    output: Path,
    device: str,
    dry_run: bool,
    resume: bool,
) -> dict[str, Any]:
    """Expand the experiment grid into a deterministic manifest."""
    levels = matrix["evidence_levels"]
    if level not in levels:
        raise ValueError(f"Unknown evidence level: {level}")
    level_spec = levels[level]
    steps = int(level_spec["steps"])
    seeds = [int(seed) for seed in level_spec["seeds"]]
    claim_level = str(level_spec["purpose"])
    runs: list[dict[str, Any]] = []
    output_root = output.parent

    for scenario_id in scenario_ids:
        scenario_spec = scenario_matrix["scenarios"][scenario_id]
        allowed = set(scenario_spec["allowed_algorithms"])
        for algorithm in algorithms:
            if algorithm not in allowed:
                continue
            for seed in seeds:
                run_id = f"{level}__{scenario_id}__{algorithm}__seed{seed}"
                run_output = (
                    output_root
                    / scenario_spec["output_subdir"]
                    / _sanitize(algorithm)
                    / f"seed_{seed}.json"
                )
                command = _benchmark_command(
                    algorithm=algorithm,
                    seed=seed,
                    steps=steps,
                    output_path=run_output,
                    device=device,
                    scenario_spec=scenario_spec,
                )
                runs.append(
                    {
                        "run_id": run_id,
                        "matrix_version": matrix["schema_version"],
                        "scenario_version": scenario_matrix["schema_version"],
                        "level": level,
                        "algorithm": algorithm,
                        "scenario": scenario_id,
                        "seed": seed,
                        "steps": steps,
                        "command": command,
                        "output_path": str(run_output),
                        "claim_level": claim_level,
                    }
                )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "matrix_version": matrix["schema_version"],
        "scenario_version": scenario_matrix["schema_version"],
        "level": level,
        "claim_level": claim_level,
        "final_claim": bool(level_spec.get("final_claim", False)),
        "dry_run": dry_run,
        "resume": resume,
        "source_matrix": "configs/paper2_main_experiment_matrix.yaml",
        "source_scenario_matrix": "configs/paper2_scenario_matrix.yaml",
        "run_count": len(runs),
        "runs": runs,
    }


def _write_manifest(manifest: dict[str, Any], output: Path) -> None:
    """Write a manifest JSON file with stable formatting."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _execute_manifest(manifest: dict[str, Any], resume: bool) -> None:
    """Execute manifest benchmark commands in order."""
    for run in manifest["runs"]:
        output_path = Path(run["output_path"])
        if resume and output_path.exists():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(run["command"], check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run paper2 main experiment matrix")
    parser.add_argument("--matrix", type=Path, default=Path("configs/paper2_main_experiment_matrix.yaml"))
    parser.add_argument(
        "--scenario-matrix",
        type=Path,
        default=Path("configs/paper2_scenario_matrix.yaml"),
    )
    parser.add_argument(
        "--level",
        choices=["L1_screening", "L2_candidate_validation", "L3_formal_verification"],
        required=True,
    )
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--algorithms", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/paper2_main_matrix/manifest.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the CLI."""
    args = parse_args()
    matrix = _load_yaml(args.matrix)
    scenario_matrix = _load_yaml(args.scenario_matrix)
    scenario_ids, algorithms = _resolve_selection(
        matrix=matrix,
        scenario_matrix=scenario_matrix,
        selected_scenarios=_split_csv(args.scenarios),
        selected_algorithms=_split_csv(args.algorithms),
    )
    manifest = build_manifest(
        matrix=matrix,
        scenario_matrix=scenario_matrix,
        level=args.level,
        scenario_ids=scenario_ids,
        algorithms=algorithms,
        output=args.output,
        device=args.device,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    _write_manifest(manifest, args.output)
    if args.dry_run:
        print(f"DRY RUN paper2 main matrix: {manifest['run_count']} planned runs")
        print(f"Manifest saved to: {args.output}")
        return
    _execute_manifest(manifest, resume=args.resume)
    print(f"paper2 main matrix completed: {manifest['run_count']} runs")


if __name__ == "__main__":
    main()
