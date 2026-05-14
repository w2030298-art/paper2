"""Run or dry-run the paper2 v4.9 main experiment matrix."""

from __future__ import annotations

import argparse
import copy
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


def _benchmark_config_name(algorithm: str) -> str:
    """Return the algorithm config filename expected by benchmark.py."""
    return f"{algorithm.lower()}.yaml"


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
    generated_config_dir: Path,
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
        "--configs-dir",
        str(generated_config_dir),
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


def _base_algorithm_config_path(algorithm: str) -> Path:
    """Resolve the source algorithm config for generated per-run configs."""
    config_dir = Path("configs") / "algorithms"
    candidates = [
        config_dir / _benchmark_config_name(algorithm),
        config_dir / f"{_sanitize(algorithm)}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No algorithm config found for {algorithm}")


def _deep_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set a nested mapping value, creating intermediate mappings as needed."""
    cursor = payload
    for key in path[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _apply_ablation_config(config: dict[str, Any], algorithm: str, ablation: str) -> dict[str, Any]:
    """Apply the pre-registered ablation switches to an algorithm config."""
    patched = copy.deepcopy(config)
    if algorithm == "CL-PPO":
        if ablation == "no_constraint_signal":
            _deep_set(patched, ("constraints", "enabled"), False)
        elif ablation == "no_risk_critic":
            _deep_set(patched, ("risk", "enabled"), False)
        elif ablation == "no_safety_layer":
            _deep_set(patched, ("safety_layer", "enabled"), False)
    elif algorithm == "GAM-COMA":
        if ablation == "full":
            _deep_set(patched, ("game_theory", "use_shapley_credit"), True)
            _deep_set(patched, ("graph_attention", "enabled"), True)
            _deep_set(patched, ("action_masking", "enabled"), True)
        elif ablation == "no_graph_attention":
            _deep_set(patched, ("graph_attention", "enabled"), False)
            _deep_set(patched, ("game_theory", "use_shapley_credit"), True)
        elif ablation == "no_feasible_action_masking":
            _deep_set(patched, ("action_masking", "enabled"), False)
            _deep_set(patched, ("game_theory", "use_shapley_credit"), True)
        elif ablation == "no_warm_start":
            _deep_set(patched, ("game_theory", "warm_start_steps"), 0)
            _deep_set(patched, ("game_theory", "use_shapley_credit"), True)
        elif ablation == "no_shapley_credit":
            _deep_set(patched, ("game_theory", "use_shapley_credit"), False)
    return patched


def _write_generated_config(
    output_root: Path,
    level: str,
    scenario_id: str,
    algorithm: str,
    ablation: str,
) -> tuple[Path, Path]:
    """Write per-run algorithm config files and return file and directory paths."""
    source_path = _base_algorithm_config_path(algorithm)
    config = _load_yaml(source_path)
    config = _apply_ablation_config(config, algorithm, ablation)
    config_dir = (
        output_root
        / "generated_configs"
        / level
        / _sanitize(scenario_id)
        / _sanitize(algorithm)
        / _sanitize(ablation)
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    algorithm_path = config_dir / "algorithm.yaml"
    serialized = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    algorithm_path.write_text(serialized, encoding="utf-8")
    (config_dir / _benchmark_config_name(algorithm)).write_text(serialized, encoding="utf-8")
    return algorithm_path, config_dir


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
        stage = str(scenario_spec["stage"])
        allowed = set(scenario_spec["allowed_algorithms"])
        for algorithm in algorithms:
            if algorithm not in allowed:
                continue
            ablations = scenario_spec.get("ablations") if stage == "N2" else None
            ablation_labels = [str(item) for item in (ablations or ["full"])]
            for seed in seeds:
                for ablation in ablation_labels:
                    run_id = f"{level}__{scenario_id}__{algorithm}__{ablation}__seed{seed}"
                    run_output = (
                        output_root
                        / scenario_spec["output_subdir"]
                        / _sanitize(algorithm)
                        / _sanitize(ablation)
                        / f"seed_{seed}.json"
                    )
                    generated_config_path, generated_config_dir = _write_generated_config(
                        output_root=output_root,
                        level=level,
                        scenario_id=scenario_id,
                        algorithm=algorithm,
                        ablation=ablation,
                    )
                    result_kind = (
                        "benchmark_diagnostic"
                        if stage == "N1" and bool(scenario_spec.get("oracle_reference"))
                        else "benchmark"
                    )
                    command = _benchmark_command(
                        algorithm=algorithm,
                        seed=seed,
                        steps=steps,
                        output_path=run_output,
                        device=device,
                        scenario_spec=scenario_spec,
                        generated_config_dir=generated_config_dir,
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "matrix_version": matrix["schema_version"],
                            "scenario_version": scenario_matrix["schema_version"],
                            "level": level,
                            "stage": stage,
                            "algorithm": algorithm,
                            "scenario": scenario_id,
                            "seed": seed,
                            "steps": steps,
                            "ablation": ablation,
                            "source_config": str(scenario_spec["profile"].get("source_config", "")),
                            "profile": copy.deepcopy(scenario_spec["profile"]),
                            "result_kind": result_kind,
                            "command": command,
                            "output_path": str(run_output),
                            "claim_level": claim_level,
                            "generated_config_path": str(generated_config_path),
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
        "claim_boundaries": {
            "n1_oracle_claim": "blocked_without_oracle_wrapper",
            "dry_run_is_result": False,
            "final_claim_requires": matrix.get("final_claim_level", "L3_formal_verification"),
        },
        "run_count": len(runs),
        "runs": runs,
    }


def _write_n1_oracle_gap_note() -> None:
    """Record the local N1 oracle boundary when no oracle wrapper is available."""
    path = Path("ref/paper2_n1_oracle_gap.md")
    if path.exists():
        return
    path.write_text(
        "# Paper2 N1 Oracle Gap\n\n"
        "The v4.9 matrix runner marks `N1-oracle-small` records as `benchmark_diagnostic` "
        "because no local oracle wrapper is wired into this patch. These records may support "
        "diagnostic benchmark checks, but they must not be claimed as oracle-gap evidence until "
        "an oracle-capable runner is added and validated.\n",
        encoding="utf-8",
    )


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
    if any(run["stage"] == "N1" for run in manifest["runs"]):
        _write_n1_oracle_gap_note()
    if args.dry_run:
        print(f"DRY RUN paper2 main matrix: {manifest['run_count']} planned runs")
        print(f"Manifest saved to: {args.output}")
        return
    _execute_manifest(manifest, resume=args.resume)
    print(f"paper2 main matrix completed: {manifest['run_count']} runs")


if __name__ == "__main__":
    main()
