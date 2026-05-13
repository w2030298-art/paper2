#!/usr/bin/env python3
"""Stage-1 PPO/COMA tuner for Mainline-A candidate selection."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.pareto import is_pareto_efficient  # noqa: E402
from src.experiment.physical_objective import compute_j_phys, extract_stage1_metrics  # noqa: E402
from src.experiment.runtime_config import sha256_file  # noqa: E402


CSV_FIELDS = [
    "trial_id",
    "trial_name",
    "algorithm",
    "environment_profile",
    "seed",
    "status",
    "j_phys",
    "feasible",
    "pareto_eligible",
    "p95_latency",
    "mean_latency",
    "energy_per_task",
    "tail_instability",
    "deadline_miss_rate",
    "constraint_violation",
    "reward_mean",
    "result_json",
    "config_path",
    "error",
]


def load_search_config(path: str | Path) -> dict[str, Any]:
    """Load a Stage-1 YAML search config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["_path"] = str(config_path)
    return cfg


def _sample_param(spec: dict[str, Any], rng: random.Random) -> Any:
    kind = str(spec.get("type", "float"))
    if kind == "categorical":
        choices = list(spec["choices"])
        return choices[rng.randrange(len(choices))]
    low = float(spec["low"])
    high = float(spec["high"])
    if kind == "log_float":
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    if kind == "int":
        return int(rng.randint(int(low), int(high)))
    return float(rng.uniform(low, high))


def _sample_with_optuna(search_space: dict[str, dict[str, Any]], trial_index: int, seed: int) -> dict[str, Any] | None:
    try:
        import optuna
    except ImportError:
        return None

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    params: dict[str, Any] = {}
    for _ in range(max(1, trial_index)):
        trial = study.ask()
        params = {}
        for name, spec in search_space.items():
            kind = str(spec.get("type", "float"))
            if kind == "categorical":
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
            elif kind == "log_float":
                params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
            elif kind == "int":
                params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
            else:
                params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
        study.tell(trial, float(trial_index))
    return params


def generate_trial_configs(search_cfg: dict[str, Any], trials: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate trial parameter sets, with trial 0000 fixed to the recommended start."""
    if trials < 1:
        return []
    generated = [
        {
            "trial_id": "0000",
            "name": search_cfg["recommended_start"]["name"],
            "params": dict(search_cfg["recommended_start"]["params"]),
            "sampling": "recommended_start",
        }
    ]
    search_space = search_cfg.get("search_space", {}) or {}
    rng = random.Random(seed)
    for idx in range(1, trials):
        params = _sample_with_optuna(search_space, idx, seed)
        sampling = "optuna_tpe" if params is not None else "deterministic_random"
        if params is None:
            params = {name: _sample_param(spec, rng) for name, spec in search_space.items()}
        generated.append(
            {
                "trial_id": f"{idx:04d}",
                "name": f"sample-{idx:04d}",
                "params": params,
                "sampling": sampling,
            }
        )
    return generated


def _set_dotted(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def materialize_trial_config(
    search_cfg: dict[str, Any],
    trial: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a concrete algorithm config for one trial and return its path."""
    base_config = PROJECT_ROOT / str(search_cfg["base_config"])
    with base_config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = copy.deepcopy(cfg)
    for key, value in trial["params"].items():
        _set_dotted(cfg, key, value)
    cfg.setdefault("algorithm", {})["name"] = search_cfg["algorithm"]
    cfg.setdefault("stage1_tuning", {}).update(
        {
            "trial_id": trial["trial_id"],
            "trial_name": trial["name"],
            "source_search_config": search_cfg.get("_path"),
            "environment_profile": search_cfg["environment_profile"],
        }
    )

    trial_dir = Path(output_dir) / search_cfg["algorithm"].lower() / f"trial_{trial['trial_id']}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return config_path


def _result_metric(payload: dict[str, Any], key: str) -> Any:
    final_eval = payload.get("final_eval", {}) if isinstance(payload, dict) else {}
    if key in final_eval:
        return final_eval[key]
    return payload.get(key)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _write_best_config(path: Path, rows: list[dict[str, Any]]) -> None:
    def finite_j_phys(row: dict[str, Any]) -> float:
        try:
            return float(row.get("j_phys", "inf"))
        except (TypeError, ValueError):
            return float("inf")

    successful = [
        row
        for row in rows
        if row.get("status") == "success" and math.isfinite(finite_j_phys(row))
    ]
    if not successful:
        path.write_text(
            yaml.safe_dump(
                {
                    "status": "no_successful_trial",
                    "reason": "all_trials_failed_or_infeasible",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return

    best = min(successful, key=finite_j_phys)
    source = Path(best.get("config_path", "")) if best.get("config_path") else None
    if source is not None and source.exists():
        path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        path.write_text(
            yaml.safe_dump(
                {
                    "status": "no_successful_trial",
                    "reason": "all_trials_failed_or_infeasible",
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )


def _load_existing_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_audit(path: Path, algorithm: str, audit: dict[str, Any]) -> None:
    payload = _load_existing_audit(path)
    payload[algorithm.upper()] = audit
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_trial(
    *,
    search_cfg: dict[str, Any],
    trial: dict[str, Any],
    seed: int,
    config_path: Path,
    timesteps: int,
    device: str,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    trial_dir = config_path.parent
    result_json = trial_dir / f"result_seed_{seed}.json"
    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(config_path),
        "--algorithm",
        search_cfg["algorithm"],
        "--environment-profile",
        search_cfg["environment_profile"],
        "--timesteps",
        str(timesteps),
        "--seed",
        str(seed),
        "--device",
        device,
        "--result-json",
        str(result_json),
    ]
    row = {
        "trial_id": trial["trial_id"],
        "trial_name": trial["name"],
        "algorithm": search_cfg["algorithm"],
        "environment_profile": search_cfg["environment_profile"],
        "seed": seed,
        "result_json": str(result_json),
        "config_path": str(config_path),
    }
    if dry_run:
        row.update(
            {
                "status": "dry-run",
                "j_phys": "",
                "feasible": False,
                "pareto_eligible": False,
                "error": "",
            }
        )
        return row, None, command

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip().splitlines()
        row.update(
            {
                "status": "failed",
                "j_phys": "inf",
                "feasible": False,
                "pareto_eligible": False,
                "error": error[-1] if error else f"train.py exited {completed.returncode}",
            }
        )
        return row, None, command

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    metrics = extract_stage1_metrics(payload.get("final_eval", payload))
    j_phys = compute_j_phys(metrics, search_cfg["algorithm"])
    metrics["pareto_eligible"] = bool(metrics.get("pareto_eligible", True)) and math.isfinite(j_phys)
    row.update(
        {
            "status": "success",
            "j_phys": j_phys,
            "feasible": bool(metrics["feasible"]),
            "pareto_eligible": bool(metrics["pareto_eligible"]),
            "p95_latency": metrics["p95_latency"],
            "mean_latency": metrics["mean_latency"],
            "energy_per_task": metrics["energy_per_task"],
            "tail_instability": metrics["tail_instability"],
            "deadline_miss_rate": metrics["deadline_miss_rate"],
            "constraint_violation": metrics["constraint_violation"],
            "reward_mean": _result_metric(payload, "eval/reward_mean"),
            "error": "",
        }
    )
    return row, payload, command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune PPO/COMA Stage-1 Mainline-A candidates")
    parser.add_argument("--algorithm", choices=["PPO", "COMA"], required=True)
    parser.add_argument("--search-config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "starter", "smoke", "broad", "narrow", "confirm"], required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--environment-profile", choices=["mainline-a"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/stage1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_cfg = load_search_config(args.search_config)
    if search_cfg["algorithm"] != args.algorithm:
        raise ValueError(f"Search config algorithm {search_cfg['algorithm']} does not match {args.algorithm}")
    if search_cfg["environment_profile"] != args.environment_profile:
        raise ValueError("Stage-1 tuner only accepts environment_profile mainline-a for this packet")

    output_dir = Path(args.output_dir)
    trials = generate_trial_configs(search_cfg, args.trials, seed=min(args.seeds))
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    runtime_config: dict[str, Any] | None = None
    dry_run = bool(args.dry_run or args.mode == "dry-run")

    for trial in trials:
        config_path = materialize_trial_config(search_cfg, trial, output_dir)
        for seed in args.seeds:
            row, payload, command = _run_trial(
                search_cfg=search_cfg,
                trial=trial,
                seed=int(seed),
                config_path=config_path,
                timesteps=args.timesteps,
                device=args.device,
                dry_run=dry_run,
            )
            rows.append(row)
            commands.append({"trial_id": trial["trial_id"], "seed": int(seed), "command": command})
            if payload and runtime_config is None:
                runtime_config = payload.get("resolved_runtime_config")

    algo_lower = args.algorithm.lower()
    trials_csv = output_dir / f"{algo_lower}_trials.csv"
    pareto_csv = output_dir / f"{algo_lower}_pareto.csv"
    best_config = output_dir / f"{algo_lower}_best_config.yaml"
    audit_path = output_dir / "search_audit.json"
    _write_csv(trials_csv, rows)
    _write_csv(pareto_csv, is_pareto_efficient(rows))
    _write_best_config(best_config, rows)
    _write_audit(
        audit_path,
        args.algorithm,
        {
            "algorithm": args.algorithm,
            "environment_profile": args.environment_profile,
            "action_space": (runtime_config or {}).get("action_space"),
            "agent_runtime": (runtime_config or {}).get("agent_runtime"),
            "search_config_path": str(Path(args.search_config)),
            "search_config_sha256": sha256_file(args.search_config),
            "base_config_path": search_cfg["base_config"],
            "base_config_sha256": sha256_file(PROJECT_ROOT / str(search_cfg["base_config"])),
            "mode": args.mode,
            "sampling_mode": sorted({trial["sampling"] for trial in trials}),
            "pruning_eligibility_rules": {
                "constraint_violation": "must be zero",
                "PPO_deadline_miss_rate_max": 0.001,
                "COMA_deadline_miss_rate_max": 0.003,
                "reward": "tie-break only within algorithm family",
            },
            "dry_run": dry_run,
            "commands": commands,
            "artifacts": {
                "trials_csv": str(trials_csv),
                "pareto_csv": str(pareto_csv),
                "best_config": str(best_config),
            },
        },
    )


if __name__ == "__main__":
    main()
