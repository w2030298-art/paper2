#!/usr/bin/env python3
"""CLI entrypoint for experiment orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.experiment.errors import (  # noqa: E402
    ExperimentLockError,
    ExperimentNotFoundError,
    ExperimentStateError,
)
from src.experiment.manager import ExperimentManager  # noqa: E402
from src.experiment.presets import PRESETS  # noqa: E402
from src.experiment.result_writer import BenchmarkResultWriter  # noqa: E402
from src.experiment.state_store import JsonStateStore  # noqa: E402


DEFAULT_START_OPTIONS = {
    "algorithms": ["GRPO", "PPO", "SAC", "DDQN"],
    "timesteps": 5000,
    "seed": 42,
    "device": "auto",
    "eval_episodes": 3,
    "env": "auto",
    "output_dir": "results",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment manager CLI")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--preset", choices=["quick", "full17"], default=None)
    start_parser.add_argument("--fresh", action="store_true")
    start_parser.add_argument("--run-id", default=None)
    start_parser.add_argument("--name", default=None)
    start_parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
    )
    start_parser.add_argument("--timesteps", type=int, default=None)
    start_parser.add_argument("--seed", type=int, default=None)
    start_parser.add_argument("--device", default=None)
    start_parser.add_argument("--eval-episodes", type=int, default=None)
    start_parser.add_argument("--env", default=None)
    start_parser.add_argument("--output-dir", default=None)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--run-id", required=True)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--run-id", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-id", required=True)

    subparsers.add_parser("list")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--run-id", required=True)
    export_parser.add_argument("--output", default=None)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--run-id", required=True)
    reset_parser.add_argument("--algorithm", required=True)

    subparsers.add_parser("rebuild-index")
    return parser


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False))


def _print_error_json(message: str) -> None:
    print(json.dumps({"status": "error", "error": message}, ensure_ascii=False), file=sys.stderr)


def _resolve_start_options(args: argparse.Namespace) -> dict:
    preset = PRESETS[args.preset] if args.preset is not None else DEFAULT_START_OPTIONS
    run_id = args.run_id or preset.get("run_id")
    if run_id is None:
        raise ValueError("start requires --run-id or --preset")

    name = args.name or preset.get("name") or run_id
    algorithms = args.algorithms if args.algorithms is not None else list(preset["algorithms"])
    timesteps = args.timesteps if args.timesteps is not None else preset["timesteps"]
    seed = args.seed if args.seed is not None else preset["seed"]
    device = args.device if args.device is not None else preset["device"]
    eval_episodes = (
        args.eval_episodes if args.eval_episodes is not None else preset["eval_episodes"]
    )
    env = args.env if args.env is not None else preset["env"]
    output_dir = args.output_dir if args.output_dir is not None else preset["output_dir"]

    return {
        "run_id": run_id,
        "name": name,
        "algorithms": algorithms,
        "timesteps": timesteps,
        "seed": seed,
        "device": device,
        "eval_episodes": eval_episodes,
        "env": env,
        "output_dir": output_dir,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            manager = ExperimentManager()
            options = _resolve_start_options(args)
            run_id = options["run_id"]
            if args.fresh and manager.store.exists(run_id):
                manager.delete_experiment(run_id)
            if args.fresh or not manager.store.exists(run_id):
                manager.create_experiment(**options)
            manager.start_or_resume(run_id)
            _print_json(manager.get_status(run_id))
        elif args.command == "resume":
            manager = ExperimentManager()
            manager.start_or_resume(args.run_id)
            _print_json(manager.get_status(args.run_id))
        elif args.command == "stop":
            manager = ExperimentManager()
            state = manager.request_stop(args.run_id)
            _print_json(state.to_dict())
        elif args.command == "status":
            manager = ExperimentManager()
            _print_json(manager.get_status(args.run_id))
        elif args.command == "list":
            manager = ExperimentManager()
            _print_json(manager.list_runs(rebuild_index=True))
        elif args.command == "reset":
            manager = ExperimentManager()
            state = manager.reset_failed_algorithm(args.run_id, args.algorithm)
            _print_json(state.to_dict())
        elif args.command == "rebuild-index":
            manager = ExperimentManager()
            manager.rebuild_index()
            _print_json({"status": "ok"})
        elif args.command == "export":
            writer = BenchmarkResultWriter(JsonStateStore("experiments"))
            output = writer.export_run(args.run_id, output_path=args.output)
            _print_json({"status": "ok", "output": str(output)})
        return 0
    except ExperimentNotFoundError as exc:
        _print_error_json(str(exc))
        return 2
    except (ExperimentStateError, ExperimentLockError) as exc:
        _print_error_json(str(exc))
        return 3
    except ValueError as exc:
        _print_error_json(str(exc))
        return 4
    except Exception as exc:  # noqa: BLE001
        _print_error_json(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
