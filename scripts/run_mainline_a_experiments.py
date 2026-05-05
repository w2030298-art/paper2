#!/usr/bin/env python3
"""Dry-run capable runner for mainline-A N0/N1/N2/N3 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_CONFIGS = {
    "N0": PROJECT_ROOT / "configs/experiments/mainline_a_n0_smoke.yaml",
    "N1": PROJECT_ROOT / "configs/experiments/mainline_a_n1_oracle.yaml",
    "N2": PROJECT_ROOT / "configs/experiments/mainline_a_n2_ablation.yaml",
    "N3": PROJECT_ROOT / "configs/experiments/mainline_a_n3_ood.yaml",
}


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def build_stage_plan(stage: str, config: dict[str, Any], output_root: Path, results_root: Path) -> dict[str, Any]:
    """Build a serializable execution plan for one stage."""
    return {
        "stage": stage,
        "name": config.get("name", f"mainline_a_{stage.lower()}"),
        "algorithms": config.get("algorithms", ["game_aware_pd_marl"]),
        "seeds": config.get("seeds", [42]),
        "steps": config.get("steps", 1000),
        "output_root": str(output_root),
        "results_root": str(results_root),
        "dry_run_supported": True,
    }


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

    manifest = write_manifest(plans, PROJECT_ROOT / args.output_root)
    print(f"mainline-A manifest written: {manifest}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()

