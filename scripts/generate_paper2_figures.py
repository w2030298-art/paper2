"""Generate or plan paper2 final figures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


FIGURE_GROUPS = [
    {
        "id": "convergence",
        "title": "Convergence curves",
        "metrics": ["reward_mean", "social_welfare_mean", "e2e_latency_mean", "constraint_violation_rate", "agent_reward_jain_mean"],
    },
    {
        "id": "final_comparison",
        "title": "Final comparison",
        "metrics": ["social_welfare_mean", "reward_mean", "e2e_latency_mean", "energy_total_mean"],
    },
    {
        "id": "n2_ablation",
        "title": "N2 ablation",
        "metrics": ["social_welfare_mean", "constraint_violation_rate", "e2e_latency_mean"],
    },
    {
        "id": "n3_ood",
        "title": "N3 OOD robustness",
        "metrics": ["social_welfare_mean", "deadline_miss_rate", "queue_wait_mean"],
    },
    {
        "id": "statistics",
        "title": "Statistical summary",
        "metrics": ["confidence_interval", "cohens_d", "adjusted_p_value"],
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _split_formats(value: str) -> list[str]:
    """Split a comma-delimited format list."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_figure_plan(args: argparse.Namespace, results_payload: dict[str, Any]) -> dict[str, Any]:
    """Write a figure plan describing expected final groups."""
    plan = {
        "schema_version": "paper2-figure-plan/v1",
        "dry_run": args.dry_run,
        "results": str(args.results),
        "statistics": str(args.statistics) if args.statistics else None,
        "output_dir": str(args.output_dir),
        "formats": _split_formats(args.format),
        "input_schema": results_payload.get("schema_version", "unknown"),
        "input_is_dry_run": bool(results_payload.get("dry_run", False)),
        "figure_groups": FIGURE_GROUPS,
        "claim_status": (
            "planned_only_no_paper_claims"
            if args.dry_run or results_payload.get("dry_run")
            else "ready_for_real_result_rendering"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate paper2 final figures")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/paper2_final"))
    parser.add_argument("--format", type=str, default="pdf,png")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the figure CLI."""
    args = parse_args()
    results_payload = _load_json(args.results)
    plan = _write_figure_plan(args, results_payload)
    if args.dry_run:
        print(f"DRY RUN paper2 figures: plan saved to {args.output_dir / 'figure_plan.json'}")
        return
    if results_payload.get("dry_run"):
        raise SystemExit("Refusing to generate final figures from a dry-run manifest")
    first_format = plan["formats"][0] if plan["formats"] else "pdf"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_results.py",
            "--input",
            str(args.results),
            "--output",
            str(args.output_dir),
            "--format",
            first_format,
        ],
        check=True,
    )
    print(f"paper2 figures generated in {args.output_dir}")


if __name__ == "__main__":
    main()
