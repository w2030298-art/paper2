"""Export the paper2 final delivery package or a dry-run package manifest."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


BASE_ITEMS = [
    ("required", ".ai/ledger.json"),
    ("required", ".ai/checkpoint.json"),
    ("required", "configs/paper2_main_experiment_matrix.yaml"),
    ("required", "configs/paper2_scenario_matrix.yaml"),
    ("required", "configs/formal_convergence_matrix.yaml"),
    ("required", "configs/benchmark_mainline_a_final_screening.yaml"),
    ("required", "scripts/run_paper2_main_matrix.py"),
    ("required", "scripts/analyze_paper2_statistics.py"),
    ("required", "scripts/generate_paper2_figures.py"),
    ("required", "scripts/export_paper2_final_package.py"),
    ("required", "ref/paper2_stage2_residuals_closeout.md"),
    ("required", "ref/paper2_statistics_method.md"),
    ("required", "ref/paper2_final_delivery_manifest.md"),
    ("required", "ref/paper2_final_adaptation_checklist.md"),
]

RESULT_ITEMS = [
    ("optional", "results/paper2_main_matrix/dry_run_manifest.json"),
    ("optional", "results/paper2_statistics/stats_plan.json"),
    ("optional", "results/paper2_statistics/summary.csv"),
    ("optional", "results/paper2_statistics/pairwise_tests.csv"),
    ("optional", "results/paper2_statistics/effect_sizes.csv"),
    ("optional", "results/paper2_statistics/statistics_report.json"),
]

FIGURE_ITEMS = [
    ("optional", "figures/paper2_final/figure_plan.json"),
]


def _item_status(kind: str, path: Path) -> str:
    """Return package status for an item path."""
    if path.exists():
        return "present"
    return "missing_optional" if kind == "optional" else "missing_required"


def _collect_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Collect package item metadata."""
    requested = list(BASE_ITEMS)
    if args.include_results:
        requested.extend(RESULT_ITEMS)
    if args.include_figures:
        requested.extend(FIGURE_ITEMS)
    if args.include_ref:
        requested.extend(
            [
                ("optional", "ref/mainline_a_final_algorithm_adaptation.md"),
                ("optional", "ref/stage2-roadmap-cl-ppo-gam-coma.md"),
            ]
        )
    items = []
    for kind, rel_path in requested:
        path = Path(rel_path)
        items.append(
            {
                "path": rel_path,
                "kind": kind,
                "status": _item_status(kind, path),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
            }
        )
    return items


def _manifest(args: argparse.Namespace, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the package manifest."""
    return {
        "schema_version": "paper2-final-package-manifest/v1",
        "dry_run": args.dry_run,
        "output": str(args.output),
        "items": items,
        "missing_required": [item["path"] for item in items if item["status"] == "missing_required"],
        "missing_optional": [item["path"] for item in items if item["status"] == "missing_optional"],
        "claim_status": "dry_run_inventory_only" if args.dry_run else "package_created_from_existing_files",
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write package manifest JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_zip(output: Path, manifest: dict[str, Any]) -> None:
    """Write a zip containing present package items and the package manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("paper2_final_package_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for item in manifest["items"]:
            if item["status"] == "present":
                archive.write(item["path"], arcname=item["path"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export paper2 final delivery package")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-results", action="store_true", default=True)
    parser.add_argument("--include-figures", action="store_true", default=True)
    parser.add_argument("--include-ref", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=Path("paper2_final_delivery_package.zip"))
    return parser.parse_args()


def main() -> None:
    """Run the export CLI."""
    args = parse_args()
    items = _collect_items(args)
    manifest = _manifest(args, items)
    manifest_path = Path(str(args.output) + ".manifest.json")
    _write_manifest(manifest_path, manifest)
    if args.dry_run:
        print(f"DRY RUN paper2 export: manifest saved to {manifest_path}")
        return
    if manifest["missing_required"]:
        missing = ", ".join(manifest["missing_required"])
        raise SystemExit(f"Missing required package items: {missing}")
    _write_zip(args.output, manifest)
    print(f"paper2 final package written to {args.output}")


if __name__ == "__main__":
    main()
