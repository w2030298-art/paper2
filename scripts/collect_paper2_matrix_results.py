"""Collect paper2 matrix per-run outputs into statistics-ready records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_METRICS = [
    "social_welfare_mean",
    "reward_mean",
    "e2e_latency_mean",
    "e2e_latency_p95",
    "deadline_miss_rate",
    "energy_total_mean",
    "throughput_tasks_per_step",
    "agent_reward_jain_mean",
    "constraint_violation_rate",
    "queue_wait_mean",
    "comm_score",
]

METRIC_ALIASES = {
    "social_welfare_mean": ["final_social_welfare_mean", "social_welfare_mean"],
    "reward_mean": ["final_reward_mean", "reward_mean"],
    "e2e_latency_mean": ["final_e2e_latency_mean", "final_latency_mean", "e2e_latency_mean", "latency_mean"],
    "e2e_latency_p95": ["final_e2e_latency_p95", "final_latency_p95", "e2e_latency_p95", "latency_p95"],
    "deadline_miss_rate": ["final_deadline_miss_rate", "deadline_miss_rate"],
    "energy_total_mean": ["final_energy_total_mean", "final_energy_mean", "energy_total_mean", "energy_mean"],
    "throughput_tasks_per_step": ["final_throughput_tasks_per_step", "throughput_tasks_per_step"],
    "agent_reward_jain_mean": ["final_agent_reward_jain_mean", "agent_reward_jain_mean"],
    "constraint_violation_rate": ["final_constraint_violation_rate", "constraint_violation_rate"],
    "queue_wait_mean": ["final_queue_wait_mean", "queue_wait_mean"],
    "comm_score": ["final_comm_score", "comm_score"],
}

STATUS_FIELDS = [
    "run_id",
    "status",
    "reason",
    "scenario",
    "stage",
    "algorithm",
    "ablation",
    "seed",
    "output_path",
]


def _load_json(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = STATUS_FIELDS) -> None:
    """Write CSV rows with fixed fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_records(payload: Any) -> list[dict[str, Any]]:
    """Return candidate result records from supported benchmark/oracle shapes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "records", "runs"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _select_record(candidates: list[dict[str, Any]], run: dict[str, Any]) -> dict[str, Any] | None:
    """Select the record matching the manifest algorithm and seed."""
    for record in candidates:
        if str(record.get("algorithm")) == str(run["algorithm"]) and int(record.get("seed", run["seed"])) == int(run["seed"]):
            return record
    return candidates[0] if candidates else None


def _metric_from_record(record: dict[str, Any], metric: str) -> float | None:
    """Extract a metric value by canonical name or alias."""
    nested = record.get("metrics")
    if isinstance(nested, dict) and metric in nested:
        return float(nested[metric])
    nested = record.get("final_metrics")
    if isinstance(nested, dict) and metric in nested:
        return float(nested[metric])
    for key in METRIC_ALIASES[metric]:
        if key in record and record[key] is not None:
            return float(record[key])
    return None


def _normalize_record(run: dict[str, Any], record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize one manifest run and one result record."""
    metrics: dict[str, float] = {}
    missing_metrics: list[str] = []
    for metric in REQUIRED_METRICS:
        value = _metric_from_record(record, metric)
        if value is None:
            missing_metrics.append(metric)
        else:
            metrics[metric] = value
    normalized = {
        "run_id": run["run_id"],
        "scenario": run["scenario"],
        "stage": run["stage"],
        "algorithm": run["algorithm"],
        "seed": int(run["seed"]),
        "steps": int(run["steps"]),
        "level": run["level"],
        "ablation": run.get("ablation", "full"),
        "result_kind": run.get("result_kind", "benchmark"),
        "output_path": run["output_path"],
        "metrics": metrics,
        "source_status": str(record.get("status", "success")),
    }
    return normalized, missing_metrics


def _status_row(run: dict[str, Any], status: str, reason: str = "") -> dict[str, Any]:
    """Build a status CSV row."""
    return {
        "run_id": run.get("run_id", ""),
        "status": status,
        "reason": reason,
        "scenario": run.get("scenario", ""),
        "stage": run.get("stage", ""),
        "algorithm": run.get("algorithm", ""),
        "ablation": run.get("ablation", "full"),
        "seed": run.get("seed", ""),
        "output_path": run.get("output_path", ""),
    }


def _seed_grid_issues(records: list[dict[str, Any]]) -> list[str]:
    """Find incomplete seed grids within each scenario/stage/ablation group."""
    grouped: dict[tuple[str, str, str], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        key = (record["scenario"], record["stage"], record["ablation"])
        grouped[key][record["algorithm"]].add(int(record["seed"]))

    issues: list[str] = []
    for key, by_algorithm in grouped.items():
        if len(by_algorithm) <= 1:
            continue
        expected = set().union(*by_algorithm.values())
        for algorithm, seeds in by_algorithm.items():
            if seeds != expected:
                missing = sorted(expected - seeds)
                issues.append(f"{'/'.join(key)}/{algorithm}: missing seeds {missing}")
    return issues


def collect(manifest: dict[str, Any], strict: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Collect records, status rows, missing rows, and strict issues."""
    if manifest.get("dry_run") is True:
        raise ValueError("Refusing to collect evidence from a dry-run manifest")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Manifest must contain a runs list")

    records: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for run in runs:
        output_path = Path(run["output_path"])
        if not output_path.exists():
            row = _status_row(run, "missing", "missing output")
            status_rows.append(row)
            missing_rows.append(row)
            continue
        try:
            payload = _load_json(output_path)
            selected = _select_record(_candidate_records(payload), run)
            if selected is None:
                raise ValueError("no result records found")
            normalized, missing_metrics = _normalize_record(run, selected)
            if normalized["source_status"] not in {"success", "ok", "passed"}:
                reason = f"source status {normalized['source_status']}"
                status_rows.append(_status_row(run, "failed", reason))
                issues.append(f"{run['run_id']}: {reason}")
                continue
            if missing_metrics:
                reason = "missing metrics: " + ",".join(missing_metrics)
                status_rows.append(_status_row(run, "partial", reason))
                if strict:
                    issues.append(f"{run['run_id']}: {reason}")
                records.append(normalized)
                continue
            records.append(normalized)
            status_rows.append(_status_row(run, "collected"))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            status_rows.append(_status_row(run, "failed", reason))
            issues.append(f"{run.get('run_id', output_path)}: {reason}")

    if missing_rows:
        issues.append(f"missing outputs: {len(missing_rows)}")
    seed_issues = _seed_grid_issues(records)
    if seed_issues:
        issues.append("incomplete seed grid: " + "; ".join(seed_issues))
    if strict and issues:
        return records, status_rows, missing_rows, issues
    return records, status_rows, missing_rows, []


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect paper2 matrix results")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-csv", type=Path, required=True)
    parser.add_argument("--missing-csv", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the result collector CLI."""
    args = parse_args()
    try:
        manifest = _load_json(args.manifest)
        records, status_rows, missing_rows, issues = collect(
            manifest,
            strict=bool(args.strict and not args.allow_partial),
        )
        _write_csv(args.status_csv, status_rows)
        _write_csv(args.missing_csv, missing_rows)
        payload = {
            "schema_version": "paper2-matrix-results/v1",
            "manifest": str(args.manifest),
            "level": manifest.get("level"),
            "record_count": len(records),
            "strict": bool(args.strict),
            "results": records,
        }
        _write_json(args.output, payload)
        if issues:
            print("; ".join(issues), file=sys.stderr)
            raise SystemExit(1)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Collected {len(records)} paper2 matrix results into {args.output}")


if __name__ == "__main__":
    main()
