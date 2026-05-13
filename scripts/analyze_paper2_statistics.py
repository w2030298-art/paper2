"""Analyze paper2 v4.8 main-matrix statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SUMMARY_FIELDS = [
    "scenario",
    "algorithm",
    "metric",
    "n",
    "mean",
    "std",
    "stderr",
    "ci95_low",
    "ci95_high",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
]

PAIRWISE_FIELDS = [
    "scenario",
    "baseline",
    "comparison",
    "metric",
    "n_paired",
    "mean_delta",
    "p_value",
    "p_value_status",
    "adjusted_p_value",
    "correction",
]

EFFECT_FIELDS = [
    "scenario",
    "baseline",
    "comparison",
    "metric",
    "cohens_d",
    "baseline_mean",
    "comparison_mean",
]


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write CSV rows with a fixed field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_stats_plan(payload: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> None:
    """Write a manifest-only statistics plan without statistical values."""
    runs = payload.get("runs", [])
    algorithms = sorted({run.get("algorithm") for run in runs if run.get("algorithm")})
    scenarios = sorted({run.get("scenario") for run in runs if run.get("scenario")})
    levels = sorted({run.get("level", payload.get("level")) for run in runs if run.get("level", payload.get("level"))})
    plan = {
        "schema_version": "paper2-statistics-plan/v1",
        "mode": "manifest-only/dry-run",
        "statistics_generated": False,
        "input_schema": payload.get("schema_version", "unknown"),
        "primary_metric": args.primary_metric,
        "baseline": args.baseline,
        "comparison_groups": args.comparison_groups,
        "planned_run_count": len(runs),
        "algorithms": algorithms,
        "scenarios": scenarios,
        "levels": levels,
        "planned_tables": ["summary", "pairwise", "magnitude"],
        "notes": [
            "Dry-run validates planned schema only.",
            "No inferential statistics are computed without real multi-seed results.",
        ],
    }
    _write_json(output_dir / "stats_plan.json", plan)


def _extract_metric(record: dict[str, Any], metric: str) -> float:
    """Extract a metric value from a result record."""
    metrics = record.get("metrics")
    if isinstance(metrics, dict) and metric in metrics:
        return float(metrics[metric])
    if metric in record:
        return float(record[metric])
    final_metrics = record.get("final_metrics")
    if isinstance(final_metrics, dict) and metric in final_metrics:
        return float(final_metrics[metric])
    raise ValueError(
        f"Missing metric {metric!r} for {record.get('algorithm')} "
        f"in {record.get('scenario')} seed {record.get('seed')}"
    )


def _normalize_results(payload: dict[str, Any], primary_metric: str) -> list[dict[str, Any]]:
    """Normalize supported result payloads into flat statistic records."""
    raw_results = payload.get("results")
    if raw_results is None:
        raw_results = [
            run for run in payload.get("runs", [])
            if isinstance(run, dict) and ("metrics" in run or primary_metric in run)
        ]
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError("No result records with metrics were found")

    normalized: list[dict[str, Any]] = []
    for record in raw_results:
        if not isinstance(record, dict):
            raise ValueError("Each result record must be a JSON object")
        normalized.append(
            {
                "scenario": str(record["scenario"]),
                "algorithm": str(record["algorithm"]),
                "seed": int(record["seed"]),
                primary_metric: _extract_metric(record, primary_metric),
            }
        )
    return normalized


def _validate_seed_grid(records: list[dict[str, Any]], allow_partial: bool) -> list[str]:
    """Validate that all algorithms share the same seeds within each scenario."""
    by_scenario: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        by_scenario[record["scenario"]][record["algorithm"]].add(record["seed"])

    issues: list[str] = []
    for scenario, by_algorithm in by_scenario.items():
        expected = set().union(*by_algorithm.values())
        for algorithm, seeds in by_algorithm.items():
            if seeds != expected:
                missing = sorted(expected - seeds)
                extra = sorted(seeds - expected)
                issues.append(
                    f"{scenario}/{algorithm}: missing={missing or []}, extra={extra or []}"
                )
    if issues and not allow_partial:
        raise ValueError("Incomplete seed grid: " + "; ".join(issues))
    return issues


def _ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Return std, stderr, and normal-approximation 95% CI."""
    if len(values) < 2:
        return 0.0, 0.0, values[0], values[0]
    std_value = stdev(values)
    stderr = std_value / math.sqrt(len(values))
    delta = 1.96 * stderr
    return std_value, stderr, mean(values) - delta, mean(values) + delta


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Compute a deterministic percentile bootstrap CI for the mean."""
    if len(values) < 2:
        return values[0], values[0]
    estimates: list[float] = []
    count = len(values)
    for index in range(400):
        sample = [values[(index + offset * 17) % count] for offset in range(count)]
        estimates.append(mean(sample))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return low, high


def _summary_rows(records: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    """Build per-scenario and per-algorithm summary rows."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record["scenario"], record["algorithm"])].append(record[primary_metric])

    rows: list[dict[str, Any]] = []
    for (scenario, algorithm), values in sorted(grouped.items()):
        std_value, stderr, ci_low, ci_high = _ci95(values)
        boot_low, boot_high = _bootstrap_ci(values)
        rows.append(
            {
                "scenario": scenario,
                "algorithm": algorithm,
                "metric": primary_metric,
                "n": len(values),
                "mean": f"{mean(values):.10g}",
                "std": f"{std_value:.10g}",
                "stderr": f"{stderr:.10g}",
                "ci95_low": f"{ci_low:.10g}",
                "ci95_high": f"{ci_high:.10g}",
                "bootstrap_ci_low": f"{boot_low:.10g}",
                "bootstrap_ci_high": f"{boot_high:.10g}",
            }
        )
    return rows


def _cohens_d(left: list[float], right: list[float]) -> float:
    """Compute Cohen's d for two independent value lists."""
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_std = stdev(left)
    right_std = stdev(right)
    pooled = math.sqrt(((len(left) - 1) * left_std**2 + (len(right) - 1) * right_std**2) / (len(left) + len(right) - 2))
    if pooled == 0:
        return 0.0
    return (mean(right) - mean(left)) / pooled


def _optional_p_value(baseline_values: list[float], comparison_values: list[float]) -> tuple[str, str]:
    """Compute an optional paired p-value when scipy is available."""
    if len(baseline_values) < 2 or len(comparison_values) < 2:
        return "", "not_run_insufficient_pairs"
    try:
        from scipy import stats  # type: ignore
    except ImportError:
        return "", "not_run_scipy_unavailable"

    try:
        result = stats.wilcoxon(comparison_values, baseline_values, zero_method="zsplit")
        return f"{float(result.pvalue):.10g}", "computed_wilcoxon_signed_rank"
    except ValueError as exc:
        return "", f"not_run_{str(exc).replace(' ', '_')}"


def _holm_bonferroni(rows: list[dict[str, Any]]) -> None:
    """Add Holm-Bonferroni adjusted p-values to pairwise rows."""
    numeric = [
        (index, float(row["p_value"]))
        for index, row in enumerate(rows)
        if row.get("p_value")
    ]
    numeric.sort(key=lambda item: item[1])
    total = len(numeric)
    adjusted_by_index: dict[int, float] = {}
    running_max = 0.0
    for rank, (index, p_value) in enumerate(numeric):
        adjusted = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, adjusted)
        adjusted_by_index[index] = running_max
    for index, row in enumerate(rows):
        row["correction"] = "holm_bonferroni"
        row["adjusted_p_value"] = (
            f"{adjusted_by_index[index]:.10g}" if index in adjusted_by_index else ""
        )


def _pairwise_and_effect_rows(
    records: list[dict[str, Any]],
    primary_metric: str,
    baseline: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build baseline-vs-comparison pairwise and effect-size rows."""
    by_group: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for record in records:
        by_group[(record["scenario"], record["algorithm"])][record["seed"]] = record[primary_metric]

    pairwise_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    scenarios = sorted({scenario for scenario, _algorithm in by_group})
    for scenario in scenarios:
        baseline_by_seed = by_group.get((scenario, baseline), {})
        if not baseline_by_seed:
            continue
        algorithms = sorted(algorithm for item_scenario, algorithm in by_group if item_scenario == scenario)
        for algorithm in algorithms:
            if algorithm == baseline:
                continue
            comparison_by_seed = by_group[(scenario, algorithm)]
            shared_seeds = sorted(set(baseline_by_seed) & set(comparison_by_seed))
            baseline_values = [baseline_by_seed[seed] for seed in shared_seeds]
            comparison_values = [comparison_by_seed[seed] for seed in shared_seeds]
            deltas = [comparison_by_seed[seed] - baseline_by_seed[seed] for seed in shared_seeds]
            p_value, p_status = _optional_p_value(baseline_values, comparison_values)
            pairwise_rows.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "comparison": algorithm,
                    "metric": primary_metric,
                    "n_paired": len(shared_seeds),
                    "mean_delta": f"{mean(deltas):.10g}" if deltas else "",
                    "p_value": p_value,
                    "p_value_status": p_status,
                    "adjusted_p_value": "",
                    "correction": "holm_bonferroni",
                }
            )
            effect_rows.append(
                {
                    "scenario": scenario,
                    "baseline": baseline,
                    "comparison": algorithm,
                    "metric": primary_metric,
                    "cohens_d": f"{_cohens_d(baseline_values, comparison_values):.10g}",
                    "baseline_mean": f"{mean(baseline_values):.10g}" if baseline_values else "",
                    "comparison_mean": f"{mean(comparison_values):.10g}" if comparison_values else "",
                }
            )
    _holm_bonferroni(pairwise_rows)
    return pairwise_rows, effect_rows


def _analyze_results(payload: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> None:
    """Analyze real result records and write statistics exports."""
    records = _normalize_results(payload, args.primary_metric)
    seed_issues = _validate_seed_grid(records, allow_partial=args.allow_partial)
    summary_rows = _summary_rows(records, args.primary_metric)
    pairwise_rows, effect_rows = _pairwise_and_effect_rows(records, args.primary_metric, args.baseline)

    _write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(output_dir / "pairwise_tests.csv", pairwise_rows, PAIRWISE_FIELDS)
    _write_csv(output_dir / "effect_sizes.csv", effect_rows, EFFECT_FIELDS)
    report = {
        "schema_version": "paper2-statistics-report/v1",
        "mode": "results-present",
        "input_schema": payload.get("schema_version", "unknown"),
        "primary_metric": args.primary_metric,
        "baseline": args.baseline,
        "record_count": len(records),
        "summary_rows": len(summary_rows),
        "pairwise_rows": len(pairwise_rows),
        "effect_rows": len(effect_rows),
        "seed_grid": {
            "status": "partial_allowed" if seed_issues else "complete",
            "issues": seed_issues,
        },
        "multiple_comparison_correction": {
            "method": "holm_bonferroni",
            "scope": "all numeric pairwise p-values in this report",
        },
    }
    _write_json(output_dir / "statistics_report.json", report)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze paper2 statistics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper2_statistics"))
    parser.add_argument("--primary-metric", type=str, default="social_welfare_mean")
    parser.add_argument("--baseline", type=str, default="PPO")
    parser.add_argument(
        "--comparison-groups",
        type=str,
        default="proposed,stage1_best_defaults,strong_single_agent_baselines,strong_multi_agent_baselines",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the statistics analyzer CLI."""
    args = parse_args()
    payload = _load_json(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run or payload.get("dry_run") is True:
        _write_stats_plan(payload, args.output_dir, args)
        print(f"DRY RUN paper2 statistics: stats plan saved to {args.output_dir / 'stats_plan.json'}")
        return
    try:
        _analyze_results(payload, args.output_dir, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"paper2 statistics saved to {args.output_dir}")


if __name__ == "__main__":
    main()
