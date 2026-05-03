#!/usr/bin/env python3
"""Build a failure matrix for non-converged benchmark algorithms."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_results import (
    CONVERGENCE_METRIC_SPECS,
    _aggregate_metric_by_seed,
    _build_timestep_axis,
    _collect_convergence_data,
    _is_span_extreme,
    _pick_metric_series,
    _sanitize_metric_series,
    compute_convergence_status,
)


STATUS_PRIORITY = {
    "catastrophic_outlier": 60,
    "diverging": 50,
    "oscillating": 40,
    "bad_plateau": 30,
    "improving": 20,
    "converged_good": 10,
    "insufficient": 0,
}

DEFAULT_TARGET_ALGORITHMS = {
    "GRPO",
    "SAC",
    "A3C",
    "TRPO",
    "MAPPO",
    "COMA",
    "IPPO",
    "VDN",
    "MADDPG",
    "IQL",
    "MATD3",
}

CATASTROPHIC_REWARD_FLOOR = -100.0


def _load_json(path: Path) -> Any:
    """Load a JSON file with a clear path-bound error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _coerce_results(payload: Any) -> list[dict]:
    """Return benchmark result records from supported payload shapes."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "records", "benchmarks"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def discover_benchmark_inputs(results_dir: Path = Path("results")) -> list[Path]:
    """Find benchmark JSON files without triggering any experiment reruns."""
    if not results_dir.exists():
        return []
    return sorted(
        results_dir.glob("benchmark*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_benchmark_results(input_paths: Sequence[Path] | None = None) -> tuple[list[dict], list[str]]:
    """Load one or more benchmark result files."""
    paths = list(input_paths or discover_benchmark_inputs())
    records: list[dict] = []
    loaded: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        file_records = _coerce_results(_load_json(path))
        if file_records:
            records.extend(file_records)
            loaded.append(str(path))
    return records, loaded


def load_quality_records(path: Path | None) -> list[dict]:
    """Load convergence quality records if a report exists."""
    if path is None or not path.exists():
        return []
    payload = _load_json(path)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _quality_summary(quality_records: Iterable[dict]) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate quality records by algorithm and metric."""
    summary: dict[tuple[str, str], dict[str, Any]] = {}
    for record in quality_records:
        algo = str(record.get("algorithm", "unknown"))
        metric = str(record.get("metric", "unknown"))
        key = (algo, metric)
        item = summary.setdefault(
            key,
            {
                "extreme_outlier_count": 0,
                "failed_seeds": set(),
                "outlier_count": 0,
                "quality_records": 0,
            },
        )
        item["quality_records"] += 1
        outlier_count = int(record.get("outlier_count") or 0)
        item["outlier_count"] += outlier_count
        if _quality_record_is_extreme(record):
            item["extreme_outlier_count"] += 1
        if str(record.get("run_status", "success")) != "success":
            item["failed_seeds"].add(str(record.get("seed", "unknown")))

    for item in summary.values():
        item["failed_seed_count"] = len(item["failed_seeds"])
        del item["failed_seeds"]
    return summary


def _quality_record_is_extreme(record: dict) -> bool:
    """Return whether a quality record describes a truly extreme event."""
    if not record.get("severe_outlier"):
        return False
    raw_min = record.get("raw_min")
    raw_max = record.get("raw_max")
    if raw_min is None or raw_max is None:
        return True
    try:
        low = float(raw_min)
        high = float(raw_max)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(low) or not np.isfinite(high):
        return True
    max_abs = max(abs(low), abs(high))
    min_abs = max(min(abs(low), abs(high)), 1e-9)
    metric = str(record.get("metric", "unknown"))
    if metric == "reward" and low <= CATASTROPHIC_REWARD_FLOOR:
        return True
    return max_abs > 1000.0 or max_abs / min_abs >= 100.0


def _raw_metric_is_extreme(metric_name: str, values: np.ndarray) -> bool:
    """Detect domain-scale metric failures without naming algorithms."""
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    if metric_name == "reward" and float(np.nanmin(finite)) <= CATASTROPHIC_REWARD_FLOOR:
        return True
    return _is_span_extreme(finite)


def _seed_failure_count(seeds: Sequence[dict]) -> int:
    """Count failed seeds from benchmark convergence metadata."""
    return len([seed for seed in seeds if str(seed.get("run_status", "success")) != "success"])


def _status_rank(status: str) -> int:
    """Return an ordering rank where larger means more severe."""
    return STATUS_PRIORITY.get(status, 0)


def _overall_status(metric_records: dict[str, dict]) -> str:
    """Pick the most severe metric status for an algorithm."""
    if not metric_records:
        return "insufficient"
    return max(
        (str(record.get("status", "insufficient")) for record in metric_records.values()),
        key=_status_rank,
    )


def _analyze_metric(
    seeds: Sequence[dict],
    metric_name: str,
    spec: dict,
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Analyze one metric across all successful seeds for one algorithm."""
    seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
    raw_parts: list[np.ndarray] = []
    for seed_data in seeds:
        raw_values = _pick_metric_series(seed_data, spec)
        if raw_values.size == 0:
            continue
        raw_parts.append(raw_values)
        if str(seed_data.get("run_status", "success")) != "success":
            continue
        x_values = _build_timestep_axis(seed_data, len(raw_values))
        clean_values, _ = _sanitize_metric_series(raw_values, outlier_policy="none")
        seed_curves.append((x_values, clean_values))

    empty = {
        "status": "insufficient",
        "tail_relative_change": None,
        "tail_slope": None,
        "best_tail_gap": None,
        "volatility": None,
        "extreme_outlier_count": int(quality.get("extreme_outlier_count", 0)),
        "failed_seed_count": int(quality.get("failed_seed_count", 0)),
        "plateau_badness": None,
        "reward_regression_from_initial": None,
        "reward_regression_from_best": None,
        "n_seeds": 0,
    }
    if not seed_curves:
        return empty

    aggregated = _aggregate_metric_by_seed(seed_curves, aggregate="median")
    if aggregated["n_seeds"] == 0:
        return empty
    status = compute_convergence_status(
        aggregated["x"],
        aggregated["median"],
        higher_is_better=bool(spec["higher_is_better"]),
    )
    extreme_count = int(quality.get("extreme_outlier_count", 0))
    if raw_parts:
        raw_all = np.concatenate([np.asarray(part, dtype=float).reshape(-1) for part in raw_parts])
        if _raw_metric_is_extreme(metric_name, raw_all):
            extreme_count += 1
    status_name = str(status["status"])
    if metric_name == "reward" and extreme_count > 0:
        status_name = "catastrophic_outlier"

    return {
        "status": status_name,
        "tail_relative_change": status.get("relative_change"),
        "tail_slope": status.get("tail_slope"),
        "best_tail_gap": status.get("best_tail_gap"),
        "volatility": status.get("tail_volatility"),
        "extreme_outlier_count": extreme_count,
        "failed_seed_count": int(quality.get("failed_seed_count", 0)),
        "plateau_badness": status.get("plateau_badness"),
        "reward_regression_from_initial": status.get("reward_regression_from_initial"),
        "reward_regression_from_best": status.get("reward_regression_from_best"),
        "n_seeds": int(aggregated["n_seeds"]),
    }


def build_failure_matrix(
    results: Sequence[dict],
    *,
    quality_records: Sequence[dict] | None = None,
    algorithms: set[str] | None = None,
) -> dict[str, Any]:
    """Build a serializable convergence failure matrix."""
    selected_algorithms = algorithms or DEFAULT_TARGET_ALGORITHMS
    algo_data = _collect_convergence_data(list(results))
    quality_by_metric = _quality_summary(quality_records or [])
    records: list[dict] = []

    for algo in sorted(selected_algorithms):
        seeds = algo_data.get(algo, [])
        metric_records: dict[str, dict] = {}
        failed_seed_count = _seed_failure_count(seeds)
        extreme_outlier_count = 0
        for metric_name, spec in CONVERGENCE_METRIC_SPECS.items():
            quality = quality_by_metric.get((algo, metric_name), {})
            metric_record = _analyze_metric(seeds, metric_name, spec, quality)
            failed_seed_count = max(failed_seed_count, int(metric_record["failed_seed_count"]))
            extreme_outlier_count += int(metric_record["extreme_outlier_count"])
            metric_records[metric_name] = metric_record

        overall = _overall_status(metric_records)
        requires_event_audit = bool(
            metric_records.get("reward", {}).get("status") == "catastrophic_outlier"
        )
        records.append(
            {
                "algorithm": algo,
                "overall_status": overall,
                "failed_seed_count": failed_seed_count,
                "extreme_outlier_count": extreme_outlier_count,
                "requires_event_audit": requires_event_audit,
                "metrics": metric_records,
            }
        )

    status_counts = Counter(record["overall_status"] for record in records)
    return {
        "schema_version": 1,
        "target_algorithms": sorted(selected_algorithms),
        "records": records,
        "status_counts": dict(status_counts),
    }


def _fmt_float(value: Any) -> str:
    """Format a nullable float for Markdown tables."""
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def _recommendation(record: dict) -> str:
    """Return a conservative next action for one algorithm."""
    if record.get("requires_event_audit"):
        return "先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。"
    status = str(record.get("overall_status"))
    if status == "bad_plateau":
        return "检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。"
    if status == "oscillating":
        return "优先检查方差、advantage/critic 诊断和 eval episode 数量。"
    if status == "diverging":
        return "优先检查学习率、target update、replay 或多智能体非平稳问题。"
    if status == "improving":
        return "继续 targeted rerun 验证趋势，不进入 full 17。"
    if status == "converged_good":
        return "保留为对照，等待 review 确认可进入论文候选图。"
    return "补齐更多 train_logs/result 数据后再判断。"


def render_failure_markdown(matrix: dict[str, Any]) -> str:
    """Render the failure matrix as a decision report."""
    lines = [
        "# Convergence Failure Analysis",
        "",
        "本报告由 `scripts/analyze_convergence_failures.py` 生成，只做诊断与稳定化计划，不触发 full 17 重跑。",
        "",
        "## Failure Matrix",
        "",
        "| Algorithm | Overall | Reward | Failed Seeds | Extreme Outliers | Tail Change | Best-Tail Gap | Recommendation |",
        "|-----------|---------|--------|--------------|------------------|-------------|---------------|----------------|",
    ]
    for record in matrix.get("records", []):
        reward = record.get("metrics", {}).get("reward", {})
        lines.append(
            "| {algorithm} | {overall} | {reward_status} | {failed} | {outliers} | "
            "{tail_change} | {gap} | {recommendation} |".format(
                algorithm=record.get("algorithm", "unknown"),
                overall=record.get("overall_status", "insufficient"),
                reward_status=reward.get("status", "insufficient"),
                failed=record.get("failed_seed_count", 0),
                outliers=record.get("extreme_outlier_count", 0),
                tail_change=_fmt_float(reward.get("tail_relative_change")),
                gap=_fmt_float(reward.get("best_tail_gap")),
                recommendation=_recommendation(record),
            )
        )

    lines.extend(
        [
            "",
            "## Status Definitions",
            "",
            "- `converged_good`: 尾部稳定且接近历史最佳。",
            "- `bad_plateau`: 尾部稳定但明显差于历史最佳或初始表现。",
            "- `oscillating`: 尾部波动超过阈值。",
            "- `diverging`: 尾部方向持续变坏。",
            "- `catastrophic_outlier`: 出现极端异常 reward，先按异常事件排查。",
            "",
            "## Next Step",
            "",
            "先做 targeted rerun 计划：每个异常算法短程 50k steps 验证异常是否复现，再决定是否扩大到 100k/200k；不直接重跑 full 17。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_failure_outputs(matrix: dict[str, Any], *, json_path: Path, markdown_path: Path) -> None:
    """Write JSON and Markdown failure analysis outputs."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_failure_markdown(matrix), encoding="utf-8")


def _parse_paths(values: Sequence[str] | None) -> list[Path] | None:
    """Convert CLI path strings to Path objects."""
    if not values:
        return None
    return [Path(value) for value in values]


def main() -> None:
    """Run convergence failure analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Analyze convergence failures and write convergence_failure_matrix outputs."
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=None,
        help="Benchmark JSON files. Defaults to results/benchmark*.json.",
    )
    parser.add_argument(
        "--quality-report",
        type=str,
        default="figures/convergence_quality_report.json",
        help="Convergence quality report JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/convergence_failure_matrix.json",
        help="Failure matrix JSON output path.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="docs/convergence_failure_analysis.md",
        help="Markdown decision report output path.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Optional algorithm filter. Defaults to the module 12 target set.",
    )
    args = parser.parse_args()

    results, loaded_inputs = load_benchmark_results(_parse_paths(args.input))
    quality_records = load_quality_records(Path(args.quality_report))
    algorithms = set(args.algorithms) if args.algorithms else None
    matrix = build_failure_matrix(results, quality_records=quality_records, algorithms=algorithms)
    matrix["source_inputs"] = loaded_inputs
    matrix["quality_report"] = args.quality_report if Path(args.quality_report).exists() else None
    write_failure_outputs(
        matrix,
        json_path=Path(args.output_json),
        markdown_path=Path(args.output_md),
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
