#!/usr/bin/env python3
"""Build a failure matrix for non-converged benchmark algorithms."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

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
FORMAL_GATE_DEFAULTS = {
    "reward_best_tail_gap_max": 0.10,
    "l1_reward_best_tail_gap_max": 0.15,
    "metric_regression_max": 0.10,
    "catastrophic_outlier_count": 0,
    "failed_seed_count": 0,
}
FORMAL_STATUS_BY_LEVEL = {
    "L1": {"candidate": "l1_candidate", "failed": "failed_l1"},
    "L2": {
        "candidate": "candidate_converged_under_protocol",
        "failed": "excluded_from_formal_claim",
    },
    "L3": {
        "candidate": "verified_converged_under_protocol",
        "failed": "excluded_from_formal_claim",
    },
}
METRIC_FINAL_VALUE_KEYS = {
    "reward": ["final_reward_mean_mean", "final_reward_mean", "reward"],
    "latency": [
        "final_latency_per_task_mean_mean",
        "final_latency_mean_mean",
        "final_latency_mean",
        "latency",
    ],
    "energy": [
        "final_energy_per_task_mean_mean",
        "final_energy_mean_mean",
        "final_energy_mean",
        "energy",
    ],
    "comm_score": ["final_comm_score_mean", "comm_score"],
    "deadline_miss_rate": ["final_deadline_miss_rate_mean", "deadline_miss_rate"],
}
METRIC_DIRECTION = {
    "reward": True,
    "latency": False,
    "energy": False,
    "comm_score": True,
    "deadline_miss_rate": False,
}


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


def _coerce_float_series(series: Any) -> np.ndarray:
    """Convert a JSON-like series into a finite-aware float array."""
    if series is None:
        return np.asarray([], dtype=float)
    if isinstance(series, np.ndarray):
        raw_values = series.reshape(-1).tolist()
    elif isinstance(series, (list, tuple)):
        raw_values = list(series)
    else:
        raw_values = [series]

    converted: list[float] = []
    for value in raw_values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = np.nan
        converted.append(numeric if np.isfinite(numeric) else np.nan)
    return np.asarray(converted, dtype=float)


def compute_tail_metrics(series: Any, higher_is_better: bool) -> dict[str, Any]:
    """Compute protocol tail metrics for one metric series."""
    values = _coerce_float_series(series)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "status": "insufficient",
            "tail_relative_change": None,
            "tail_slope": None,
            "tail_volatility": None,
            "best_tail_gap": None,
        }

    status = compute_convergence_status(
        np.arange(len(values), dtype=float),
        values,
        higher_is_better=higher_is_better,
    )
    return {
        "status": status.get("status"),
        "tail_relative_change": status.get("relative_change"),
        "tail_slope": status.get("tail_slope"),
        "tail_volatility": status.get("tail_volatility"),
        "best_tail_gap": status.get("best_tail_gap"),
        "tail_mean": status.get("tail_mean"),
        "best_value": status.get("best_value"),
        "higher_is_better": higher_is_better,
    }


def _seed_count_from_metadata(run_metadata: dict) -> int:
    """Infer seed count from a result or metadata record."""
    seeds = run_metadata.get("seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes)):
        return len(seeds)
    convergence = run_metadata.get("convergence_by_seed")
    if isinstance(convergence, dict):
        return len(convergence)
    if isinstance(convergence, list):
        return len(convergence)
    try:
        return int(run_metadata.get("n_seeds") or 0)
    except (TypeError, ValueError):
        return 0


def _steps_from_metadata(run_metadata: dict) -> int:
    """Infer training steps from known result fields."""
    for key in ("steps", "total_steps", "train_timesteps_mean", "total_timesteps"):
        try:
            value = int(float(run_metadata.get(key)))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value

    convergence = run_metadata.get("convergence_by_seed")
    seed_records: Iterable[Any]
    if isinstance(convergence, dict):
        seed_records = convergence.values()
    elif isinstance(convergence, list):
        seed_records = convergence
    else:
        seed_records = []
    inferred = 0
    for seed_record in seed_records:
        if not isinstance(seed_record, dict):
            continue
        for key in ("total_timesteps", "steps", "total_steps"):
            try:
                value = int(float(seed_record.get(key)))
            except (TypeError, ValueError):
                continue
            inferred = max(inferred, value)
    return inferred


def classify_evidence_level(run_metadata: dict) -> str:
    """Classify run metadata into L0/L1/L2/L3 evidence levels."""
    explicit = str(run_metadata.get("evidence_level", "")).upper()
    if explicit in {"L0", "L1", "L2", "L3"}:
        return explicit

    steps = _steps_from_metadata(run_metadata)
    seed_count = _seed_count_from_metadata(run_metadata)
    if steps >= 200_000 and seed_count >= 5:
        return "L3"
    if steps >= 100_000 and seed_count >= 3:
        return "L2"
    if steps >= 50_000 and seed_count >= 1:
        return "L1"
    return "L0"


def _pick_final_value(record: dict, metric_name: str) -> float | None:
    """Pick a final scalar metric from a benchmark-like record."""
    for key in METRIC_FINAL_VALUE_KEYS.get(metric_name, [metric_name]):
        value = record.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


def detect_metric_regression(current: dict, baseline: dict) -> dict[str, dict[str, Any]]:
    """Compare current metrics against a baseline using protocol directions."""
    results: dict[str, dict[str, Any]] = {}
    for metric_name, higher_is_better in METRIC_DIRECTION.items():
        current_value = _pick_final_value(current, metric_name)
        baseline_value = _pick_final_value(baseline, metric_name)
        if current_value is None or baseline_value is None:
            results[metric_name] = {
                "current": current_value,
                "baseline": baseline_value,
                "regression_ratio": None,
                "passed": None,
                "higher_is_better": higher_is_better,
            }
            continue

        denom = max(abs(baseline_value), 1e-9)
        if higher_is_better:
            regression_ratio = max(0.0, (baseline_value - current_value) / denom)
        else:
            regression_ratio = max(0.0, (current_value - baseline_value) / denom)
        results[metric_name] = {
            "current": current_value,
            "baseline": baseline_value,
            "regression_ratio": regression_ratio,
            "passed": regression_ratio <= FORMAL_GATE_DEFAULTS["metric_regression_max"],
            "higher_is_better": higher_is_better,
        }
    return results


def _load_protocol_gates(protocol: dict | None) -> dict[str, Any]:
    """Return protocol gate thresholds from a config-like mapping."""
    if not protocol:
        return dict(FORMAL_GATE_DEFAULTS)
    gates = protocol.get("gates") if isinstance(protocol.get("gates"), dict) else protocol
    resolved = dict(FORMAL_GATE_DEFAULTS)
    for key in resolved:
        if key in gates:
            resolved[key] = gates[key]
    return resolved


def _load_protocol(path: Path | None) -> dict[str, Any] | None:
    """Load a YAML protocol/matrix when provided."""
    if path is None or not path.exists() or path.suffix.lower() not in {".yaml", ".yml"}:
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def classify_formal_convergence(
    record: dict,
    protocol: dict | None = None,
    baseline_record: dict | None = None,
) -> dict[str, Any]:
    """Classify one algorithm under the engineering convergence protocol."""
    gates = _load_protocol_gates(protocol)
    target_algorithms = protocol.get("target_algorithms", {}) if protocol else {}
    l1_candidate_algorithms = set(target_algorithms.get("l1_candidates", []))
    evidence_level = classify_evidence_level(record)
    status_names = FORMAL_STATUS_BY_LEVEL.get(evidence_level, FORMAL_STATUS_BY_LEVEL["L1"])
    metrics = record.get("metrics", {})
    reward = metrics.get("reward", {}) if isinstance(metrics, dict) else {}
    best_tail_gap = reward.get("best_tail_gap")
    failed_seed_count = int(record.get("failed_seed_count") or 0)
    catastrophic_count = int(record.get("extreme_outlier_count") or 0)
    reward_status = str(reward.get("status", "insufficient"))

    metric_regressions = (
        detect_metric_regression(record, baseline_record) if baseline_record is not None else {}
    )
    regression_failures = [
        name
        for name, item in metric_regressions.items()
        if item.get("passed") is False
        and float(item.get("regression_ratio") or 0.0) > float(gates["metric_regression_max"])
    ]

    reasons: list[str] = []
    decision = status_names["candidate"]
    if failed_seed_count > int(gates["failed_seed_count"]):
        reasons.append("failed_seed_count above protocol gate")
        decision = status_names["failed"]
    if catastrophic_count > int(gates["catastrophic_outlier_count"]):
        reasons.append("catastrophic outlier requires event audit")
        decision = "needs_event_audit" if evidence_level == "L1" else status_names["failed"]
    if reward_status == "catastrophic_outlier":
        reasons.append("reward classified as catastrophic_outlier")
        decision = "needs_event_audit" if evidence_level == "L1" else status_names["failed"]
    if best_tail_gap is None:
        reasons.append("missing reward best_tail_gap")
        decision = status_names["failed"] if evidence_level != "L1" else "failed_l1"
    gap_gate = (
        float(gates["l1_reward_best_tail_gap_max"])
        if evidence_level == "L1"
        else float(gates["reward_best_tail_gap_max"])
    )
    if best_tail_gap is not None and float(best_tail_gap) > gap_gate:
        reasons.append("reward best_tail_gap above protocol gate")
        if evidence_level == "L1" and decision not in {"needs_event_audit", "failed_l1"}:
            decision = "needs_single_variable_fix"
        elif evidence_level != "L1":
            decision = status_names["failed"]
    if (
        evidence_level == "L1"
        and l1_candidate_algorithms
        and record.get("algorithm") not in l1_candidate_algorithms
        and reward_status != "converged_good"
        and decision == "l1_candidate"
    ):
        reasons.append("algorithm is not in the L1 candidate group")
        decision = "needs_single_variable_fix"
    if regression_failures:
        reasons.append("metric regression above protocol gate: " + ", ".join(regression_failures))
        decision = status_names["failed"] if evidence_level != "L1" else "failed_l1"
    if evidence_level == "L0":
        reasons.append("offline diagnosis cannot carry convergence validation")
        decision = "excluded_from_formal_claim"

    if not reasons:
        if evidence_level == "L1":
            reasons.append("L1 screening passed; formal validation still requires L2/L3")
        elif evidence_level == "L2":
            reasons.append("L2 candidate gate passed; L3 still required for final claim")
        else:
            reasons.append("L3 gate passed under the formal protocol")

    return {
        "algorithm": record.get("algorithm", "unknown"),
        "evidence_level": evidence_level,
        "decision": decision,
        "reward_status": reward_status,
        "best_tail_gap": best_tail_gap,
        "tail_relative_change": reward.get("tail_relative_change"),
        "tail_slope": reward.get("tail_slope"),
        "tail_volatility": reward.get("volatility") or reward.get("tail_volatility"),
        "failed_seed_count": failed_seed_count,
        "catastrophic_outlier_count": catastrophic_count,
        "metric_regressions": metric_regressions,
        "reason": "; ".join(reasons),
    }


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


def _records_by_algorithm(records: Sequence[dict]) -> dict[str, dict]:
    """Index benchmark-like records by algorithm."""
    return {
        str(record.get("algorithm", "unknown")): record
        for record in records
        if isinstance(record, dict)
    }


def build_formal_decision_report(
    matrix: dict[str, Any],
    *,
    evidence_level: str,
    baseline_records: Sequence[dict] | None = None,
    protocol: dict | None = None,
) -> dict[str, Any]:
    """Build a formal protocol decision report from a failure matrix."""
    baseline_by_algorithm = _records_by_algorithm(baseline_records or [])
    decisions: list[dict[str, Any]] = []
    for record in matrix.get("records", []):
        enriched = dict(record)
        enriched["evidence_level"] = evidence_level
        baseline_record = baseline_by_algorithm.get(str(record.get("algorithm")))
        decisions.append(
            classify_formal_convergence(
                enriched,
                protocol=protocol,
                baseline_record=baseline_record,
            )
        )

    decision_counts = Counter(item["decision"] for item in decisions)
    return {
        "schema_version": 1,
        "evidence_level": evidence_level,
        "records": decisions,
        "decision_counts": dict(decision_counts),
        "protocol_scope": "engineering_validation",
    }


def render_formal_decision_markdown(report: dict[str, Any]) -> str:
    """Render formal protocol decisions as Markdown."""
    evidence_level = str(report.get("evidence_level", "L0")).upper()
    if evidence_level == "L1":
        title = "# L1 Baseline Convergence Assessment"
        scope = (
            "L1 is a 50k single-seed screening pass. It can only produce candidate, "
            "failure, event-audit, or single-variable-fix decisions."
        )
    elif evidence_level == "L2":
        title = "# L2 Candidate Convergence Report"
        scope = (
            "L2 is a 100k multi-seed candidate gate. Passing L2 is not enough for the "
            "main convergence claim."
        )
    elif evidence_level == "L3":
        title = "# L3 Verified Convergence Report"
        scope = (
            "L3 is the 200k multi-seed formal engineering gate. Only rows that pass this "
            "level may use `verified_converged_under_protocol`."
        )
    else:
        title = "# Formal Convergence Decision Report"
        scope = "L0/offline diagnosis can locate issues but cannot validate convergence."

    lines = [
        title,
        "",
        f"> evidence level: `{evidence_level}` | scope: engineering protocol",
        "",
        scope,
        "",
        "| Algorithm | Decision | Reward Status | Failed Seeds | Catastrophic Outliers | Best-Tail Gap | Tail Change | Reason |",
        "|-----------|----------|---------------|--------------|-----------------------|---------------|-------------|--------|",
    ]
    for record in report.get("records", []):
        lines.append(
            "| {algorithm} | {decision} | {reward_status} | {failed} | {outliers} | "
            "{gap} | {tail} | {reason} |".format(
                algorithm=record.get("algorithm", "unknown"),
                decision=record.get("decision", "unknown"),
                reward_status=record.get("reward_status", "unknown"),
                failed=record.get("failed_seed_count", 0),
                outliers=record.get("catastrophic_outlier_count", 0),
                gap=_fmt_float(record.get("best_tail_gap")),
                tail=_fmt_float(record.get("tail_relative_change")),
                reason=record.get("reason", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Protocol Boundaries",
            "",
            "- 50k single-seed data is screening evidence only.",
            "- L2 requires 100k steps across seeds 42, 43, and 44.",
            "- L3 requires 200k steps across seeds 42, 43, 44, 45, and 46.",
            "- Algorithms that have not passed L3 stay out of the paper main convergence figure.",
        ]
    )
    return "\n".join(lines) + "\n"


def _log_summary(path: Path) -> str:
    """Return a short diagnostic summary from a log file."""
    if not path.exists():
        return "missing"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unreadable"
    lowered = text.lower()
    flags = [flag for flag in ("traceback", "nan", "inf", "error") if flag in lowered]
    if flags:
        return "contains " + ", ".join(flags[:4])
    stripped = " ".join(text.split())
    return stripped[:120] if stripped else "empty"


def _find_artifact_dir(algorithm: str) -> Path | None:
    """Find the newest local artifact directory for an algorithm."""
    candidates = sorted(
        Path("experiments").glob(f"**/artifacts/{algorithm}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _first_bad_event(seed_data: dict, metric_name: str = "reward") -> dict[str, Any]:
    """Locate the first protocol-relevant bad event in a seed metric series."""
    spec = CONVERGENCE_METRIC_SPECS[metric_name]
    values = _pick_metric_series(seed_data, spec)
    if values.size == 0:
        return {
            "first_bad_timestep": None,
            "raw_min": None,
            "previous_value": None,
            "next_value": None,
        }
    x_values = _build_timestep_axis(seed_data, len(values))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "first_bad_timestep": None,
            "raw_min": None,
            "previous_value": None,
            "next_value": None,
        }
    bad_indexes = np.where(values <= CATASTROPHIC_REWARD_FLOOR)[0]
    index = int(bad_indexes[0]) if bad_indexes.size else int(np.nanargmin(values))
    previous_value = values[index - 1] if index > 0 else None
    next_value = values[index + 1] if index + 1 < len(values) else None
    return {
        "first_bad_timestep": _fmt_float(x_values[index]) if index < len(x_values) else None,
        "raw_min": _fmt_float(values[index]),
        "previous_value": _fmt_float(previous_value),
        "next_value": _fmt_float(next_value),
    }


def _suspected_layer_for_algorithm(algorithm: str, stderr_summary: str) -> str:
    """Return a conservative suspected layer from logs and algorithm family."""
    lowered = stderr_summary.lower()
    if "traceback" in lowered or "nan" in lowered or "inf" in lowered:
        return "unknown"
    if algorithm in {"IQL", "VDN"}:
        return "replay"
    if algorithm in {"IPPO", "MADDPG"}:
        return "optimizer"
    return "unknown"


def _source_classification(suspected_layer: str) -> str:
    """Map event suspected layer into formal audit source classes."""
    if suspected_layer in {"replay", "optimizer"}:
        return "training_instability"
    if suspected_layer == "logging":
        return "evaluation_noise"
    if suspected_layer in {"env", "reward", "action_mask"}:
        return "environment_metric_bug"
    return "unknown"


def build_event_audit(
    results: Sequence[dict],
    *,
    algorithms: set[str],
) -> dict[str, Any]:
    """Build an event audit for catastrophic or unstable algorithms."""
    algo_data = _collect_convergence_data(list(results))
    records: list[dict[str, Any]] = []
    for algorithm in sorted(algorithms):
        seeds = algo_data.get(algorithm, [])
        seed_data = seeds[0] if seeds else {}
        if seeds:
            for candidate in seeds:
                event = _first_bad_event(candidate)
                raw_min = event.get("raw_min")
                try:
                    if raw_min is not None and float(raw_min) <= CATASTROPHIC_REWARD_FLOOR:
                        seed_data = candidate
                        break
                except (TypeError, ValueError):
                    continue
        event = _first_bad_event(seed_data) if seed_data else {
            "first_bad_timestep": None,
            "raw_min": None,
            "previous_value": None,
            "next_value": None,
        }
        artifact_dir = _find_artifact_dir(algorithm)
        stderr_summary = _log_summary(artifact_dir / "stderr.log") if artifact_dir else "missing"
        stdout_summary = _log_summary(artifact_dir / "stdout.log") if artifact_dir else "missing"
        suspected_layer = _suspected_layer_for_algorithm(algorithm, stderr_summary)
        source = _source_classification(suspected_layer)
        decision = "allow_l2_l3_gate" if source in {"training_instability", "evaluation_noise"} else "needs_escalation"
        records.append(
            {
                "algorithm": algorithm,
                "seed": seed_data.get("seed") if seed_data else None,
                **event,
                "stderr_summary": stderr_summary,
                "stdout_summary": stdout_summary,
                "suspected_layer": suspected_layer,
                "source_classification": source,
                "decision": decision,
                "artifact_dir": str(artifact_dir) if artifact_dir else None,
            }
        )

    return {
        "schema_version": 1,
        "records": records,
        "needs_escalation": any(record["decision"] == "needs_escalation" for record in records),
    }


def render_event_audit_markdown(audit: dict[str, Any]) -> str:
    """Render an event audit report."""
    lines = [
        "# Convergence Event Audit",
        "",
        "> Last updated: 2026-05-04 | plan.md version: slimming-plan-v3 | phase: module 14 Step 5",
        "",
        "This audit reads existing local artifacts only. It does not run training and does not enable stability overrides.",
        "",
        "| Algorithm | Seed | First Bad Timestep | Raw Min | Previous Value | Next Value | Suspected Layer | Source | Decision |",
        "|-----------|------|--------------------|---------|----------------|------------|-----------------|--------|----------|",
    ]
    for record in audit.get("records", []):
        lines.append(
            "| {algorithm} | {seed} | {first_bad_timestep} | {raw_min} | {previous_value} | "
            "{next_value} | {suspected_layer} | {source} | {decision} |".format(
                algorithm=record.get("algorithm", "unknown"),
                seed=record.get("seed", "unknown"),
                first_bad_timestep=record.get("first_bad_timestep") or "NA",
                raw_min=record.get("raw_min") or "NA",
                previous_value=record.get("previous_value") or "NA",
                next_value=record.get("next_value") or "NA",
                suspected_layer=record.get("suspected_layer", "unknown"),
                source=record.get("source_classification", "unknown"),
                decision=record.get("decision", "unknown"),
            )
        )

    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            "- `environment_metric_bug` or `unknown` stops L2/L3 and requires `NEEDS_ESCALATION`.",
            "- `training_instability` or `evaluation_noise` may proceed to L2/L3 gates, but still cannot enter the paper main claim without L3.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        help="Optional protocol config or Markdown path for formal convergence decisions.",
    )
    parser.add_argument(
        "--evidence-level",
        choices=["L1", "L2", "L3"],
        default=None,
        help="Classify outputs under a formal evidence level.",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Baseline JSON used for metric regression checks.",
    )
    parser.add_argument(
        "--formal-output",
        type=str,
        default=None,
        help="Optional JSON output for formal decisions or event audit.",
    )
    args = parser.parse_args()

    results, loaded_inputs = load_benchmark_results(_parse_paths(args.input))
    quality_records = load_quality_records(Path(args.quality_report))
    algorithms = set(args.algorithms) if args.algorithms else None
    protocol = _load_protocol(Path(args.protocol) if args.protocol else None)
    matrix = build_failure_matrix(results, quality_records=quality_records, algorithms=algorithms)
    matrix["source_inputs"] = loaded_inputs
    matrix["quality_report"] = args.quality_report if Path(args.quality_report).exists() else None
    formal_path = Path(args.formal_output) if args.formal_output else None
    if formal_path and "event_audit" in formal_path.name:
        audit_algorithms = algorithms or {"IQL", "VDN", "IPPO", "MADDPG"}
        audit = build_event_audit(results, algorithms=audit_algorithms)
        formal_path.parent.mkdir(parents=True, exist_ok=True)
        formal_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        event_md = Path("docs/convergence_event_audit.md")
        event_md.write_text(render_event_audit_markdown(audit), encoding="utf-8")

    if args.evidence_level:
        baseline_records: list[dict] = []
        if args.baseline_json:
            baseline_records = _coerce_results(_load_json(Path(args.baseline_json)))
        formal_report = build_formal_decision_report(
            matrix,
            evidence_level=args.evidence_level,
            baseline_records=baseline_records,
            protocol=protocol,
        )
        if formal_path:
            formal_path.parent.mkdir(parents=True, exist_ok=True)
            formal_path.write_text(
                json.dumps(formal_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(matrix, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(
            render_formal_decision_markdown(formal_report),
            encoding="utf-8",
        )
    else:
        write_failure_outputs(
            matrix,
            json_path=Path(args.output_json),
            markdown_path=Path(args.output_md),
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    if formal_path:
        print(f"Wrote {formal_path}")


if __name__ == "__main__":
    main()
