#!/usr/bin/env python3
"""
Plot Benchmark Results — 从 JSON 生成对比图表

用法:
    python scripts/plot_results.py --input results/benchmark.json --output figures/
    python scripts/plot_results.py --input results/benchmark.json --output figures/ --format pdf
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt


# 配色方案
ALGO_COLORS = {
    "GRPO": "#e74c3c",
    "PPO": "#3498db",
    "SAC": "#2ecc71",
    "DDQN": "#f39c12",
    "DDPG": "#9b59b6",
    "TD3": "#1abc9c",
    "A3C": "#e67e22",
    "TRPO": "#34495e",
    "SimPO": "#e91e63",
    "MAPPO": "#00bcd4",
    "QMIX": "#795548",
    "COMA": "#FF5722",
    "IPPO": "#607D8B",
    "VDN": "#8BC34A",
    "MADDPG": "#FF9800",
    "IQL": "#03A9F4",
    "MATD3": "#9C27B0",
}

CONVERGENCE_METRIC_SPECS = {
    "reward": {
        "title": "Reward",
        "ylabel": "Reward",
        "aliases": ["eval/reward_mean", "eval_eval/reward_mean"],
        "higher_is_better": True,
    },
    "social_welfare": {
        "title": "Social Welfare",
        "ylabel": "Welfare",
        "aliases": ["eval/social_welfare_mean", "eval_eval/social_welfare_mean"],
        "higher_is_better": True,
    },
    "latency": {
        "title": "Latency / Task",
        "ylabel": "Latency / Task",
        "aliases": ["eval/e2e_latency_mean", "eval/latency_mean", "eval_eval/e2e_latency_mean", "eval_eval/latency_mean"],
        "higher_is_better": False,
    },
    "latency_p95": {
        "title": "P95 Latency",
        "ylabel": "Latency P95",
        "aliases": ["eval/e2e_latency_p95", "eval_eval/e2e_latency_p95"],
        "higher_is_better": False,
    },
    "deadline_miss_rate": {
        "title": "Deadline Miss Rate",
        "ylabel": "Miss Rate",
        "aliases": ["eval/deadline_miss_rate", "eval_eval/deadline_miss_rate"],
        "higher_is_better": False,
    },
    "throughput": {
        "title": "Throughput",
        "ylabel": "Tasks / Step",
        "aliases": ["eval/throughput_tasks_per_step", "eval_eval/throughput_tasks_per_step"],
        "higher_is_better": True,
    },
    "energy": {
        "title": "Energy / Task",
        "ylabel": "Energy / Task",
        "aliases": ["eval/energy_per_task_mean", "eval/energy_mean", "eval_eval/energy_per_task_mean", "eval_eval/energy_mean"],
        "higher_is_better": False,
    },
    "comm_score": {
        "title": "Comm Score",
        "ylabel": "Comm Score",
        "aliases": ["eval/comm_score", "eval_eval/comm_score"],
        "higher_is_better": True,
    },
    "agent_fairness": {
        "title": "Agent Reward Jain Fairness",
        "ylabel": "Jain Index",
        "aliases": ["eval/agent_reward_jain_mean", "eval_eval/agent_reward_jain_mean"],
        "higher_is_better": True,
    },
    "constraint_violation": {
        "title": "Constraint Violation Rate",
        "ylabel": "Violation Rate",
        "aliases": ["eval/constraint/any_violation_mean", "eval_eval/constraint/any_violation_mean"],
        "higher_is_better": False,
    },
    "queue_wait": {
        "title": "Queue Wait",
        "ylabel": "Queue Wait",
        "aliases": ["eval/queue_wait_mean_mean", "eval_eval/queue_wait_mean_mean"],
        "higher_is_better": False,
    },
    "offload_ratio": {
        "title": "Offload Ratio",
        "ylabel": "Offload Ratio",
        "aliases": ["eval/offload_ratio_mean_mean", "eval/action/individual_offload_mean_mean", "eval_eval/offload_ratio_mean_mean"],
        "higher_is_better": True,
    },
    "price": {
        "title": "Dynamic Price Mean",
        "ylabel": "Price",
        "aliases": ["eval/pricing/price_mean", "eval_eval/pricing/price_mean"],
        "higher_is_better": False,
    },
}


DEFAULT_EVAL_INTERVAL = 1000


def load_results(path: str) -> List[Dict[str, Any]]:
    """加载 benchmark JSON 结果"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_float(value):
    try:
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _fmt_or_na(value, fmt: str) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    return format(numeric, fmt)


def _pick_first_numeric(record: Dict[str, Any], keys: List[str]) -> float | None:
    for key in keys:
        value = _to_float(record.get(key))
        if value is not None:
            return value
    return None


def build_reward_breakdown_series(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Collect reward breakdown series from mainline-A result records."""
    series: Dict[str, List[float]] = {}
    for record in results:
        breakdown = record.get("reward_breakdown") or record.get("mainline_a_reward_components") or {}
        if not isinstance(breakdown, dict):
            continue
        for key, value in breakdown.items():
            numeric = _to_float(value)
            if numeric is not None:
                series.setdefault(str(key), []).append(numeric)
    return series


def build_price_state_series(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Collect price and state-coupling series."""
    keys = ["price", "queue_pressure", "channel_quality", "migration_risk"]
    return {
        key: [value for value in (_to_float(record.get(key)) for record in results) if value is not None]
        for key in keys
    }


def build_dual_variable_series(results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Collect dual-variable time series."""
    series: Dict[str, List[float]] = {}
    for record in results:
        duals = record.get("dual_variables", {})
        if isinstance(duals, dict):
            for key, value in duals.items():
                numeric = _to_float(value)
                if numeric is not None:
                    series.setdefault(str(key), []).append(numeric)
    return series


def build_oracle_gap_table(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build oracle-gap table rows."""
    rows = []
    for record in results:
        if "optimality_gap" in record or "oracle_gap" in record:
            rows.append(
                {
                    "algorithm": record.get("algorithm", "unknown"),
                    "optimality_gap": record.get("optimality_gap", record.get("oracle_gap")),
                    "constraint_violation": record.get("constraint_violation", 0.0),
                }
            )
    return rows


def build_ood_generalization_table(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build OOD generalization table rows."""
    rows = []
    for record in results:
        if record.get("stage") == "N3" or record.get("ood"):
            rows.append(
                {
                    "algorithm": record.get("algorithm", "unknown"),
                    "social_welfare": record.get("social_welfare"),
                    "average_latency": record.get("average_latency"),
                    "constraint_violation_rate": record.get("constraint_violation_rate"),
                }
            )
    return rows


def _pick_latency_metric(record: Dict[str, Any]) -> float | None:
    return _pick_first_numeric(
        record,
        [
            "final_latency_per_task_mean_mean",
            "final_latency_per_task_mean",
            "final_latency_total_mean_mean",
            "final_latency_mean_mean",
            "final_latency_total_mean",
            "final_latency_mean",
        ],
    )


def _pick_energy_metric(record: Dict[str, Any]) -> float | None:
    return _pick_first_numeric(
        record,
        [
            "final_energy_per_task_mean_mean",
            "final_energy_per_task_mean",
            "final_energy_total_mean_mean",
            "final_energy_mean_mean",
            "final_energy_total_mean",
            "final_energy_mean",
        ],
    )


def plot_reward_comparison(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """柱状图: 各算法平均奖励对比 (按环境分组)"""
    # 按环境分组
    env_groups: Dict[str, List[Dict]] = {}
    for r in results:
        env_name = r.get("environment", "unknown")
        env_groups.setdefault(env_name, []).append(r)

    for env_name, group in env_groups.items():
        algos = []
        rewards = []
        stds = []
        colors = []

        for r in group:
            algo = r["algorithm"]
            # 多 seed 取 mean_mean，单 seed 取 final_reward_mean
            rm = _to_float(r.get("final_reward_mean_mean", r.get("final_reward_mean", 0)))
            rs = _to_float(r.get("final_reward_mean_std", r.get("final_reward_std", 0)))
            if rm is None:
                continue
            algos.append(algo)
            rewards.append(rm)
            stds.append(0.0 if rs is None else rs)
            colors.append(ALGO_COLORS.get(algo, "#95a5a6"))

        if not algos:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(algos))
        bars = ax.bar(x, rewards, yerr=stds, capsize=5,
                      color=colors, edgecolor="white", linewidth=0.8)

        ax.set_ylabel("Mean Reward", fontsize=12)
        ax.set_title(f"Benchmark: {env_name}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # 在柱顶标注数值
        for bar, val in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        safe_env = env_name.replace("/", "_")
        fig.savefig(output_dir / f"reward_comparison_{safe_env}.{fmt}", dpi=150)
        plt.close(fig)


def plot_training_time(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """柱状图: 训练耗时对比"""
    algos = []
    times = []
    colors = []

    for r in results:
        algo = r["algorithm"]
        tm = _to_float(r.get("train_time_seconds_mean", r.get("train_time_seconds", 0)))
        if tm is None:
            continue
        algos.append(algo)
        times.append(tm)
        colors.append(ALGO_COLORS.get(algo, "#95a5a6"))

    if not algos:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(algos))
    ax.bar(x, times, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Training Time (s)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"training_time.{fmt}", dpi=150)
    plt.close(fig)


def plot_latency_energy(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """散点图: Latency/task vs Energy/task (气泡大小 = reward)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in results:
        algo = r["algorithm"]
        lat = _pick_latency_metric(r)
        eng = _pick_energy_metric(r)
        rew = _to_float(r.get("final_reward_mean_mean", r.get("final_reward_mean", 0)))
        if lat is None or eng is None or rew is None:
            continue
        color = ALGO_COLORS.get(algo, "#95a5a6")

        # 气泡大小: abs(reward) 映射到 [50, 500]
        size = max(50, min(500, abs(rew) * 200 + 50))
        ax.scatter(lat, eng, s=size, c=color, alpha=0.7, edgecolors="white",
                   linewidth=1.5, label=algo, zorder=3)
        ax.annotate(algo, (lat, eng), fontsize=9, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points")

    ax.set_xlabel("Mean Latency per Task", fontsize=12)
    ax.set_ylabel("Mean Energy per Task", fontsize=12)
    ax.set_title("Latency vs Energy per Task (bubble size ∝ |reward|)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    if ax.collections:
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / f"latency_vs_energy.{fmt}", dpi=150)
    plt.close(fig)


def plot_summary_table(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """表格图: 所有指标汇总"""
    # 收集列
    columns = ["Algorithm", "Reward", "Latency/task", "Energy/task", "Time (s)"]
    rows = []
    for r in results:
        algo = r["algorithm"]
        rew = r.get("final_reward_mean_mean", r.get("final_reward_mean", 0))
        lat = _pick_latency_metric(r)
        eng = _pick_energy_metric(r)
        tm = r.get("train_time_seconds_mean", r.get("train_time_seconds", 0))
        rows.append(
            [
                algo,
                _fmt_or_na(rew, ".4f"),
                _fmt_or_na(lat, ".2f"),
                _fmt_or_na(eng, ".2f"),
                _fmt_or_na(tm, ".1f"),
            ]
        )

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center",
                     cellLoc="center", colColours=["#3498db"] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 表头样式
    for j in range(len(columns)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Benchmark Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / f"summary_table.{fmt}", dpi=150)
    plt.close(fig)


def _json_number(value: Any) -> float | None:
    """Return a JSON-safe finite float, or None."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _series_to_float_array(series: Any) -> np.ndarray:
    """Convert a scalar-like series to float array, coercing bad values to NaN."""
    if series is None:
        return np.asarray([], dtype=float)
    if isinstance(series, np.ndarray):
        raw_values = series.reshape(-1).tolist()
    elif isinstance(series, (list, tuple)):
        raw_values = list(series)
    else:
        return np.asarray([], dtype=float)

    converted = []
    for value in raw_values:
        numeric = _json_number(value)
        converted.append(np.nan if numeric is None else numeric)
    return np.asarray(converted, dtype=float)


def _pick_metric_series(seed_data: dict, spec: dict) -> np.ndarray:
    """Pick the first metric alias present in seed data."""
    for alias in spec["aliases"]:
        if alias in seed_data:
            return _series_to_float_array(seed_data.get(alias))
    return np.asarray([], dtype=float)


def _build_timestep_axis(seed_data: dict, length: int) -> np.ndarray:
    """Build an x-axis aligned with a convergence series."""
    if length <= 0:
        return np.asarray([], dtype=float)

    for key in ("eval_steps", "timesteps"):
        timesteps = _series_to_float_array(seed_data.get(key))
        if len(timesteps) >= length:
            aligned = timesteps[:length]
            if np.all(np.isfinite(aligned)):
                return aligned

    eval_interval = _json_number(seed_data.get("effective_eval_interval"))
    if eval_interval is None or eval_interval <= 0:
        eval_interval = _json_number(seed_data.get("eval_interval"))
    if eval_interval is None or eval_interval <= 0:
        eval_interval = DEFAULT_EVAL_INTERVAL
    return np.arange(length, dtype=float) * float(eval_interval)


def _sanitize_metric_series(
    values: np.ndarray,
    *,
    outlier_policy: str = "winsorize",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[np.ndarray, dict]:
    """Clean one metric series and return quality statistics."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    clean = raw.copy()
    finite_mask = np.isfinite(raw)
    finite_values = raw[finite_mask]

    stats = {
        "raw_count": int(raw.size),
        "finite_count": int(finite_values.size),
        "nan_count": int(raw.size - finite_values.size),
        "raw_min": _json_number(np.nanmin(finite_values)) if finite_values.size else None,
        "raw_max": _json_number(np.nanmax(finite_values)) if finite_values.size else None,
        "clean_min": None,
        "clean_max": None,
        "outlier_count": 0,
        "outlier_ratio": 0.0,
        "outlier_policy": outlier_policy,
    }

    clean[~finite_mask] = np.nan
    if finite_values.size and outlier_policy != "none":
        if outlier_policy == "winsorize":
            lower = float(np.nanquantile(finite_values, lower_quantile))
            upper = float(np.nanquantile(finite_values, upper_quantile))
            outlier_mask = finite_mask & ((raw < lower) | (raw > upper))
            clean[finite_mask] = np.clip(raw[finite_mask], lower, upper)
        elif outlier_policy == "iqr-mask":
            q1 = float(np.nanquantile(finite_values, 0.25))
            q3 = float(np.nanquantile(finite_values, 0.75))
            iqr = q3 - q1
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            outlier_mask = finite_mask & ((raw < lower) | (raw > upper))
            clean[outlier_mask] = np.nan
        else:
            raise ValueError(f"Unsupported outlier policy: {outlier_policy}")
        stats["outlier_count"] = int(np.count_nonzero(outlier_mask))
        stats["outlier_ratio"] = (
            float(stats["outlier_count"] / finite_values.size) if finite_values.size else 0.0
        )

    clean_finite = clean[np.isfinite(clean)]
    if clean_finite.size:
        stats["clean_min"] = _json_number(np.nanmin(clean_finite))
        stats["clean_max"] = _json_number(np.nanmax(clean_finite))
    return clean, stats


def _is_span_extreme(values: np.ndarray) -> bool:
    """Detect extreme numeric spread without naming specific algorithms."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return False
    median_abs = float(np.nanmedian(np.abs(finite)))
    max_abs = float(np.nanmax(np.abs(finite)))
    if median_abs <= 1e-12:
        return max_abs > 1e3
    return max_abs / median_abs >= 100.0


def _write_convergence_quality_report(report: list[dict], output_dir: Path) -> None:
    """Write convergence quality records as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "convergence_quality_report.json"
    md_path = output_dir / "convergence_quality_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        "# Convergence Quality Report",
        "",
        "| Algorithm | Seed | Metric | Evidence | Status | Policy | Outliers | Ratio | Notes |",
        "|-----------|------|--------|----------|--------|--------|----------|-------|-------|",
    ]
    for record in report:
        notes = []
        if record.get("skipped_from_clean_plot"):
            notes.append(record.get("skip_reason", "skipped"))
        if record.get("excluded_from_clean_plot"):
            notes.append(record.get("reason", "excluded"))
        if record.get("severe_outlier"):
            notes.append("severe_outlier")
        note_text = "; ".join(str(n) for n in notes) if notes else "-"
        ratio = record.get("outlier_ratio")
        ratio_text = "NA" if ratio is None else f"{float(ratio):.3f}"
        lines.append(
            "| {algorithm} | {seed} | {metric} | {evidence_level} | {run_status} | "
            "{outlier_policy} | {outlier_count} | {ratio} | {notes} |".format(
                algorithm=record.get("algorithm", "unknown"),
                seed=record.get("seed", "unknown"),
                metric=record.get("metric", "unknown"),
                evidence_level=record.get("evidence_level", "NA"),
                run_status=record.get("run_status", "unknown"),
                outlier_policy=record.get("outlier_policy", "none"),
                outlier_count=record.get("outlier_count", 0),
                ratio=ratio_text,
                notes=note_text,
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_metric_by_seed(
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
    *,
    aggregate: str = "median",
) -> dict:
    """Interpolate seed curves onto a common grid and aggregate robustly."""
    if aggregate not in {"median", "mean"}:
        raise ValueError(f"Unsupported aggregate: {aggregate}")

    valid_curves = []
    positive_steps = []
    for x_values, y_values in seed_curves:
        x = np.asarray(x_values, dtype=float).reshape(-1)
        y = np.asarray(y_values, dtype=float).reshape(-1)
        n = min(len(x), len(y))
        if n < 2:
            continue
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        unique_x, unique_idx = np.unique(x, return_index=True)
        x = unique_x
        y = y[unique_idx]
        if len(x) < 2:
            continue
        diffs = np.diff(x)
        positive_steps.extend(diffs[diffs > 0].tolist())
        valid_curves.append((x, y))

    empty = {
        "x": np.asarray([], dtype=float),
        "median": np.asarray([], dtype=float),
        "q25": np.asarray([], dtype=float),
        "q75": np.asarray([], dtype=float),
        "mean": np.asarray([], dtype=float),
        "std": np.asarray([], dtype=float),
        "n_seeds": 0,
    }
    if not valid_curves:
        return empty

    start = max(float(x[0]) for x, _ in valid_curves)
    end = min(float(x[-1]) for x, _ in valid_curves)
    if end < start:
        start = min(float(x[0]) for x, _ in valid_curves)
        end = max(float(x[-1]) for x, _ in valid_curves)

    step = float(np.nanmedian(positive_steps)) if positive_steps else DEFAULT_EVAL_INTERVAL
    if not np.isfinite(step) or step <= 0:
        step = DEFAULT_EVAL_INTERVAL
    if math.isclose(start, end):
        grid = np.asarray([start], dtype=float)
    else:
        grid = np.arange(start, end + step * 0.5, step, dtype=float)
        if grid.size < 2:
            grid = np.linspace(start, end, num=2, dtype=float)

    interpolated = []
    for x, y in valid_curves:
        interpolated.append(np.interp(grid, x, y))
    arr = np.vstack(interpolated)
    return {
        "x": grid,
        "median": np.nanmedian(arr, axis=0),
        "q25": np.nanquantile(arr, 0.25, axis=0),
        "q75": np.nanquantile(arr, 0.75, axis=0),
        "mean": np.nanmean(arr, axis=0),
        "std": np.nanstd(arr, axis=0),
        "n_seeds": int(arr.shape[0]),
    }


def compute_convergence_status(
    x: np.ndarray,
    y: np.ndarray,
    *,
    higher_is_better: bool,
    window_fraction: float = 0.10,
    rel_change_threshold: float = 0.05,
    volatility_threshold: float = 0.10,
    best_gap_threshold: float = 0.10,
    min_points: int = 5,
) -> dict:
    """Classify convergence using direction, tail stability, and bad-plateau checks."""
    x_values = np.asarray(x, dtype=float).reshape(-1)
    values = np.asarray(y, dtype=float).reshape(-1)
    n = min(len(x_values), len(values))
    if n:
        x_values = x_values[:n]
        values = values[:n]
        mask = np.isfinite(x_values) & np.isfinite(values)
        x_values = x_values[mask]
        values = values[mask]
    else:
        x_values = np.asarray([], dtype=float)
        values = np.asarray([], dtype=float)
    if len(values) < min_points:
        return {
            "status": "insufficient",
            "tail_mean": None,
            "prev_tail_mean": None,
            "relative_change": None,
            "tail_volatility": None,
            "tail_slope": None,
            "best_value": None,
            "initial_value": None,
            "best_tail_gap": None,
            "plateau_badness": None,
            "reward_regression_from_initial": None,
            "reward_regression_from_best": None,
            "higher_is_better": higher_is_better,
        }

    window = max(2, int(math.ceil(len(values) * window_fraction)))
    if len(values) < window * 2:
        window = max(1, len(values) // 2)
    tail = values[-window:]
    prev_tail = values[-2 * window : -window]
    tail_mean = float(np.nanmean(tail))
    prev_tail_mean = float(np.nanmean(prev_tail))
    denom = max(abs(prev_tail_mean), 1e-9)
    relative_change = float((tail_mean - prev_tail_mean) / denom)
    volatility_denom = max(abs(tail_mean), 1e-9)
    tail_volatility = float(np.nanstd(tail) / volatility_denom)
    initial_value = float(values[0])
    best_value = float(np.nanmax(values) if higher_is_better else np.nanmin(values))
    if higher_is_better:
        best_gap_raw = best_value - tail_mean
        initial_gap_raw = initial_value - tail_mean
    else:
        best_gap_raw = tail_mean - best_value
        initial_gap_raw = tail_mean - initial_value
    best_tail_gap = float(max(0.0, best_gap_raw) / max(abs(best_value), 1e-9))
    regression_from_initial = float(max(0.0, initial_gap_raw) / max(abs(initial_value), 1e-9))
    plateau_badness = max(best_tail_gap, regression_from_initial)
    if len(tail) >= 2:
        tail_x = x_values[-window:]
        x_span = float(np.nanmax(tail_x) - np.nanmin(tail_x))
        x_norm = (tail_x - float(np.nanmin(tail_x))) / max(x_span, 1.0)
        tail_slope = float(np.polyfit(x_norm, tail, 1)[0])
    else:
        tail_slope = 0.0

    is_improving = relative_change > 0 if higher_is_better else relative_change < 0
    if _is_span_extreme(values):
        status = "catastrophic_outlier"
    elif not is_improving and abs(relative_change) >= rel_change_threshold:
        status = "diverging"
    elif tail_volatility > volatility_threshold:
        status = "oscillating"
    elif abs(relative_change) < rel_change_threshold:
        if plateau_badness > best_gap_threshold:
            status = "bad_plateau"
        else:
            status = "converged_good"
    else:
        status = "improving"

    return {
        "status": status,
        "tail_mean": tail_mean,
        "prev_tail_mean": prev_tail_mean,
        "relative_change": relative_change,
        "tail_volatility": tail_volatility,
        "tail_slope": tail_slope,
        "best_value": best_value,
        "initial_value": initial_value,
        "best_tail_gap": best_tail_gap,
        "plateau_badness": plateau_badness,
        "reward_regression_from_initial": regression_from_initial,
        "reward_regression_from_best": best_tail_gap,
        "higher_is_better": higher_is_better,
    }


def _robust_ylim(
    values: list[np.ndarray],
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[float, float] | None:
    """Return robust y-limits for publication figures."""
    finite_parts = [np.asarray(v, dtype=float).reshape(-1) for v in values if len(v)]
    if not finite_parts:
        return None
    combined = np.concatenate(finite_parts)
    combined = combined[np.isfinite(combined)]
    if combined.size < 3:
        return None
    low = float(np.nanquantile(combined, lower))
    high = float(np.nanquantile(combined, upper))
    if not np.isfinite(low) or not np.isfinite(high) or math.isclose(low, high):
        return None
    padding = (high - low) * 0.05
    return low - padding, high + padding


def _collect_convergence_data(results: List[Dict]) -> dict[str, list[dict]]:
    """Collect convergence entries from benchmark results or train_logs files."""
    algo_data: dict[str, list[dict]] = {}
    for result in results:
        algo = result.get("algorithm", "unknown")
        seeds = result.get("convergence_by_seed")
        if isinstance(seeds, dict) and seeds:
            entries = []
            for seed_key, seed_data in seeds.items():
                if isinstance(seed_data, dict):
                    entry = dict(seed_data)
                    entry.setdefault("seed", seed_key)
                    entry.setdefault("algorithm", algo)
                    entry.setdefault("run_status", "success")
                    entry.setdefault("failure_reason", None)
                    entries.append(entry)
            if entries:
                algo_data[algo] = entries
        elif isinstance(seeds, list) and seeds:
            entries = []
            for idx, seed_data in enumerate(seeds):
                if isinstance(seed_data, dict):
                    entry = dict(seed_data)
                    entry.setdefault("seed", idx)
                    entry.setdefault("algorithm", algo)
                    entry.setdefault("run_status", "success")
                    entry.setdefault("failure_reason", None)
                    entries.append(entry)
            if entries:
                algo_data[algo] = entries

    if algo_data:
        return algo_data

    for result in results:
        algo = result.get("algorithm", "unknown")
        checkpoint_dir = result.get("checkpoint_dir", "")
        if not checkpoint_dir:
            continue
        train_logs_path = Path(checkpoint_dir) / "train_logs.json"
        if not train_logs_path.exists() and Path("experiments").exists():
            for exp_dir in Path("experiments").iterdir():
                if not exp_dir.is_dir():
                    continue
                candidate = exp_dir / "artifacts" / algo / "checkpoints" / "train_logs.json"
                if candidate.exists():
                    train_logs_path = candidate
                    break
        if train_logs_path.exists():
            try:
                with open(train_logs_path, encoding="utf-8") as f:
                    logs = json.load(f)
                if isinstance(logs, dict):
                    logs.setdefault("algorithm", algo)
                    logs.setdefault("run_status", "success")
                    logs.setdefault("failure_reason", None)
                    algo_data[algo] = [logs]
            except (OSError, json.JSONDecodeError):
                continue
    return algo_data


def _plot_convergence_figure(
    algo_data: dict[str, list[dict]],
    output_path: Path,
    *,
    clean: bool,
    aggregate: str,
    outlier_policy: str,
    quality_records: list[dict],
    record_quality: bool = True,
) -> dict[str, dict[str, bool]]:
    """Render one convergence figure and append quality records."""
    metric_items = list(CONVERGENCE_METRIC_SPECS.items())
    n_metrics = len(metric_items)
    ncols = 3 if n_metrics > 4 else 2
    nrows = int(math.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    warning_map: dict[str, dict[str, bool]] = {
        metric_name: {} for metric_name, _ in metric_items
    }
    plotted_any = False

    for idx, (metric_name, spec) in enumerate(metric_items):
        ax = axes[idx]
        clean_values_for_limits: list[np.ndarray] = []

        for algo, seeds in algo_data.items():
            color = ALGO_COLORS.get(algo, "#95a5a6")
            seed_curves: list[tuple[np.ndarray, np.ndarray]] = []
            first_raw_label = True
            metric_has_warning = False

            for seed_data in seeds:
                seed = seed_data.get("seed", "unknown")
                run_status = str(seed_data.get("run_status", "success"))
                raw_values = _pick_metric_series(seed_data, spec)
                if raw_values.size == 0:
                    continue
                x_values = _build_timestep_axis(seed_data, len(raw_values))
                clean_values, stats = _sanitize_metric_series(
                    raw_values,
                    outlier_policy=outlier_policy if clean else "none",
                )
                severe_outlier = bool(_is_span_extreme(raw_values))
                metric_has_warning = metric_has_warning or severe_outlier

                record = {
                    "algorithm": algo,
                    "seed": seed,
                    "metric": metric_name,
                    "run_status": run_status,
                    "failure_reason": seed_data.get("failure_reason"),
                    "schema_version": seed_data.get("schema_version"),
                    "higher_is_better": bool(spec["higher_is_better"]),
                    "severe_outlier": severe_outlier,
                    **stats,
                }
                if clean and run_status != "success":
                    record["skipped_from_clean_plot"] = True
                    record["skip_reason"] = f"run_status={run_status}"
                    if record_quality:
                        quality_records.append(record)
                    continue
                if record_quality:
                    quality_records.append(record)

                if clean:
                    seed_curves.append((x_values, clean_values))
                    clean_values_for_limits.append(clean_values)
                else:
                    label = algo if first_raw_label else "_nolegend_"
                    ax.plot(x_values, raw_values, color=color, alpha=0.45, linewidth=1.1, label=label)
                    first_raw_label = False
                    plotted_any = True

            if clean and seed_curves:
                aggregated = _aggregate_metric_by_seed(seed_curves, aggregate=aggregate)
                if aggregated["n_seeds"] == 0:
                    continue
                x = aggregated["x"]
                main_key = "median" if aggregate == "median" else "mean"
                y = aggregated[main_key]
                status = compute_convergence_status(
                    x,
                    y,
                    higher_is_better=bool(spec["higher_is_better"]),
                )
                warning_map[metric_name][algo] = metric_has_warning
                warning_suffix = " ⚠" if metric_has_warning else ""
                label = f"{algo} [{status['status']}]{warning_suffix}"
                ax.plot(x, y, color=color, label=label, linewidth=1.7)
                if aggregate == "median":
                    ax.fill_between(x, aggregated["q25"], aggregated["q75"], color=color, alpha=0.14)
                else:
                    ax.fill_between(
                        x,
                        aggregated["mean"] - aggregated["std"],
                        aggregated["mean"] + aggregated["std"],
                        color=color,
                        alpha=0.14,
                    )
                plotted_any = True

        if clean:
            ylim = _robust_ylim(clean_values_for_limits)
            if ylim is not None:
                ax.set_ylim(*ylim)
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel(str(spec["ylabel"]), fontsize=11)
        mode_label = "Clean" if clean else "Raw"
        ax.set_title(f"{spec['title']} Convergence ({mode_label})", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        if ax.lines or ax.collections:
            ax.legend(fontsize=8, loc="best")

    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    if plotted_any:
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return warning_map


def plot_convergence_curves(
    results: list[dict],
    output_dir: Path,
    fmt: str = "png",
    *,
    mode: str = "both",
    aggregate: str = "median",
    outlier_policy: str = "winsorize",
    include_algorithms: set[str] | None = None,
    exclude_algorithms: set[str] | None = None,
    evidence_level: str | None = None,
    run_id: str | None = None,
    seed_set: list[int] | None = None,
    config_hash: str | None = None,
    override_id: str | None = None,
) -> None:
    """Plot diagnostic and publication convergence curves with quality reports."""
    if mode not in {"raw", "clean", "both"}:
        raise ValueError(f"Unsupported convergence mode: {mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_data = _collect_convergence_data(results)
    quality_records: list[dict] = []
    if not algo_data:
        _write_convergence_quality_report(quality_records, output_dir)
        return

    selected_data: dict[str, list[dict]] = {}
    include = set(include_algorithms) if include_algorithms else None
    exclude = set(exclude_algorithms) if exclude_algorithms else set()
    for algo, seeds in algo_data.items():
        if include is not None and algo not in include:
            continue
        if algo in exclude:
            quality_records.append(
                {
                    "algorithm": algo,
                    "seed": "all",
                    "metric": "all",
                    "run_status": "excluded",
                    "excluded_from_clean_plot": True,
                    "reason": "excluded by convergence_exclude",
                    "outlier_policy": outlier_policy,
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                }
            )
            continue
        selected_data[algo] = seeds

    if not selected_data:
        _write_convergence_quality_report(quality_records, output_dir)
        return

    prefix = f"{evidence_level.lower()}_" if evidence_level else ""
    if mode in {"raw", "both"}:
        _plot_convergence_figure(
            selected_data,
            output_dir / f"{prefix}convergence_curves_raw_all.{fmt}",
            clean=False,
            aggregate=aggregate,
            outlier_policy=outlier_policy,
            quality_records=quality_records,
            record_quality=mode == "raw",
        )
    if mode in {"clean", "both"}:
        _plot_convergence_figure(
            selected_data,
            output_dir / f"{prefix}convergence_curves_clean_all.{fmt}",
            clean=True,
            aggregate=aggregate,
            outlier_policy=outlier_policy,
            quality_records=quality_records,
            record_quality=True,
        )
    metadata = {
        "evidence_level": evidence_level,
        "run_id": run_id,
        "seed_set": seed_set,
        "config_hash": config_hash,
        "override_id": override_id,
    }
    for record in quality_records:
        for key, value in metadata.items():
            if value is not None:
                record[key] = value
    _write_convergence_quality_report(quality_records, output_dir)


def plot_composite_ranking(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """3 子图: 各 profile 下综合排名 (balanced / latency_critical / energy_constrained)"""
    profiles = ["balanced", "latency_critical", "energy_constrained"]
    profile_labels = ["Balanced", "Latency-Critical", "Energy-Constrained"]

    # 收集有 composite_scores 数据的算法
    algo_scores: Dict[str, Dict] = {}
    for r in results:
        algo = r["algorithm"]
        cs = r.get("composite_scores")
        if cs and isinstance(cs, dict):
            algo_scores[algo] = cs

    if not algo_scores:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (profile, label) in enumerate(zip(profiles, profile_labels)):
        ax = axes[idx]
        # 收集各算法在该 profile 下的 composite_score
        entries = []
        for algo, cs in algo_scores.items():
            score = cs.get(profile)
            if score is not None and isinstance(score, (int, float)):
                entries.append((algo, float(score)))

        if not entries:
            ax.set_title(f"{label}\n(no data)", fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        # 按 composite_score 降序排列
        entries.sort(key=lambda x: x[1], reverse=True)
        algos = [e[0] for e in entries]
        scores = [e[1] for e in entries]
        colors = [ALGO_COLORS.get(a, "#95a5a6") for a in algos]

        x = np.arange(len(algos))
        bars = ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.8)

        ax.set_ylabel("Composite Score", fontsize=11)
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # 柱顶标注数值
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"composite_ranking.{fmt}", dpi=150)
    plt.close(fig)


def plot_radar_chart(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """雷达图: Top-6 算法在 balanced profile 下的多维对比"""
    # 收集有 composite_scores 的算法
    algo_scores: Dict[str, Dict] = {}
    for r in results:
        algo = r["algorithm"]
        cs = r.get("composite_scores")
        if cs and isinstance(cs, dict):
            algo_scores[algo] = cs

    if not algo_scores:
        return

    # 按 balanced profile 排名，取 top 6
    balanced_scores = []
    for algo, cs in algo_scores.items():
        score = cs.get("balanced")
        if score is not None and isinstance(score, (int, float)):
            balanced_scores.append((algo, float(score)))

    if not balanced_scores:
        return

    balanced_scores.sort(key=lambda x: x[1], reverse=True)
    top_algos = [e[0] for e in balanced_scores[:6]]

    # 4 个维度
    dimensions = ["reward_norm", "latency_norm", "energy_norm", "stability_norm"]
    dim_labels = ["Reward", "Latency", "Energy", "Stability"]
    n_dims = len(dimensions)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for algo in top_algos:
        cs = algo_scores[algo]
        values = []
        for dim in dimensions:
            val = cs.get(dim)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(0.0)
        values += values[:1]  # 闭合

        color = ALGO_COLORS.get(algo, "#95a5a6")
        ax.plot(angles, values, color=color, linewidth=1.8, label=algo)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_title("Top-6 Algorithm Radar (Balanced Profile)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / f"radar_top6.{fmt}", dpi=150)
    plt.close(fig)


def plot_weight_sensitivity(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """折线图: 各算法在不同 profile 下的排名变化 (权重敏感性)"""
    profiles = ["balanced", "latency_critical", "energy_constrained"]
    profile_labels = ["Balanced", "Latency-Critical", "Energy-Constrained"]

    # 收集有 composite_scores 的算法
    algo_scores: Dict[str, Dict] = {}
    for r in results:
        algo = r["algorithm"]
        cs = r.get("composite_scores")
        if cs and isinstance(cs, dict):
            algo_scores[algo] = cs

    if not algo_scores:
        return

    # 计算每个 profile 下的排名
    algo_ranks: Dict[str, List[int]] = {}
    for profile in profiles:
        entries = []
        for algo, cs in algo_scores.items():
            score = cs.get(profile)
            if score is not None and isinstance(score, (int, float)):
                entries.append((algo, float(score)))

        entries.sort(key=lambda x: x[1], reverse=True)
        for rank, (algo, _) in enumerate(entries, start=1):
            algo_ranks.setdefault(algo, []).append(rank)

    if not algo_ranks:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(profiles))

    for algo, ranks in algo_ranks.items():
        if len(ranks) != len(profiles):
            continue
        color = ALGO_COLORS.get(algo, "#95a5a6")
        ax.plot(x, ranks, color=color, marker="o", linewidth=1.8,
                markersize=6, label=algo)

    ax.set_xticks(x)
    ax.set_xticklabels(profile_labels, fontsize=11)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_title("Weight Sensitivity: Algorithm Rank Across Profiles",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()  # rank 1 在顶部
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / f"weight_sensitivity.{fmt}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Benchmark Results")
    parser.add_argument("--input", type=str, required=True, help="benchmark JSON path")
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=str,
        default="figures",
        help="output directory",
    )
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--convergence-mode", choices=["raw", "clean", "both"], default="both")
    parser.add_argument("--convergence-aggregate", choices=["median", "mean"], default="median")
    parser.add_argument(
        "--convergence-outlier-policy",
        choices=["none", "winsorize", "iqr-mask"],
        default="winsorize",
    )
    parser.add_argument("--convergence-include", nargs="*", default=None)
    parser.add_argument("--convergence-exclude", nargs="*", default=None)
    parser.add_argument("--evidence-level", choices=["L1", "L2", "L3"], default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed-set", nargs="*", type=int, default=None)
    parser.add_argument("--config-hash", default=None)
    parser.add_argument("--override-id", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    plot_reward_comparison(results, output_dir, args.format)
    plot_training_time(results, output_dir, args.format)
    plot_latency_energy(results, output_dir, args.format)
    plot_summary_table(results, output_dir, args.format)
    plot_convergence_curves(
        results,
        output_dir,
        args.format,
        mode=args.convergence_mode,
        aggregate=args.convergence_aggregate,
        outlier_policy=args.convergence_outlier_policy,
        include_algorithms=set(args.convergence_include) if args.convergence_include else None,
        exclude_algorithms=set(args.convergence_exclude) if args.convergence_exclude else None,
        evidence_level=args.evidence_level,
        run_id=args.run_id,
        seed_set=args.seed_set,
        config_hash=args.config_hash,
        override_id=args.override_id,
    )
    plot_composite_ranking(results, output_dir, args.format)
    plot_radar_chart(results, output_dir, args.format)
    plot_weight_sensitivity(results, output_dir, args.format)

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
