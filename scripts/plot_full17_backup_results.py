#!/usr/bin/env python3
"""Render Full17 backup convergence and CSV summary figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_results import ALGO_COLORS


DEFAULT_EXPERIMENT_DIR = Path(
    "experiments/paper2_full_17_mainline_a_auto_20260512_112657"
)
DEFAULT_SUMMARY_CSV = Path(
    "/mnt/c/Users/22003/Documents/codex_深度调研/full17_derived_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("figures/paper2_full_17_mainline_a_auto_20260512_112657")
FIGURE_DPI = 180
TEXT_PAD_FRACTION = 0.015


@dataclass(frozen=True)
class MetricSpec:
    """Plot metadata and source aliases for one convergence metric."""

    key: str
    title: str
    ylabel: str
    aliases: tuple[str, ...]
    higher_is_better: bool


@dataclass(frozen=True)
class AlgorithmCurve:
    """Loaded convergence curves for one algorithm."""

    algorithm: str
    steps: np.ndarray
    metrics: dict[str, np.ndarray]
    final_metrics: dict[str, float]


CORE_METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="reward",
        title="Reward",
        ylabel="Eval reward mean",
        aliases=("eval_eval/reward_mean", "eval/reward_mean", "reward_mean"),
        higher_is_better=True,
    ),
    MetricSpec(
        key="latency",
        title="Latency",
        ylabel="E2E latency mean",
        aliases=(
            "eval_eval/e2e_latency_mean",
            "eval_eval/latency_per_task_mean",
            "eval_eval/latency_mean",
            "eval/e2e_latency_mean",
            "eval/latency_per_task_mean",
            "eval/latency_mean",
        ),
        higher_is_better=False,
    ),
    MetricSpec(
        key="comm_score",
        title="Comm Score",
        ylabel="Comm score",
        aliases=("eval_eval/comm_score", "eval/comm_score"),
        higher_is_better=True,
    ),
    MetricSpec(
        key="energy",
        title="Energy",
        ylabel="Energy per task",
        aliases=(
            "eval_eval/energy_per_task_mean",
            "eval_eval/energy_mean",
            "eval/energy_per_task_mean",
            "eval/energy_mean",
        ),
        higher_is_better=False,
    ),
)

CSV_CORE_METRICS: tuple[str, ...] = (
    "reward_mean",
    "latency_mean",
    "latency_p95",
    "energy_per_task",
    "energy_total",
    "deadline_miss_rate",
    "throughput",
    "comm_score",
    "jain",
    "constraint_any",
    "physical_score_no_reward",
    "reliability_adjusted_physical_score",
)
CSV_RANK_METRICS: tuple[str, ...] = (
    "physical_rank_no_reward",
    "reliability_adjusted_rank",
    "screening_confidence_factor",
)
NON_NUMERIC_COLUMNS = {
    "algorithm",
    "family",
    "action_env",
    "convergence_class",
    "risk_flags",
}


def _to_float(value: Any) -> float | None:
    """Return a finite float or None."""
    try:
        if value in ("", None):
            return None
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _float_array(values: Any) -> np.ndarray:
    """Convert a JSON list-like value to a finite float array with NaNs for bad cells."""
    if not isinstance(values, (list, tuple)):
        return np.asarray([], dtype=float)
    converted: list[float] = []
    for value in values:
        numeric = _to_float(value)
        converted.append(np.nan if numeric is None else numeric)
    return np.asarray(converted, dtype=float)


def _metric_from_aliases(logs: dict[str, Any], aliases: Iterable[str]) -> np.ndarray:
    """Pick the first available metric series from the preferred aliases."""
    for alias in aliases:
        values = _float_array(logs.get(alias))
        if values.size:
            return values
    return np.asarray([], dtype=float)


def _build_steps(logs: dict[str, Any], target_length: int) -> np.ndarray:
    """Build a timestep axis aligned with a target series length."""
    for key in ("eval_steps", "timesteps"):
        values = _float_array(logs.get(key))
        if values.size >= target_length:
            return values[:target_length]
    return np.arange(target_length, dtype=float)


def _final_metrics(result_path: Path) -> dict[str, float]:
    """Read scalar final metrics from an algorithm result file."""
    if not result_path.exists():
        return {}
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    final_eval = payload.get("final_eval")
    metrics: dict[str, float] = {}
    if isinstance(final_eval, dict):
        for key, value in final_eval.items():
            numeric = _to_float(value)
            if numeric is not None:
                metrics[str(key)] = numeric
    for key, value in payload.items():
        numeric = _to_float(value)
        if numeric is not None:
            metrics[str(key)] = numeric
    return metrics


def _ordered_artifact_dirs(experiment_dir: Path) -> list[Path]:
    """Return algorithm artifact directories in run order where available."""
    artifacts_dir = experiment_dir / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Missing artifacts directory: {artifacts_dir}")

    by_name = {path.name: path for path in artifacts_dir.iterdir() if path.is_dir()}
    run_path = experiment_dir / "run.json"
    ordered: list[Path] = []
    if run_path.exists():
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        algorithms = payload.get("algorithms") if isinstance(payload, dict) else None
        if isinstance(algorithms, list):
            for item in algorithms:
                if isinstance(item, dict):
                    name = str(item.get("name", ""))
                    if name in by_name:
                        ordered.append(by_name.pop(name))
    ordered.extend(path for _, path in sorted(by_name.items()))
    return ordered


def load_algorithm_curves(experiment_dir: Path) -> dict[str, AlgorithmCurve]:
    """Load convergence curves from a specific Full17 experiment backup directory."""
    curves: dict[str, AlgorithmCurve] = {}
    for artifact_dir in _ordered_artifact_dirs(experiment_dir):
        logs_path = artifact_dir / "checkpoints" / "train_logs.json"
        if not logs_path.exists():
            continue
        logs = json.loads(logs_path.read_text(encoding="utf-8"))
        if not isinstance(logs, dict):
            continue

        metrics: dict[str, np.ndarray] = {}
        max_length = 0
        for spec in CORE_METRIC_SPECS:
            values = _metric_from_aliases(logs, spec.aliases)
            if values.size:
                metrics[spec.key] = values
                max_length = max(max_length, int(values.size))
        if not metrics:
            continue

        steps = _build_steps(logs, max_length)
        curves[artifact_dir.name] = AlgorithmCurve(
            algorithm=artifact_dir.name,
            steps=steps,
            metrics=metrics,
            final_metrics=_final_metrics(artifact_dir / "result.json"),
        )
    return curves


def load_summary_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Load CSV rows and field order."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def numeric_columns(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    """Return field names that contain at least one finite numeric value."""
    numeric: list[str] = []
    for field in fieldnames:
        if field in NON_NUMERIC_COLUMNS:
            continue
        if any(_to_float(row.get(field)) is not None for row in rows):
            numeric.append(field)
    return numeric


def _safe_name(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _algo_color(algorithm: str) -> str:
    """Return the configured algorithm color with a neutral fallback."""
    return ALGO_COLORS.get(algorithm, "#6b7280")


def _aligned_xy(curve: AlgorithmCurve, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Align a curve metric with its timestep axis."""
    y_values = curve.metrics.get(metric_key, np.asarray([], dtype=float))
    length = min(int(curve.steps.size), int(y_values.size))
    if length <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x_values = curve.steps[:length].astype(float) / 1000.0
    return x_values, y_values[:length].astype(float)


def _format_value(value: float) -> str:
    """Format a chart label compactly."""
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _set_robust_ylim(ax: plt.Axes, values: list[np.ndarray]) -> bool:
    """Apply 5-95 percent robust y-limits and report whether it changed limits."""
    parts = [np.asarray(item, dtype=float).reshape(-1) for item in values if item.size]
    if not parts:
        return False
    combined = np.concatenate(parts)
    combined = combined[np.isfinite(combined)]
    if combined.size < 5:
        return False
    raw_low = float(np.nanmin(combined))
    raw_high = float(np.nanmax(combined))
    low = float(np.nanquantile(combined, 0.05))
    high = float(np.nanquantile(combined, 0.95))
    if not math.isfinite(low) or not math.isfinite(high) or math.isclose(low, high):
        return False
    padding = (high - low) * 0.06
    ax.set_ylim(low - padding, high + padding)
    return low > raw_low or high < raw_high


def _style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Apply common axis styling."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Timesteps (k)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.28, linewidth=0.8)
    ax.tick_params(labelsize=9)


def plot_per_algorithm_curves(
    curves: dict[str, AlgorithmCurve],
    output_dir: Path,
    *,
    fmt: str,
    dpi: int,
) -> list[Path]:
    """Write one four-panel convergence figure per algorithm."""
    target_dir = output_dir / "per_algorithm"
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for algorithm, curve in curves.items():
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), constrained_layout=True)
        axes_flat = axes.flatten()
        for ax, spec in zip(axes_flat, CORE_METRIC_SPECS):
            x_values, y_values = _aligned_xy(curve, spec.key)
            if not y_values.size:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                _style_axis(ax, spec.title, spec.ylabel)
                continue
            ax.plot(
                x_values,
                y_values,
                color=_algo_color(algorithm),
                linewidth=2.0,
                marker="o",
                markersize=3.4,
            )
            finite = y_values[np.isfinite(y_values)]
            if finite.size:
                last_y = float(finite[-1])
                last_x = float(x_values[np.where(np.isfinite(y_values))[0][-1]])
                ax.annotate(
                    f"last={_format_value(last_y)}",
                    xy=(last_x, last_y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                )
            _style_axis(ax, spec.title, spec.ylabel)
        fig.suptitle(
            f"{algorithm} convergence on Full17 Mainline-A backup",
            fontsize=15,
            fontweight="bold",
        )
        output_path = target_dir / f"{_safe_name(algorithm)}_convergence.{fmt}"
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        outputs.append(output_path)
    return outputs


def plot_algorithm_comparison(
    curves: dict[str, AlgorithmCurve],
    output_dir: Path,
    *,
    fmt: str,
    dpi: int,
    robust: bool,
) -> Path:
    """Write a four-panel all-algorithm convergence comparison figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 10.8))
    axes_flat = axes.flatten()
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for ax, spec in zip(axes_flat, CORE_METRIC_SPECS):
        values_for_limits: list[np.ndarray] = []
        for algorithm, curve in curves.items():
            x_values, y_values = _aligned_xy(curve, spec.key)
            if not y_values.size:
                continue
            line = ax.plot(
                x_values,
                y_values,
                color=_algo_color(algorithm),
                linewidth=1.55,
                alpha=0.88,
                label=algorithm,
            )[0]
            values_for_limits.append(y_values)
            if algorithm not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(algorithm)
        if robust and _set_robust_ylim(ax, values_for_limits):
            ax.text(
                0.01,
                0.97,
                "robust y-axis: 5-95%",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
            )
        _style_axis(ax, f"{spec.title} convergence", spec.ylabel)

    suffix = "" if robust else "_full_range"
    title_suffix = "robust axis" if robust else "full y-range"
    fig.suptitle(
        f"Full17 Mainline-A algorithm comparison ({title_suffix})",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.84, 0.94))
    output_path = output_dir / f"algorithm_comparison_curves{suffix}.{fmt}"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _row_sort_key(row: dict[str, str]) -> tuple[float, str]:
    """Sort rows by reliability rank, then algorithm."""
    rank = _to_float(row.get("reliability_adjusted_rank"))
    if rank is None:
        rank = float("inf")
    return rank, str(row.get("algorithm", ""))


def _bar_label_offset(values: list[float]) -> float:
    """Return a small x-axis padding value for bar labels."""
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 1.0
    span = max(finite) - min(finite)
    return max(span * TEXT_PAD_FRACTION, max(abs(value) for value in finite) * 0.01, 1e-6)


def _plot_horizontal_metric(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
) -> None:
    """Plot one CSV metric as a horizontal bar chart."""
    labels = [str(row.get("algorithm", "unknown")) for row in rows]
    values = [_to_float(row.get(metric)) for row in rows]
    numeric_values = [0.0 if value is None else value for value in values]
    y_positions = np.arange(len(rows), dtype=float)
    colors = [_algo_color(label) for label in labels]
    ax.barh(y_positions, numeric_values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.invert_yaxis()
    ax.set_title(metric, fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    offset = _bar_label_offset(numeric_values)
    for index, value in enumerate(numeric_values):
        ha = "left" if value >= 0 else "right"
        x_value = value + offset if value >= 0 else value - offset
        ax.text(x_value, index, _format_value(value), va="center", ha=ha, fontsize=6.8)


def plot_csv_metric_bars(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    fmt: str,
    dpi: int,
) -> Path:
    """Write a readable bar-chart overview for key CSV summary metrics."""
    ordered_rows = sorted(rows, key=_row_sort_key)
    metrics = [metric for metric in CSV_CORE_METRICS if any(row.get(metric) for row in rows)]
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.0, max(11.0, 4.3 * nrows)))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, metric in zip(axes_flat, metrics):
        _plot_horizontal_metric(ax, ordered_rows, metric)
    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(
        "Full17 derived CSV summary as bar charts",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
    output_path = output_dir / f"csv_metric_bars_core.{fmt}"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_csv_rank_bars(
    rows: list[dict[str, str]],
    output_dir: Path,
    *,
    fmt: str,
    dpi: int,
) -> Path:
    """Write rank and confidence bar charts from the CSV summary."""
    ordered_rows = sorted(rows, key=_row_sort_key)
    metrics = [metric for metric in CSV_RANK_METRICS if any(row.get(metric) for row in rows)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16.5, 6.8))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, metric in zip(axes_flat, metrics):
        _plot_horizontal_metric(ax, ordered_rows, metric)
    fig.suptitle(
        "Full17 CSV ranks and screening confidence",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))
    output_path = output_dir / f"csv_rank_confidence_bars.{fmt}"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_manifest(
    output_dir: Path,
    *,
    experiment_dir: Path,
    summary_csv: Path | None,
    curves: dict[str, AlgorithmCurve],
    outputs: list[Path],
) -> Path:
    """Write a JSON manifest for generated figures."""
    manifest = {
        "schema_version": "full17-backup-figures/v1",
        "experiment_dir": str(experiment_dir),
        "summary_csv": str(summary_csv) if summary_csv else None,
        "algorithm_count": len(curves),
        "algorithms": list(curves),
        "metrics": [spec.key for spec in CORE_METRIC_SPECS],
        "outputs": [str(path) for path in outputs],
    }
    output_path = output_dir / "figure_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot Full17 backup convergence curves and derived CSV bars."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render all requested Full17 backup figures."""
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_algorithm_curves(args.experiment_dir)
    if not curves:
        raise FileNotFoundError(f"No train_logs.json data found under {args.experiment_dir}")

    outputs: list[Path] = []
    outputs.extend(
        plot_per_algorithm_curves(
            curves,
            args.output_dir,
            fmt=args.format,
            dpi=args.dpi,
        )
    )
    outputs.append(
        plot_algorithm_comparison(
            curves,
            args.output_dir,
            fmt=args.format,
            dpi=args.dpi,
            robust=True,
        )
    )
    outputs.append(
        plot_algorithm_comparison(
            curves,
            args.output_dir,
            fmt=args.format,
            dpi=args.dpi,
            robust=False,
        )
    )

    summary_csv: Path | None = None
    if args.summary_csv and args.summary_csv.exists():
        summary_csv = args.summary_csv
        rows, fieldnames = load_summary_rows(args.summary_csv)
        if rows and numeric_columns(rows, fieldnames):
            outputs.append(
                plot_csv_metric_bars(rows, args.output_dir, fmt=args.format, dpi=args.dpi)
            )
            outputs.append(
                plot_csv_rank_bars(rows, args.output_dir, fmt=args.format, dpi=args.dpi)
            )

    outputs.append(
        write_manifest(
            args.output_dir,
            experiment_dir=args.experiment_dir,
            summary_csv=summary_csv,
            curves=curves,
            outputs=outputs,
        )
    )
    print(json.dumps({"output_dir": str(args.output_dir), "outputs": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
