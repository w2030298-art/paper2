#!/usr/bin/env python3
"""Create a compact export for the latest Mainline-A full-17 benchmark run."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "paper2_full_17_mainline_a"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / RUN_ID
OUTPUT_ZIP = PROJECT_ROOT / f"{RUN_ID}_export.zip"
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024

ALGORITHMS = [
    "GRPO",
    "PPO",
    "SAC",
    "DDQN",
    "DDPG",
    "TD3",
    "A3C",
    "TRPO",
    "SimPO",
    "MAPPO",
    "QMIX",
    "COMA",
    "IPPO",
    "VDN",
    "MADDPG",
    "IQL",
    "MATD3",
]

FINAL_METRIC_FIELDS = [
    "algorithm",
    "status",
    "environment",
    "environment_profile",
    "seed",
    "device",
    "train_timesteps",
    "eval_points",
    "first_eval_reward_mean",
    "last_eval_reward_mean",
    "best_eval_reward_mean",
    "best_eval_step",
    "reward_delta_first_to_last",
    "tail_reward_mean",
    "tail_reward_std",
    "final_reward_mean",
    "final_reward_std",
    "final_latency_mean",
    "final_energy_mean",
    "final_deadline_miss_rate",
    "final_throughput_tasks_per_step",
    "final_comm_score",
    "final_social_welfare_mean",
    "final_agent_reward_jain_mean",
    "final_constraint_any_violation_mean",
]

CONVERGENCE_FIELDS = [
    "algorithm",
    "eval_index",
    "step",
    "reward_mean",
    "reward_std",
    "latency_mean",
    "energy_mean",
    "deadline_miss_rate",
    "throughput_tasks_per_step",
    "comm_score",
    "social_welfare_mean",
    "agent_reward_jain_mean",
]


def load_json(path: Path) -> Any:
    """Load JSON from a UTF-8 file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(data: Any) -> bytes:
    """Serialize JSON with stable formatting for archive members."""
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    """Serialize dictionaries to CSV bytes."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def metric(data: dict[str, Any], key: str) -> Any:
    """Read a final_eval metric from a result document."""
    return data.get("final_eval", {}).get(key)


def mean_tail(values: list[float], width: int = 5) -> float | None:
    """Return the mean over the last N values."""
    if not values:
        return None
    tail = values[-min(width, len(values)) :]
    return statistics.fmean(tail)


def std_tail(values: list[float], width: int = 5) -> float | None:
    """Return population stddev over the last N values."""
    if not values:
        return None
    tail = values[-min(width, len(values)) :]
    if len(tail) == 1:
        return 0.0
    return statistics.pstdev(tail)


def finite_float(value: Any) -> float | None:
    """Convert finite numeric values to float and keep missing values as None."""
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(converted):
        return converted
    return None


def build_analysis() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build final metric, convergence curve, and normalized benchmark rows."""
    state = load_json(EXPERIMENT_DIR / "state.json")
    status_by_algorithm = {
        record.get("name"): record.get("status") for record in state.get("records", [])
    }

    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        result_path = EXPERIMENT_DIR / "artifacts" / algorithm / "result.json"
        logs_path = EXPERIMENT_DIR / "artifacts" / algorithm / "checkpoints" / "train_logs.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result.json for {algorithm}: {result_path}")
        if not logs_path.exists():
            raise FileNotFoundError(f"Missing train_logs.json for {algorithm}: {logs_path}")

        result = load_json(result_path)
        logs = load_json(logs_path)
        steps = [int(step) for step in logs.get("eval_steps", [])]
        rewards = [finite_float(value) for value in logs.get("eval_eval/reward_mean", [])]
        rewards = [value for value in rewards if value is not None]

        best_reward = max(rewards) if rewards else None
        best_step = None
        if rewards and steps:
            best_index = rewards.index(best_reward)
            best_step = steps[best_index] if best_index < len(steps) else None

        final_row = {
            "algorithm": algorithm,
            "status": status_by_algorithm.get(algorithm),
            "environment": result.get("environment"),
            "environment_profile": result.get("environment_profile"),
            "seed": result.get("seed"),
            "device": result.get("device"),
            "train_timesteps": result.get("train_timesteps"),
            "eval_points": len(steps),
            "first_eval_reward_mean": rewards[0] if rewards else None,
            "last_eval_reward_mean": rewards[-1] if rewards else None,
            "best_eval_reward_mean": best_reward,
            "best_eval_step": best_step,
            "reward_delta_first_to_last": (rewards[-1] - rewards[0]) if len(rewards) >= 2 else None,
            "tail_reward_mean": mean_tail(rewards),
            "tail_reward_std": std_tail(rewards),
            "final_reward_mean": metric(result, "eval/reward_mean"),
            "final_reward_std": metric(result, "eval/reward_std"),
            "final_latency_mean": metric(result, "eval/latency_mean"),
            "final_energy_mean": metric(result, "eval/energy_mean"),
            "final_deadline_miss_rate": metric(result, "eval/deadline_miss_rate"),
            "final_throughput_tasks_per_step": metric(result, "eval/throughput_tasks_per_step"),
            "final_comm_score": metric(result, "eval/comm_score"),
            "final_social_welfare_mean": metric(result, "eval/social_welfare_mean"),
            "final_agent_reward_jain_mean": metric(result, "eval/agent_reward_jain_mean"),
            "final_constraint_any_violation_mean": metric(
                result, "eval/constraint/any_violation_mean"
            ),
        }
        final_rows.append(final_row)
        benchmark_rows.append(
            {
                "algorithm": algorithm,
                "status": status_by_algorithm.get(algorithm),
                "result": result,
                "summary": final_row,
            }
        )

        max_points = max(
            len(steps),
            len(logs.get("eval_eval/reward_mean", [])),
            len(logs.get("eval_eval/reward_std", [])),
        )
        for index in range(max_points):
            row = {
                "algorithm": algorithm,
                "eval_index": index,
                "step": steps[index] if index < len(steps) else None,
                "reward_mean": value_at(logs, "eval_eval/reward_mean", index),
                "reward_std": value_at(logs, "eval_eval/reward_std", index),
                "latency_mean": value_at(logs, "eval_eval/latency_mean", index),
                "energy_mean": value_at(logs, "eval_eval/energy_mean", index),
                "deadline_miss_rate": value_at(logs, "eval_eval/deadline_miss_rate", index),
                "throughput_tasks_per_step": value_at(
                    logs, "eval_eval/throughput_tasks_per_step", index
                ),
                "comm_score": value_at(logs, "eval_eval/comm_score", index),
                "social_welfare_mean": value_at(logs, "eval_eval/social_welfare_mean", index),
                "agent_reward_jain_mean": value_at(logs, "eval_eval/agent_reward_jain_mean", index),
            }
            curve_rows.append(row)

    final_rows.sort(
        key=lambda row: (
            row["final_reward_mean"] is None,
            -(row["final_reward_mean"] or float("-inf")),
        )
    )
    return final_rows, curve_rows, benchmark_rows


def value_at(data: dict[str, Any], key: str, index: int) -> Any:
    """Return a list value from a log dictionary."""
    values = data.get(key, [])
    if index >= len(values):
        return None
    return values[index]


def checkpoint_inventory() -> list[dict[str, Any]]:
    """Return all PyTorch checkpoint files with sizes and exclusion reasons."""
    rows: list[dict[str, Any]] = []
    for path in sorted(EXPERIMENT_DIR.glob("artifacts/*/checkpoints/*.pt")):
        algorithm = path.parts[-3]
        size_bytes = path.stat().st_size
        rows.append(
            {
                "algorithm": algorithm,
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "included_in_zip": False,
                "exclusion_reason": (
                    "Model weights are excluded to keep the package under 500MB; "
                    "metric logs, TensorBoard events, and result JSON are included."
                ),
            }
        )
    return rows


def source_files() -> list[Path]:
    """Collect source files to include in the archive."""
    files: list[Path] = []
    files.extend([EXPERIMENT_DIR / "run.json", EXPERIMENT_DIR / "state.json"])

    for algorithm in ALGORITHMS:
        algorithm_dir = EXPERIMENT_DIR / "artifacts" / algorithm
        if algorithm_dir.exists():
            for path in sorted(algorithm_dir.rglob("*")):
                if path.is_file() and path.suffix != ".pt":
                    files.append(path)

    for optional in [
        PROJECT_ROOT / "results" / "final_algorithm_screening_decisions.csv",
        PROJECT_ROOT / "configs" / "benchmark_mainline_a_final_screening.yaml",
        PROJECT_ROOT / "configs" / "system_model_mainline_a.yaml",
        PROJECT_ROOT / "configs" / "pricing_dynamic_mainline_a.yaml",
    ]:
        if optional.exists():
            files.append(optional)

    for directory in [
        PROJECT_ROOT / "results" / "mainline_a",
        PROJECT_ROOT / "configs" / "algorithms",
        PROJECT_ROOT / "configs" / "experiments",
    ]:
        if directory.exists():
            files.extend(path for path in sorted(directory.rglob("*")) if path.is_file())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(path)
            seen.add(resolved)
    return deduped


def archive_name(path: Path) -> str:
    """Return a portable archive name for a project file."""
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def build_readme(final_rows: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]) -> str:
    """Build a README for consumers of the export."""
    top_rows = final_rows[:5]
    ranking_lines = "\n".join(
        f"- {index}. {row['algorithm']}: final_reward_mean={row['final_reward_mean']}, "
        f"best_eval_reward_mean={row['best_eval_reward_mean']}, eval_points={row['eval_points']}"
        for index, row in enumerate(top_rows, start=1)
    )
    checkpoint_bytes = sum(int(row["size_bytes"]) for row in checkpoint_rows)
    return f"""# Paper2 Full 17 Mainline-A Benchmark Export

Generated: {datetime.now(UTC).isoformat()}
Run ID: `{RUN_ID}`
Purpose: compare all 17 algorithms on the Mainline-A environment and preserve the
data needed to screen algorithms by final performance and convergence trend.

## Included

- `experiments/{RUN_ID}/run.json` and `state.json`
- Per-algorithm `result.json`, `checkpoints/train_logs.json`, `stdout.log`, `stderr.log`
- Per-algorithm TensorBoard event files
- `analysis/final_metrics.csv`
- `analysis/convergence_curves.csv`
- `analysis/benchmark_normalized.json`
- `analysis/checkpoint_inventory.csv`
- Mainline-A configs and N0-N3 support result files, when present

## Excluded

PyTorch `.pt` checkpoints are excluded from the archive. Their total raw size is
{checkpoint_bytes / (1024 * 1024):.2f} MB, and several individual files are too
large for a 500 MB package. See `analysis/checkpoint_inventory.csv` for the full
inventory and sizes.

## Screening Fields

- Final performance: `analysis/final_metrics.csv` -> `final_reward_mean`,
  `final_latency_mean`, `final_energy_mean`, `final_deadline_miss_rate`
- Convergence: `analysis/convergence_curves.csv` -> one row per algorithm and
  evaluation point
- Raw full metrics: each `result.json` and `train_logs.json`

## Top 5 By Final Reward

{ranking_lines}
"""


def build_manifest(
    files: list[Path],
    final_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a machine-readable export manifest."""
    state = load_json(EXPERIMENT_DIR / "state.json")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": RUN_ID,
        "archive_name": OUTPUT_ZIP.name,
        "max_archive_bytes": MAX_ARCHIVE_BYTES,
        "experiment_status": state.get("status"),
        "completed_count": len(state.get("completed_algorithms", [])),
        "total_algorithms": len(ALGORITHMS),
        "included_project_files": [archive_name(path) for path in files],
        "included_analysis_files": [
            "README.md",
            "analysis/final_metrics.csv",
            "analysis/convergence_curves.csv",
            "analysis/benchmark_normalized.json",
            "analysis/checkpoint_inventory.csv",
            "analysis/export_manifest.json",
        ],
        "final_metric_rows": len(final_rows),
        "convergence_curve_rows": len(curve_rows),
        "checkpoint_files_excluded": len(checkpoint_rows),
        "checkpoint_bytes_excluded": sum(int(row["size_bytes"]) for row in checkpoint_rows),
        "notes": [
            "All non-.pt experiment artifacts are included.",
            "Model checkpoint weights are excluded because the full checkpoint set exceeds 500MB.",
            "Higher reward is better; rewards are negative in this environment.",
        ],
    }


def create_export_zip() -> None:
    """Create the latest compact benchmark export zip."""
    if not EXPERIMENT_DIR.exists():
        raise FileNotFoundError(f"Experiment directory not found: {EXPERIMENT_DIR}")

    final_rows, curve_rows, benchmark_rows = build_analysis()
    checkpoint_rows = checkpoint_inventory()
    files = source_files()

    generated_members = {
        "README.md": build_readme(final_rows, checkpoint_rows).encode("utf-8"),
        "analysis/final_metrics.csv": csv_bytes(final_rows, FINAL_METRIC_FIELDS),
        "analysis/convergence_curves.csv": csv_bytes(curve_rows, CONVERGENCE_FIELDS),
        "analysis/benchmark_normalized.json": json_bytes(benchmark_rows),
        "analysis/checkpoint_inventory.csv": csv_bytes(
            checkpoint_rows,
            [
                "algorithm",
                "path",
                "size_bytes",
                "size_mb",
                "included_in_zip",
                "exclusion_reason",
            ],
        ),
    }
    manifest = build_manifest(files, final_rows, curve_rows, checkpoint_rows)
    generated_members["analysis/export_manifest.json"] = json_bytes(manifest)

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, archive_name(path))
        for name, payload in generated_members.items():
            archive.writestr(name, payload)

    size_bytes = OUTPUT_ZIP.stat().st_size
    if size_bytes > MAX_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Archive exceeds 500MB: {size_bytes / (1024 * 1024):.2f} MB"
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(OUTPUT_ZIP.relative_to(PROJECT_ROOT)),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 3),
                "project_files": len(files),
                "analysis_files": len(generated_members),
                "algorithms": len(final_rows),
                "convergence_rows": len(curve_rows),
                "checkpoint_files_excluded": len(checkpoint_rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main() -> int:
    """Run the exporter and convert failures to a non-zero exit code."""
    try:
        create_export_zip()
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
