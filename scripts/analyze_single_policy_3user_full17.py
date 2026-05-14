#!/usr/bin/env python3
"""Analyze v5.0 single-policy 3-user full17-equivalent benchmark output."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ALGORITHM_ORDER = ["GRPO", "PPO", "SAC", "DDQN", "DDPG", "TD3", "A3C", "TRPO", "SimPO"]
SUMMARY_FILENAME = "single_policy_3user_full17_summary.csv"
DECISION_FILENAME = "paper2_single_agent_reassessment_decision.md"
METRIC_KEYS = [
    "reward",
    "social_welfare",
    "latency",
    "energy",
    "comm_score",
    "deadline_miss_rate",
    "fairness",
]


def _load_results(path: Path) -> list[dict[str, Any]]:
    """Load benchmark JSON as a list of records."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [dict(item) for item in payload["results"]]
    raise ValueError(f"{path} must contain a list of benchmark records")


def _first(record: dict[str, Any], *keys: str) -> Any:
    """Return the first present metric value."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _float_or_none(value: Any) -> float | None:
    """Coerce a metric to float if possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one benchmark record into the summary schema."""
    status = str(record.get("status", "ok" if record.get("n_seeds", 0) else "unknown"))
    errors = record.get("errors") or []
    error_text = ""
    if errors:
        error_text = "; ".join(str(item.get("error", item)) for item in errors)
    return {
        "algorithm": record.get("algorithm"),
        "status": status,
        "interface": record.get("interface", "single_policy_multi_user"),
        "num_agents": record.get("num_agents", record.get("single_policy_num_users", 3)),
        "n_seeds": record.get("n_seeds", 0),
        "reward": _float_or_none(_first(record, "final_reward_mean_mean", "final_reward_mean")),
        "social_welfare": _float_or_none(
            _first(record, "final_social_welfare_mean_mean", "final_social_welfare_mean")
        ),
        "latency": _float_or_none(
            _first(record, "final_latency_per_task_mean_mean", "final_e2e_latency_mean_mean", "final_latency_mean_mean")
        ),
        "energy": _float_or_none(
            _first(record, "final_energy_per_task_mean_mean", "final_energy_mean_mean")
        ),
        "comm_score": _float_or_none(
            _first(record, "final_comm_score_mean", "final_comm_score")
        ),
        "deadline_miss_rate": _float_or_none(
            _first(record, "final_deadline_miss_rate_mean", "final_deadline_miss_rate")
        ),
        "fairness": _float_or_none(
            _first(record, "final_agent_reward_jain_mean_mean", "final_agent_reward_jain_mean")
        ),
        "error": error_text,
    }


def _ordered_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return summary rows in the canonical algorithm order."""
    by_algo = {str(record.get("algorithm")): _summary_row(record) for record in records}
    rows = [by_algo[name] for name in ALGORITHM_ORDER if name in by_algo]
    rows.extend(row for name, row in by_algo.items() if name not in ALGORITHM_ORDER)
    return rows


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write summary CSV and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUMMARY_FILENAME
    fieldnames = [
        "algorithm",
        "status",
        "interface",
        "num_agents",
        "n_seeds",
        *METRIC_KEYS,
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _rank_successes(rows: list[dict[str, Any]], key: str, reverse: bool = True) -> list[dict[str, Any]]:
    """Rank successful rows with finite metric values."""
    candidates = [
        row
        for row in rows
        if row.get("status") in {"ok", "partial", "success"} and isinstance(row.get(key), float)
    ]
    return sorted(candidates, key=lambda item: item[key], reverse=reverse)


def _write_figures(rows: list[dict[str, Any]], figure_dir: Path) -> list[Path]:
    """Write simple metric figures when matplotlib is available."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        manifest = figure_dir / "figure_manifest.json"
        manifest.write_text(
            json.dumps({"status": "skipped", "reason": "matplotlib is not installed"}, indent=2),
            encoding="utf-8",
        )
        return [manifest]

    for key, ylabel, filename in [
        ("reward", "Reward", "reward_ranking.png"),
        ("comm_score", "Communication score", "comm_score_ranking.png"),
    ]:
        ranked = _rank_successes(rows, key, reverse=True)
        if not ranked:
            continue
        labels = [str(row["algorithm"]) for row in ranked]
        values = [float(row[key]) for row in ranked]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color="#3b82f6")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Algorithm")
        ax.set_title(f"Single-policy 3-user {ylabel.lower()} ranking")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        out = figure_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        written.append(out)
    manifest = figure_dir / "figure_manifest.json"
    manifest.write_text(
        json.dumps({"status": "ok", "figures": [path.name for path in written]}, indent=2),
        encoding="utf-8",
    )
    written.append(manifest)
    return written


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Format selected rows as a markdown table."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def _write_report(rows: list[dict[str, Any]], report_path: Path, summary_path: Path, figures: list[Path]) -> Path:
    """Write the human review report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    successes = [row for row in rows if row.get("status") in {"ok", "partial", "success"}]
    failures = [row for row in rows if row.get("status") not in {"ok", "partial", "success"}]
    reward_rank = _rank_successes(rows, "reward", reverse=True)
    ppo_row = next((row for row in rows if row.get("algorithm") == "PPO"), None)
    ppo_gate = "pending human review" if ppo_row and ppo_row.get("status") in {"ok", "partial", "success"} else "not restartable"

    content = f"""# Single-Policy 3-User Full17 Report

`.ai/ledger.json` remains the single machine state source.

## Interface Semantics

This report analyzes `single_policy_multi_user`: one shared single-agent policy selects one action for each of 3 user observations, then the joint action is stepped once in the same Mainline-A environment. This is a shared-policy baseline, not MAPPO/COMA-style multi-policy MARL.

## Scope

- Algorithms: {", ".join(ALGORITHM_ORDER)}
- Seeds: single seed engineering comparison
- Steps: 100000 per algorithm in the benchmark input
- Statistical significance: not claimed
- Summary CSV: `{summary_path.as_posix()}`
- Figure artifacts: {", ".join(path.as_posix() for path in figures)}

## Status

Successful or partial records: {len(successes)}

Failure records: {len(failures)}

{_markdown_table(rows, ["algorithm", "status", "reward", "social_welfare", "latency", "energy", "comm_score", "fairness"])}

## Failure Table

{_markdown_table(failures, ["algorithm", "status", "error"]) if failures else "No failed algorithms were recorded."}

## Reward Ranking

{_markdown_table(reward_rank, ["algorithm", "reward", "comm_score", "latency", "energy"]) if reward_rank else "No successful reward records are available."}

## Old-Evaluation Boundary

The old 1-agent single-agent full17 results are not statistically or semantically comparable with this corrected 3-user shared-control interface. They may be used as historical artifacts only.

## CL-PPO Gate

PPO baseline status under the corrected interface: {ppo_gate}. CL-PPO remains frozen until a human review explicitly accepts PPO as a valid baseline under this report.
"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


def _write_decision(rows: list[dict[str, Any]], report_path: Path) -> Path:
    """Write the CL-PPO reassessment decision file."""
    decision_path = report_path.parent / DECISION_FILENAME
    ppo_row = next((row for row in rows if row.get("algorithm") == "PPO"), None)
    ppo_ok = ppo_row is not None and ppo_row.get("status") in {"ok", "partial", "success"}
    decision = "PENDING_HUMAN_REVIEW" if ppo_ok else "KEEP_CL_PPO_FROZEN"
    reason = (
        "PPO has a corrected-interface record, but CL-PPO still needs human review before restart."
        if ppo_ok
        else "PPO does not have a usable corrected-interface record."
    )
    content = f"""# Single-Agent Reassessment Decision

Decision: `{decision}`

Reason: {reason}

Boundary: old 1-agent full17 results remain retracted as research evidence for the 3-user Mainline-A game. No statistical significance claim is made from this single-seed engineering comparison.

Report: `{report_path.as_posix()}`
"""
    decision_path.write_text(content, encoding="utf-8")
    return decision_path


def analyze(input_path: Path, output_dir: Path, figure_dir: Path, report_path: Path) -> dict[str, Path | list[Path]]:
    """Run the full analysis pipeline."""
    records = _load_results(input_path)
    rows = _ordered_rows(records)
    summary_path = _write_summary(rows, output_dir)
    figures = _write_figures(rows, figure_dir)
    _write_report(rows, report_path, summary_path, figures)
    decision_path = _write_decision(rows, report_path)
    return {
        "summary": summary_path,
        "figures": figures,
        "report": report_path,
        "decision": decision_path,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Analyze single-policy 3-user full17 results")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    outputs = analyze(args.input, args.output_dir, args.figure_dir, args.report)
    print(json.dumps({key: str(value) for key, value in outputs.items() if key != "figures"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
