"""Pareto filtering helpers for Stage-1 tuning outputs."""

from __future__ import annotations

from typing import Any, Iterable


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _candidate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status", "success")).lower() not in {"success", "ok"}:
            continue
        if not bool(row.get("feasible", False)):
            continue
        if not bool(row.get("pareto_eligible", True)):
            continue
        if _as_float(row.get("constraint_violation", 0.0)) > 0.0:
            continue
        candidates.append(row)
    return candidates


def is_pareto_efficient(
    rows: Iterable[dict[str, Any]],
    objectives: tuple[str, ...] = ("p95_latency", "mean_latency", "energy_per_task", "tail_instability"),
) -> list[dict[str, Any]]:
    """Return feasible zero-constraint rows that are not dominated."""
    candidates = _candidate_rows(rows)
    efficient: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        row_values = [_as_float(row.get(objective)) for objective in objectives]
        dominated = False
        for other_idx, other in enumerate(candidates):
            if other_idx == idx:
                continue
            other_values = [_as_float(other.get(objective)) for objective in objectives]
            if all(o <= r for o, r in zip(other_values, row_values)) and any(
                o < r for o, r in zip(other_values, row_values)
            ):
                dominated = True
                break
        if not dominated:
            efficient.append(row)
    return efficient
