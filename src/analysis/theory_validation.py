"""Numerical theory validation helpers for mainline-A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def validate_price_monotonicity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate price monotonicity with queue pressure."""
    ordered = sorted(records, key=lambda item: float(item.get("queue_pressure", 0.0)))
    violations = 0
    for prev, nxt in zip(ordered, ordered[1:]):
        if float(nxt.get("price", 0.0)) + 1e-9 < float(prev.get("price", 0.0)):
            violations += 1
    return {"check": "price_monotonicity", "passed": violations == 0, "violations": violations}


def validate_demand_elasticity(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate non-increasing demand with price."""
    ordered = sorted(records, key=lambda item: float(item.get("price", 0.0)))
    violations = 0
    for prev, nxt in zip(ordered, ordered[1:]):
        if float(nxt.get("demand", 0.0)) > float(prev.get("demand", 0.0)) + 1e-9:
            violations += 1
    return {"check": "demand_elasticity", "passed": violations == 0, "violations": violations}


def validate_constraint_residual_trend(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate that residuals do not trend upward."""
    residuals = [float(item.get("constraint_residual", 0.0)) for item in records]
    if len(residuals) < 2:
        slope = 0.0
    else:
        slope = (residuals[-1] - residuals[0]) / max(len(residuals) - 1, 1)
    return {"check": "constraint_residual_trend", "passed": slope <= 0.0, "slope": slope}


def export_theory_validation_report(records: Sequence[dict[str, Any]], output_path: str | Path) -> dict[str, Any]:
    """Export a validation report for theory assumptions."""
    report = {
        "price_monotonicity": validate_price_monotonicity(records),
        "demand_elasticity": validate_demand_elasticity(records),
        "constraint_residual_trend": validate_constraint_residual_trend(records),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

