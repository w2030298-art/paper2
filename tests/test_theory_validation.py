"""Tests for theory validation helpers."""

from src.analysis.theory_validation import (
    export_theory_validation_report,
    validate_constraint_residual_trend,
    validate_demand_elasticity,
    validate_price_monotonicity,
)


def test_theory_validation_reports_pass(tmp_path) -> None:
    records = [
        {"queue_pressure": 0.1, "price": 1.0, "demand": 3.0, "constraint_residual": 2.0},
        {"queue_pressure": 0.2, "price": 2.0, "demand": 2.0, "constraint_residual": 1.0},
    ]
    output = tmp_path / "report.json"

    assert validate_price_monotonicity(records)["passed"]
    assert validate_demand_elasticity(records)["passed"]
    assert validate_constraint_residual_trend(records)["passed"]
    assert export_theory_validation_report(records, output)["price_monotonicity"]["passed"]
    assert output.is_file()

