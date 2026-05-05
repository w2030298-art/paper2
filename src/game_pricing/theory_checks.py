"""Numerical checks for dynamic-pricing theory assumptions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class TheoryCheckResult:
    """Result of a numerical theory check."""

    passed: bool
    metric: float
    message: str


def check_demand_price_monotonicity(samples: Sequence[tuple[float, float]]) -> TheoryCheckResult:
    """Check that demand is monotone non-increasing in price."""
    ordered = sorted((float(price), float(demand)) for price, demand in samples)
    violations = 0
    for (_, prev_demand), (_, next_demand) in zip(ordered, ordered[1:]):
        if next_demand > prev_demand + 1e-9:
            violations += 1
    return TheoryCheckResult(violations == 0, float(violations), "demand-price monotonicity")


def check_strong_concavity_proxy(hessian_or_fd_matrix: Sequence[Sequence[float]]) -> TheoryCheckResult:
    """Check a diagonal negative-definite proxy for strong concavity."""
    diagonal = [float(row[idx]) for idx, row in enumerate(hessian_or_fd_matrix) if idx < len(row)]
    max_diag = max(diagonal) if diagonal else 0.0
    return TheoryCheckResult(max_diag < 0.0, max_diag, "strong concavity diagonal proxy")


def check_unique_best_response_grid(user_state: object, price_grid: Sequence[Sequence[float]]) -> TheoryCheckResult:
    """Check that each grid point has one strict cheapest price."""
    _ = user_state
    ties = 0
    for row in price_grid:
        values = [float(value) for value in row]
        if len(values) != len(set(values)):
            ties += 1
    return TheoryCheckResult(ties == 0, float(ties), "unique best response grid")


def check_price_lipschitz_bound(
    pricing_policy: Callable[[object], Iterable[float]], state_samples: Sequence[object]
) -> TheoryCheckResult:
    """Check a conservative finite-difference Lipschitz proxy."""
    last: tuple[float, ...] | None = None
    max_delta = 0.0
    for sample in state_samples:
        current = tuple(float(value) for value in pricing_policy(sample))
        if last is not None:
            max_delta = max(max_delta, max(abs(a - b) for a, b in zip(last, current)))
        last = current
    return TheoryCheckResult(max_delta < 10.0, max_delta, "price Lipschitz finite-difference proxy")


def export_theory_checks(results: Sequence[TheoryCheckResult], output_path: str | Path) -> None:
    """Export theory check results as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.__dict__ for result in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

