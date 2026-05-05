"""Primal-dual helpers for constrained MARL."""

from __future__ import annotations

from dataclasses import dataclass, field


CONSTRAINT_KEYS = (
    "latency_deadline",
    "energy_budget",
    "queue_stability",
    "migration_rate",
    "budget_feasibility",
)


@dataclass(frozen=True)
class ConstraintResiduals:
    """Constraint residual values; positive means violation."""

    latency_deadline: float = 0.0
    energy_budget: float = 0.0
    queue_stability: float = 0.0
    migration_rate: float = 0.0
    budget_feasibility: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return residuals as a dictionary."""
        return {key: float(getattr(self, key)) for key in CONSTRAINT_KEYS}


@dataclass(frozen=True)
class PrimalDualState:
    """Dual variables for constrained updates."""

    dual_variables: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in CONSTRAINT_KEYS})


@dataclass
class PrimalDualUpdater:
    """Projected dual-ascent updater."""

    dual_lr: float = 0.01
    dual_clip: tuple[float, float] = (0.0, 20.0)
    state: PrimalDualState = field(default_factory=PrimalDualState)

    def update_dual_variables(self, residuals: ConstraintResiduals) -> PrimalDualState:
        """Apply projected dual ascent."""
        low, high = self.dual_clip
        updated = {}
        for key, residual in residuals.as_dict().items():
            current = self.state.dual_variables.get(key, 0.0)
            updated[key] = min(max(current + self.dual_lr * residual, low), high)
        self.state = PrimalDualState(updated)
        return self.state


def update_dual_variables(
    residuals: ConstraintResiduals,
    state: PrimalDualState | None = None,
    dual_lr: float = 0.01,
    dual_clip: tuple[float, float] = (0.0, 20.0),
) -> PrimalDualState:
    """Functional dual-variable update."""
    updater = PrimalDualUpdater(dual_lr=dual_lr, dual_clip=dual_clip, state=state or PrimalDualState())
    return updater.update_dual_variables(residuals)


def compute_lagrangian_reward(
    base_reward: float, residuals: ConstraintResiduals, dual_vars: PrimalDualState | dict[str, float]
) -> float:
    """Subtract dual-weighted residual penalties from base reward."""
    variables = dual_vars.dual_variables if isinstance(dual_vars, PrimalDualState) else dual_vars
    penalty = sum(max(0.0, residual) * float(variables.get(key, 0.0)) for key, residual in residuals.as_dict().items())
    return float(base_reward - penalty)

