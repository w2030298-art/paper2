"""Mobility and service-continuity models for mainline-A."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Sequence

from .state import MigrationSnapshot, MobilitySnapshot


@dataclass(frozen=True)
class HandoverEvent:
    """Cell handover event."""

    occurred: bool
    source_cell: str
    target_cell: str
    risk: float = 0.0


@dataclass(frozen=True)
class ServiceContinuityState:
    """Continuity state used by migration-aware rewards."""

    migration_policy: str = "no_migration"
    interruption_s: float = 0.0
    handover_count: int = 0


@dataclass
class MarkovMobilityModel:
    """Markov cell-transition model."""

    cells: tuple[str, ...] = ("cell-0", "cell-1")
    stay_probability: float = 0.8

    def sample_next(self, state: MobilitySnapshot, rng: random.Random | None = None) -> MobilitySnapshot:
        """Sample the next mobility snapshot."""
        generator = rng or random.Random()
        if generator.random() < self.stay_probability or len(self.cells) == 1:
            next_cell = state.cell_id
        else:
            candidates = [cell for cell in self.cells if cell != state.cell_id]
            next_cell = generator.choice(candidates)
        risk = 0.0 if next_cell == state.cell_id else 1.0 - self.stay_probability
        return MobilitySnapshot(
            cell_id=next_cell,
            position_xy=state.position_xy,
            velocity_mps=state.velocity_mps,
            handover_risk=risk,
            migration_policy=state.migration_policy,
        )


@dataclass
class TraceMobilityModel:
    """Trace adapter for precomputed mobility snapshots."""

    trace: Sequence[MobilitySnapshot] = field(default_factory=list)
    index: int = 0

    def sample_next(self, state: MobilitySnapshot, rng: random.Random | None = None) -> MobilitySnapshot:
        """Return the next trace snapshot, or the input state if exhausted."""
        _ = rng
        if not self.trace:
            return state
        snapshot = self.trace[self.index % len(self.trace)]
        self.index += 1
        return snapshot


def sample_next_location(
    state: MobilitySnapshot, action: str = "stay", rng: random.Random | None = None
) -> MobilitySnapshot:
    """Sample a next location using the default Markov model."""
    cells = ("cell-0", "cell-1", "cell-2")
    model = MarkovMobilityModel(cells=cells, stay_probability=0.9 if action == "stay" else 0.6)
    return model.sample_next(state, rng)


def detect_handover(prev_state: MobilitySnapshot, next_state: MobilitySnapshot) -> HandoverEvent:
    """Detect whether a handover occurred."""
    occurred = prev_state.cell_id != next_state.cell_id
    risk = max(prev_state.handover_risk, next_state.handover_risk)
    return HandoverEvent(occurred, prev_state.cell_id, next_state.cell_id, risk)


def compute_service_interruption_penalty(
    handover_event: HandoverEvent, migration_state: MigrationSnapshot | ServiceContinuityState | None
) -> float:
    """Compute service-interruption penalty from handover and migration state."""
    if not handover_event.occurred:
        return 0.0
    base = 1.0 + float(handover_event.risk)
    if isinstance(migration_state, MigrationSnapshot):
        base += float(migration_state.risk)
    elif isinstance(migration_state, ServiceContinuityState):
        if migration_state.migration_policy == "cooperative_migration":
            base *= 0.5
        elif migration_state.migration_policy == "nearest_edge":
            base *= 0.75
        base += float(migration_state.interruption_s)
    return float(max(0.0, base))

