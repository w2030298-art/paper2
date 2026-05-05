"""Tests for mainline-A mobility helpers."""

from src.mec_model.mobility import (
    ServiceContinuityState,
    compute_service_interruption_penalty,
    detect_handover,
)
from src.mec_model.state import MobilitySnapshot


def test_handover_penalty_reduced_by_cooperative_migration() -> None:
    event = detect_handover(MobilitySnapshot(cell_id="a"), MobilitySnapshot(cell_id="b"))

    penalty = compute_service_interruption_penalty(
        event, ServiceContinuityState(migration_policy="cooperative_migration")
    )

    assert event.occurred is True
    assert 0.0 < penalty < 1.0

