"""Tests for mainline-A energy helpers."""

from src.mec_model.energy import (
    EnergyBreakdown,
    compute_edge_compute_energy,
    compute_local_energy,
    compute_migration_energy,
    compute_tx_energy,
)


def test_energy_components_are_nonnegative() -> None:
    breakdown = EnergyBreakdown(
        local_compute_j=compute_local_energy(10.0, 2.0, 0.1),
        tx_j=compute_tx_energy(2.0, 3.0),
        edge_compute_j=compute_edge_compute_energy(10.0, 2.0, 0.05),
        migration_j=compute_migration_energy(100.0, 10.0, 2.0),
    )

    assert breakdown.total_j > 0.0

