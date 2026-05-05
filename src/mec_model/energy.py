"""Network-level energy abstractions for mainline-A."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyBreakdown:
    """Energy components in joules."""

    local_compute_j: float = 0.0
    tx_j: float = 0.0
    edge_compute_j: float = 0.0
    migration_j: float = 0.0

    @property
    def total_j(self) -> float:
        """Return total energy."""
        return self.local_compute_j + self.tx_j + self.edge_compute_j + self.migration_j


def compute_local_energy(cpu_cycles: float, frequency_hz: float, kappa: float) -> float:
    """Compute DVFS-style local compute energy."""
    return float(max(cpu_cycles, 0.0) * max(kappa, 0.0) * max(frequency_hz, 0.0) ** 2)


def compute_tx_energy(tx_power_w: float, tx_time_s: float) -> float:
    """Compute radio transmit energy."""
    return float(max(tx_power_w, 0.0) * max(tx_time_s, 0.0))


def compute_edge_compute_energy(cpu_cycles: float, frequency_hz: float, kappa_edge: float) -> float:
    """Compute network-level edge compute energy."""
    return compute_local_energy(cpu_cycles, frequency_hz, kappa_edge)


def compute_migration_energy(data_bits: float, link_rate_bps: float, tx_power_w: float) -> float:
    """Compute migration-transfer energy."""
    rate = max(float(link_rate_bps), 1e-9)
    return compute_tx_energy(tx_power_w, max(float(data_bits), 0.0) / rate)

