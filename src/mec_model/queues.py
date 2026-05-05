"""Queueing approximations for the mainline-A MEC model."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .state import QueueSnapshot
from .types import QueueModelType


BOUNDED_DELAY_S = 1_000.0


@dataclass(frozen=True)
class QueueEstimate:
    """Computed queue metrics."""

    waiting_delay_s: float
    system_delay_s: float
    utilization: float
    drop_probability: float = 0.0


class MM1QueueModel:
    """M/M/1 queue approximation with bounded overload penalty."""

    def estimate(self, snapshot: QueueSnapshot) -> QueueEstimate:
        """Estimate waiting and system delay."""
        arrival = max(0.0, float(snapshot.arrival_rate))
        service = max(1e-9, float(snapshot.service_rate))
        rho = arrival / service
        if rho >= 1.0:
            wait = BOUNDED_DELAY_S
        else:
            wait = rho / max(service - arrival, 1e-9)
        return QueueEstimate(wait, wait + 1.0 / service, min(rho, 1.0))


class MMCQueueModel:
    """M/M/c approximation using an effective multi-server capacity."""

    def estimate(self, snapshot: QueueSnapshot) -> QueueEstimate:
        """Estimate waiting delay for c identical servers."""
        servers = max(1, int(snapshot.servers))
        arrival = max(0.0, float(snapshot.arrival_rate))
        service = max(1e-9, float(snapshot.service_rate))
        capacity = servers * service
        rho = arrival / capacity
        if rho >= 1.0:
            wait = BOUNDED_DELAY_S
        else:
            wait = (rho ** math.sqrt(2.0 * (servers + 1.0))) / max(capacity - arrival, 1e-9)
        return QueueEstimate(wait, wait + 1.0 / service, min(rho, 1.0))


class ParallelQueueApproxModel:
    """Parallel-worker approximation for edge compute queues."""

    def estimate(self, snapshot: QueueSnapshot) -> QueueEstimate:
        """Estimate waiting delay with a square-root pooling gain."""
        servers = max(1, int(snapshot.servers))
        pooled = QueueSnapshot(
            arrival_rate=snapshot.arrival_rate,
            service_rate=max(float(snapshot.service_rate), 1e-9) * math.sqrt(servers),
            queue_length=snapshot.queue_length,
            servers=1,
            capacity=snapshot.capacity,
        )
        return MM1QueueModel().estimate(pooled)


class FiniteCapacityQueueModel:
    """Finite-capacity M/M/1/K queue approximation."""

    def estimate(self, snapshot: QueueSnapshot) -> QueueEstimate:
        """Estimate queue delay and drop probability."""
        base = MM1QueueModel().estimate(snapshot)
        return QueueEstimate(
            waiting_delay_s=base.waiting_delay_s,
            system_delay_s=base.system_delay_s,
            utilization=base.utilization,
            drop_probability=compute_drop_probability(snapshot),
        )


def compute_queue_pressure(queue_snapshot: QueueSnapshot) -> float:
    """Return normalized queue pressure in [0, 1]."""
    service = max(float(queue_snapshot.service_rate), 1e-9)
    rho = max(0.0, float(queue_snapshot.arrival_rate)) / service
    length_pressure = float(queue_snapshot.queue_length) / max(float(queue_snapshot.capacity or 10), 1.0)
    return float(max(0.0, min(1.0, 0.5 * rho + 0.5 * length_pressure)))


def compute_waiting_delay(queue_snapshot: QueueSnapshot, model_type: QueueModelType) -> float:
    """Dispatch to the selected queue model and return waiting delay."""
    models = {
        "mm1": MM1QueueModel(),
        "mmc": MMCQueueModel(),
        "parallel": ParallelQueueApproxModel(),
        "finite_capacity": FiniteCapacityQueueModel(),
    }
    return float(models[model_type].estimate(queue_snapshot).waiting_delay_s)


def compute_drop_probability(queue_snapshot: QueueSnapshot) -> float:
    """Return finite-capacity blocking probability."""
    capacity = int(queue_snapshot.capacity or 0)
    if capacity <= 0:
        return 0.0
    service = max(float(queue_snapshot.service_rate), 1e-9)
    rho = max(0.0, float(queue_snapshot.arrival_rate)) / service
    if rho <= 0.0:
        return 0.0
    if abs(rho - 1.0) < 1e-9:
        return float(1.0 / (capacity + 1.0))
    numerator = (1.0 - rho) * (rho**capacity)
    denominator = 1.0 - rho ** (capacity + 1)
    return float(max(0.0, min(1.0, numerator / max(denominator, 1e-12))))


def estimate_deadline_miss_rate(queue_snapshot: QueueSnapshot, deadline_s: float) -> float:
    """Estimate deadline miss rate from queue delay and deadline."""
    delay = compute_waiting_delay(queue_snapshot, "finite_capacity" if queue_snapshot.capacity else "mm1")
    deadline = max(float(deadline_s), 1e-9)
    return float(max(0.0, min(1.0, 1.0 - math.exp(-delay / deadline))))

