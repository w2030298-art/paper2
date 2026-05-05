"""Tests for mainline-A queue models."""

import math

from src.mec_model.queues import (
    compute_drop_probability,
    compute_queue_pressure,
    compute_waiting_delay,
)
from src.mec_model.state import QueueSnapshot


def test_overloaded_mm1_returns_bounded_finite_delay() -> None:
    delay = compute_waiting_delay(QueueSnapshot(arrival_rate=2, service_rate=1), "mm1")

    assert math.isfinite(delay)
    assert delay == 1000.0


def test_queue_pressure_and_drop_probability_are_bounded() -> None:
    snapshot = QueueSnapshot(arrival_rate=0.5, service_rate=1.0, queue_length=5, capacity=10)

    assert 0.0 <= compute_queue_pressure(snapshot) <= 1.0
    assert 0.0 <= compute_drop_probability(snapshot) <= 1.0

