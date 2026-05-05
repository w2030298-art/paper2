"""Analytic and simulation channel models for mainline-A."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


def compute_shannon_rate_bps(bandwidth_hz: float, sinr_linear: float) -> float:
    """Compute Shannon rate in bits per second."""
    return float(max(bandwidth_hz, 0.0) * math.log2(1.0 + max(sinr_linear, 0.0)))


def compute_sinr_linear(
    tx_power_w: float, pathloss_linear: float, interference_w: float, noise_w: float
) -> float:
    """Compute linear SINR with bounded denominator."""
    received = max(float(tx_power_w), 0.0) / max(float(pathloss_linear), 1e-12)
    denominator = max(float(interference_w) + float(noise_w), 1e-12)
    return float(received / denominator)


@dataclass(frozen=True)
class AnalyticRateModel:
    """Pathloss plus average-SINR analytical rate model."""

    bandwidth_hz: float = 20e6
    noise_w: float = 1e-9

    def rate_bps(self, tx_power_w: float, pathloss_linear: float, interference_w: float = 0.0) -> float:
        """Compute analytical rate."""
        sinr = compute_sinr_linear(tx_power_w, pathloss_linear, interference_w, self.noise_w)
        return compute_shannon_rate_bps(self.bandwidth_hz, sinr)


@dataclass(frozen=True)
class ThreeGppLiteRateModel:
    """Simplified UMi/UMa LoS/NLoS model with shadowing."""

    scenario: str = "UMi"
    bandwidth_hz: float = 20e6
    carrier_ghz: float = 3.5
    rng_seed: int | None = None

    def pathloss_db(self, distance_m: float) -> float:
        """Compute simplified 3GPP-like pathloss."""
        distance = max(float(distance_m), 1.0)
        los_probability = min(1.0, 18.0 / distance + math.exp(-distance / 63.0) * (1.0 - 18.0 / distance))
        rng = random.Random(self.rng_seed)
        los = rng.random() <= los_probability
        scenario_offset = 28.0 if self.scenario == "UMi" else 32.4
        exponent = 22.0 if los else 36.7
        shadow = rng.gauss(0.0, 4.0 if los else 7.0)
        return float(scenario_offset + exponent * math.log10(distance) + 20.0 * math.log10(self.carrier_ghz) + shadow)

    def rate_bps(self, tx_power_w: float, distance_m: float, interference_w: float = 0.0) -> float:
        """Compute simulation rate."""
        pathloss_linear = 10.0 ** (self.pathloss_db(distance_m) / 10.0)
        return AnalyticRateModel(self.bandwidth_hz).rate_bps(tx_power_w, pathloss_linear, interference_w)


@dataclass(frozen=True)
class RayleighRateModel:
    """Rayleigh fading rate model."""

    bandwidth_hz: float = 20e6
    fading_gain: float = 1.0

    def rate_bps(self, tx_power_w: float, pathloss_linear: float, interference_w: float = 0.0) -> float:
        """Compute rate under a Rayleigh gain."""
        effective_pathloss = max(float(pathloss_linear), 1e-12) / max(float(self.fading_gain), 1e-12)
        return AnalyticRateModel(self.bandwidth_hz).rate_bps(tx_power_w, effective_pathloss, interference_w)


@dataclass(frozen=True)
class PathlossOnlyRateModel:
    """Pathloss-only rate model for ablation."""

    bandwidth_hz: float = 20e6

    def rate_bps(self, tx_power_w: float, pathloss_linear: float) -> float:
        """Compute rate without interference or fading."""
        return AnalyticRateModel(self.bandwidth_hz).rate_bps(tx_power_w, pathloss_linear, 0.0)

