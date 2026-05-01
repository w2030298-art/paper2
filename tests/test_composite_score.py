"""
CompositeScorer unit tests — Module 9 Step 6.
"""

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.composite_score import CompositeScorer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_profiles() -> dict[str, dict[str, float]]:
    """Load scoring profiles from the project YAML config."""
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "scoring_profiles.yaml"
    )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["profiles"]


# ---------------------------------------------------------------------------
# 1. test_normalize_direction
# ---------------------------------------------------------------------------

def test_normalize_direction():
    """Latency/energy are inverted (lowest → 1.0); reward/stability are normal (highest → 1.0)."""
    profiles = _load_profiles()
    scorer = CompositeScorer(profiles)

    results = [
        {
            "algorithm": "fast",
            "reward_mean": 10.0,
            "latency_mean": 1.0,   # lowest latency → should become 1.0
            "energy_mean": 2.0,    # lowest energy  → should become 1.0
            "reward_std": 0.0,
        },
        {
            "algorithm": "slow",
            "reward_mean": 50.0,   # highest reward → should become 1.0
            "latency_mean": 10.0,  # highest latency → should become 0.0
            "energy_mean": 20.0,   # highest energy  → should become 0.0
            "reward_std": 0.0,
        },
    ]

    norm = scorer.normalize(results)

    # reward: higher is better → slow(50) gets 1.0, fast(10) gets 0.0
    assert norm["slow"]["reward_norm"] == pytest.approx(1.0)
    assert norm["fast"]["reward_norm"] == pytest.approx(0.0)

    # latency: lower is better (inverted) → fast(1) gets 1.0, slow(10) gets 0.0
    assert norm["fast"]["latency_norm"] == pytest.approx(1.0)
    assert norm["slow"]["latency_norm"] == pytest.approx(0.0)

    # energy: lower is better (inverted) → fast(2) gets 1.0, slow(20) gets 0.0
    assert norm["fast"]["energy_norm"] == pytest.approx(1.0)
    assert norm["slow"]["energy_norm"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. test_score_weights_sum_to_one
# ---------------------------------------------------------------------------

def test_score_weights_sum_to_one():
    """Each profile's weights must sum to 1.0."""
    profiles = _load_profiles()

    for name, weights in profiles.items():
        total = sum(weights.values())
        assert total == pytest.approx(1.0), (
            f"Profile '{name}' weights sum to {total}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# 3. test_ranking_order
# ---------------------------------------------------------------------------

def test_ranking_order():
    """Construct known data and verify rank 1/2/3 assignment."""
    profiles = {
        "test": {
            "reward": 0.25,
            "latency": 0.25,
            "energy": 0.25,
            "stability": 0.25,
        }
    }
    scorer = CompositeScorer(profiles)

    # All three have reward_std=0 and non-zero reward_mean → stability=1.0.
    # Normalised values (equal weights 0.25 each):
    #   A: reward=1.0, latency=0.0, energy=0.0, stability=1.0 → 0.50
    #   B: reward=0.0, latency=1.0, energy=1.0, stability=1.0 → 0.75
    #   C: reward=0.5, latency=0.5, energy=0.5, stability=1.0 → 0.625
    results = [
        {
            "algorithm": "A",
            "reward_mean": 100.0,   # best reward (norm=1.0)
            "latency_mean": 10.0,   # worst latency (norm=0.0)
            "energy_mean": 10.0,    # worst energy  (norm=0.0)
            "reward_std": 0.0,      # stability=1.0
        },
        {
            "algorithm": "B",
            "reward_mean": 10.0,    # worst reward (norm=0.0)
            "latency_mean": 1.0,    # best latency (norm=1.0)
            "energy_mean": 1.0,     # best energy  (norm=1.0)
            "reward_std": 0.0,      # stability=1.0
        },
        {
            "algorithm": "C",
            "reward_mean": 55.0,    # middle reward (norm=0.5)
            "latency_mean": 5.5,    # middle latency (norm=0.5)
            "energy_mean": 5.5,     # middle energy  (norm=0.5)
            "reward_std": 0.0,      # stability=1.0
        },
    ]

    scored = scorer.score(results, "test")

    # Expected ranking: B=1, C=2, A=3
    rank_map = {e["algorithm"]: e["rank"] for e in scored}
    assert rank_map["B"] == 1
    assert rank_map["C"] == 2
    assert rank_map["A"] == 3


# ---------------------------------------------------------------------------
# 4. test_robustness_flag
# ---------------------------------------------------------------------------

def test_robustness_flag():
    """One algorithm always in top-3 across all profiles → robust=True."""
    profiles = _load_profiles()
    scorer = CompositeScorer(profiles)

    # "champion" has the best values across all metrics → always rank 1
    results = [
        {
            "algorithm": "champion",
            "reward_mean": 100.0,
            "latency_mean": 1.0,
            "energy_mean": 1.0,
            "reward_std": 0.0,
        },
    ]
    # Add 6 more algorithms with progressively worse metrics
    for i in range(1, 7):
        results.append({
            "algorithm": f"algo_{i}",
            "reward_mean": 100.0 - i * 15,
            "latency_mean": 1.0 + i * 2,
            "energy_mean": 1.0 + i * 2,
            "reward_std": i * 0.5,
        })

    summary = scorer.robustness_summary(results)

    champion_entry = next(e for e in summary if e["algorithm"] == "champion")
    assert champion_entry["robust"] is True
    assert champion_entry["best_rank"] == 1
    assert champion_entry["worst_rank"] <= 3


# ---------------------------------------------------------------------------
# 5. test_stability_metric
# ---------------------------------------------------------------------------

def test_stability_metric():
    """Verify stability formula edge cases.

    stability = 1.0 - min(1.0, reward_std / |reward_mean|)

    - reward_std=0           → stability=1.0
    - reward_std=|reward_mean| → stability=0.0
    - reward_mean=0          → stability=0.0 (avoid div-by-zero)
    """
    # reward_std=0 → stability=1.0
    assert CompositeScorer._compute_stability(
        reward_mean=10.0, reward_std=0.0
    ) == pytest.approx(1.0)

    # reward_std=|reward_mean| → stability=0.0
    assert CompositeScorer._compute_stability(
        reward_mean=10.0, reward_std=10.0
    ) == pytest.approx(0.0)
    assert CompositeScorer._compute_stability(
        reward_mean=-10.0, reward_std=10.0
    ) == pytest.approx(0.0)

    # reward_mean=0 → stability=0.0
    assert CompositeScorer._compute_stability(
        reward_mean=0.0, reward_std=5.0
    ) == pytest.approx(0.0)

    # reward_std > |reward_mean| → clamped to 0.0
    assert CompositeScorer._compute_stability(
        reward_mean=1.0, reward_std=100.0
    ) == pytest.approx(0.0)

    # Normal case: reward_std < |reward_mean|
    assert CompositeScorer._compute_stability(
        reward_mean=10.0, reward_std=2.0
    ) == pytest.approx(0.8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
