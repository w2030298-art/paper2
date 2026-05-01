"""Composite scoring for multi-profile algorithm ranking."""

from __future__ import annotations

import numpy as np


class CompositeScorer:
    """Weighted normalised composite scorer for benchmark results.

    Supports multiple weighting profiles (e.g. balanced, latency_critical,
    energy_constrained).  Each profile maps metric names to weights that
    sum to 1.0.

    Metrics handled
    ---------------
    - reward:    positive (higher is better)  → min-max normalisation
    - latency:   negative (lower is better)   → inverted min-max (max - x)
    - energy:    negative (lower is better)   → inverted min-max (max - x)
    - stability: positive (higher is better)  → min-max normalisation
      derived as ``1.0 - min(1.0, reward_std / |reward_mean|)``
    """

    # Metric direction: True = higher is better, False = lower is better
    _METRIC_DIRECTION: dict[str, bool] = {
        "reward": True,
        "latency": False,
        "energy": False,
        "stability": True,
    }

    def __init__(self, profiles: dict[str, dict[str, float]]) -> None:
        """Initialise the scorer with weighting profiles.

        Parameters
        ----------
        profiles : dict
            Mapping of profile name → {metric: weight}.
            Example::

                {
                    "balanced": {"reward": 0.30, "latency": 0.30,
                                 "energy": 0.25, "stability": 0.15},
                    "latency_critical": {"reward": 0.20, "latency": 0.45,
                                         "energy": 0.15, "stability": 0.20},
                    "energy_constrained": {"reward": 0.20, "latency": 0.15,
                                           "energy": 0.45, "stability": 0.20},
                }
        """
        self.profiles: dict[str, dict[str, float]] = dict(profiles)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stability(reward_mean: float, reward_std: float) -> float:
        """Compute stability metric from reward statistics.

        ``stability = 1.0 - min(1.0, reward_std / |reward_mean|)``

        Returns 0.0 when *reward_mean* is zero to avoid division by zero.
        """
        if reward_mean == 0.0:
            return 0.0
        return 1.0 - min(1.0, reward_std / abs(reward_mean))

    @staticmethod
    def _extract_metrics(results: list[dict]) -> dict[str, np.ndarray]:
        """Extract raw metric arrays from result dicts.

        Returns
        -------
        dict
            Keys are metric names ("reward", "latency", "energy", "stability"),
            values are 1-D numpy arrays of length ``len(results)``.
        """
        n = len(results)
        reward = np.array(
            [r.get("reward_mean", 0.0) for r in results], dtype=np.float64
        )
        latency = np.array(
            [r.get("latency_mean", 0.0) for r in results], dtype=np.float64
        )
        energy = np.array(
            [r.get("energy_mean", 0.0) for r in results], dtype=np.float64
        )

        stability = np.empty(n, dtype=np.float64)
        for i, r in enumerate(results):
            stability[i] = CompositeScorer._compute_stability(
                r.get("reward_mean", 0.0),
                r.get("reward_std", 0.0),
            )

        return {
            "reward": reward,
            "latency": latency,
            "energy": energy,
            "stability": stability,
        }

    @staticmethod
    def _minmax(values: np.ndarray, invert: bool = False) -> np.ndarray:
        """Min-max normalise to [0, 1].

        When *invert* is True the formula becomes ``(max - x) / (max - min)``
        so that lower raw values map to higher normalised scores.

        Edge cases
        ----------
        - All values identical → every element becomes 0.5.
        - Single element → returns 0.5.
        """
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        span = vmax - vmin

        if span == 0.0:
            return np.full_like(values, 0.5)

        if invert:
            return (vmax - values) / span
        return (values - vmin) / span

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, results: list[dict]) -> dict[str, dict[str, float]]:
        """Min-max normalise all metrics across *results*.

        Parameters
        ----------
        results : list of dict
            Each dict must contain ``reward_mean``, ``latency_mean``,
            ``energy_mean``, and ``reward_std``.  An optional ``algorithm``
            key is used as the result identifier; if absent the list index
            is used.

        Returns
        -------
        dict
            ``{algorithm_id: {"reward_norm": float, "latency_norm": float,
            "energy_norm": float, "stability_norm": float}, ...}``
        """
        if not results:
            return {}

        raw = self._extract_metrics(results)
        norm: dict[str, np.ndarray] = {}

        for metric, values in raw.items():
            invert = not self._METRIC_DIRECTION[metric]
            norm[metric] = self._minmax(values, invert=invert)

        out: dict[str, dict[str, float]] = {}
        for i, r in enumerate(results):
            key = r.get("algorithm", str(i))
            out[key] = {
                "reward_norm": float(norm["reward"][i]),
                "latency_norm": float(norm["latency"][i]),
                "energy_norm": float(norm["energy"][i]),
                "stability_norm": float(norm["stability"][i]),
            }

        return out

    def score(self, results: list[dict], profile_name: str) -> list[dict]:
        """Score *results* using the named weighting profile.

        Parameters
        ----------
        results : list of dict
            Raw benchmark results (same format as :meth:`normalize`).
        profile_name : str
            Key into ``self.profiles``.

        Returns
        -------
        list of dict
            Each element contains ``algorithm``, ``composite_score``,
            ``rank``, and ``breakdown`` (per-metric weighted scores).
            Sorted by ``composite_score`` descending.

        Raises
        ------
        KeyError
            If *profile_name* is not in ``self.profiles``.
        """
        if profile_name not in self.profiles:
            raise KeyError(
                f"Unknown profile '{profile_name}'. "
                f"Available: {list(self.profiles.keys())}"
            )

        if not results:
            return []

        weights = self.profiles[profile_name]
        norm = self.normalize(results)

        scored: list[dict] = []
        for r in results:
            key = r.get("algorithm", str(results.index(r)))
            n = norm[key]

            breakdown: dict[str, float] = {}
            composite = 0.0
            for metric, weight in weights.items():
                norm_key = f"{metric}_norm"
                val = n.get(norm_key, 0.0)
                weighted = weight * val
                breakdown[metric] = round(weighted, 6)
                composite += weighted

            scored.append({
                "algorithm": key,
                "composite_score": round(composite, 6),
                "breakdown": breakdown,
            })

        # Sort descending by composite_score
        scored.sort(key=lambda x: x["composite_score"], reverse=True)

        # Assign ranks (1-indexed, ties get same rank)
        for i, entry in enumerate(scored):
            if i > 0 and entry["composite_score"] == scored[i - 1]["composite_score"]:
                entry["rank"] = scored[i - 1]["rank"]
            else:
                entry["rank"] = i + 1

        return scored

    def score_all_profiles(self, results: list[dict]) -> dict[str, list[dict]]:
        """Run :meth:`score` for every configured profile.

        Parameters
        ----------
        results : list of dict
            Raw benchmark results.

        Returns
        -------
        dict
            ``{profile_name: [scored_results], ...}``
        """
        return {
            name: self.score(results, name)
            for name in self.profiles
        }

    def robustness_summary(self, results: list[dict]) -> list[dict]:
        """Aggregate ranks across all profiles to assess robustness.

        Parameters
        ----------
        results : list of dict
            Raw benchmark results.

        Returns
        -------
        list of dict
            Each element contains ``algorithm``, ``avg_rank``, ``worst_rank``,
            ``best_rank``, ``rank_variance``, and ``robust`` (``True`` when
            the algorithm is top-3 in **every** profile).  Sorted by
            ``avg_rank`` ascending.
        """
        if not results:
            return []

        all_scored = self.score_all_profiles(results)

        # Collect per-algorithm ranks across profiles
        algo_ranks: dict[str, list[int]] = {}
        for profile_results in all_scored.values():
            for entry in profile_results:
                algo = entry["algorithm"]
                algo_ranks.setdefault(algo, []).append(entry["rank"])

        summary: list[dict] = []
        for algo, ranks in algo_ranks.items():
            arr = np.array(ranks, dtype=np.float64)
            summary.append({
                "algorithm": algo,
                "avg_rank": round(float(np.mean(arr)), 4),
                "worst_rank": int(np.max(arr)),
                "best_rank": int(np.min(arr)),
                "rank_variance": round(float(np.var(arr)), 4),
                "robust": bool(np.all(arr <= 3)),
            })

        summary.sort(key=lambda x: x["avg_rank"])
        return summary
