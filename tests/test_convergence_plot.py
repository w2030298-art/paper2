"""Tests for convergence curve quality plotting."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.plot_results import (  # noqa: E402
    _aggregate_metric_by_seed,
    _sanitize_metric_series,
    compute_convergence_status,
    plot_convergence_curves,
)


def _make_mock_results() -> list[dict]:
    """Build benchmark-like results with normal, outlier, and failed seeds."""
    return [
        {
            "algorithm": "GRPO",
            "convergence_by_seed": {
                "42": {
                    "schema_version": 2,
                    "seed": 42,
                    "algorithm": "GRPO",
                    "run_status": "success",
                    "failure_reason": None,
                    "eval/reward_mean": [-10.0, -8.0, -6.0, -5.5, -5.3, -5.2],
                    "eval/latency_mean": [0.5, 0.42, 0.36, 0.33, 0.32, 0.31],
                    "eval/energy_mean": [1.0, 0.9, 0.85, 0.82, 0.81, 0.80],
                    "eval/comm_score": [10.0, 12.0, 14.0, 14.5, 14.8, 15.0],
                    "eval_interval": 1000,
                },
                "43": {
                    "schema_version": 2,
                    "seed": 43,
                    "algorithm": "GRPO",
                    "run_status": "success",
                    "failure_reason": None,
                    "eval/reward_mean": [-11.0, -9.0, -6.5, -5.8, -5.4],
                    "eval/latency_mean": [0.55, 0.45, 0.37, 0.34, 0.33],
                    "eval/energy_mean": [1.1, 0.95, 0.88, 0.84, 0.82],
                    "eval/comm_score": [9.0, 11.0, 13.0, 14.0, 14.4],
                    "timesteps": [0, 500, 1500, 3000, 5000],
                },
            },
        },
        {
            "algorithm": "IQL",
            "convergence_by_seed": {
                "42": {
                    "schema_version": 2,
                    "seed": 42,
                    "algorithm": "IQL",
                    "run_status": "success",
                    "failure_reason": None,
                    "eval/reward_mean": [-8.0, -7.0, -65000.0, -6.2, -6.1, -6.0],
                    "eval/latency_mean": [0.8, float("inf"), 0.72, 0.7, 0.69, 0.68],
                    "eval/energy_mean": [1.5, 1.3, np.nan, 1.2, 1.18, 1.17],
                    "eval/comm_score": [5.0, 5.5, 5.8, 6.0, 6.1, 6.2],
                    "eval_interval": 1000,
                },
                "43": {
                    "schema_version": 2,
                    "seed": 43,
                    "algorithm": "IQL",
                    "run_status": "failed",
                    "failure_reason": "mock failure",
                    "eval/reward_mean": [-9.0, -8.0, -7.0],
                    "eval/latency_mean": [0.9, 0.85, 0.8],
                    "eval/energy_mean": [1.6, 1.5, 1.4],
                    "eval/comm_score": [4.0, 4.1, 4.2],
                    "eval_interval": 1000,
                },
            },
        },
    ]


class TestConvergenceCurves:
    """plot_convergence_curves quality pipeline tests."""

    def test_convergence_curves_no_crash(self, tmp_path):
        """Calling plot_convergence_curves should not raise."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png")

    def test_convergence_curves_output_exists(self, tmp_path):
        """Default mode writes raw, clean, and quality report outputs."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png")

        for name in [
            "convergence_curves_raw_all.png",
            "convergence_curves_clean_all.png",
            "convergence_quality_report.json",
            "convergence_quality_report.md",
        ]:
            output_file = tmp_path / name
            assert output_file.exists(), f"Expected output file not found: {output_file}"
            assert output_file.stat().st_size > 0, f"Output file is empty: {output_file}"

    def test_quality_report_flags_extreme_outlier(self, tmp_path):
        """The report should flag outlier-heavy algorithm/metric records."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png")

        report = json.loads((tmp_path / "convergence_quality_report.json").read_text())
        reward_records = [
            r for r in report if r.get("algorithm") == "IQL" and r.get("metric") == "reward"
        ]
        assert reward_records
        assert any(r["outlier_count"] > 0 or r["severe_outlier"] for r in reward_records)

    def test_clean_plot_uses_robust_axis_without_dropping_raw(self, tmp_path):
        """Clean mode should coexist with raw diagnostic evidence."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png")

        assert (tmp_path / "convergence_curves_raw_all.png").exists()
        assert (tmp_path / "convergence_curves_clean_all.png").exists()
        report_text = (tmp_path / "convergence_quality_report.md").read_text(encoding="utf-8")
        assert "severe_outlier" in report_text

    def test_output_names_are_not_duplicate(self, tmp_path):
        """New output names should replace the old duplicate convergence names."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png")

        assert not (tmp_path / "convergence_curves.png").exists()
        assert not (tmp_path / "convergence_all.png").exists()

    def test_direction_aware_convergence_for_lower_is_better_metric(self):
        """Lower latency/energy should be classified as improving."""
        status = compute_convergence_status(
            np.arange(10),
            np.asarray([10.0, 9.5, 9.0, 8.4, 7.8, 7.2, 6.8, 6.2, 5.8, 5.3]),
            higher_is_better=False,
        )

        assert status["status"] == "improving"
        assert status["relative_change"] < 0

    def test_seed_aggregation_interpolates_mismatched_timesteps(self):
        """Seed aggregation should align different x grids before aggregation."""
        aggregated = _aggregate_metric_by_seed(
            [
                (np.asarray([0, 1000, 2000, 3000]), np.asarray([0.0, 1.0, 2.0, 3.0])),
                (np.asarray([0, 500, 1500, 3000]), np.asarray([0.0, 0.5, 1.5, 3.0])),
            ]
        )

        assert aggregated["n_seeds"] == 2
        assert len(aggregated["x"]) >= 2
        assert np.all(np.isfinite(aggregated["median"]))

    def test_failed_seed_excluded_from_clean_but_reported(self, tmp_path):
        """Failed seeds should be omitted from clean curves and retained in the report."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png", mode="clean")

        report = json.loads((tmp_path / "convergence_quality_report.json").read_text())
        failed_records = [r for r in report if r.get("run_status") == "failed"]
        assert failed_records
        assert all(r.get("skipped_from_clean_plot") for r in failed_records)

    def test_nan_and_inf_do_not_crash_plotting(self, tmp_path):
        """NaN and inf values should be converted to reportable missing data."""
        plot_convergence_curves(_make_mock_results(), tmp_path, fmt="png")

        report = json.loads((tmp_path / "convergence_quality_report.json").read_text())
        missing_records = [r for r in report if r.get("nan_count", 0) > 0]
        assert missing_records

    def test_sanitize_metric_series_supports_iqr_mask(self):
        """The iqr-mask policy should replace extreme points with NaN."""
        clean, stats = _sanitize_metric_series(
            np.asarray([1.0, 1.1, 1.2, 1.15, 99.0]),
            outlier_policy="iqr-mask",
        )

        assert np.isnan(clean[-1])
        assert stats["outlier_count"] == 1
