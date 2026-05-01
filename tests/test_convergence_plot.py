"""
Tests for convergence curve plotting (Module 8 Step 5).

Validates that plot_convergence_curves() runs without error
and produces the expected output file.
"""

import os
import sys

import pytest  # noqa: F401 — needed for tmp_path fixture

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.plot_results import plot_convergence_curves


def _make_mock_results():
    """Build a minimal results list with convergence_by_seed data."""
    return [
        {
            "algorithm": "GRPO",
            "convergence_by_seed": {
                "42": {
                    "eval/reward_mean": [-10.0, -8.0, -6.0, -5.5, -5.3],
                    "eval/latency_mean": [0.5, 0.4, 0.35, 0.33, 0.32],
                    "eval/energy_mean": [1.0, 0.9, 0.85, 0.82, 0.81],
                    "eval/comm_score": [10.0, 12.0, 14.0, 14.5, 14.8],
                    "eval_interval": 1000,
                    "total_timesteps": 5000,
                }
            },
        }
    ]


class TestConvergenceCurves:
    """plot_convergence_curves 基本功能测试"""

    def test_convergence_curves_no_crash(self, tmp_path):
        """调用 plot_convergence_curves 不应抛出异常"""
        mock_results = _make_mock_results()
        # 只要不抛异常即通过
        plot_convergence_curves(mock_results, tmp_path, fmt="png")

    def test_convergence_curves_output_exists(self, tmp_path):
        """输出文件 convergence_curves.png 应被创建"""
        mock_results = _make_mock_results()
        plot_convergence_curves(mock_results, tmp_path, fmt="png")

        output_file = tmp_path / "convergence_curves.png"
        assert output_file.exists(), f"Expected output file not found: {output_file}"
        assert output_file.stat().st_size > 0, "Output file is empty"
