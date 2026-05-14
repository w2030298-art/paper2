"""Tests for v5.0 single-policy 3-user analysis artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.analyze_single_policy_3user_full17 import analyze


def test_analysis_writes_summary_report_decision_and_figures(tmp_path: Path) -> None:
    """Synthetic benchmark records should produce reviewable artifacts."""
    input_path = tmp_path / "benchmark.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "algorithm": "PPO",
                    "status": "ok",
                    "interface": "single_policy_multi_user",
                    "num_agents": 3,
                    "n_seeds": 1,
                    "final_reward_mean_mean": 2.0,
                    "final_social_welfare_mean_mean": 2.5,
                    "final_e2e_latency_mean_mean": 0.4,
                    "final_energy_mean_mean": 1.1,
                    "final_comm_score_mean": 7.0,
                    "final_deadline_miss_rate_mean": 0.1,
                    "final_agent_reward_jain_mean_mean": 0.95,
                },
                {
                    "algorithm": "SAC",
                    "status": "failed",
                    "interface": "single_policy_multi_user",
                    "num_agents": 3,
                    "n_seeds": 0,
                    "errors": [{"seed": 42, "error": "synthetic failure"}],
                },
            ]
        ),
        encoding="utf-8",
    )

    outputs = analyze(
        input_path=input_path,
        output_dir=tmp_path / "results",
        figure_dir=tmp_path / "figures",
        report_path=tmp_path / "ref" / "paper2_single_policy_3user_full17_report.md",
    )

    summary = outputs["summary"]
    report = outputs["report"]
    decision = outputs["decision"]
    assert summary.exists()
    assert report.exists()
    assert decision.exists()
    assert (tmp_path / "figures" / "figure_manifest.json").exists()

    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    assert [row["algorithm"] for row in rows] == ["PPO", "SAC"]
    assert rows[0]["interface"] == "single_policy_multi_user"
    assert rows[1]["status"] == "failed"
    assert "synthetic failure" in report.read_text(encoding="utf-8")
    assert "PENDING_HUMAN_REVIEW" in decision.read_text(encoding="utf-8")
