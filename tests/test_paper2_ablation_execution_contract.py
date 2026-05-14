"""Ablation execution contract tests for paper2 v4.9."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


RUNNER = Path("scripts/run_paper2_main_matrix.py")
MATRIX = Path("configs/paper2_main_experiment_matrix.yaml")
SCENARIOS = Path("configs/paper2_scenario_matrix.yaml")


def _manifest_for_n2(tmp_path: Path) -> dict:
    """Generate and load an N2 dry-run manifest."""
    output = tmp_path / "n2_manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--matrix",
            str(MATRIX),
            "--scenario-matrix",
            str(SCENARIOS),
            "--level",
            "L1_screening",
            "--scenarios",
            "N2-cl-ppo-ablation,N2-gam-coma-ablation",
            "--dry-run",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _config_for_run(run: dict) -> dict:
    """Load the generated algorithm config for a run."""
    return yaml.safe_load(Path(run["generated_config_path"]).read_text(encoding="utf-8"))


def test_cl_ppo_ablation_configs_toggle_declared_modules(tmp_path: Path) -> None:
    """CL-PPO ablation configs should change non-no-op module switches."""
    manifest = _manifest_for_n2(tmp_path)
    by_ablation = {
        run["ablation"]: _config_for_run(run)
        for run in manifest["runs"]
        if run["algorithm"] == "CL-PPO"
    }

    assert set(by_ablation) == {"full", "no_constraint_signal", "no_risk_critic", "no_safety_layer"}
    assert by_ablation["full"]["constraints"]["enabled"] is True
    assert by_ablation["full"]["risk"]["enabled"] is True
    assert by_ablation["full"]["safety_layer"]["enabled"] is True
    assert by_ablation["no_constraint_signal"]["constraints"]["enabled"] is False
    assert by_ablation["no_risk_critic"]["risk"]["enabled"] is False
    assert by_ablation["no_safety_layer"]["safety_layer"]["enabled"] is False


def test_gam_coma_ablation_configs_toggle_declared_modules(tmp_path: Path) -> None:
    """GAM-COMA ablation configs should change non-no-op module switches."""
    manifest = _manifest_for_n2(tmp_path)
    by_ablation = {
        run["ablation"]: _config_for_run(run)
        for run in manifest["runs"]
        if run["algorithm"] == "GAM-COMA"
    }

    assert set(by_ablation) == {
        "full",
        "no_graph_attention",
        "no_feasible_action_masking",
        "no_warm_start",
        "no_shapley_credit",
    }
    assert by_ablation["full"]["graph_attention"]["enabled"] is True
    assert by_ablation["full"]["action_masking"]["enabled"] is True
    assert by_ablation["full"]["game_theory"]["warm_start_steps"] > 0
    assert by_ablation["full"]["game_theory"]["use_shapley_credit"] is True
    assert by_ablation["no_graph_attention"]["graph_attention"]["enabled"] is False
    assert by_ablation["no_feasible_action_masking"]["action_masking"]["enabled"] is False
    assert by_ablation["no_warm_start"]["game_theory"]["warm_start_steps"] == 0
    assert by_ablation["no_shapley_credit"]["game_theory"]["use_shapley_credit"] is False


def test_gam_coma_disabled_graph_attention_uses_non_attention_critic() -> None:
    """The no-graph-attention variant should instantiate a fallback critic path."""
    from rl_algorithms.gam_coma import CentralizedCOMACritic, GAMCOMAAgent, GraphAttentionCOMACritic

    agent = GAMCOMAAgent(
        state_dim=8,
        action_dim=4,
        hidden_dim=16,
        num_agents=2,
        device="cpu",
        graph_attention={"enabled": False},
    )

    assert agent.graph_attention_enabled is False
    assert isinstance(agent.critic, CentralizedCOMACritic)
    assert not isinstance(agent.critic, GraphAttentionCOMACritic)
