"""Contract tests for the paper2 v4.8 scenario matrix."""

from __future__ import annotations

from pathlib import Path

import yaml


SCENARIO_PATH = Path("configs/paper2_scenario_matrix.yaml")
MAIN_MATRIX_PATH = Path("configs/paper2_main_experiment_matrix.yaml")

EXPECTED_SCENARIOS = [
    "ID-mainline-a",
    "N1-oracle-small",
    "N2-cl-ppo-ablation",
    "N2-gam-coma-ablation",
    "N3-ood-topology",
    "N3-ood-load",
    "N3-ood-channel-mobility",
    "N3-ood-deadline",
    "N3-ood-cooperation-queue",
]


def _scenario_matrix() -> dict:
    """Load the scenario matrix."""
    return yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))


def _main_matrix() -> dict:
    """Load the main experiment matrix."""
    return yaml.safe_load(MAIN_MATRIX_PATH.read_text(encoding="utf-8"))


def test_scenario_matrix_declares_required_ids_and_schema() -> None:
    """The scenario IDs should match the v4.8 plan exactly."""
    matrix = _scenario_matrix()

    assert matrix["schema_version"] == "paper2-scenario-matrix/v1"
    assert list(matrix["scenarios"]) == EXPECTED_SCENARIOS


def test_each_scenario_has_required_execution_contract() -> None:
    """Every scenario should define fields consumed by the matrix runner."""
    matrix = _scenario_matrix()
    required = {"stage", "profile", "allowed_algorithms", "metrics", "seeds_source", "steps_source", "output_subdir"}

    for scenario_id, spec in matrix["scenarios"].items():
        assert required <= set(spec), scenario_id
        assert spec["seeds_source"] == "main_matrix.evidence_levels"
        assert spec["steps_source"] == "main_matrix.evidence_levels"
        assert spec["output_subdir"].startswith("paper2_main_matrix/")
        assert spec["metrics"]


def test_scenario_matrix_covers_n0_n1_n2_n3() -> None:
    """The matrix should cover ID, N1 oracle, N2 ablations, and N3 OOD stress."""
    stages = {spec["stage"] for spec in _scenario_matrix()["scenarios"].values()}

    assert {"ID", "N1", "N2", "N3"} <= stages
    assert _scenario_matrix()["scenarios"]["N1-oracle-small"]["oracle_reference"] is True


def test_n1_oracle_stays_small_scale() -> None:
    """N1 oracle should stay within the reviewed small-scale boundary."""
    spec = _scenario_matrix()["scenarios"]["N1-oracle-small"]

    assert max(spec["bounds"]["num_users"]) <= 4
    assert max(spec["bounds"]["num_edges"]) <= 3


def test_n2_ablations_run_only_proposed_algorithms_by_default() -> None:
    """N2 ablations should isolate CL-PPO and GAM-COMA without frozen diagnostics."""
    scenarios = _scenario_matrix()["scenarios"]
    proposed = set(_main_matrix()["algorithm_groups"]["proposed"])
    diagnostics = set(_main_matrix()["algorithm_groups"]["diagnostic_only_or_frozen"])

    for scenario_id in ("N2-cl-ppo-ablation", "N2-gam-coma-ablation"):
        allowed = set(scenarios[scenario_id]["allowed_algorithms"])
        assert allowed <= proposed
        assert not (allowed & diagnostics)

    assert scenarios["N2-cl-ppo-ablation"]["ablations"] == [
        "full",
        "no_constraint_signal",
        "no_risk_critic",
        "no_safety_layer",
    ]
    assert scenarios["N2-gam-coma-ablation"]["ablations"] == [
        "full",
        "no_graph_attention",
        "no_feasible_action_masking",
        "no_warm_start",
        "no_shapley_credit",
    ]


def test_scenario_matrix_declares_frozen_mainline_a_constraints() -> None:
    """Scenario design must not imply edits to frozen Mainline-A core files."""
    frozen = _scenario_matrix()["frozen_mainline_a_core"]

    assert frozen == [
        "src/environments/mec_v3/game_theory_env.py",
        "configs/system_model_mainline_a.yaml",
        "configs/pricing_dynamic_mainline_a.yaml",
    ]
    for spec in _scenario_matrix()["scenarios"].values():
        assert spec["profile"]["mechanism_family"] == "mainline-a"
