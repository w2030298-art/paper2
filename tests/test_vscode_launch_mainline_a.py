"""Tests for the slim Mainline-A VSCode launch surface."""

import json
from pathlib import Path


EXPECTED_ENTRY_NAMES = [
    "Mainline-A Full 17 Fresh",
    "Single-Policy 3-User Full17 Fresh",
    "Resume",
    "Status",
    "Backup Full17 Mainline-A",
    "Stage-1 Tune PPO Starter",
    "Stage-1 Tune COMA Starter",
    "Direct Benchmark Dry Run",
    "Paper2 Main Matrix L1 Dry Run",
    "Paper2 Main Matrix L2 Candidate Validation",
    "Paper2 Main Matrix L3 Formal Verification",
    "Paper2 Scenario Matrix Dry Run",
    "Paper2 Statistics Dry Run",
    "Paper2 Final Figures Dry Run",
    "Paper2 Final Export Dry Run",
    "Plot Latest",
    "Legacy Full 17 Fresh (Explicit Fallback)",
    "Experiment List",
]

EXPECTED_PYTHONPATH = (
    "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;"
    "${workspaceFolder}/rl_algorithms"
)


def _launch_configurations() -> list[dict]:
    payload = json.loads(Path(".vscode/launch.json").read_text(encoding="utf-8"))
    assert payload["version"] == "0.2.0"
    return payload["configurations"]


def _by_name(name: str) -> dict:
    return next(item for item in _launch_configurations() if item["name"] == name)


def test_launch_json_has_exact_review_repair_entries() -> None:
    """VSCode launch entries should match the v4.8 final closeout surface."""
    configurations = _launch_configurations()
    assert [item["name"] for item in configurations] == EXPECTED_ENTRY_NAMES
    assert len(configurations) == len(EXPECTED_ENTRY_NAMES)
    assert "Export Results" not in [item["name"] for item in configurations]


def test_new_paper2_entries_use_debugpy_workspace_and_pythonpath() -> None:
    """New paper2 entries should use the repo-local debug contract."""
    for name in EXPECTED_ENTRY_NAMES:
        configuration = _by_name(name)
        assert configuration["type"] == "debugpy"
        assert configuration["cwd"] == "${workspaceFolder}"
        assert configuration["env"]["PYTHONPATH"] == EXPECTED_PYTHONPATH


def test_mainline_a_full17_entry_is_default_profile() -> None:
    """Full17 fresh launch should use the Mainline-A profile explicitly."""
    configuration = _by_name("Mainline-A Full 17 Fresh")
    assert configuration["program"] == "${workspaceFolder}/scripts/experiment_manager.py"
    assert configuration["args"] == [
        "start",
        "--preset",
        "full17",
        "--environment-profile",
        "mainline-a",
        "--fresh",
    ]


def test_single_policy_3user_full17_entry_uses_direct_benchmark() -> None:
    """The v5.0 corrected single-policy comparison should be directly launchable."""
    configuration = _by_name("Single-Policy 3-User Full17 Fresh")
    assert configuration["program"] == "${workspaceFolder}/scripts/benchmark.py"
    assert configuration["args"] == [
        "--config",
        "configs/benchmark_mainline_a_single_policy_3user_full17.yaml",
        "--output",
        "results/benchmark_mainline_a_single_policy_3user_full17.json",
        "--no-latest-alias",
    ]


def test_direct_benchmark_dry_run_uses_mainline_a_profile() -> None:
    """Direct benchmark entry should be a dry-run preflight."""
    configuration = _by_name("Direct Benchmark Dry Run")
    assert configuration["program"] == "${workspaceFolder}/scripts/benchmark.py"
    assert configuration["args"] == ["--all", "--environment-profile", "mainline-a", "--dry-run"]


def test_paper2_main_matrix_launch_entries_are_exact() -> None:
    """Paper2 matrix launches should expose dry-run and explicit heavy entries."""
    l1 = _by_name("Paper2 Main Matrix L1 Dry Run")
    l2 = _by_name("Paper2 Main Matrix L2 Candidate Validation")
    l3 = _by_name("Paper2 Main Matrix L3 Formal Verification")

    assert l1["program"] == "${workspaceFolder}/scripts/run_paper2_main_matrix.py"
    assert l1["args"] == [
        "--matrix",
        "configs/paper2_main_experiment_matrix.yaml",
        "--scenario-matrix",
        "configs/paper2_scenario_matrix.yaml",
        "--level",
        "L1_screening",
        "--dry-run",
        "--output",
        "results/paper2_main_matrix/dry_run_manifest.json",
        "--device",
        "auto",
    ]
    assert l2["args"] == [
        "--matrix",
        "configs/paper2_main_experiment_matrix.yaml",
        "--scenario-matrix",
        "configs/paper2_scenario_matrix.yaml",
        "--level",
        "L2_candidate_validation",
        "--output",
        "results/paper2_main_matrix/l2_candidate_manifest.json",
        "--device",
        "auto",
    ]
    assert l3["args"] == [
        "--matrix",
        "configs/paper2_main_experiment_matrix.yaml",
        "--scenario-matrix",
        "configs/paper2_scenario_matrix.yaml",
        "--level",
        "L3_formal_verification",
        "--output",
        "results/paper2_main_matrix/l3_formal_manifest.json",
        "--device",
        "auto",
    ]
    assert EXPECTED_ENTRY_NAMES.index("Paper2 Main Matrix L3 Formal Verification") > 0


def test_paper2_auxiliary_launch_entries_are_exact() -> None:
    """Statistics, figures, export, and scenario dry-run launches should be exact."""
    scenario = _by_name("Paper2 Scenario Matrix Dry Run")
    statistics = _by_name("Paper2 Statistics Dry Run")
    figures = _by_name("Paper2 Final Figures Dry Run")
    export = _by_name("Paper2 Final Export Dry Run")

    assert scenario["program"] == "${workspaceFolder}/scripts/run_paper2_main_matrix.py"
    assert scenario["args"] == [
        "--matrix",
        "configs/paper2_main_experiment_matrix.yaml",
        "--scenario-matrix",
        "configs/paper2_scenario_matrix.yaml",
        "--level",
        "L1_screening",
        "--scenarios",
        "ID-mainline-a,N1-oracle-small,N2-cl-ppo-ablation,N2-gam-coma-ablation",
        "--dry-run",
        "--output",
        "results/paper2_main_matrix/scenario_dry_run_manifest.json",
        "--device",
        "auto",
    ]
    assert statistics["program"] == "${workspaceFolder}/scripts/analyze_paper2_statistics.py"
    assert statistics["args"] == [
        "--input",
        "results/paper2_main_matrix/dry_run_manifest.json",
        "--output-dir",
        "results/paper2_statistics",
        "--dry-run",
    ]
    assert figures["program"] == "${workspaceFolder}/scripts/generate_paper2_figures.py"
    assert figures["args"] == [
        "--results",
        "results/paper2_main_matrix/dry_run_manifest.json",
        "--output-dir",
        "figures/paper2_final",
        "--dry-run",
    ]
    assert export["program"] == "${workspaceFolder}/scripts/export_paper2_final_package.py"
    assert export["args"] == [
        "--dry-run",
        "--output",
        "paper2_final_delivery_package.zip",
    ]


def test_backup_full17_mainline_a_uses_backup_script() -> None:
    """Manual backup entry should call backup_experiment.py for the Mainline-A run."""
    configuration = _by_name("Backup Full17 Mainline-A")
    assert configuration["program"] == "${workspaceFolder}/scripts/backup_experiment.py"
    assert configuration["args"] == [
        "--run-id",
        "paper2_full_17_mainline_a",
        "--suffix",
        "backup",
        "--require-existing",
    ]


def test_stage1_ppo_starter_launch_invokes_tuner() -> None:
    """PPO starter launch should run the Stage-1 tuner with the PPO search config."""
    configuration = _by_name("Stage-1 Tune PPO Starter")
    assert configuration["program"] == "${workspaceFolder}/scripts/tune_mainline_a_stage1.py"
    assert configuration["args"] == [
        "--algorithm",
        "PPO",
        "--search-config",
        "configs/tuning/stage1_ppo_mainline_a.yaml",
        "--mode",
        "starter",
        "--trials",
        "4",
        "--timesteps",
        "10000",
        "--seeds",
        "42",
        "--environment-profile",
        "mainline-a",
        "--device",
        "auto",
        "--output-dir",
        "outputs/stage1",
    ]


def test_stage1_coma_starter_launch_invokes_tuner() -> None:
    """COMA starter launch should run the Stage-1 tuner with the COMA search config."""
    configuration = _by_name("Stage-1 Tune COMA Starter")
    assert configuration["program"] == "${workspaceFolder}/scripts/tune_mainline_a_stage1.py"
    assert configuration["args"] == [
        "--algorithm",
        "COMA",
        "--search-config",
        "configs/tuning/stage1_coma_mainline_a.yaml",
        "--mode",
        "starter",
        "--trials",
        "4",
        "--timesteps",
        "10000",
        "--seeds",
        "42",
        "--environment-profile",
        "mainline-a",
        "--device",
        "auto",
        "--output-dir",
        "outputs/stage1",
    ]


def test_legacy_entry_is_explicit_fallback() -> None:
    """Legacy launch should require an explicit profile and fallback run id."""
    configuration = _by_name("Legacy Full 17 Fresh (Explicit Fallback)")
    assert configuration["args"] == [
        "start",
        "--preset",
        "full17",
        "--environment-profile",
        "legacy",
        "--run-id",
        "paper2_full_17_legacy_fallback",
        "--fresh",
    ]


def test_launch_json_has_no_per_algorithm_reset_entries() -> None:
    """Per-algorithm reset entries should stay out of launch.json."""
    names = [item["name"] for item in _launch_configurations()]
    assert all("Reset" not in name for name in names)
