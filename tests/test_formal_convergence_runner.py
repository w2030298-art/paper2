"""Tests for the formal convergence protocol runner."""

import json
from pathlib import Path

import yaml

from scripts.run_formal_convergence_protocol import (
    build_formal_run_plan,
    plan_to_manifest,
    select_l2_algorithms,
)


def test_select_l2_algorithms_uses_l1_and_event_gates() -> None:
    """L2 should include L1 candidates and event-audited algorithms only."""
    matrix = yaml.safe_load(Path("configs/formal_convergence_matrix.yaml").read_text())
    selected, deferred = select_l2_algorithms(matrix)

    assert {"COMA", "MAPPO", "TRPO"}.issubset(set(selected))
    assert {"IQL", "VDN", "IPPO", "MADDPG"}.issubset(set(selected))
    assert deferred["A3C"] == "requires_single_variable_fix_before_l2"


def test_l2_plan_uses_100k_three_seed_gate() -> None:
    """The L2 run plan should match the formal matrix."""
    plan = build_formal_run_plan(phase="L2", run_id="test")
    manifest = plan_to_manifest(plan, dry_run=True, no_submit=False)

    assert plan.evidence_level == "L2"
    assert plan.steps == 100000
    assert plan.seeds == [42, 43, 44]
    assert manifest["override_id"] == "none"
    assert "PPO" not in manifest["algorithms"]


def test_l3_plan_waits_for_l2_candidates(tmp_path: Path) -> None:
    """L3 selection should be based on L2 candidate decisions."""
    payload = {
        "records": [
            {"algorithm": "COMA", "decision": "candidate_converged_under_protocol"},
            {"algorithm": "TRPO", "decision": "excluded_from_formal_claim"},
        ]
    }
    report = tmp_path / "l2.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    from scripts.run_formal_convergence_protocol import select_l3_algorithms

    assert select_l3_algorithms(report) == ["COMA"]
