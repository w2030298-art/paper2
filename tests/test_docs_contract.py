"""Regression tests for the v9 AI state contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AI_DIR = ROOT / ".ai"
LEDGER_PATH = AI_DIR / "ledger.json"
INBOX_PLAN = AI_DIR / "inbox" / "plan.md"
REF_DIR = ROOT / "ref"

VALID_LEDGER_STATES = {
    "PENDING",
    "IN_PROGRESS",
    "ALL_CLEAR",
    "NEEDS_REVIEW",
    "NEEDS_WEB",
    "BLOCKED",
    "DONE",
}


def _ledger() -> dict[str, Any]:
    """Load the v9 ledger JSON document."""
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_v9_ledger_exists_and_has_required_sections() -> None:
    """The v9 ledger is the repository's single machine state source."""
    assert LEDGER_PATH.is_file()

    ledger = _ledger()

    assert ledger["schema_version"] == "wkstruc-ledger/v9"
    for key in ("task_id", "state", "progress", "last_run", "validation", "open_items"):
        assert key in ledger


def test_v9_ledger_state_progress_and_validation_are_well_formed() -> None:
    """Ledger state, progress, and validation fields should be machine-checkable."""
    ledger = _ledger()

    assert ledger["state"] in VALID_LEDGER_STATES
    assert isinstance(ledger["progress"]["done"], int)
    assert isinstance(ledger["progress"]["total"], int)
    assert isinstance(ledger["progress"]["label"], str)
    assert ledger["progress"]["done"] <= ledger["progress"]["total"]
    assert isinstance(ledger["last_run"]["at"], str)
    assert isinstance(ledger["last_run"]["agent"], str)
    assert isinstance(ledger["last_run"]["summary"], str)
    assert ledger["validation"]["status"] in {"PASS", "FAIL", "PARTIAL", "NOT_RUN", "UNKNOWN"}
    assert isinstance(ledger["validation"]["commands"], list)
    assert isinstance(ledger["validation"]["summary"], str)
    assert isinstance(ledger["open_items"], list)


def test_inbox_plan_is_transient_after_consumption() -> None:
    """The one-shot inbox plan may be absent after successful consumption."""
    if not INBOX_PLAN.exists():
        assert (AI_DIR / "inbox").is_dir()
        return

    text = INBOX_PLAN.read_text(encoding="utf-8")
    assert "# Web Plan Packet" in text
    assert "After Consumed: delete or clear this file" in text


def test_ref_directory_is_reference_material_not_state() -> None:
    """Reference reports may live in ref/, but they are not the state source."""
    assert REF_DIR.is_dir()
    assert LEDGER_PATH.is_file()
    assert not (REF_DIR / "ledger.json").exists()
