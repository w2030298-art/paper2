"""Tests for experiment manager CLI."""

import pytest
import json

import scripts.experiment_manager as cli


def test_cli_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


def test_cli_start_creates_when_missing(monkeypatch) -> None:
    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return False

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()
            self.create_called = False
            self.start_called = False

        def create_experiment(self, **_kwargs):
            self.create_called = True

        def start_or_resume(self, _run_id: str):
            self.start_called = True

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    fake = FakeManager()
    monkeypatch.setattr(cli, "ExperimentManager", lambda: fake)
    cli.main(["start", "--run-id", "demo"])
    assert fake.create_called is True
    assert fake.start_called is True


def test_cli_start_resumes_when_existing(monkeypatch) -> None:
    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return True

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()
            self.create_called = False
            self.start_called = False

        def create_experiment(self, **_kwargs):
            self.create_called = True

        def start_or_resume(self, _run_id: str):
            self.start_called = True

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    fake = FakeManager()
    monkeypatch.setattr(cli, "ExperimentManager", lambda: fake)
    cli.main(["start", "--run-id", "demo"])
    assert fake.create_called is False
    assert fake.start_called is True


def test_cli_stop_calls_request_stop(monkeypatch) -> None:
    class FakeState:
        def to_dict(self):
            return {"status": "stop_requested"}

    class FakeManager:
        def __init__(self) -> None:
            self.called = False

        def request_stop(self, _run_id: str):
            self.called = True
            return FakeState()

    fake = FakeManager()
    monkeypatch.setattr(cli, "ExperimentManager", lambda: fake)
    cli.main(["stop", "--run-id", "demo"])
    assert fake.called is True


def test_cli_status_prints_json(monkeypatch, capsys) -> None:
    class FakeManager:
        def get_status(self, _run_id: str):
            return {"run_id": "demo", "status": "running"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["status", "--run-id", "demo"])
    output = capsys.readouterr().out.strip()
    parsed = json.loads(output)
    assert parsed["status"] == "running"


def test_cli_list_prints_array(monkeypatch, capsys) -> None:
    class FakeManager:
        def list_runs(self, rebuild_index: bool = True):
            assert rebuild_index is True
            return [{"run_id": "a"}, {"run_id": "b"}]

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["list"])
    output = capsys.readouterr().out.strip()
    parsed = json.loads(output)
    assert isinstance(parsed, list)
    assert len(parsed) == 2


def test_cli_reset_calls_manager(monkeypatch) -> None:
    class FakeState:
        def to_dict(self):
            return {"status": "reset"}

    class FakeManager:
        def __init__(self) -> None:
            self.called = False

        def reset_failed_algorithm(self, _run_id: str, _algorithm: str):
            self.called = True
            return FakeState()

    fake = FakeManager()
    monkeypatch.setattr(cli, "ExperimentManager", lambda: fake)
    cli.main(["reset", "--run-id", "demo", "--algorithm", "GRPO"])
    assert fake.called is True


def test_cli_rebuild_index_outputs_ok(monkeypatch, capsys) -> None:
    class FakeManager:
        def rebuild_index(self):
            return None

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["rebuild-index"])
    output = capsys.readouterr().out.strip()
    parsed = json.loads(output)
    assert parsed == {"status": "ok"}


def test_cli_returns_nonzero_on_value_error(monkeypatch, capsys) -> None:
    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return True

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def start_or_resume(self, _run_id: str):
            raise ValueError("bad args")

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    code = cli.main(["start", "--run-id", "demo"])
    stderr = capsys.readouterr().err.strip()
    assert code == 4
    parsed = json.loads(stderr)
    assert parsed["status"] == "error"


def test_cli_export_calls_writer(monkeypatch, capsys) -> None:
    called = {"ok": False}

    class FakeWriter:
        def __init__(self, _store) -> None:
            pass

        def export_run(self, run_id: str, output_path=None):
            called["ok"] = True
            assert run_id == "demo"
            return "results/custom.json"

    monkeypatch.setattr(cli, "BenchmarkResultWriter", FakeWriter)
    cli.main(["export", "--run-id", "demo"])
    output = capsys.readouterr().out.strip()
    parsed = json.loads(output)
    assert called["ok"] is True
    assert parsed["status"] == "ok"
