"""Tests for experiment manager CLI."""

import json

import pytest

import scripts.experiment_manager as cli


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    output = capsys.readouterr().out
    assert exc_info.value.code == 0
    assert "--preset" in output
    assert "--fresh" in output


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


def test_cli_start_fresh_deletes_existing_before_create(monkeypatch) -> None:
    events: list[str] = []

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return True

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def backup_experiment(self, run_id: str, *, include_plots: bool = False, suffix: str = "backup") -> None:
            assert run_id == "vscode_quick"
            assert include_plots is False
            assert suffix == "auto"
            events.append("backup")

        def delete_experiment(self, run_id: str) -> None:
            assert run_id == "vscode_quick"
            events.append("delete")

        def create_experiment(self, **kwargs):
            assert kwargs["run_id"] == "vscode_quick"
            events.append("create")

        def start_or_resume(self, run_id: str):
            assert run_id == "vscode_quick"
            events.append("start")

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["start", "--preset", "quick", "--fresh"])
    assert events == ["backup", "delete", "create", "start"]


def test_cli_start_fresh_no_backup_skips_backup(monkeypatch) -> None:
    events: list[str] = []

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return True

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def backup_experiment(self, run_id: str, *, include_plots: bool = False, suffix: str = "backup") -> None:
            raise AssertionError("backup_experiment should not be called when --no-backup is set")

        def delete_experiment(self, run_id: str) -> None:
            events.append("delete")

        def create_experiment(self, **kwargs):
            events.append("create")

        def start_or_resume(self, run_id: str):
            events.append("start")

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["start", "--preset", "quick", "--fresh", "--no-backup"])
    assert events == ["delete", "create", "start"]


def test_cli_start_help_mentions_no_backup(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["start", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--no-backup" in output
    assert "--environment-profile" in output


def test_cli_start_quick_preset_uses_quick_defaults(monkeypatch) -> None:
    captured: dict = {}

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return False

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def create_experiment(self, **kwargs):
            captured.update(kwargs)

        def start_or_resume(self, _run_id: str):
            return None

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["start", "--preset", "quick"])
    assert captured["run_id"] == "vscode_quick"
    assert captured["name"] == "VSCode Quick Benchmark"
    assert captured["algorithms"] == ["GRPO", "PPO", "SAC"]
    assert captured["timesteps"] == 5000
    assert captured["eval_episodes"] == 3
    assert captured["environment_profile"] == "mainline-a"


def test_cli_start_full17_preset_uses_full17_defaults(monkeypatch) -> None:
    captured: dict = {}

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return False

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def create_experiment(self, **kwargs):
            captured.update(kwargs)

        def start_or_resume(self, _run_id: str):
            return None

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["start", "--preset", "full17"])
    assert captured["run_id"] == "paper2_full_17_mainline_a"
    assert len(captured["algorithms"]) == 17
    assert "MATD3" in captured["algorithms"]
    assert captured["timesteps"] == 100000
    assert captured["eval_episodes"] == 10
    assert captured["environment_profile"] == "mainline-a"


def test_cli_start_single_policy_3user_preset_records_interface_metadata(monkeypatch) -> None:
    captured: dict = {}

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return False

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def create_experiment(self, **kwargs):
            captured.update(kwargs)

        def start_or_resume(self, _run_id: str):
            return None

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["start", "--preset", "single_policy_3user_full17"])
    assert captured["run_id"] == "paper2_single_policy_3user_full17_mainline_a"
    assert captured["algorithms"] == ["GRPO", "PPO", "SAC", "DDQN", "DDPG", "TD3", "A3C", "TRPO", "SimPO"]
    assert captured["timesteps"] == 100000
    assert captured["eval_episodes"] == 10
    assert captured["metadata"] == {
        "interface": "single_policy_multi_user",
        "num_users": 3,
        "shared_reward": "mean",
    }


def test_cli_start_preset_allows_explicit_run_id_override(monkeypatch) -> None:
    captured: dict = {}

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return False

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def create_experiment(self, **kwargs):
            captured.update(kwargs)

        def start_or_resume(self, run_id: str):
            assert run_id == "custom_run"

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(["start", "--preset", "quick", "--run-id", "custom_run"])
    assert captured["run_id"] == "custom_run"
    assert captured["algorithms"] == ["GRPO", "PPO", "SAC"]


def test_cli_start_supports_explicit_legacy_fallback_profile(monkeypatch) -> None:
    captured: dict = {}

    class FakeStore:
        def exists(self, _run_id: str) -> bool:
            return False

    class FakeManager:
        def __init__(self) -> None:
            self.store = FakeStore()

        def create_experiment(self, **kwargs):
            captured.update(kwargs)

        def start_or_resume(self, run_id: str):
            assert run_id == "paper2_full_17_legacy_fallback"

        def get_status(self, _run_id: str):
            return {"status": "ok"}

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    cli.main(
        [
            "start",
            "--preset",
            "full17",
            "--environment-profile",
            "legacy",
            "--run-id",
            "paper2_full_17_legacy_fallback",
        ]
    )
    assert captured["environment_profile"] == "legacy"
    assert captured["run_id"] == "paper2_full_17_legacy_fallback"


def test_cli_start_requires_run_id_without_preset(monkeypatch, capsys) -> None:
    class FakeManager:
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(cli, "ExperimentManager", lambda: FakeManager())
    code = cli.main(["start"])
    stderr = capsys.readouterr().err.strip()
    parsed = json.loads(stderr)
    assert code != 0
    assert parsed["error"] == "start requires --run-id or --preset"


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
