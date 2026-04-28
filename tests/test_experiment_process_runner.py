"""Tests for experiment process runner."""

from src.experiment.process_runner import ProcessResult
from src.experiment.process_runner import ProcessRunner
from src.experiment.registry import TrainCommandBuilder
from src.experiment.models import AlgorithmSpec
from src.experiment.state_store import JsonStateStore


def test_process_result_succeeded() -> None:
    success = ProcessResult(
        exit_code=0,
        interrupted=False,
        stdout_log="out.log",
        stderr_log="err.log",
        result_json="result.json",
        checkpoint_dir="ckpt",
    )
    failed = ProcessResult(
        exit_code=1,
        interrupted=False,
        stdout_log="out.log",
        stderr_log="err.log",
        result_json="result.json",
        checkpoint_dir="ckpt",
    )
    interrupted = ProcessResult(
        exit_code=0,
        interrupted=True,
        stdout_log="out.log",
        stderr_log="err.log",
        result_json="result.json",
        checkpoint_dir="ckpt",
    )

    assert success.succeeded() is True
    assert failed.succeeded() is False
    assert interrupted.succeeded() is False


def test_algorithm_paths_are_under_experiment_dir(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    runner = ProcessRunner(store=store)
    paths = runner.algorithm_paths(run_id="demo", algorithm_name="GRPO")

    base_dir = store.experiment_dir("demo")
    for path in paths.values():
        path.relative_to(base_dir)


def test_send_interrupt_fallback_calls_terminate(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminate_called = False

        def send_signal(self, _sig) -> None:
            raise RuntimeError("send failed")

        def terminate(self) -> None:
            self.terminate_called = True

    monkeypatch.setattr("src.experiment.process_runner.sys.platform", "linux")
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    runner = ProcessRunner(store=store)
    process = FakeProcess()

    runner._send_interrupt(process)  # type: ignore[arg-type]
    assert process.terminate_called is True


def test_run_algorithm_success_with_fake_subprocess(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 1234
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

        def send_signal(self, _sig) -> None:
            return

        def terminate(self) -> None:
            return

        def kill(self) -> None:
            return

    monkeypatch.setattr("src.experiment.process_runner.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("src.experiment.process_runner.time.sleep", lambda _: None)

    store = JsonStateStore(root_dir=tmp_path / "experiments")
    runner = ProcessRunner(store=store, command_builder=TrainCommandBuilder(project_root=tmp_path))
    spec = AlgorithmSpec(
        name="GRPO",
        config_path="configs/algorithms/grpo.yaml",
        timesteps=5000,
        seed=42,
    )

    paths = runner.algorithm_paths(run_id="demo", algorithm_name="GRPO")
    paths["result_json"].parent.mkdir(parents=True, exist_ok=True)
    paths["result_json"].write_text("{}", encoding="utf-8")

    result = runner.run_algorithm(run_id="demo", spec=spec)
    assert result.succeeded() is True


def test_run_algorithm_interrupted_when_stop_requested(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            self.pid = 2345
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

        def send_signal(self, _sig) -> None:
            self.returncode = 130

        def terminate(self) -> None:
            self.returncode = 130

        def kill(self) -> None:
            self.returncode = 137

    monkeypatch.setattr("src.experiment.process_runner.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("src.experiment.process_runner.time.sleep", lambda _: None)

    store = JsonStateStore(root_dir=tmp_path / "experiments")
    monkeypatch.setattr(store, "has_stop_request", lambda _run_id: True)

    runner = ProcessRunner(store=store, command_builder=TrainCommandBuilder(project_root=tmp_path))
    spec = AlgorithmSpec(
        name="GRPO",
        config_path="configs/algorithms/grpo.yaml",
        timesteps=5000,
        seed=42,
    )

    result = runner.run_algorithm(run_id="demo", spec=spec)
    assert result.interrupted is True


def test_cleanup_process_file_is_idempotent(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    runner = ProcessRunner(store=store)
    process_file = store.experiment_dir("demo") / "process.json"
    process_file.parent.mkdir(parents=True, exist_ok=True)
    process_file.write_text("{}", encoding="utf-8")

    runner.cleanup_process_file("demo")
    runner.cleanup_process_file("demo")
