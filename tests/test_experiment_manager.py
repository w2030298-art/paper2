"""Tests for experiment manager."""

from src.experiment.manager import ExperimentManager
from src.experiment.local_index import LocalExperimentIndex
from src.experiment.models import (
    AlgorithmRunRecord,
    AlgorithmSpec,
    AlgorithmStatus,
    ExperimentManifest,
    ExperimentState,
    ExperimentStatus,
)
from src.experiment.process_runner import ProcessRunner
from src.experiment.process_runner import ProcessResult
from src.experiment.registry import AlgorithmRegistry
from src.experiment.state_store import JsonStateStore


def test_manager_default_init() -> None:
    manager = ExperimentManager()
    assert manager.store is not None
    assert manager.registry is not None
    assert manager.runner is not None
    assert manager.index is not None


def test_create_experiment_writes_manifest_and_state(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    registry = AlgorithmRegistry()
    runner = ProcessRunner(store=store)
    manager = ExperimentManager(store=store, registry=registry, runner=runner, index=index)

    manager.create_experiment(
        run_id="demo",
        name="Demo Run",
        algorithms=["grpo", "ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )

    manifest = store.load_manifest("demo")
    state = store.load_state("demo")
    assert len(manifest.algorithms) == 2
    assert len(state.records) == 2
    assert state.current_index == 0


def test_start_or_resume_completes_two_algorithms(tmp_path, monkeypatch) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    registry = AlgorithmRegistry()
    runner = ProcessRunner(store=store)
    manager = ExperimentManager(store=store, registry=registry, runner=runner, index=index)

    manager.create_experiment(
        run_id="demo",
        name="Demo Run",
        algorithms=["grpo", "ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )

    def fake_run_algorithm(*, run_id, spec):
        return ProcessResult(
            exit_code=0,
            interrupted=False,
            stdout_log=f"experiments/{run_id}/artifacts/{spec.name}/stdout.log",
            stderr_log=f"experiments/{run_id}/artifacts/{spec.name}/stderr.log",
            result_json=f"experiments/{run_id}/artifacts/{spec.name}/result.json",
            checkpoint_dir=f"experiments/{run_id}/artifacts/{spec.name}/checkpoints",
        )

    monkeypatch.setattr(runner, "run_algorithm", fake_run_algorithm)

    state = manager.start_or_resume("demo")
    assert state.completed_algorithms == ["GRPO", "PPO"]


def test_resume_restarts_first_unfinished_algorithm(tmp_path, monkeypatch) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    registry = AlgorithmRegistry()
    runner = ProcessRunner(store=store)
    manager = ExperimentManager(store=store, registry=registry, runner=runner, index=index)

    now = "2026-04-28T12:00:00Z"
    specs = registry.build_specs(
        ["grpo", "ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )
    manifest = manager.create_experiment(
        run_id="resume_demo",
        name="Resume Demo",
        algorithms=["grpo", "ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )
    state = ExperimentState(
        schema_version=1,
        run_id="resume_demo",
        status=ExperimentStatus.STOPPED,
        current_index=1,
        records=[
            AlgorithmRunRecord(name="GRPO", status=AlgorithmStatus.COMPLETED),
            AlgorithmRunRecord(name="PPO", status=AlgorithmStatus.INTERRUPTED),
        ],
        completed_algorithms=["GRPO"],
        updated_at=now,
    )
    store.save_manifest(manifest)
    store.save_state(state)

    run_order: list[str] = []

    def fake_run_algorithm(*, run_id, spec):
        run_order.append(spec.name)
        return ProcessResult(
            exit_code=0,
            interrupted=False,
            stdout_log=f"experiments/{run_id}/artifacts/{spec.name}/stdout.log",
            stderr_log=f"experiments/{run_id}/artifacts/{spec.name}/stderr.log",
            result_json=f"experiments/{run_id}/artifacts/{spec.name}/result.json",
            checkpoint_dir=f"experiments/{run_id}/artifacts/{spec.name}/checkpoints",
        )

    monkeypatch.setattr(runner, "run_algorithm", fake_run_algorithm)
    manager.start_or_resume("resume_demo")
    assert run_order[0] == "PPO"


def test_request_stop_sets_stop_requested(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    registry = AlgorithmRegistry()
    runner = ProcessRunner(store=store)
    manager = ExperimentManager(store=store, registry=registry, runner=runner, index=index)

    manager.create_experiment(
        run_id="stop_demo",
        name="Stop Demo",
        algorithms=["grpo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )
    state = manager.request_stop("stop_demo")
    assert state.stop_requested is True
    assert store.stop_request_path("stop_demo").exists()


def test_get_status_reports_next_algorithm(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    manager = ExperimentManager(
        store=store,
        registry=AlgorithmRegistry(),
        runner=ProcessRunner(store=store),
        index=index,
    )
    manager.create_experiment(
        run_id="status_demo",
        name="Status Demo",
        algorithms=["grpo", "ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )

    status = manager.get_status("status_demo")
    assert status["next_algorithm"] == "GRPO"


def test_list_runs_sorted_by_updated_at(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    manager = ExperimentManager(
        store=store,
        registry=AlgorithmRegistry(),
        runner=ProcessRunner(store=store),
        index=index,
    )

    manager.create_experiment(
        run_id="old_run",
        name="Old Run",
        algorithms=["grpo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )
    state_old = store.load_state("old_run")
    state_old.updated_at = "2026-04-28T10:00:00Z"
    store.save_state(state_old)

    manager.create_experiment(
        run_id="new_run",
        name="New Run",
        algorithms=["ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )
    state_new = store.load_state("new_run")
    state_new.updated_at = "2026-04-28T12:00:00Z"
    store.save_state(state_new)

    runs = manager.list_runs(rebuild_index=True)
    assert [item["run_id"] for item in runs] == ["new_run", "old_run"]


def test_reset_failed_algorithm_moves_current_index_back(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    manager = ExperimentManager(
        store=store,
        registry=AlgorithmRegistry(),
        runner=ProcessRunner(store=store),
        index=index,
    )
    manager.create_experiment(
        run_id="reset_demo",
        name="Reset Demo",
        algorithms=["grpo", "ppo"],
        timesteps=5000,
        seed=42,
        device="auto",
        eval_episodes=3,
    )

    state = store.load_state("reset_demo")
    state.records[0].status = AlgorithmStatus.COMPLETED
    state.records[1].status = AlgorithmStatus.FAILED
    state.records[1].error = "boom"
    state.records[1].exit_code = 1
    state.current_index = 1
    state.completed_algorithms = ["GRPO"]
    store.save_state(state)

    reset_state = manager.reset_failed_algorithm("reset_demo", "PPO")
    assert reset_state.current_index == 1
    assert reset_state.records[1].status == AlgorithmStatus.PENDING
    assert reset_state.records[1].error is None


def test_rebuild_index_after_manual_json_copy(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")
    manager = ExperimentManager(
        store=store,
        registry=AlgorithmRegistry(),
        runner=ProcessRunner(store=store),
        index=index,
    )

    manifest = ExperimentManifest(
        schema_version=1,
        run_id="manual_demo",
        name="Manual Demo",
        created_at="2026-04-28T12:00:00Z",
        updated_at="2026-04-28T12:00:00Z",
        algorithms=[
            AlgorithmSpec(
                name="GRPO",
                config_path="configs/algorithms/grpo.yaml",
                timesteps=5000,
                seed=42,
            )
        ],
        experiment_dir=str(store.experiment_dir("manual_demo")),
    )
    state = ExperimentState(
        schema_version=1,
        run_id="manual_demo",
        status=ExperimentStatus.INITIALIZED,
        current_index=0,
        records=[AlgorithmRunRecord(name="GRPO")],
        updated_at="2026-04-28T12:00:00Z",
    )
    store.create(manifest, state)

    manager.rebuild_index()
    runs = manager.list_runs(rebuild_index=False)
    assert any(item["run_id"] == "manual_demo" for item in runs)
