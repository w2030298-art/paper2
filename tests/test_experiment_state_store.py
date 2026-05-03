"""Tests for experiment state store."""

import pytest

from src.experiment.errors import ExperimentLockError, ExperimentStateError
from src.experiment.models import (
    AlgorithmRunRecord,
    AlgorithmSpec,
    ExperimentManifest,
    ExperimentState,
    ExperimentStatus,
)
from src.experiment.state_store import FileLock, JsonStateStore, read_json, write_json_atomic
from src.experiment.time_utils import utc_now_iso


def test_utc_now_iso_suffix() -> None:
    assert utc_now_iso().endswith("Z")


def test_write_json_atomic_roundtrip(tmp_path) -> None:
    payload = {"message": "中文字段", "value": 42}
    output = tmp_path / "demo" / "state.json"
    write_json_atomic(output, payload)
    loaded = read_json(output)
    assert loaded == payload


def test_file_lock_exclusive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.experiment.state_store.time.sleep", lambda _: None)

    lock_path = tmp_path / "demo.lock"
    first = FileLock(lock_path)
    second = FileLock(lock_path)

    first.acquire()
    with pytest.raises(ExperimentLockError):
        second.acquire()

    first.release()
    second.acquire()
    second.release()


def test_json_state_store_create_load_update(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")

    manifest = ExperimentManifest(
        schema_version=1,
        run_id="demo",
        name="Demo Run",
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
    )
    state = ExperimentState(
        schema_version=1,
        run_id="demo",
        status=ExperimentStatus.INITIALIZED,
        current_index=0,
        records=[AlgorithmRunRecord(name="GRPO")],
        updated_at="2026-04-28T12:00:00Z",
    )

    store.create(manifest, state)

    loaded_manifest = store.load_manifest("demo")
    loaded_state = store.load_state("demo")

    assert loaded_manifest.run_id == "demo"
    assert len(loaded_manifest.algorithms) == 1
    assert loaded_manifest.algorithms[0].name == "GRPO"
    assert loaded_state.run_id == "demo"
    assert loaded_state.status == ExperimentStatus.INITIALIZED

    loaded_state.status = ExperimentStatus.RUNNING
    loaded_state.updated_at = "2026-04-28T12:05:00Z"
    store.save_state(loaded_state)

    updated_state = store.load_state("demo")
    assert updated_state.status == ExperimentStatus.RUNNING


def test_state_store_delete_missing_run_is_noop(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    store.delete("missing")
    assert not store.experiment_dir("missing").exists()


def test_state_store_delete_existing_run_removes_directory(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    run_dir = store.experiment_dir("demo")
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text("{}", encoding="utf-8")

    store.delete("demo")
    assert not run_dir.exists()


def test_state_store_delete_refuses_running_experiment_with_process_json(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    run_dir = store.experiment_dir("demo")
    run_dir.mkdir(parents=True)
    (run_dir / "process.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExperimentStateError, match="Cannot delete running experiment: demo"):
        store.delete("demo")
    assert run_dir.exists()


def test_state_store_delete_refuses_path_outside_root(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    with pytest.raises(ExperimentStateError, match="Refusing to delete outside experiment root"):
        store.delete("../outside")
    assert outside_dir.exists()
