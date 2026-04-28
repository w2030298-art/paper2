"""Tests for experiment local index."""

from src.experiment.local_index import LocalExperimentIndex
from src.experiment.models import (
    AlgorithmRunRecord,
    AlgorithmSpec,
    AlgorithmStatus,
    ExperimentManifest,
    ExperimentState,
    ExperimentStatus,
)
from src.experiment.state_store import JsonStateStore


def test_local_index_rebuild_from_store(tmp_path) -> None:
    store = JsonStateStore(root_dir=tmp_path / "experiments")
    index = LocalExperimentIndex(db_path=tmp_path / "experiments/.index.sqlite3")

    manifest_a = ExperimentManifest(
        schema_version=1,
        run_id="run_a",
        name="Run A",
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
        experiment_dir="experiments/run_a",
    )
    state_a = ExperimentState(
        schema_version=1,
        run_id="run_a",
        status=ExperimentStatus.COMPLETED,
        current_index=1,
        records=[AlgorithmRunRecord(name="GRPO", status=AlgorithmStatus.COMPLETED)],
        completed_algorithms=["GRPO"],
        updated_at="2026-04-28T12:30:00Z",
    )

    manifest_b = ExperimentManifest(
        schema_version=1,
        run_id="run_b",
        name="Run B",
        created_at="2026-04-28T13:00:00Z",
        updated_at="2026-04-28T13:00:00Z",
        algorithms=[
            AlgorithmSpec(
                name="PPO",
                config_path="configs/algorithms/ppo.yaml",
                timesteps=4000,
                seed=7,
            )
        ],
        experiment_dir="experiments/run_b",
    )
    state_b = ExperimentState(
        schema_version=1,
        run_id="run_b",
        status=ExperimentStatus.RUNNING,
        current_index=0,
        records=[AlgorithmRunRecord(name="PPO", status=AlgorithmStatus.PENDING)],
        updated_at="2026-04-28T13:05:00Z",
    )

    store.create(manifest_a, state_a)
    store.create(manifest_b, state_b)

    index.rebuild_from_store(store)
    runs = index.list_runs()

    assert len(runs) == 2
    assert {item["run_id"] for item in runs} == {"run_a", "run_b"}
