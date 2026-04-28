"""Tests for experiment model primitives."""

from src.experiment.models import (
    AlgorithmRunRecord,
    ExperimentState,
    AlgorithmSpec,
    AlgorithmStatus,
    ExperimentStatus,
)


def test_status_enum_values_are_stable() -> None:
    assert ExperimentStatus.INITIALIZED.value == "initialized"
    assert ExperimentStatus.RUNNING.value == "running"
    assert ExperimentStatus.STOP_REQUESTED.value == "stop_requested"
    assert ExperimentStatus.STOPPED.value == "stopped"
    assert ExperimentStatus.COMPLETED.value == "completed"
    assert ExperimentStatus.FAILED.value == "failed"

    assert AlgorithmStatus.PENDING.value == "pending"
    assert AlgorithmStatus.RUNNING.value == "running"
    assert AlgorithmStatus.COMPLETED.value == "completed"
    assert AlgorithmStatus.INTERRUPTED.value == "interrupted"
    assert AlgorithmStatus.FAILED.value == "failed"
    assert AlgorithmStatus.SKIPPED.value == "skipped"


def test_algorithm_spec_roundtrip() -> None:
    spec = AlgorithmSpec(
        name="GRPO",
        config_path="configs/algorithms/grpo.yaml",
        timesteps=5000,
        seed=42,
    )

    restored = AlgorithmSpec.from_dict(spec.to_dict())

    assert restored.name == spec.name
    assert restored.config_path == spec.config_path
    assert restored.timesteps == spec.timesteps
    assert restored.seed == spec.seed
    assert restored.device == spec.device
    assert restored.env == spec.env
    assert restored.eval_episodes == spec.eval_episodes
    assert restored.extra_args == spec.extra_args


def test_algorithm_run_record_transitions() -> None:
    record = AlgorithmRunRecord(name="GRPO")

    record.mark_running(started_at="2026-04-28T12:00:00Z", device="cpu")
    assert record.attempts == 1
    assert record.status == AlgorithmStatus.RUNNING

    record.mark_completed(
        finished_at="2026-04-28T12:10:00Z",
        exit_code=0,
        result_path="experiments/demo/artifacts/GRPO/result.json",
        checkpoint_dir="experiments/demo/artifacts/GRPO/checkpoints",
    )
    assert record.status == AlgorithmStatus.COMPLETED


def test_experiment_state_next_pending_index() -> None:
    state = ExperimentState(
        schema_version=1,
        run_id="demo",
        status=ExperimentStatus.RUNNING,
        current_index=0,
        records=[
            AlgorithmRunRecord(name="GRPO", status=AlgorithmStatus.COMPLETED),
            AlgorithmRunRecord(name="PPO", status=AlgorithmStatus.COMPLETED),
            AlgorithmRunRecord(name="SAC", status=AlgorithmStatus.PENDING),
        ],
    )

    assert state.next_pending_index() == 2

    state.records[2].status = AlgorithmStatus.COMPLETED
    assert state.next_pending_index() is None
