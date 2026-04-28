"""Core data models for experiment orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentStatus(str, Enum):
    """Experiment-level lifecycle status."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class AlgorithmStatus(str, Enum):
    """Single algorithm execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class AlgorithmSpec:
    """Algorithm execution specification."""

    name: str
    config_path: str
    timesteps: int
    seed: int
    device: str = "auto"
    env: str = "auto"
    eval_episodes: int = 10
    extra_args: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_path": self.config_path,
            "timesteps": self.timesteps,
            "seed": self.seed,
            "device": self.device,
            "env": self.env,
            "eval_episodes": self.eval_episodes,
            "extra_args": list(self.extra_args),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AlgorithmSpec:
        extra_args_raw = data.get("extra_args", [])
        if extra_args_raw is None:
            extra_args: list[str] = []
        else:
            extra_args = [str(arg) for arg in extra_args_raw]

        return cls(
            name=str(data["name"]).strip(),
            config_path=str(data["config_path"]).strip(),
            timesteps=int(data["timesteps"]),
            seed=int(data["seed"]),
            device=str(data.get("device", "auto")).strip(),
            env=str(data.get("env", "auto")).strip(),
            eval_episodes=int(data.get("eval_episodes", 10)),
            extra_args=extra_args,
        )


@dataclass(slots=True)
class AlgorithmRunRecord:
    """Per-algorithm execution record."""

    name: str
    status: AlgorithmStatus = AlgorithmStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    attempts: int = 0
    device: str = "auto"
    result_path: str | None = None
    checkpoint_dir: str | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "attempts": self.attempts,
            "device": self.device,
            "result_path": self.result_path,
            "checkpoint_dir": self.checkpoint_dir,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AlgorithmRunRecord:
        status_raw = data.get("status", AlgorithmStatus.PENDING.value)
        exit_code_raw = data.get("exit_code")

        return cls(
            name=str(data["name"]).strip(),
            status=AlgorithmStatus(str(status_raw)),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            exit_code=None if exit_code_raw is None else int(exit_code_raw),
            attempts=int(data.get("attempts", 0)),
            device=str(data.get("device", "auto")).strip(),
            result_path=data.get("result_path"),
            checkpoint_dir=data.get("checkpoint_dir"),
            stdout_log=data.get("stdout_log"),
            stderr_log=data.get("stderr_log"),
            error=data.get("error"),
        )

    def mark_running(self, *, started_at: str, device: str) -> None:
        self.status = AlgorithmStatus.RUNNING
        self.started_at = started_at
        self.device = device
        self.attempts += 1

    def mark_completed(
        self,
        *,
        finished_at: str,
        exit_code: int,
        result_path: str,
        checkpoint_dir: str,
    ) -> None:
        self.status = AlgorithmStatus.COMPLETED
        self.finished_at = finished_at
        self.exit_code = int(exit_code)
        self.result_path = result_path
        self.checkpoint_dir = checkpoint_dir
        self.error = None

    def mark_interrupted(
        self,
        *,
        finished_at: str,
        exit_code: int | None,
        error: str | None = None,
    ) -> None:
        self.status = AlgorithmStatus.INTERRUPTED
        self.finished_at = finished_at
        self.exit_code = None if exit_code is None else int(exit_code)
        self.error = error

    def mark_failed(self, *, finished_at: str, exit_code: int | None, error: str) -> None:
        self.status = AlgorithmStatus.FAILED
        self.finished_at = finished_at
        self.exit_code = None if exit_code is None else int(exit_code)
        self.error = error


@dataclass(slots=True)
class ExperimentManifest:
    """Immutable metadata of an experiment run."""

    schema_version: int
    run_id: str
    name: str
    created_at: str
    updated_at: str
    algorithms: list[AlgorithmSpec]
    project_root: str = "."
    output_dir: str = "results"
    experiment_dir: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "algorithms": [algorithm.to_dict() for algorithm in self.algorithms],
            "project_root": self.project_root,
            "output_dir": self.output_dir,
            "experiment_dir": self.experiment_dir,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentManifest:
        algorithms_raw = data.get("algorithms", [])
        metadata_raw = data.get("metadata", {})
        return cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]).strip(),
            name=str(data["name"]).strip(),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            algorithms=[AlgorithmSpec.from_dict(item) for item in algorithms_raw],
            project_root=str(data.get("project_root", ".")).strip(),
            output_dir=str(data.get("output_dir", "results")).strip(),
            experiment_dir=str(data.get("experiment_dir", "")).strip(),
            metadata=dict(metadata_raw) if isinstance(metadata_raw, dict) else {},
        )


@dataclass(slots=True)
class ExperimentState:
    """Mutable state of an experiment run."""

    schema_version: int
    run_id: str
    status: ExperimentStatus
    current_index: int
    records: list[AlgorithmRunRecord]
    completed_algorithms: list[str] = field(default_factory=list)
    stop_requested: bool = False
    last_error: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "current_index": self.current_index,
            "records": [record.to_dict() for record in self.records],
            "completed_algorithms": list(self.completed_algorithms),
            "stop_requested": self.stop_requested,
            "last_error": self.last_error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentState:
        status_raw = data["status"]
        records_raw = data.get("records", [])
        return cls(
            schema_version=int(data["schema_version"]),
            run_id=str(data["run_id"]).strip(),
            status=ExperimentStatus(str(status_raw)),
            current_index=int(data["current_index"]),
            records=[AlgorithmRunRecord.from_dict(item) for item in records_raw],
            completed_algorithms=[str(name) for name in data.get("completed_algorithms", [])],
            stop_requested=bool(data.get("stop_requested", False)),
            last_error=data.get("last_error"),
            updated_at=str(data.get("updated_at", "")),
        )

    def next_pending_index(self) -> int | None:
        for index, record in enumerate(self.records):
            if record.status != AlgorithmStatus.COMPLETED:
                return index
        return None

    def get_record(self, algorithm_name: str) -> AlgorithmRunRecord:
        for record in self.records:
            if record.name == algorithm_name:
                return record
        raise KeyError(f"Algorithm record not found: {algorithm_name}")

    def request_stop(self, updated_at: str) -> None:
        self.stop_requested = True
        self.updated_at = updated_at
        if self.status != ExperimentStatus.COMPLETED:
            self.status = ExperimentStatus.STOP_REQUESTED

    def clear_stop(self, updated_at: str) -> None:
        self.stop_requested = False
        self.updated_at = updated_at
