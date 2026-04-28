"""Experiment manager core logic."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .local_index import LocalExperimentIndex
from .models import (
    AlgorithmRunRecord,
    AlgorithmStatus,
    ExperimentManifest,
    ExperimentState,
    ExperimentStatus,
)
from .process_runner import ProcessRunner
from .registry import AlgorithmRegistry
from .state_store import JsonStateStore
from .time_utils import utc_now_iso


class ExperimentManager:
    """Manage resumable experiment runs."""

    def __init__(
        self,
        store: JsonStateStore | None = None,
        registry: AlgorithmRegistry | None = None,
        runner: ProcessRunner | None = None,
        index: LocalExperimentIndex | None = None,
    ) -> None:
        self.store = store or JsonStateStore("experiments")
        self.registry = registry or AlgorithmRegistry()
        self.runner = runner or ProcessRunner(store=self.store)
        self.index = index or LocalExperimentIndex("experiments/.index.sqlite3")

    def create_experiment(
        self,
        *,
        run_id: str,
        name: str,
        algorithms: Sequence[str],
        timesteps: int,
        seed: int,
        device: str,
        eval_episodes: int,
        env: str = "auto",
        output_dir: str = "results",
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentManifest:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("run_id contains invalid characters")
        if not algorithms:
            raise ValueError("algorithms cannot be empty")
        if timesteps <= 0:
            raise ValueError("timesteps must be > 0")
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be > 0")

        specs = self.registry.build_specs(
            algorithms,
            timesteps=timesteps,
            seed=seed,
            device=device,
            eval_episodes=eval_episodes,
            env=env,
        )
        records = [AlgorithmRunRecord(name=spec.name) for spec in specs]

        now = utc_now_iso()
        manifest = ExperimentManifest(
            schema_version=1,
            run_id=run_id,
            name=name,
            created_at=now,
            updated_at=now,
            algorithms=specs,
            output_dir=output_dir,
            experiment_dir=str(self.store.experiment_dir(run_id)),
            metadata=metadata or {},
        )
        state = ExperimentState(
            schema_version=1,
            run_id=run_id,
            status=ExperimentStatus.INITIALIZED,
            current_index=0,
            records=records,
            updated_at=now,
        )

        self.store.create(manifest, state)
        self.index.initialize()
        self.index.upsert(manifest, state)
        return manifest

    def start_or_resume(self, run_id: str) -> ExperimentState:
        with self.store.with_lock(run_id):
            manifest = self.store.load_manifest(run_id)
            state = self.store.load_state(run_id)
            self.store.clear_stop_request(run_id)
            if state.status == ExperimentStatus.COMPLETED:
                return state
            state.status = ExperimentStatus.RUNNING
            state.updated_at = utc_now_iso()
            self.store.save_state(state)
            self.index.initialize()
            self.index.upsert(manifest, state)

        while True:
            with self.store.with_lock(run_id):
                manifest = self.store.load_manifest(run_id)
                state = self.store.load_state(run_id)

                if self.store.has_stop_request(run_id) or state.stop_requested:
                    state.status = ExperimentStatus.STOPPED
                    state.updated_at = utc_now_iso()
                    self.store.save_state(state)
                    self.index.upsert(manifest, state)
                    return state

                index = state.next_pending_index()
                if index is None:
                    state.status = ExperimentStatus.COMPLETED
                    state.updated_at = utc_now_iso()
                    self.store.save_state(state)
                    self.index.upsert(manifest, state)
                    return state

                record = state.records[index]
                spec = manifest.algorithms[index]
                record.mark_running(started_at=utc_now_iso(), device=spec.device)
                state.current_index = index
                state.status = ExperimentStatus.RUNNING
                state.updated_at = utc_now_iso()
                self.store.save_state(state)
                self.index.upsert(manifest, state)

            result = self.runner.run_algorithm(run_id=run_id, spec=spec)

            with self.store.with_lock(run_id):
                manifest = self.store.load_manifest(run_id)
                state = self.store.load_state(run_id)
                record = state.records[index]
                finished_at = utc_now_iso()

                if result.succeeded() and result.error is None:
                    record.mark_completed(
                        finished_at=finished_at,
                        exit_code=result.exit_code or 0,
                        result_path=result.result_json,
                        checkpoint_dir=result.checkpoint_dir,
                    )
                    if record.name not in state.completed_algorithms:
                        state.completed_algorithms.append(record.name)
                    state.current_index = index + 1
                    state.status = ExperimentStatus.RUNNING
                    state.last_error = None
                    if state.next_pending_index() is None:
                        state.status = ExperimentStatus.COMPLETED
                    state.updated_at = utc_now_iso()
                    self.store.save_state(state)
                    self.index.upsert(manifest, state)
                    if state.status == ExperimentStatus.COMPLETED:
                        return state
                    continue

                if result.interrupted:
                    record.mark_interrupted(
                        finished_at=finished_at,
                        exit_code=result.exit_code,
                        error=result.error,
                    )
                    state.current_index = index
                    state.status = ExperimentStatus.STOPPED
                    state.updated_at = utc_now_iso()
                    self.store.save_state(state)
                    self.index.upsert(manifest, state)
                    return state

                message = result.error or f"Process failed with exit code: {result.exit_code}"
                record.mark_failed(
                    finished_at=finished_at,
                    exit_code=result.exit_code,
                    error=message,
                )
                state.current_index = index
                state.status = ExperimentStatus.FAILED
                state.last_error = message
                state.updated_at = utc_now_iso()
                self.store.save_state(state)
                self.index.upsert(manifest, state)
                return state

    def request_stop(self, run_id: str) -> ExperimentState:
        with self.store.with_lock(run_id):
            manifest = self.store.load_manifest(run_id)
            state = self.store.load_state(run_id)
            self.store.write_stop_request(run_id)
            if state.status != ExperimentStatus.COMPLETED:
                state.request_stop(updated_at=utc_now_iso())
            else:
                state.updated_at = utc_now_iso()
            self.store.save_state(state)
            self.index.initialize()
            self.index.upsert(manifest, state)
            return state

    def get_status(self, run_id: str) -> dict[str, Any]:
        manifest = self.store.load_manifest(run_id)
        state = self.store.load_state(run_id)
        next_index = state.next_pending_index()
        next_algorithm = None
        if next_index is not None and next_index < len(manifest.algorithms):
            next_algorithm = manifest.algorithms[next_index].name
        return {
            "run_id": manifest.run_id,
            "name": manifest.name,
            "status": state.status.value,
            "current_index": state.current_index,
            "total_algorithms": len(manifest.algorithms),
            "completed_count": len(state.completed_algorithms),
            "completed_algorithms": list(state.completed_algorithms),
            "next_algorithm": next_algorithm,
            "updated_at": state.updated_at,
        }

    def list_runs(self, rebuild_index: bool = True) -> list[dict[str, Any]]:
        self.index.initialize()
        if rebuild_index:
            self.index.rebuild_from_store(self.store)
        runs = self.index.list_runs()
        return sorted(runs, key=lambda item: item["updated_at"], reverse=True)

    def reset_failed_algorithm(self, run_id: str, algorithm_name: str) -> ExperimentState:
        with self.store.with_lock(run_id):
            manifest = self.store.load_manifest(run_id)
            state = self.store.load_state(run_id)
            record = state.get_record(algorithm_name)

            if record.status not in {AlgorithmStatus.FAILED, AlgorithmStatus.INTERRUPTED}:
                raise ValueError("Only failed/interrupted algorithm can be reset")

            target_index = next(
                index for index, item in enumerate(state.records) if item.name == algorithm_name
            )
            record.status = AlgorithmStatus.PENDING
            record.error = None
            record.exit_code = None
            record.started_at = None
            record.finished_at = None

            state.current_index = target_index
            allowed_completed = {
                item.name
                for index, item in enumerate(state.records)
                if index < target_index and item.status == AlgorithmStatus.COMPLETED
            }
            state.completed_algorithms = [
                name for name in state.completed_algorithms if name in allowed_completed
            ]
            state.last_error = None
            state.status = ExperimentStatus.STOPPED
            state.updated_at = utc_now_iso()
            self.store.save_state(state)
            self.index.upsert(manifest, state)
            return state

    def rebuild_index(self) -> None:
        self.index.initialize()
        self.index.rebuild_from_store(self.store)
