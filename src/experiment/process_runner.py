"""Process runner for experiment orchestration."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .models import AlgorithmSpec
from .registry import TrainCommandBuilder
from .state_store import JsonStateStore, write_json_atomic
from .time_utils import utc_now_iso


class ProcessRunner:
    """Placeholder process runner for module 5 step 1."""

    def __init__(
        self,
        store: JsonStateStore,
        command_builder: TrainCommandBuilder | None = None,
    ) -> None:
        self.store = store
        self.command_builder = command_builder or TrainCommandBuilder()

    def algorithm_paths(self, *, run_id: str, algorithm_name: str) -> dict[str, Path]:
        experiment_dir = self.store.experiment_dir(run_id)
        artifact_dir = experiment_dir / "artifacts" / algorithm_name
        return {
            "artifact_dir": artifact_dir,
            "checkpoint_dir": artifact_dir / "checkpoints",
            "result_json": artifact_dir / "result.json",
            "stdout_log": artifact_dir / "stdout.log",
            "stderr_log": artifact_dir / "stderr.log",
            "process_json": experiment_dir / "process.json",
        }

    def _send_interrupt(self, process: subprocess.Popen) -> None:
        if sys.platform.startswith("win"):
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                return
            except Exception:
                process.terminate()
                return
        try:
            process.send_signal(signal.SIGINT)
            return
        except Exception:
            process.terminate()

    def run_algorithm(self, *, run_id: str, spec: AlgorithmSpec) -> ProcessResult:
        paths = self.algorithm_paths(run_id=run_id, algorithm_name=spec.name)
        paths["artifact_dir"].mkdir(parents=True, exist_ok=True)

        command = self.command_builder.build(
            run_id=run_id,
            spec=spec,
            experiment_dir=self.store.experiment_dir(run_id),
        )

        stdout_handle = None
        stderr_handle = None
        process = None
        interrupted = False
        try:
            stdout_handle = paths["stdout_log"].open("w", encoding="utf-8")
            stderr_handle = paths["stderr_log"].open("w", encoding="utf-8")
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if sys.platform.startswith("win")
                else 0
            )
            process = subprocess.Popen(
                command,
                cwd=str(self.command_builder.project_root),
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
            )
            write_json_atomic(
                paths["process_json"],
                {"command": command, "pid": process.pid, "started_at": utc_now_iso()},
            )

            while True:
                exit_code = process.poll()
                if exit_code is not None:
                    break

                if self.store.has_stop_request(run_id):
                    interrupted = True
                    self._send_interrupt(process)
                    deadline = time.time() + 30
                    while process.poll() is None and time.time() < deadline:
                        time.sleep(1)
                    if process.poll() is None:
                        process.kill()
                    break

                time.sleep(1)

            if process.poll() is None:
                process.wait()
            exit_code = process.poll()

            error = None
            if not interrupted and exit_code == 0 and not paths["result_json"].exists():
                error = "Result JSON not found"

            return ProcessResult(
                exit_code=exit_code,
                interrupted=interrupted,
                stdout_log=str(paths["stdout_log"]),
                stderr_log=str(paths["stderr_log"]),
                result_json=str(paths["result_json"]),
                checkpoint_dir=str(paths["checkpoint_dir"]),
                error=error,
            )
        finally:
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()
            self.cleanup_process_file(run_id)

    def cleanup_process_file(self, run_id: str) -> None:
        process_file = self.store.experiment_dir(run_id) / "process.json"
        try:
            process_file.unlink()
        except FileNotFoundError:
            return


@dataclass(slots=True)
class ProcessResult:
    exit_code: int | None
    interrupted: bool
    stdout_log: str
    stderr_log: str
    result_json: str
    checkpoint_dir: str
    error: str | None = None

    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.interrupted
