"""State store placeholders for experiment orchestration."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ExperimentLockError, ExperimentNotFoundError, ExperimentStateError
from .models import ExperimentManifest, ExperimentState
from .time_utils import utc_now_iso


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON via <name>.tmp and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


class FileLock:
    """Simple cross-platform file lock based on O_EXCL."""

    def __init__(self, lock_path: Path, stale_seconds: int = 86400) -> None:
        self.lock_path = lock_path
        self.stale_seconds = stale_seconds
        self._acquired = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in range(50):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._is_stale():
                    self._remove_if_exists()
                    continue
                time.sleep(0.1)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            self._acquired = True
            return

        raise ExperimentLockError(f"Unable to acquire lock: {self.lock_path}")

    def release(self) -> None:
        if self._acquired:
            self._remove_if_exists()
            self._acquired = False

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _is_stale(self) -> bool:
        try:
            modified_at = self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return (time.time() - modified_at) > self.stale_seconds

    def _remove_if_exists(self) -> None:
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            return


class JsonStateStore:
    """Placeholder state store class for module 2 step 1."""

    def __init__(self, root_dir: Path | str = "experiments") -> None:
        self.root_dir = Path(root_dir)

    def experiment_dir(self, run_id: str) -> Path:
        return self.root_dir / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.experiment_dir(run_id) / "run.json"

    def state_path(self, run_id: str) -> Path:
        return self.experiment_dir(run_id) / "state.json"

    def control_dir(self, run_id: str) -> Path:
        return self.experiment_dir(run_id) / "control"

    def stop_request_path(self, run_id: str) -> Path:
        return self.control_dir(run_id) / "stop.request"

    def create(self, manifest: ExperimentManifest, state: ExperimentState) -> None:
        if self.exists(manifest.run_id):
            raise ExperimentStateError(f"Experiment already exists: {manifest.run_id}")
        self.save_manifest(manifest)
        self.save_state(state)

    def load_manifest(self, run_id: str) -> ExperimentManifest:
        path = self.manifest_path(run_id)
        if not path.exists():
            raise ExperimentNotFoundError(f"Manifest not found: {path}")
        return ExperimentManifest.from_dict(read_json(path))

    def load_state(self, run_id: str) -> ExperimentState:
        path = self.state_path(run_id)
        if not path.exists():
            raise ExperimentNotFoundError(f"State not found: {path}")
        return ExperimentState.from_dict(read_json(path))

    def save_manifest(self, manifest: ExperimentManifest) -> None:
        write_json_atomic(self.manifest_path(manifest.run_id), manifest.to_dict())

    def save_state(self, state: ExperimentState) -> None:
        write_json_atomic(self.state_path(state.run_id), state.to_dict())

    def with_lock(self, run_id: str) -> AbstractContextManager[FileLock]:
        lock_path = self.experiment_dir(run_id) / "state.lock"
        return FileLock(lock_path)

    def exists(self, run_id: str) -> bool:
        return self.manifest_path(run_id).exists() and self.state_path(run_id).exists()

    def list_run_ids(self) -> list[str]:
        if not self.root_dir.exists():
            return []
        run_ids: list[str] = []
        for candidate in self.root_dir.iterdir():
            if not candidate.is_dir():
                continue
            run_id = candidate.name
            if self.exists(run_id):
                run_ids.append(run_id)
        return sorted(run_ids)

    def write_stop_request(self, run_id: str) -> Path:
        output = self.stop_request_path(run_id)
        payload = {"requested_at": utc_now_iso(), "pid": os.getpid()}
        write_json_atomic(output, payload)
        return output

    def clear_stop_request(self, run_id: str) -> None:
        try:
            self.stop_request_path(run_id).unlink()
        except FileNotFoundError:
            return

    def has_stop_request(self, run_id: str) -> bool:
        return self.stop_request_path(run_id).exists()
