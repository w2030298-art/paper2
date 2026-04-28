"""Local SQLite index for experiment summaries."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .models import ExperimentManifest, ExperimentState
from .state_store import JsonStateStore


class LocalExperimentIndex:
    """SQLite cache index for fast experiment listing."""

    def __init__(self, db_path: Path | str = "experiments/.index.sqlite3") -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_index INTEGER NOT NULL,
                    total_algorithms INTEGER NOT NULL,
                    completed_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    experiment_dir TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert(self, manifest: ExperimentManifest, state: ExperimentState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_id, name, status, current_index, total_algorithms,
                    completed_count, updated_at, experiment_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    current_index=excluded.current_index,
                    total_algorithms=excluded.total_algorithms,
                    completed_count=excluded.completed_count,
                    updated_at=excluded.updated_at,
                    experiment_dir=excluded.experiment_dir
                """,
                (
                    manifest.run_id,
                    manifest.name,
                    state.status.value,
                    state.current_index,
                    len(manifest.algorithms),
                    len(state.completed_algorithms),
                    state.updated_at,
                    manifest.experiment_dir,
                ),
            )
            conn.commit()

    def rebuild_from_store(self, store: JsonStateStore) -> None:
        self.initialize()
        with self._connect() as conn:
            conn.execute("DELETE FROM experiment_runs")
            conn.commit()

        for run_id in store.list_run_ids():
            manifest = store.load_manifest(run_id)
            state = store.load_state(run_id)
            self.upsert(manifest, state)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, name, status, current_index, total_algorithms,
                       completed_count, updated_at, experiment_dir
                FROM experiment_runs
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, name, status, current_index, total_algorithms,
                       completed_count, updated_at, experiment_dir
                FROM experiment_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
