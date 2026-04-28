"""Result writer for benchmark compatibility exports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import AlgorithmRunRecord, AlgorithmStatus
from .state_store import JsonStateStore


class BenchmarkResultWriter:
    """Convert per-algorithm result JSON into benchmark-compatible entries."""

    def __init__(self, store: JsonStateStore) -> None:
        self.store = store

    def load_algorithm_result(self, result_path: Path) -> dict[str, Any]:
        with result_path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle))

    def to_benchmark_entry(
        self,
        *,
        record: AlgorithmRunRecord,
        result_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        final_eval = result_payload.get("final_eval", {})
        if not isinstance(final_eval, Mapping):
            final_eval = {}

        return {
            "algorithm": result_payload.get("algorithm", record.name),
            "environment": result_payload.get("environment"),
            "seed": result_payload.get("seed"),
            "device": result_payload.get("device"),
            "train_timesteps": result_payload.get("train_timesteps"),
            "status": result_payload.get("status"),
            "final_reward_mean": final_eval.get("eval/reward_mean"),
            "final_reward_std": final_eval.get("eval/reward_std"),
            "final_latency_mean": final_eval.get("eval/latency_mean"),
            "final_energy_mean": final_eval.get("eval/energy_mean"),
            "final_comm_score": final_eval.get("eval/comm_score"),
            "checkpoint_dir": result_payload.get("checkpoint_dir"),
        }

    def export_run(self, run_id: str, output_path: Path | str | None = None) -> Path:
        manifest = self.store.load_manifest(run_id)
        state = self.store.load_state(run_id)

        records_by_name = {record.name: record for record in state.records}
        entries: list[dict[str, Any]] = []

        for spec in manifest.algorithms:
            record = records_by_name.get(spec.name)
            if record is None or record.status != AlgorithmStatus.COMPLETED:
                continue

            if not record.result_path:
                entries.append(
                    {
                        "algorithm": record.name,
                        "status": "failed",
                        "error": "Completed record result JSON not found: ",
                        "checkpoint_dir": record.checkpoint_dir,
                    }
                )
                continue

            result_path = Path(record.result_path)
            if not result_path.exists():
                entries.append(
                    {
                        "algorithm": record.name,
                        "status": "failed",
                        "error": f"Completed record result JSON not found: {result_path}",
                        "checkpoint_dir": record.checkpoint_dir,
                    }
                )
                continue

            payload = self.load_algorithm_result(result_path)
            entries.append(self.to_benchmark_entry(record=record, result_payload=payload))

        target = (
            Path(output_path)
            if output_path is not None
            else Path("results") / f"benchmark_{run_id}.json"
        )
        latest = Path("results") / "benchmark.json"

        self._write_json_atomic(target, entries)
        self._write_json_atomic(latest, entries)
        return target

    @staticmethod
    def _write_json_atomic(path: Path, payload: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
