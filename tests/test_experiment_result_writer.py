"""Tests for experiment result writer."""

import json

from src.experiment.models import (
    AlgorithmRunRecord,
    AlgorithmSpec,
    AlgorithmStatus,
    ExperimentManifest,
    ExperimentState,
    ExperimentStatus,
)
from src.experiment.result_writer import BenchmarkResultWriter
from src.experiment.state_store import JsonStateStore


def test_to_benchmark_entry_maps_final_eval_fields(tmp_path) -> None:
    writer = BenchmarkResultWriter(store=JsonStateStore(root_dir=tmp_path / "experiments"))
    record = AlgorithmRunRecord(name="GRPO")
    payload = {
        "algorithm": "GRPO",
        "environment": "MEC-v1",
        "seed": 42,
        "device": "cpu",
        "train_timesteps": 5000,
        "status": "success",
        "checkpoint_dir": "experiments/demo/artifacts/GRPO/checkpoints",
        "final_eval": {
            "eval/reward_mean": 1.0,
            "eval/reward_std": 0.1,
            "eval/latency_mean": 2.0,
            "eval/energy_mean": 3.0,
            "eval/comm_score": 4.0,
        },
    }

    entry = writer.to_benchmark_entry(record=record, result_payload=payload)
    assert entry["final_reward_mean"] == 1.0
    assert entry["final_reward_std"] == 0.1
    assert entry["final_latency_mean"] == 2.0
    assert entry["final_energy_mean"] == 3.0
    assert entry["final_comm_score"] == 4.0


def test_export_run_writes_benchmark_json_and_latest_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = JsonStateStore(root_dir=tmp_path / "experiments")

    result_path = tmp_path / "experiments" / "demo" / "artifacts" / "GRPO" / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "algorithm": "GRPO",
        "environment": "MEC-v1",
        "seed": 42,
        "device": "cpu",
        "train_timesteps": 5000,
        "checkpoint_dir": "experiments/demo/artifacts/GRPO/checkpoints",
        "final_eval": {"eval/reward_mean": 1.0},
        "status": "success",
    }
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")

    manifest = ExperimentManifest(
        schema_version=1,
        run_id="demo",
        name="Demo",
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
        experiment_dir=str(store.experiment_dir("demo")),
    )
    state = ExperimentState(
        schema_version=1,
        run_id="demo",
        status=ExperimentStatus.COMPLETED,
        current_index=1,
        records=[
            AlgorithmRunRecord(
                name="GRPO",
                status=AlgorithmStatus.COMPLETED,
                result_path=str(result_path),
                checkpoint_dir="experiments/demo/artifacts/GRPO/checkpoints",
            )
        ],
        completed_algorithms=["GRPO"],
        updated_at="2026-04-28T12:00:00Z",
    )
    store.create(manifest, state)

    writer = BenchmarkResultWriter(store=store)
    output = writer.export_run("demo")
    assert output.exists()
    assert (tmp_path / "results" / "benchmark.json").exists()
