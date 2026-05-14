"""Tests for strict paper2 matrix result collection."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


COLLECTOR = Path("scripts/collect_paper2_matrix_results.py")


def _run_record(tmp_path: Path, algorithm: str, seed: int, payload_kind: str = "list") -> dict:
    """Create a manifest run record and matching benchmark output."""
    output_path = tmp_path / f"{algorithm}_{seed}_{payload_kind}.json"
    metrics = {
        "social_welfare_mean": 10.0 + seed / 100.0,
        "reward_mean": 1.0 + seed / 100.0,
        "e2e_latency_mean": 0.2,
        "e2e_latency_p95": 0.3,
        "deadline_miss_rate": 0.01,
        "energy_total_mean": 0.5,
        "throughput_tasks_per_step": 2.0,
        "agent_reward_jain_mean": 0.9,
        "constraint_violation_rate": 0.0,
        "queue_wait_mean": 0.1,
        "comm_score": 75.0,
    }
    result = {
        "algorithm": algorithm,
        "seed": seed,
        "status": "success",
        **{f"final_{key}": value for key, value in metrics.items()},
    }
    if payload_kind == "list":
        payload: object = [result]
    elif payload_kind == "dict_results":
        payload = {"results": [result]}
    else:
        payload = result
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "run_id": f"L2__ID-mainline-a__{algorithm}__full__seed{seed}",
        "level": "L2_candidate_validation",
        "stage": "ID",
        "scenario": "ID-mainline-a",
        "algorithm": algorithm,
        "seed": seed,
        "steps": 100000,
        "ablation": "full",
        "result_kind": "benchmark",
        "output_path": str(output_path),
    }


def _write_manifest(path: Path, runs: list[dict], dry_run: bool = False) -> None:
    """Write a collector manifest fixture."""
    path.write_text(
        json.dumps(
            {
                "schema_version": "paper2-main-matrix-manifest/v1",
                "dry_run": dry_run,
                "level": "L2_candidate_validation",
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def _collect(manifest: Path, output: Path, strict: bool = True) -> subprocess.CompletedProcess[str]:
    """Run the collector."""
    args = [
        sys.executable,
        str(COLLECTOR),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--status-csv",
        str(output.with_name("status.csv")),
        "--missing-csv",
        str(output.with_name("missing.csv")),
    ]
    if strict:
        args.append("--strict")
    return subprocess.run(args, text=True, capture_output=True)


def test_collector_rejects_dry_run_manifest(tmp_path: Path) -> None:
    """Dry-run manifests must not be treated as evidence."""
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, [_run_record(tmp_path, "PPO", 42)], dry_run=True)

    result = _collect(manifest, tmp_path / "results.json")

    assert result.returncode != 0
    assert "dry-run manifest" in result.stderr


def test_collector_rejects_missing_output_in_strict_mode(tmp_path: Path) -> None:
    """Strict collection should fail when a planned run output is missing."""
    manifest = tmp_path / "manifest.json"
    run = _run_record(tmp_path, "PPO", 42)
    Path(run["output_path"]).unlink()
    _write_manifest(manifest, [run])

    result = _collect(manifest, tmp_path / "results.json")

    assert result.returncode != 0
    assert "missing outputs" in result.stderr
    rows = list(csv.DictReader((tmp_path / "missing.csv").open(encoding="utf-8")))
    assert rows[0]["run_id"] == run["run_id"]


def test_collector_rejects_partial_seed_grid_in_strict_mode(tmp_path: Path) -> None:
    """Strict collection should reject scenario seed grids with missing algorithm seeds."""
    manifest = tmp_path / "manifest.json"
    runs = [
        _run_record(tmp_path, "PPO", 42),
        _run_record(tmp_path, "PPO", 43),
        _run_record(tmp_path, "CL-PPO", 42),
    ]
    _write_manifest(manifest, runs)

    result = _collect(manifest, tmp_path / "results.json")

    assert result.returncode != 0
    assert "incomplete seed grid" in result.stderr


def test_collector_normalizes_list_and_dict_payloads(tmp_path: Path) -> None:
    """Collector should normalize benchmark list and dict result payloads."""
    manifest = tmp_path / "manifest.json"
    runs = [
        _run_record(tmp_path, "PPO", 42, "list"),
        _run_record(tmp_path, "PPO", 43, "dict_results"),
        _run_record(tmp_path, "CL-PPO", 42, "dict"),
        _run_record(tmp_path, "CL-PPO", 43, "list"),
    ]
    output = tmp_path / "results.json"
    _write_manifest(manifest, runs)

    result = _collect(manifest, output)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "paper2-matrix-results/v1"
    assert payload["record_count"] == 4
    assert {record["metrics"]["social_welfare_mean"] for record in payload["results"]}
    assert all(record["result_kind"] == "benchmark" for record in payload["results"])
