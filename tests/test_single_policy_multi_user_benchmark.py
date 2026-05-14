"""Benchmark wiring tests for the single-policy multi-user interface."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.benchmark as benchmark
from src.experiment.single_policy_multi_user import SinglePolicyMultiUserSettings


def _fake_seed_result(algorithm: str, seed: int = 42) -> dict:
    """Return a minimal successful seed result shaped like benchmark_single output."""
    return {
        "algorithm": algorithm,
        "environment": benchmark.ALGO_ENV_MAP[algorithm],
        "seed": seed,
        "device": "cpu",
        "train_timesteps": 1,
        "train_time_seconds": 0.0,
        "final_reward_mean": 1.0,
        "final_reward_std": 0.0,
        "final_latency_mean": 1.0,
        "final_e2e_latency_mean": 1.0,
        "final_e2e_latency_p95": 1.0,
        "final_deadline_miss_rate": 0.0,
        "final_throughput_tasks_per_step": 3.0,
        "final_comm_score": 1.0,
        "final_energy_mean": 1.0,
        "final_latency_total_mean": 3.0,
        "final_energy_total_mean": 3.0,
        "final_social_welfare_mean": 1.0,
        "total_episodes": 1,
        "total_updates": 0,
        "convergence": {"schema_version": 2, "run_status": "success"},
        "resolved_runtime_config": {"algorithm": algorithm},
    }


def test_run_benchmark_forces_three_user_single_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_benchmark should force num_agents=3 and annotate aggregate records."""
    captured: dict = {}

    def fake_benchmark_single(*args, **kwargs):
        captured["env_overrides"] = kwargs["env_overrides"]
        captured["single_policy_settings"] = kwargs["single_policy_settings"]
        return _fake_seed_result(args[0], seed=args[2])

    monkeypatch.setattr(benchmark, "benchmark_single", fake_benchmark_single)
    results = benchmark.run_benchmark(
        ["PPO"],
        seeds=[42],
        timesteps=1,
        episodes=1,
        device="cpu",
        verbose=False,
        sync_latest_alias=False,
        single_policy_settings=SinglePolicyMultiUserSettings(enabled=True, num_users=3),
    )

    assert captured["env_overrides"]["force_num_agents"] == 3
    assert captured["single_policy_settings"].interface == "single_policy_multi_user"
    assert results[0]["interface"] == "single_policy_multi_user"
    assert results[0]["num_agents"] == 3
    assert results[0]["single_policy_shared_reward"] == "mean"


def test_run_benchmark_rejects_cl_ppo_for_single_policy() -> None:
    """CL-PPO is frozen and excluded from the v5.0 single-policy comparison."""
    with pytest.raises(ValueError, match="CL-PPO"):
        benchmark.run_benchmark(
            ["CL-PPO"],
            seeds=[42],
            timesteps=1,
            episodes=1,
            device="cpu",
            verbose=False,
            sync_latest_alias=False,
            single_policy_settings=SinglePolicyMultiUserSettings(enabled=True, num_users=3),
        )


def test_single_policy_output_json_keeps_failure_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A failing algorithm should remain in the result JSON with error details."""
    def failing_benchmark_single(*_args, **_kwargs):
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(benchmark, "benchmark_single", failing_benchmark_single)
    output = tmp_path / "benchmark.json"
    results = benchmark.run_benchmark(
        ["PPO"],
        seeds=[42],
        timesteps=1,
        episodes=1,
        device="cpu",
        output_file=output,
        verbose=False,
        sync_latest_alias=False,
        single_policy_settings=SinglePolicyMultiUserSettings(enabled=True, num_users=3),
    )

    assert results[0]["status"] == "failed"
    assert results[0]["interface"] == "single_policy_multi_user"
    assert "intentional failure" in results[0]["errors"][0]["error"]
    assert output.exists()
