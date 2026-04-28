# Project Status Snapshot

## Basic Information

- Date: 2026-04-23
- Branch: `N/A` (no `.git` metadata in current workspace)
- Baseline Commit: `N/A`
- Audience: Advisor

## Goal and Scope

- Current goal: finish the README-defined train/test/benchmark/plot end-to-end loop.
- Covered in this cycle:
  - environment and trainer tests
  - 17-algorithm compatibility test
  - A3C and SAC smoke training
  - full benchmark for 16 algorithms (GRPO excluded)
  - figure generation and reporting docs
- Not covered in this cycle:
  - GRPO full benchmark run (blocked by memory instability)

## Code and Experiment Progress

- Completed:
  - runtime and dependency readiness validated (Python 3.12.13, torch 2.11.0+cpu, gymnasium 1.2.3)
  - tests passed:
    - `tests/test_mec_envs.py`: 8 passed
    - `tests/test_trainers.py`: 8 passed
    - `tests/test_algorithms_on_envs.py`: 18 passed
  - smoke training completed:
    - `checkpoints/a3c/final.pt`
    - `checkpoints/a3c/train_logs.json`
    - `checkpoints/sac/final.pt`
    - `checkpoints/sac/train_logs.json`
  - full benchmark completed for 16 algorithms with `timesteps=100000`, `seeds=42 123 456`, all `status=ok`
  - plotting completed, PDF figures generated in `figures/`
- In progress: none
- Blocker:
  - GRPO update stage still triggers abnormal tensor expansion and large memory allocation (~34GB)

## Core Metrics

- Trainable algorithms this round: `16 / 17` (GRPO excluded)
- Runnable environments: `3`
  - `GameTheoryMECEnv`
  - `GameTheoryDiscreteMAEnv`
  - `GameTheoryContinuousMAEnv`
- Benchmark script status: complete and reproducible
- Benchmark output summary:
  - entries: `16`
  - failures: `0`
  - best discrete reward mean: `VDN = 57.2650`
  - best continuous reward mean: `MADDPG = 25.2393`

## Evidence Paths

- Configs: `configs/algorithms/`
- Training script: `scripts/train.py`
- Evaluation script: `scripts/evaluate.py`
- Benchmark script: `scripts/benchmark.py`
- Final benchmark outputs:
  - `results/benchmark.json`
  - `results/benchmark_no_grpo.json`
- Per-algorithm benchmark outputs: `results/full_per_algo/`
- Benchmark logs:
  - `results/benchmark_full_group1_run.log`
  - `results/full_per_algo/*.log`
- Figures: `figures/`
- Tests: `tests/`

## Risks and Mitigation

- Risk 1: GRPO memory instability (OOM / system instability)
  - Impact: cannot safely include GRPO in full run on this device
  - Mitigation: exclude GRPO for this round and track as a dedicated follow-up fix
- Risk 2: high runtime on CPU-only full benchmark
  - Impact: long iteration cycle and expensive reruns
  - Mitigation: run per algorithm with separate JSON/log outputs to avoid large-batch rerun loss
