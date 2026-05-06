# Mainline-A N0 Smoke Report

## Status

- Date: 2026-05-05
- Status: N0_DONE_PENDING_REVIEW
- Scope: N0 only; N1/N2/N3 were not started.
- Config: `configs/experiments/mainline_a_n0_smoke.yaml`
- Output: `results/mainline_a/n0_smoke/benchmark.json`

## Command

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --config configs\experiments\mainline_a_n0_smoke.yaml --output results\mainline_a\n0_smoke\benchmark.json --no-latest-alias
```

## Run Inputs

- Seed: 42
- Configured steps: 1000
- Executed train timesteps: 1000
- Users: 4
- Edges: 2
- Queue model: mm1
- Channel model: analytic
- Config algorithm label: `game_aware_pd_marl`
- Executed algorithm: MAPPO with Mainline-A/game-aware path enabled
- Device: cpu
- Environment: MEC-v1-game-theory-discrete-ma
- Stdout log: `logs/benchmark_20260505_173930.log`
- Stderr log: `logs/benchmark_20260505_173930.err.log`

## Metrics

| Metric | Value |
|---|---:|
| status | ok |
| n_seeds | 1 |
| final_reward_mean_mean | -34.2025 |
| final_reward_std_mean | 0.9274 |
| train_time_seconds_mean | 184.7 |
| final_e2e_latency_mean_mean | 0.4567 |
| final_e2e_latency_p95_mean | 0.8543 |
| final_deadline_miss_rate_mean | 0.0215 |
| final_throughput_tasks_per_step_mean | 4.0 |
| final_comm_score_mean | 210.8755 |
| final_energy_mean_mean | 2.3157 |
| final_latency_per_task_mean_mean | 0.4567 |
| final_energy_per_task_mean_mean | 0.0058 |

## Safety Checks

- Traceback: no traceback found in N0 logs.
- NaN/Inf: no NaN/Inf markers found in N0 logs or JSON metrics.
- Reward all zero: no; final reward mean is -34.2025.
- Latest benchmark alias: not updated; `--no-latest-alias` was used.
- Price boundary clipping: no. A post-run N0 config price probe sampled 40 dynamic prices with bounds [0.05, 10.0]; min=1.053037, max=1.289476, mean=1.178899, all-at-boundary=false.

## Notes

- The N0 config label `game_aware_pd_marl` is bound to the existing MAPPO training path because there is no standalone registered agent class with that name.
- `benchmark.py` now consumes the N0 config for users, edges, seed, steps, queue model, channel model, system-model config, dynamic-pricing config, and game-aware enablement.
- N0 only proves the Mainline-A training pipeline can run. It is not N1 oracle, N2 ablation, or N3 OOD evidence.
