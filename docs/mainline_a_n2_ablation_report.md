# Mainline-A N2 Deterministic Controlled Probe Report

## Status

- Date: 2026-05-05
- Status: N2_DONE_PENDING_REVIEW
- Scope: N2 deterministic controlled probe only; N3 was not started and full 17 was not run.
- Evidence level: deterministic controlled probe; not training-grade or publication-grade ablation evidence.
- Config: `configs/experiments/mainline_a_n2_ablation.yaml`
- Output directory: `results/mainline_a/n2_ablation/`
- Benchmark alias: `results/benchmark.json` was not created or overwritten.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n2_ablation.yaml --stage N2 --dry-run
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n2_ablation.yaml --stage N2 --preflight --preflight-steps 256 --results-root results\mainline_a
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n2_ablation.yaml --stage N2 --results-root results\mainline_a
```

## Probe Matrix

| Ablation | Switches |
|---|---|
| full_model | dynamic pricing, queue state, channel state, migration state, primal-dual, cooperation, hybrid channel |
| no_dynamic_price | dynamic pricing disabled |
| no_queue_state | queue state disabled; true queue cost still measured |
| no_channel_state | channel state disabled |
| no_migration_state | migration state disabled; true migration cost still measured |
| no_primal_dual | primal-dual update disabled |
| no_cooperation | cooperation disabled |
| analytic_channel_only | analytic channel only |
| 3gpp_lite_channel | 3GPP-lite channel only |

## Preflight Probe

- Seeds: `[42]`
- Steps: 256
- Records: 9
- Required metrics: reward, latency, energy, price, constraint violation.
- Result: passed. Required metrics were present, finite, schema-consistent, and non-identical across ablations.
- Artifacts: `results/mainline_a/n2_ablation/preflight/ablation_matrix.json`, `ablation_records.json`, `metric_deltas.json`, `summary.json`.

## Deterministic Controlled Probe

- Seeds: `[42, 43, 44]`
- Steps: 50000
- Records: 27
- Result: passed for deterministic probe outputs. Required metrics were present, finite, schema-consistent, and every non-full label differs from `full_model`.
- Artifacts: `results/mainline_a/n2_ablation/ablation_matrix.json`, `ablation_records.json`, `metric_deltas.json`, `summary.json`.

## Metric Means

| Ablation | Reward | Latency | Energy | Price | Constraint |
|---|---:|---:|---:|---:|---:|
| full_model | -0.921585 | 0.378271 | 1.435977 | 1.230709 | 0.021727 |
| no_dynamic_price | -1.006966 | 0.493271 | 1.465977 | 1.000000 | 0.055727 |
| no_queue_state | -0.975610 | 0.493271 | 1.435977 | 0.785750 | 0.071727 |
| no_channel_state | -1.030842 | 0.468271 | 1.489977 | 1.278709 | 0.021727 |
| no_migration_state | -0.990337 | 0.445271 | 1.435977 | 1.196959 | 0.021727 |
| no_primal_dual | -1.052646 | 0.483271 | 1.435977 | 1.230709 | 0.096727 |
| no_cooperation | -1.059522 | 0.433271 | 1.457977 | 1.230709 | 0.021727 |
| analytic_channel_only | -0.893794 | 0.353471 | 1.429977 | 1.218709 | 0.021727 |
| 3gpp_lite_channel | -0.968817 | 0.412671 | 1.478977 | 1.246709 | 0.021727 |

## Deltas vs full_model

| Ablation | dReward | dLatency | dEnergy | dPrice | dConstraint |
|---|---:|---:|---:|---:|---:|
| no_dynamic_price | -0.085381 | 0.115000 | 0.030000 | -0.230709 | 0.034000 |
| no_queue_state | -0.054025 | 0.115000 | 0.000000 | -0.444959 | 0.050000 |
| no_channel_state | -0.109257 | 0.090000 | 0.054000 | 0.048000 | 0.000000 |
| no_migration_state | -0.068752 | 0.067000 | 0.000000 | -0.033750 | 0.000000 |
| no_primal_dual | -0.131061 | 0.105000 | 0.000000 | 0.000000 | 0.075000 |
| no_cooperation | -0.137937 | 0.055000 | 0.022000 | 0.000000 | 0.000000 |
| analytic_channel_only | 0.027791 | -0.024800 | -0.006000 | -0.012000 | 0.000000 |
| 3gpp_lite_channel | -0.047232 | 0.034400 | 0.043000 | 0.016000 | 0.000000 |

## Anomaly Audit

- Missing required metrics: no.
- NaN/Inf: no.
- Result schema mismatch: no.
- Label-only ablation: no; every label maps to explicit switches in `ablation_matrix.json`.
- Duplicate ablation label: rejected by runner validation.
- Metrics all identical: no.
- Non-full ablation identical to `full_model`: no.
- Benchmark alias overwrite: no.

## N3 Gate

N2 preflight and deterministic controlled probe passed the local checks, so the N2 artifact is ready for review at probe level. This is not publication-grade ablation evidence. N3 remains `NOT_STARTED`; do not start N3 until the user/Web approves this N2 report and explicitly requests N3.
