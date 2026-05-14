# Paper2 v4.9 L3 Preregistration

This preregistration records the default L3 plan before any L2-driven narrowing.

## Scope

- Level: `L3_formal_verification`
- Steps per run: `200000`
- Seeds: `42, 43, 44, 45, 46`
- Default policy: do not narrow L3 based on L2 post-hoc preferences.

## Algorithms

- Proposed: `CL-PPO`, `GAM-COMA`
- Stage-1 defaults: `PPO`, `COMA`
- Strong single-agent baselines: `SimPO`, `TRPO`, `GRPO`, `DDPG`, `TD3`, `DDQN`
- Strong multi-agent baselines: `MAPPO`, `MADDPG`, `MATD3`
- Diagnostic-only algorithms are excluded from default execution.

## Scenarios

- `ID-mainline-a`
- `N1-oracle-small` as `benchmark_diagnostic` only unless an oracle wrapper is implemented.
- `N2-cl-ppo-ablation`
- `N2-gam-coma-ablation`
- `N3-ood-topology`
- `N3-ood-load`
- `N3-ood-channel-mobility`
- `N3-ood-deadline`
- `N3-ood-cooperation-queue`

## Metrics

- `social_welfare_mean`
- `reward_mean`
- `e2e_latency_mean`
- `e2e_latency_p95`
- `deadline_miss_rate`
- `energy_total_mean`
- `agent_reward_jain_mean`
- `constraint_violation_rate`
- `queue_wait_mean`
- `comm_score`

## Baseline And Exclusion Policy

- Baseline for pairwise statistics: `PPO`
- Exclusions require a pre-run implementation/resource blocker, not a post-hoc L2 ranking preference.

## Current Execution Decision

L3 was not executed because L2 did not produce strict real outputs and the workspace preflight found a dirty frozen-core file plus CPU-only heavy-run constraints.
