# Single-Policy 3-User Full17 Report

`.ai/ledger.json` remains the single machine state source.

## Interface Semantics

This report analyzes `single_policy_multi_user`: one shared single-agent policy selects one action for each of 3 user observations, then the joint action is stepped once in the same Mainline-A environment. This is a shared-policy baseline, not MAPPO/COMA-style multi-policy MARL.

## Scope

- Algorithms: GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO
- Seeds: single seed engineering comparison
- Steps: 100000 per algorithm in the benchmark input
- Statistical significance: not claimed
- Summary CSV: `results/single_policy_3user_full17_summary.csv`
- Figure artifacts: figures/single_policy_3user_full17/figure_manifest.json

## Status

Successful or partial records: 0

Failure records: 9

| algorithm | status | reward | social_welfare | latency | energy | comm_score | fairness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | blocked |  |  |  |  |  |  |
| PPO | blocked |  |  |  |  |  |  |
| SAC | blocked |  |  |  |  |  |  |
| DDQN | blocked |  |  |  |  |  |  |
| DDPG | blocked |  |  |  |  |  |  |
| TD3 | blocked |  |  |  |  |  |  |
| A3C | blocked |  |  |  |  |  |  |
| TRPO | blocked |  |  |  |  |  |  |
| SimPO | blocked |  |  |  |  |  |  |

## Failure Table

| algorithm | status | error |
| --- | --- | --- |
| GRPO | blocked | RESOURCE_BLOCKED: local CPU-only run did not complete the first GRPO 2048-step rollout after several minutes; CUDA was unavailable because the NVIDIA driver is too old. |
| PPO | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before PPO started because the local CPU-only GRPO rollout was not making practical progress. |
| SAC | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before SAC started because the local CPU-only GRPO rollout was not making practical progress. |
| DDQN | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before DDQN started because the local CPU-only GRPO rollout was not making practical progress. |
| DDPG | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before DDPG started because the local CPU-only GRPO rollout was not making practical progress. |
| TD3 | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before TD3 started because the local CPU-only GRPO rollout was not making practical progress. |
| A3C | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before A3C started because the local CPU-only GRPO rollout was not making practical progress. |
| TRPO | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before TRPO started because the local CPU-only GRPO rollout was not making practical progress. |
| SimPO | blocked | RESOURCE_BLOCKED: full 9-algorithm 100K-step single-policy 3-user run was stopped before SimPO started because the local CPU-only GRPO rollout was not making practical progress. |

## Reward Ranking

No successful reward records are available.

## Old-Evaluation Boundary

The old 1-agent single-agent full17 results are not statistically or semantically comparable with this corrected 3-user shared-control interface. They may be used as historical artifacts only.

## CL-PPO Gate

PPO baseline status under the corrected interface: not restartable. CL-PPO remains frozen until a human review explicitly accepts PPO as a valid baseline under this report.
