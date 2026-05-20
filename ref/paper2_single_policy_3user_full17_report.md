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
- Figure artifacts: figures/single_policy_3user_full17/reward_ranking.png, figures/single_policy_3user_full17/comm_score_ranking.png, figures/single_policy_3user_full17/figure_manifest.json

## Status

Successful or partial records: 9

Failure records: 0

| algorithm | status | reward | social_welfare | latency | energy | comm_score | fairness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRPO | ok | -16.7084 | -16.7084 | 0.3651 | 0.0062 | 151.6147 | 0.5202 |
| PPO | ok | -36.6194 | -36.6194 | 0.2218 | 0.0113 | 209.5089 | 0.5999 |
| SAC | ok | -94.4002 | -94.4002 | 0.9902 | 0.0102 | 90.6331 | 0.5947 |
| DDQN | ok | -2347.4033 | -2347.4033 | 0.9975 | 0.1281 | 96.7937 | 0.6450 |
| DDPG | ok | -13.9416 | -13.9416 | 0.2302 | 0.0064 | 210.3894 | 0.5610 |
| TD3 | ok | -18.3406 | -18.3406 | 0.2373 | 0.0081 | 210.4669 | 0.5918 |
| A3C | ok | -2203.2188 | -2203.2188 | 1.3957 | 0.2015 | 89.1543 | 0.6445 |
| TRPO | ok | -52.6276 | -52.6276 | 0.2562 | 0.0387 | 175.3025 | 0.6087 |
| SimPO | ok | -22.2104 | -22.2104 | 0.1970 | 0.0122 | 218.2512 | 0.5871 |

## Failure Table

No failed algorithms were recorded.

## Reward Ranking

| algorithm | reward | comm_score | latency | energy |
| --- | --- | --- | --- | --- |
| DDPG | -13.9416 | 210.3894 | 0.2302 | 0.0064 |
| GRPO | -16.7084 | 151.6147 | 0.3651 | 0.0062 |
| TD3 | -18.3406 | 210.4669 | 0.2373 | 0.0081 |
| SimPO | -22.2104 | 218.2512 | 0.1970 | 0.0122 |
| PPO | -36.6194 | 209.5089 | 0.2218 | 0.0113 |
| TRPO | -52.6276 | 175.3025 | 0.2562 | 0.0387 |
| SAC | -94.4002 | 90.6331 | 0.9902 | 0.0102 |
| A3C | -2203.2188 | 89.1543 | 1.3957 | 0.2015 |
| DDQN | -2347.4033 | 96.7937 | 0.9975 | 0.1281 |

## Old-Evaluation Boundary

The old 1-agent single-agent full17 results are not statistically or semantically comparable with this corrected 3-user shared-control interface. They may be used as historical artifacts only.

## CL-PPO Gate

PPO baseline status under the corrected interface: pending human review. CL-PPO remains frozen until a human review explicitly accepts PPO as a valid baseline under this report.
