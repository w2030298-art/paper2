# GameTheory MEC Implementation Matrix

Date: 2026-04-23

## 1. Algorithm Implementation Status Matrix

| Algorithm | Family | Integration Mode | Shapley Credit | CTDE Hints | Warm-Start |
|---|---|---|---|---|---|
| GRPO | On-policy | Deep Fusion | Yes | Yes | Yes |
| MAPPO | On-policy | Deep Fusion | Yes | Yes | Yes |
| QMIX | Off-policy | Deep Fusion | Yes | Yes | Yes |
| COMA | On-policy | Deep Fusion | Yes | Yes | Yes |
| IPPO | On-policy | Deep Fusion | Yes | Yes | Yes |
| VDN | Off-policy | Deep Fusion | Yes | Yes | Yes |
| MADDPG | Off-policy | Deep Fusion | Yes | Yes | Yes |
| IQL | Off-policy | Deep Fusion | Yes | Yes | Yes |
| MATD3 | Off-policy | Deep Fusion | Yes | Yes | Yes |
| PPO | On-policy | Compatibility | Optional | Optional | Optional |
| SAC | Off-policy | Compatibility | Optional | Optional | Optional |
| DDQN | Off-policy | Compatibility | Optional | Optional | Optional |
| DDPG | Off-policy | Compatibility | Optional | Optional | Optional |
| TD3 | Off-policy | Compatibility | Optional | Optional | Optional |
| A3C | On-policy | Compatibility | Optional | Optional | Optional |
| TRPO | On-policy | Compatibility | Optional | Optional | Optional |
| SimPO | On-policy | Compatibility | Optional | Optional | Optional |

Notes:
- Deep Fusion defaults are enabled for `GRPO + 8` multi-agent algorithms.
- Compatibility algorithms accept `global_states/game_hints/shapley_values/reward_terms/eq_actions` and ignore unused keys without changing core loss logic.

## 2. Ablation Matrix Mapping

| Ablation ID | What Is Disabled | Code/Config Switch |
|---|---|---|
| A0 | None (full stack) | Default `game_theory.enabled=true` |
| A1 | Shapley credit allocation | `game_theory.use_shapley_credit=false` |
| A2 | CTDE game hints | `game_theory.ctde_with_hints=false` |
| A3 | Warm-start imitation stage | `game_theory.warm_start_steps=0` |
| A4 | Adaptive hierarchical reward | `adaptive_reward_weights=false` (env-level) |
| A5 | Queue/DVFS/3GPP upgrades | switch to legacy model path (env-level fallback) |
| A6 | Constraint projection barrier | `enable_action_projection=false` or `barrier_coeff=0` |
| A7 | Enhanced observation history | reduce to legacy base obs (env-level fallback) |
| A8 | Bilevel pricing solver | threshold pricing fallback (env-level fallback) |

## 3. Paper Notation Consistency Checklist

| Symbol | Meaning | Code Field / Component | Checked |
|---|---|---|---|
| `p_k` | edge-server price | `game_hints.equilibrium_prices`, `OptimalPricingMechanism` | Yes |
| `a_i` | user action | policy output + adapter decode | Yes |
| `phi_i` | Shapley allocation | `info.shapley_allocation`, `batch_data.shapley_values` | Yes |
| `R_i` | hierarchical reward | `reward_terms`: `r_imm/r_coop/r_eq` | Yes |
| `rho_k` | queue utilization | `queue_metrics.rho_mean/max` | Yes |
| `SINR` | channel quality | `Channel3GPP.compute_sinr` + obs delta SNR | Yes |
| `T_queue` | queue delay | `QueueingDelayModel.compute_delay` | Yes |
| `E_i` | energy term | `DVFSEnergyModel.compute_energy` + env energy outputs | Yes |
| `L_barrier` | barrier penalty | `constraint_metrics.barrier_mean` | Yes |
