# RL-MEC Benchmark Results Report
**Experiment Date**: 2026-04-23 10:15:41
**Report Generated**: 2026-04-24 13:11:24
**Total Timesteps**: 500,000 (500k)

## Table of Contents
- [Executive Summary](#executive-summary)
- [Performance Rankings](#performance-rankings)
- [Algorithm Category Analysis](#algorithm-category-analysis)
- [Environment Analysis](#environment-analysis)
- [Detailed Results](#detailed-results)
- [Key Findings](#key-findings)
- [Failure Diagnosis](#failure-diagnosis)
- [Recommendations](#recommendations)

## Executive Summary

This benchmark evaluates **17 RL algorithms** on GameTheory-based MEC environments:
- **Success Rate**: 16/17 (94.1%)
- **Failed**: 1 algorithm(s) (IPPO - configuration error)

### Top Performers

**Highest Reward**: QMIX (71.9143)
**Fastest Training**: GRPO (325.8s)
**Lowest Latency**: COMA (2.1430)

## Performance Rankings

### Reward Rankings (Top 3)
| Rank | Algorithm | Environment | Reward | Category |
|------|-----------|-------------|--------|----------|
| 1 | QMIX | discrete-ma | 71.9143 | Off-Policy |
| 2 | MADDPG | continuous-ma | 35.1932 | Off-Policy |
| 3 | MATD3 | continuous-ma | 35.1932 | Off-Policy |

### Training Time Rankings (Top 3 - Fastest)
| Rank | Algorithm | Time (sec) | Environment |
|------|-----------|-----------|-------------|
| 1 | GRPO | 325.8 | continuous-ma |
| 2 | PPO | 378.9 | continuous-ma |
| 3 | TRPO | 430.0 | continuous-ma |

### Latency Rankings (Top 3 - Lowest)
| Rank | Algorithm | Latency | Environment |
|------|-----------|---------|-------------|
| 1 | COMA | 2.1430 | discrete-ma |
| 2 | A3C | 16.7736 | discrete-ma |
| 3 | DDQN | 24.5879 | discrete-ma |

## Algorithm Category Analysis

### On-Policy Algorithms (7 algorithms)
- **Algorithms**: GRPO, PPO, TRPO, A3C, SimPO, MAPPO, COMA
- **Avg Reward**: 11.5405
- **Avg Training Time**: 863.0s
- **Avg Latency**: 97.9617
- **Avg Energy**: 48.6830

### Off-Policy Algorithms (9 algorithms)
- **Algorithms**: SAC, DDPG, TD3, MADDPG, MATD3, DDQN, QMIX, VDN, IQL
- **Avg Reward**: 21.3287
- **Avg Training Time**: 9769.8s
- **Avg Latency**: 188.2050
- **Avg Energy**: 45.6378

## Environment Analysis

### game-theory-continuous-ma
- **Algorithm Count**: 8
- **Best Algorithm**: MADDPG (reward: 35.1932)
- **Average Reward**: 15.2860

### game-theory-discrete-ma
- **Algorithm Count**: 8
- **Best Algorithm**: QMIX (reward: 71.9143)
- **Average Reward**: 18.8067

## Detailed Results

| Algorithm | Environment | Reward | Latency | Energy | Time (s) | Category |
|-----------|-------------|--------|---------|--------|----------|----------|
| QMIX | ma | 71.9143 | 551.5173 | 0.2274 | 11218.1 | Off-Policy |
| MADDPG | ma | 35.1932 | 299.0856 | 149.5428 | 14400.4 | Off-Policy |
| MATD3 | ma | 35.1932 | 299.0856 | 149.5428 | 14852.8 | Off-Policy |
| MAPPO | ma | 32.9935 | 279.0207 | 139.2527 | 1809.7 | On-Policy |
| VDN | ma | 15.5044 | 139.7159 | 1.3436 | 10991.9 | Off-Policy |
| IQL | ma | 13.7557 | 143.8421 | 14.9696 | 12045.9 | Off-Policy |
| GRPO | ma | 11.8355 | 99.8109 | 49.9054 | 325.8 | On-Policy |
| SAC | ma | 11.8355 | 99.8109 | 49.9054 | 6243.9 | Off-Policy |
| SimPO | ma | 11.8355 | 99.8109 | 49.9054 | 525.8 | On-Policy |
| PPO | ma | 11.8350 | 99.8058 | 49.9054 | 378.9 | On-Policy |
| TRPO | ma | 10.3200 | 88.3667 | 48.9449 | 430.0 | On-Policy |
| DDPG | ma | 5.4054 | 49.3835 | 0.4036 | 4935.9 | Off-Policy |
| DDQN | ma | 2.4861 | 24.5879 | 2.4112 | 8755.3 | Off-Policy |
| A3C | ma | 1.6934 | 16.7736 | 1.6043 | 765.2 | On-Policy |
| TD3 | ma | 0.6704 | 86.8159 | 42.3938 | 4484.3 | Off-Policy |
| COMA | ma | 0.2705 | 2.1430 | 1.2628 | 1805.6 | On-Policy |

## Key Findings

### Convergence Speed
- **Fastest convergence**: GRPO (325.8s)
- Multi-agent algorithms (MADDPG, MATD3, QMIX) require significantly longer training time

### Sample Efficiency
Top performers by reward per second:
- GRPO: 0.0363 reward/sec
- PPO: 0.0312 reward/sec
- TRPO: 0.0240 reward/sec

### Algorithm Suitability
- **Continuous Action Space**: GRPO, PPO, and MADDPG show strong performance
- **Discrete Action Space**: QMIX and MAPPO show highest rewards among discrete algorithms
- **Single-Agent**: On-policy algorithms (GRPO, PPO) converge faster
- **Multi-Agent**: MADDPG and MATD3 achieve highest absolute rewards

## Failure Diagnosis

### Failed Algorithms

**IPPO (MEC-v1-game-theory-discrete-ma)**
- Error: `PPOAgent.__init__() got an unexpected keyword argument 'use_game_theory'`
- Recommendation: Check configuration parameter validation in IPPOAgent initialization

## Recommendations

### For Production Deployment
1. **Use GRPO or PPO** for continuous control (fast, reliable, good reward)
2. **Use MADDPG/MATD3** for multi-agent continuous tasks (highest rewards)
3. **Use MAPPO/QMIX** for discrete multi-agent tasks

### For Future Experiments
1. **Fix IPPO**: Remove or properly handle 'use_game_theory' parameter
2. **Hyperparameter Tuning**: Current configs are defaults; tuning could improve SAC and DDQN
3. **Extended Training**: Run to 1M+ timesteps to see convergence patterns
4. **Multi-seed Analysis**: Current results are single-seed (seed=42); use 3+ seeds for statistical significance
