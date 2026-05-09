# Mainline-A 最后一轮算法修复与环境适配记录

本文档记录基于 `paper2-full17-data-analysis-focused` 的最后一轮算法筛选、算法/适配器修复、超参数调整与 benchmark 展示机制扩展。约束条件：**不修改 Mainline-A 环境核心**，只修改算法、环境适配器、benchmark/plot 工具与算法配置。

## 1. Mainline-A 环境冻结范围

本轮未修改以下 Mainline-A 环境核心与最终环境配置：

- `src/environments/mec_v3/game_theory_env.py`
- `configs/system_model_mainline_a.yaml`
- `configs/pricing_dynamic_mainline_a.yaml`

本轮修改发生在：

- 算法接口与探索噪声：`rl_algorithms/ddpg.py`, `rl_algorithms/td3.py`
- 算法-环境接口适配：`src/environments/mec_v3/game_theory_adapters.py`
- benchmark 与展示：`scripts/benchmark.py`, `scripts/plot_results.py`
- 算法配置：`configs/algorithms/*.yaml`
- 最终筛选配置：`configs/benchmark_mainline_a_final_screening.yaml`
- 筛选表：`results/final_algorithm_screening_decisions.csv`

## 2. 基于 full17 的算法筛选结论

full17 是单种子、100K 步的 artifact-level evidence，不作为统计显著性结论；它足以支撑最后一轮工程筛选。核心观察如下：

| 类别 | 算法 | 依据 | 本轮处理 |
|---|---|---|---|
| 主候选 | SimPO | reward 第一，Pareto，`stable_or_improving`，tail_std=0.0359 | 不做激进算法变更；加入更密评估与扩展指标 |
| 主候选 | PPO | Pareto，延迟优秀，comm_score 高；tail_std=0.164 | 轻微稳定性调参：更小 clip、entropy、lr |
| 主候选/资源基线 | TRPO | Pareto，latency/energy 最优，但 reward ceiling 低 | 保留为资源效率基线；max_kl 轻微收紧 |
| 稳定低上限参照 | GRPO | Pareto 但 `stable_but_low_ceiling` | 仅纳入参照与更密评估，不预期胜出 |
| 主候选/需修 | DDPG | reward 第二梯队，稳定但 Pareto-dominated，存在 oscillating 标记 | 暴露并降低 OU 噪声，延长 warmup，降低 lr |
| 主候选/需修 | TD3 | `mostly_good_with_single_late_drop`，tail_std=7.109 | 修复硬编码探索噪声，降低行为/目标噪声与 noise clip |
| 可修后筛选 | DDQN | `early_peak_then_degraded`，best_tail_gap=7.116 | 修复 benchmark 未读取 exploration epsilon 的问题；放慢 epsilon decay |
| 可修后筛选 | MAPPO | `regressed_or_bad_plateau`，但 N1 oracle 支持 game-aware/primal-dual MARL 方向 | 保留一轮修复：更小 clip/lr/entropy，延长 warm start |
| 可修后筛选 | COMA | moderate but drifting down，未灾难性崩溃 | 降 lr、减少 epochs、延长 warm start |
| 可修后筛选 | MADDPG | outlier then partial recovery | 降 lr、增 batch，受益于连续多智能体动作适配修复 |
| 可修后筛选 | MATD3 | regressed plateau；连续多智能体接口风险 | 降 noise、延迟 policy update，受益于 adapter 修复 |
| 冻结/诊断-only | IPPO | 严重 instability/catastrophic outlier | 不再投入最后调参预算，排除默认 final screening |
| 机制不适配 | QMIX | 值分解单调假设与 Mainline-A 动态价格/公平/耦合资源奖励不匹配 | 冻结，排除默认 final screening |
| 机制不适配 | VDN | value decomposition 结果灾难性，energy 高 | 冻结，排除默认 final screening |
| 机制不适配 | IQL | 最严重崩溃：reward -816.32，energy 44.47 | 冻结，排除默认 final screening |
| 无 ROI | SAC | reward/latency/deadline miss 均差，bad plateau | 冻结，排除默认 final screening |
| 无 ROI | A3C | regressed plateau，deadline miss 高 | 冻结，排除默认 final screening |

补充证据：N1 oracle 表明 `game_aware_pd_marl` 的 mean oracle gap 为 0.0289，显著低于 MAPPO 的 0.165 与 static baseline 的 1.4303，且约束违反为 0。因此本轮没有直接放弃所有 multi-agent/game-aware 方向，而是仅冻结 IQL/VDN/QMIX/IPPO 这类在 full17 中暴露出机制不匹配或灾难性不稳定的路线。

## 3. 定制化修复方案

### 3.1 环境适配器修复

`src/environments/mec_v3/game_theory_adapters.py`：

- 连续动作适配器现在支持：
  - 单智能体 `(4,)`、`(1,4)`；
  - 多智能体 batch `(num_agents, 4)`；
  - flatten joint action `(num_agents*4,)`；
  - list/tuple of per-agent actions。
- 连续动作进入 Mainline-A 前统一做：NaN/inf 修复、padding/truncation、`[-1,1]` 裁剪。
- 离散动作适配器现在支持 scalar、`(n,1)`、logits/probs-like vector，并会裁剪到合法 action id。
- 这项修复主要服务 MADDPG/MATD3 等连续多智能体算法，也避免 early exploration 中无效 action 污染环境。

### 3.2 DDPG

修改：

- `DDPGAgent` 新增 `ou_mu`, `ou_theta`, `ou_sigma` 参数。
- `scripts/benchmark.py` 会从 `exploration` 与 `algorithm/training` section 读取这些参数。
- `configs/algorithms/ddpg.yaml`：
  - `ou_sigma: 0.20 -> 0.12`
  - `ou_theta: 0.15 -> 0.20`
  - `explore_steps/start_steps: 10000 -> 15000`
  - `lr: 3e-4 -> 2.5e-4`

目的：保留 DDPG 的高 reward 能力，同时降低 Mainline-A 上的后期探索扰动和 Pareto-dominated 风险。

### 3.3 TD3

修改：

- `TD3Agent.select_action()` 原先行为策略探索噪声硬编码为 `0.3`；现在改为 `exploration_noise_std`。
- `scripts/benchmark.py` 会读取 `exploration_noise_std`、`noise_std`、`noise_clip`。
- `configs/algorithms/td3.yaml`：
  - `noise_std: 0.20 -> 0.12`
  - `noise_clip: 0.50 -> 0.25`
  - `exploration_noise_std: 0.12`
  - `explore_steps/start_steps: 10000 -> 15000`
  - `lr: 3e-4 -> 2e-4`

目的：针对 full17 中 TD3 的 single late drop，降低行为策略与目标策略平滑噪声，避免后期动作抖动导致 tail collapse。

### 3.4 DDQN

修改：

- `scripts/benchmark.py` 现在统一从 `algorithm/training/exploration/network` 读取参数；DDQN 的 epsilon 配置不再因写在 `exploration` section 而被忽略。
- `configs/algorithms/ddqn.yaml`：
  - `epsilon_decay: 50000 -> 120000`
  - `epsilon_end: 0.05 -> 0.08`
  - `lr: 3e-4 -> 2e-4`
  - `start_steps: 5000`

目的：减轻 early peak then degraded，让 final screening 判断它是否仍可作为离散单智能体备选。

### 3.5 PPO / TRPO / SimPO / GRPO

- SimPO：full17 reward leader，不做激进变更，只提高 benchmark 数据密度。
- PPO：`eps_clip 0.20 -> 0.15`，`ent_coeff 0.01 -> 0.005`，`lr 3e-4 -> 2e-4`，目标是减少尾部波动。
- TRPO：`max_kl 0.01 -> 0.008`，继续作为资源效率基线。
- GRPO：保留稳定低上限参照角色，不做主动上限提升式调参。

### 3.6 MAPPO / COMA / MADDPG / MATD3

- MAPPO：`eps_clip 0.12`，`ent_coeff 0.005`，`num_epochs 8`，`lr 1.5e-4`，`warm_start_steps 2000`，`warm_start_lr_scale 0.4`。
- COMA：`num_epochs 6`，`lr 2e-4`，`warm_start_steps 2000`，`warm_start_lr_scale 0.4`。
- MADDPG：`lr 2e-4`，`batch_size 512`，`warm_start_steps 2000`。
- MATD3：`noise_std 0.12`，`policy_delay 3`，`lr 2e-4`，`warm_start_steps 2000`。

目的：对仍有机制依据的 multi-agent 路线做最后一轮保守修复；后续只筛选，不继续调参。

## 4. Benchmark 能力展示机制扩展

原 benchmark 的问题：checkpoint 点太少，并且主要输出 reward/latency/energy/comm_score 四类收敛曲线。

本轮改动：

### 4.1 更密 checkpoint

`BaseTrainer` 新增：

- `min_eval_points`
- `eval_at_start`
- `eval_steps`
- `effective_eval_interval`

`benchmark.py` 新增 CLI/config：

- `--eval-interval`
- `--min-eval-points`
- `--eval-at-start`

最终筛选配置使用：

```yaml
benchmark:
  steps: 100000
  seeds: [42, 43, 44]
  eval_interval: 2500
  min_eval_points: 32
  eval_at_start: true
```

训练日志与 benchmark JSON 中会写入 `eval_steps`，plot 工具会优先用真实 step 坐标而不是假设固定 `eval_interval * index`。

### 4.2 扩展指标

`BaseTrainer.evaluate()` 现在除原有指标外，会读取环境 `info` 中的更多 Mainline-A/GameTheory 观测：

- 通信体验：`e2e_latency_mean`, `e2e_latency_p95`, `deadline_miss_rate`, `throughput_tasks_per_step`, `comm_score`
- 能耗：`energy_total_mean`, `energy_per_task_mean`
- 社会福利：`social_welfare_mean`, `social_welfare_per_step_mean`
- 智能体公平：`agent_reward_jain_mean`
- 约束：`constraint/any_violation`, `penalty_mean`, `barrier_mean`, latency/energy/power violations
- 队列：`queue_wait_mean`, queue lengths, queue metrics
- 行为：local/edge target rate、offload ratio、target mean
- 动态价格：price mean/std/p95、provider revenue proxy
- 公平机制：EFX satisfaction、EFX violation/transfer、CP-net score
- reward breakdown：`reward_component/*`
- Mainline-A reward components：`mainline_a_reward/*`

`benchmark.py` 最终结果会额外输出：

- `final_social_welfare_mean`
- `final_social_welfare_per_step_mean`
- `final_agent_reward_jain_mean`
- `final_constraint_violation_rate`
- `final_constraint_penalty_mean`
- `final_queue_wait_mean`
- `final_offload_ratio_mean`
- `final_price_mean`
- `final_provider_revenue_proxy_mean`
- `final_efx_satisfaction_rate`

`plot_results.py` 的收敛图从固定 2x2 扩展为动态网格，目前覆盖 reward/social welfare/latency/p95/deadline miss/throughput/energy/comm_score/fairness/constraint/queue/offload/price。

## 5. 最终筛选命令

推荐 final screening：

```bash
python scripts/benchmark.py \
  --config configs/benchmark_mainline_a_final_screening.yaml \
  --output results/benchmark_mainline_a_final_screening.json \
  --no-latest-alias
```

检查配置解析但不训练：

```bash
python scripts/benchmark.py \
  --config configs/benchmark_mainline_a_final_screening.yaml \
  --dry-run
```

生成扩展图：

```bash
python scripts/plot_results.py \
  --input results/benchmark_mainline_a_final_screening.json \
  --output figures/mainline_a_final_screening
```

## 6. 验证记录

已执行：

```bash
python scripts/benchmark.py --config configs/benchmark_mainline_a_final_screening.yaml --dry-run
pytest -q tests/test_final_algorithm_adaptation.py
pytest -q tests/test_convergence_plot.py tests/test_benchmark_mainline_a_config.py
pytest -q tests/test_trainers.py
pytest -q tests/test_game_theory_fusion.py
python -m compileall -q src rl_algorithms scripts tests
```

结果：

- 新增 final adaptation 测试：5 passed
- convergence/benchmark config 测试：14 passed
- trainer 测试：9 passed
- game-theory fusion 测试：19 passed
- `compileall` 通过

未执行完整 100K × 3 seeds final screening；本轮是代码、配置、适配器与评估机制修复。最终排名应以后续 `configs/benchmark_mainline_a_final_screening.yaml` 跑出的完整结果为准。
