# 训练进度与算法接入简报（500k 全量 Benchmark）

- 生成时间：2026-04-23 13:49:38 +08:00
- 数据来源：
  - `logs/benchmark_full_500k_20260423_101541.log`
  - `logs/benchmark_full_500k_20260423_101541.err.log`
  - `scripts/benchmark.py`
  - `src/trainer/off_policy_trainer.py`
  - `src/trainer/on_policy_trainer.py`
  - `src/environments/mec_v3/game_theory_adapters.py`

## 1. 当前训练进度与已完成算法结果

### 1.1 已完成算法（4/17）

| 算法 | Final Reward (mean ± std) | 训练耗时(s) | 备注 |
|---|---:|---:|---|
| GRPO | 11.8355 ± 1.0085 | 325.8 | 已完成 |
| PPO | 11.8350 ± 1.0084 | 378.9 | 已完成 |
| SAC | 11.8355 ± 1.0085 | 6243.9 | 已完成 |
| DDPG | 5.4054 ± 2.0366 | 4935.9 | 已完成 |

### 1.2 当前运行算法

- 当前算法：`TD3`
- 最新进度：`92160 / 500000`（约 `18.4%`）
- 最新更新计数：`update_count=71666`
- 最近速度区间：约 `80 ~ 106 it/s`（日志窗口内）

### 1.3 阶段观察（简短）

- 在当前已完成批次中，`GRPO / PPO / SAC` 的最终 reward 接近。
- `DDPG` 的 reward 明显低于前三个算法。
- 训练时长呈现明显差异：`SAC/DDPG` 远长于 `GRPO/PPO`。

---

## 2. 17个算法对环境的接入实现方式（完整）

### 2.1 统一接入主链路

1. **算法选择与映射**：在 `scripts/benchmark.py` 中通过 `ALGORITHM_CLASSES` 与 `ALGO_ENV_MAP` 完成。
2. **环境实例化**：`make_env()` 根据映射选择：
   - 离散动作：`MEC-v1-game-theory-discrete-ma`（`GameTheoryDiscreteMAEnv`）
   - 连续动作：`MEC-v1-game-theory-continuous-ma`（`GameTheoryContinuousMAEnv`）
3. **智能体数选择**：
   - `MULTI_AGENT_ALGOS` 中的算法使用 `num_agents=3`
   - 其余算法使用 `num_agents=1`
4. **训练器选择**：
   - `ON_POLICY` -> `OnPolicyTrainer`
   - `OFF_POLICY` -> `OffPolicyTrainer`
5. **动作适配**：
   - 离散适配器：`Discrete((K+1)*5^3)` -> `{"target", "ratio(3维)"}`
   - 连续适配器：`Box(-1,1,(4,))`，第1维映射 `target`，后3维映射 `ratio`

### 2.2 17算法接入总表

| 算法 | 训练范式 | 训练器 | 环境ID | 动作类型 | num_agents |
|---|---|---|---|---|---:|
| GRPO | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 1 |
| PPO | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 1 |
| SAC | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 1 |
| DDQN | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 1 |
| DDPG | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 1 |
| TD3 | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 1 |
| A3C | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 1 |
| TRPO | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 1 |
| SimPO | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 1 |
| MAPPO | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 3 |
| QMIX | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 3 |
| COMA | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 3 |
| IPPO | On-Policy | OnPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 3 |
| VDN | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 3 |
| MADDPG | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 3 |
| IQL | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-discrete-ma | 离散(单索引) | 3 |
| MATD3 | Off-Policy | OffPolicyTrainer | MEC-v1-game-theory-continuous-ma | 连续(Box 4维) | 3 |

### 2.3 当前算法（TD3）接入细节

- TD3 映射到 `MEC-v1-game-theory-continuous-ma`。
- 当前基准配置下 TD3 以 `num_agents=1` 运行。
- `OffPolicyTrainer` 负责：
  - 从 agent 获取连续动作；
  - 动作规范化（flatten、NaN/Inf 处理）；
  - 调 `env.step(action)`；
  - 用回放机制做更新并维护 `update_count`。
- 连续适配器在 step 时将 4 维向量解码为底层 GameTheory 环境 Dict 动作（`target + ratio`）。

---

## 3. `results` 中历史训练数据是否有参考价值

结论：**有参考价值，但只能“有条件使用”**。

### 3.1 目前可用历史文件

当前 `results/` 保留的是：
- `benchmark.json`
- `benchmark_no_grpo.json`
- `benchmark_quick_run.json`
- `benchmark_charts.html`
- `performance_analysis_report.md`

### 3.2 参考价值（可用于）

- 早期方法筛选（哪些算法大致可跑通/收敛方向）。
- 看训练时间量级与资源预算。
- 快速对比图和报告模板复用（图表结构、指标字段）。

### 3.3 使用风险（不能直接当最终结论）

- **实验预算不一致**：历史运行存在不同 `timesteps`（如 10k/100k/500k）混用。
- **运行状态不一致**：部分历史任务可能是中断或分组跑出来的结果。
- **代码版本漂移**：环境适配、训练器逻辑、清理策略发生过调整时，可比性会下降。
- **结果文件不完整**：你已做过清理，部分中间文件已删除，复盘链路不完整。

### 3.4 建议的使用策略

- 将历史数据定位为“探索性证据”，不作为论文/最终结论主表。
- 以当前这条 **统一 500k 全量 run** 作为主结果来源。
- 当前 run 完成后，再统一导出最终总表和图，确保“同预算、同代码、同环境映射”。

---

## 4. 结论（简短）

- 目前 500k 全量实验已完成 4 个算法，正在训练 TD3。
- 17 算法接入链路在工程上是统一且清晰的：`算法映射 -> 适配环境 -> 训练器 -> 动作解码`。
- 历史 `results` 数据可参考，但最终对比结论应以当前统一 500k 实验为准。
