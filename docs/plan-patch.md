# 增量计划：算法评估指标体系优化

## 元信息

- 基线：`docs/plan.md`（模块 1-6 已全部完成）
- 变更类型：patch — 新增模块 7/8/9，不修改模块 1-6
- 触发原因：现有 benchmark 缺少训练过程可视化、comm_score 公式 energy 权重偏低、无统一综合评分体系
- 前置覆盖：原 plan.md "不做事项"中"不改奖励函数、环境定义或论文实验指标"**被本 patch 显式覆盖**
- **重跑实验**：模块 7 改动 reward 函数和 comm_score 公式，现有 `results/benchmark_full_500k_*.json` 在模块 9 完成后需重跑

## 变更摘要

| 模块 | 名称 | Steps | 核心产出 |
|------|------|-------|---------|
| 7 | Reward + comm_score 权重修正 | 6 | 修改 3 个文件中的权重常量 + 公式同步 |
| 8 | 收敛曲线数据收集与可视化 | 5 | 训练过程指标导出 + 收敛曲线图 |
| 9 | Composite Score 综合评分体系 | 7 | 新模块 + 3 Profile 对比 + 雷达图 + 报告集成 |

---

# 模块 7：Reward + comm_score 权重修正

## 概述

- 职责：统一提升 energy 在 reward 函数、通信成本函数、comm_score 公式中的权重，消除 energy 被 latency 碾压的系统性偏差
- 前置依赖：无（纯权重常量修改）
- 预计步骤数：6

## 变更决策记录

| 位置 | 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|------|
| `game_theory_env.py:719` `r_imm` | `energy_ratio` 系数 | 0.10 | 0.25 | energy 占 r_imm 仅 10%，算法无动力省能耗 |
| `game_theory_env.py:716` `r_imm` | `latency_ratio` 系数 | 0.55 | 0.40 | 让出 0.15 给 energy，保持总和 1.0 |
| `game_theory_env.py:1107` `_communication_cost` | energy 系数 | 0.10 | 0.25 | 与 reward 保持一致的 energy 重要性 |
| `game_theory_env.py:1104` `_communication_cost` | latency 系数 | 0.60 | 0.45 | 让出 0.15 给 energy |
| `base_trainer.py:427` `comm_score` | `0.1 * energy` | 0.1 | 0.3 | 分母中 energy 影响力仅为 latency 的 1/10 |
| `benchmark.py:582` `comm_score` | `0.1 * energy` | 0.1 | 0.3 | 同步 base_trainer |
| `evaluate.py:314` `comm_score` | `0.1 * energy` | 0.1 | 0.3 | 同步 base_trainer |

## Step 1：修改 reward 函数权重

- 文件：`src/environments/mec_v3/game_theory_env.py`
- 操作：修改 `HierarchicalReward.compute()` 中 `r_imm` 的系数
- 具体变更：
  ```python
  # 旧（第 715-720 行）
  r_imm = -(
      0.55 * latency_ratio
      + 0.15 * queue_ratio
      + 0.20 * deadline_miss_ratio * deadline_miss_ratio
      + 0.10 * energy_ratio
      + non_nearest_penalty
  )

  # 新
  r_imm = -(
      0.40 * latency_ratio
      + 0.15 * queue_ratio
      + 0.20 * deadline_miss_ratio * deadline_miss_ratio
      + 0.25 * energy_ratio
      + non_nearest_penalty
  )
  ```
- 验证：`python -c "assert 0.40 + 0.15 + 0.20 + 0.25 == 1.0"`（non_nearest_penalty 是条件附加项，不计入归一化）

## Step 2：修改 `_communication_cost` 权重

- 文件：`src/environments/mec_v3/game_theory_env.py`
- 操作：修改 `GameTheoryMECEnv._communication_cost()` 中的系数
- 具体变更：
  ```python
  # 旧（第 1103-1108 行）
  return float(
      0.60 * latency / max(self.latency_budget, EPS)
      + 0.15 * queue_wait / max(self.latency_budget, EPS)
      + 0.15 * deadline_miss_ratio * deadline_miss_ratio
      + 0.10 * float(energy) / max(self.energy_budget, EPS)
      + 0.05 * non_nearest
  )

  # 新
  return float(
      0.45 * latency / max(self.latency_budget, EPS)
      + 0.15 * queue_wait / max(self.latency_budget, EPS)
      + 0.10 * deadline_miss_ratio * deadline_miss_ratio
      + 0.25 * float(energy) / max(self.energy_budget, EPS)
      + 0.05 * non_nearest
  )
  ```
- 验证：`python -c "assert 0.45 + 0.15 + 0.10 + 0.25 + 0.05 == 1.0"`

## Step 3：修改 `base_trainer.py` 中 `comm_score` 公式

- 文件：`src/trainer/base_trainer.py`
- 操作：第 427 行 `0.1 * e2e_energy_mean` → `0.3 * e2e_energy_mean`
- 具体变更：
  ```python
  # 旧
  comm_score = float(
      100.0
      * throughput_tasks_per_step
      * max(0.0, 1.0 - deadline_miss_rate)
      / (1.0 + e2e_latency_p95 + 0.1 * e2e_energy_mean)
  )

  # 新
  comm_score = float(
      100.0
      * throughput_tasks_per_step
      * max(0.0, 1.0 - deadline_miss_rate)
      / (1.0 + e2e_latency_p95 + 0.3 * e2e_energy_mean)
  )
  ```
- 验证：`grep -n "0\.3 \* e2e_energy_mean" src/trainer/base_trainer.py` 应返回恰好 1 行

## Step 4：同步 `benchmark.py` 中 `comm_score` 公式

- 文件：`scripts/benchmark.py`
- 操作：第 582 行 `0.1 * energy_per_task` → `0.3 * energy_per_task`
- 具体变更：
  ```python
  # 旧
  comm_score = float(100.0 * throughput * max(0.0, 1.0 - deadline_miss_rate) / (1.0 + e2e_p95 + 0.1 * energy_per_task))

  # 新
  comm_score = float(100.0 * throughput * max(0.0, 1.0 - deadline_miss_rate) / (1.0 + e2e_p95 + 0.3 * energy_per_task))
  ```
- 验证：`grep -n "0\.3 \* energy_per_task" scripts/benchmark.py` 应返回恰好 1 行

## Step 5：同步 `evaluate.py` 中 `comm_score` 公式

- 文件：`scripts/evaluate.py`
- 操作：第 314 行同步修改
- 验证：`grep -rn "0\.1 \* e2e_energy\|0\.1 \* energy_per_task" src/ scripts/` 应返回 **0 行**（确认无遗漏）

## Step 6：权重变更单元测试

- 文件：新建 `tests/test_reward_weights.py`
- 内容：
  1. `test_r_imm_energy_weight()`：实例化 `HierarchicalReward`，构造 latency=0, energy=1 的输入，断言 `r_imm` 中 energy 贡献占比为 0.25
  2. `test_communication_cost_energy_weight()`：构造同等 latency/energy 归一化值，断言 `_communication_cost` 返回值中 energy 贡献符合 0.25 比例
  3. `test_comm_score_energy_coefficient()`：直接验证 `base_trainer._evaluate()` 输出的 `comm_score` 在 energy 变化时的灵敏度比旧系数高 3 倍
- 验证：`pytest tests/test_reward_weights.py -v`

---

# 模块 8：收敛曲线数据收集与可视化

## 概述

- 职责：在 benchmark 流程中保留每个 eval_interval 的训练指标，输出收敛曲线图
- 前置依赖：模块 7（权重修正后的指标才有意义）
- 现状：`base_trainer.py` 已在每次 eval 时追加 `eval_logs` 并写出 `train_logs.json` 到 `save_dir`，但 `benchmark.py` 不收集这些文件
- 预计步骤数：5

## Step 1：`benchmark_single` 收集 `train_logs`

- 文件：`scripts/benchmark.py`
- 操作：在 `benchmark_single()` 函数中（第 680 行 `trainer.train()` 之后），增加收集 `trainer.eval_logs` 的逻辑
- 具体变更：
  ```python
  # 在 trainer.train() 之后、trainer.evaluate() 之前：
  convergence_data = {}
  for key, values in trainer.eval_logs.items():
      convergence_data[key] = [round(v, 6) for v in values]
  # eval_interval 用来还原 x 轴时间步
  convergence_data["eval_interval"] = trainer.eval_interval
  convergence_data["total_timesteps"] = trainer.total_steps
  ```
- 在 `result` dict（第 694-712 行）中追加字段：
  ```python
  result["convergence"] = convergence_data
  ```
- 验证：`python -c "import scripts.benchmark"` 无报错

## Step 2：`run_benchmark` 汇总收敛数据

- 文件：`scripts/benchmark.py`
- 操作：在 `run_benchmark()` 的 `avg` 聚合逻辑（第 847-881 行）中，将 `convergence` 字段从各 seed 的 result 中提取并以 `{seed: convergence_data}` 格式存入 `avg["convergence_by_seed"]`
- 不对 convergence 做跨 seed 平均（曲线需要分 seed 展示）
- 验证：临时用 1 算法 1000 步 dry-run 确认 JSON 中有 `convergence_by_seed` 字段

## Step 3：新建收敛曲线绘图函数

- 文件：`scripts/plot_results.py`
- 操作：新增函数 `plot_convergence_curves(results, output_dir, fmt)`
- 功能：
  1. 读取每个算法的 `convergence_by_seed` → 提取 `eval/reward_mean` 序列
  2. x 轴：`eval_interval * index`，y 轴：reward_mean
  3. 每个算法一条线，复用 `ALGO_COLORS`
  4. 子图 1：reward 收敛曲线（全部算法叠加）
  5. 子图 2：latency 收敛曲线
  6. 子图 3：energy 收敛曲线
  7. 子图 4：comm_score 收敛曲线
  8. 滑窗收敛判定：最后 10% 窗口的相对变化率 < 5% 则标注 ✓ 已收敛
  9. Legend 中标注收敛状态
- 输出：`figures/convergence_curves.{fmt}`
- 验证：函数存在且可被 import

## Step 4：集成到 `plot_results.py` CLI

- 文件：`scripts/plot_results.py`
- 操作：在 `main()` 中调用 `plot_convergence_curves()`
- 条件：仅当 JSON 中有 `convergence_by_seed` 字段时才画
- 验证：`python scripts/plot_results.py --help` 无报错

## Step 5：收敛曲线单元测试

- 文件：新建 `tests/test_convergence_plot.py`
- 内容：
  1. 构造含 `convergence_by_seed` 的 mock JSON
  2. 调用 `plot_convergence_curves()` 验证不抛异常
  3. 验证输出文件存在
- 验证：`pytest tests/test_convergence_plot.py -v`

---

# 模块 9：Composite Score 综合评分体系

## 概述

- 职责：实现加权归一化综合评分，支持多 profile 对比，集成到 benchmark / plot / report 流程
- 前置依赖：模块 7（权重修正）、模块 8（收敛数据用于 stability 评分）
- 预计步骤数：7

## 三个预设 Profile

| Profile | 代号 | 场景 | reward | latency | energy | stability |
|---------|------|------|--------|---------|--------|-----------|
| A | `balanced` | 通用 MEC | 0.30 | 0.30 | 0.25 | 0.15 |
| B | `latency_critical` | 自动驾驶/AR | 0.20 | 0.40 | 0.25 | 0.15 |
| C | `energy_constrained` | IoT/传感器 | 0.20 | 0.25 | 0.40 | 0.15 |

## Step 1：创建 `src/utils/composite_score.py`

- 文件：新建 `src/utils/composite_score.py`
- 类：`CompositeScorer`
- 接口：
  ```python
  class CompositeScorer:
      """加权归一化综合评分器"""

      def __init__(self, profiles: Dict[str, Dict[str, float]]):
          """profiles: {"balanced": {"reward": 0.30, "latency": 0.30, "energy": 0.25, "stability": 0.15}, ...}"""

      def normalize(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
          """对全部算法结果做 min-max 归一化
          返回 {algorithm: {"reward_norm": float, "latency_norm": float, ...}}
          注意 latency 和 energy 是越小越好（反向归一化）"""

      def score(self, results: List[Dict], profile_name: str) -> List[Dict]:
          """返回 [{algorithm, composite_score, rank, breakdown: {dim: weighted_score}}, ...]
          按 composite_score 降序排列"""

      def score_all_profiles(self, results: List[Dict]) -> Dict[str, List[Dict]]:
          """对每个 profile 算一遍，返回 {profile_name: [scored_results]}"""

      def robustness_summary(self, results: List[Dict]) -> List[Dict]:
          """返回 [{algorithm, avg_rank, worst_rank, best_rank, rank_variance}, ...]
          在所有 profile 中都排前 3 的算法标记 robust=True"""
  ```
- `stability` 度量定义：`1.0 - min(1.0, reward_std / abs(reward_mean))` — reward 变异系数越小 stability 越高
- 归一化规则：
  - `reward`：正向，min-max → [0, 1]
  - `latency`：反向，max-min → [0, 1]（越小越好）
  - `energy`：反向，max-min → [0, 1]
  - `stability`：正向，min-max → [0, 1]
- 验证：`python -c "from src.utils.composite_score import CompositeScorer"` 无报错

## Step 2：创建 `configs/scoring_profiles.yaml`

- 文件：新建 `configs/scoring_profiles.yaml`
- 内容：
  ```yaml
  profiles:
    balanced:
      reward: 0.30
      latency: 0.30
      energy: 0.25
      stability: 0.15
    latency_critical:
      reward: 0.20
      latency: 0.40
      energy: 0.25
      stability: 0.15
    energy_constrained:
      reward: 0.20
      latency: 0.25
      energy: 0.40
      stability: 0.15
  ```
- 验证：`python -c "import yaml; yaml.safe_load(open('configs/scoring_profiles.yaml'))"`

## Step 3：集成 composite_score 到 `benchmark.py`

- 文件：`scripts/benchmark.py`
- 操作：在 `run_benchmark()` 函数写出 JSON 之前（第 925 行附近），加载 `scoring_profiles.yaml`，用 `CompositeScorer` 计算所有 profile 的 composite_score，追加到每个算法的 result dict 中
- 具体追加字段：
  ```python
  # 每个算法结果中追加
  result["composite_scores"] = {
      "balanced": {"score": 0.72, "rank": 3, "breakdown": {...}},
      "latency_critical": {"score": 0.65, "rank": 5, "breakdown": {...}},
      "energy_constrained": {"score": 0.81, "rank": 1, "breakdown": {...}},
  }
  result["robustness"] = {"avg_rank": 3.0, "worst_rank": 5, "best_rank": 1, "robust": False}
  ```
- 验证：dry-run 1 个算法 1000 步，确认 JSON 输出含 `composite_scores` 字段

## Step 4：新增可视化图表

- 文件：`scripts/plot_results.py`
- 新增函数：
  1. `plot_composite_ranking(results, output_dir, fmt)`
     - 分 3 个子图，每个 profile 一个分组柱状图
     - 柱子按 composite_score 降序排列
     - 颜色 = `ALGO_COLORS`
  2. `plot_radar_chart(results, output_dir, fmt)`
     - 每个算法一个雷达图（4 维：reward / latency / energy / stability 的归一化值）
     - 最多画排名前 6 的算法，避免过于拥挤
     - 用 `balanced` profile 的排名取前 6
  3. `plot_weight_sensitivity(results, output_dir, fmt)`
     - 3 个 profile 下各算法排名变化的折线图
     - x 轴 = profile，y 轴 = rank（倒序，rank 1 在上）
     - 线条交叉多 = 排名对权重敏感
- 输出文件：
  - `figures/composite_ranking.{fmt}`
  - `figures/radar_top6.{fmt}`
  - `figures/weight_sensitivity.{fmt}`
- 验证：`python scripts/plot_results.py --help` 无报错

## Step 5：更新 `generate_report.py`

- 文件：`scripts/generate_report.py`
- 操作：
  1. 在报告 Markdown 中新增 `## 综合评分排名` 章节
  2. 输出 3 个 profile 的排名表
  3. 输出 robustness 汇总表
  4. 引用 `figures/composite_ranking.*` 和 `figures/radar_top6.*`
- 验证：`python scripts/generate_report.py --help` 无报错

## Step 6：Composite Score 单元测试

- 文件：新建 `tests/test_composite_score.py`
- 测试用例：
  1. `test_normalize_direction()`：验证 latency/energy 反向归一化正确（最低值 → 1.0）
  2. `test_score_weights_sum_to_one()`：验证每个 profile 权重之和为 1.0
  3. `test_ranking_order()`：构造已知数据，断言排名正确
  4. `test_robustness_flag()`：构造 3 个 profile 中始终排前 3 的算法，断言 `robust=True`
  5. `test_stability_metric()`：reward_std=0 → stability=1.0；reward_std=|reward_mean| → stability=0.0
- 验证：`pytest tests/test_composite_score.py -v`

## Step 7：全流程集成验收

- 操作：运行以下验收序列
  ```bash
  # 1. 全部单元测试
  pytest tests/test_reward_weights.py tests/test_convergence_plot.py tests/test_composite_score.py -v

  # 2. 确认 comm_score 公式无遗漏旧版
  grep -rn "0\.1 \* e2e_energy\|0\.1 \* energy_per_task\|0\.10 \* energy_ratio\|0\.10 \* float(energy)" src/ scripts/
  # 期望输出为空

  # 3. 确认新文件均可导入
  python -c "from src.utils.composite_score import CompositeScorer; print('OK')"

  # 4. 确认 plot_results.py 新函数存在
  python -c "from scripts.plot_results import plot_convergence_curves, plot_composite_ranking, plot_radar_chart, plot_weight_sensitivity; print('OK')"

  # 5. 确认 benchmark.py 可正常启动（不真正训练）
  python scripts/benchmark.py --help
  ```
- 验证：全部命令返回 0

---

## 模块 7-9 完成后操作

### 重跑实验

模块 7 修改了 reward 函数和 comm_score 公式，现有实验数据全部作废。需重跑：

```bash
python scripts/benchmark.py --all --timesteps 100000 --device auto
```

预计耗时 2-4 小时（17 算法 × 100k 步，CPU）。重跑完成后：

```bash
python scripts/plot_results.py --input results/benchmark_<timestamp>.json --output figures/ --format pdf
python scripts/generate_report.py
```

### 不做事项（本 patch 范围）

- 不改算法网络结构或超参
- 不改环境物理模型（channel、server、task 生成）
- 不引入新依赖（composite_score 仅用 numpy + yaml）
- 不改 VSCode 入口配置（模块 1-6 产出不动）
- 不改实验管理器流程
