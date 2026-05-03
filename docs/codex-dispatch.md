# Codex 任务派发

## 类型：patch

进入增量执行模式。**只执行 `docs/plan-patch.md` 中的模块 7/8/9**，不触碰模块 1-6 的任何代码。

## 启动

1. 读取 `docs/progress.md`，确认模块 1-6 已全部 completed。
2. 读取 `docs/plan-patch.md`，加载增量开发计划。
3. 从 **模块 7 Step 1** 开始执行。
4. 本次变更会修改 reward 函数和 comm_score 公式——这是用户确认的变更，不受原 plan.md "不做事项"约束。

## 计划模块

- 模块 7：Reward + comm_score 权重修正（6 steps）
- 模块 8：收敛曲线数据收集与可视化（5 steps）
- 模块 9：Composite Score 综合评分体系（7 steps）

## 依赖链

```
模块 7（全部 6 步）→ 模块 8（全部 5 步）→ 模块 9（全部 7 步）
```

模块 8 依赖模块 7（权重修正后的公式）。模块 9 依赖模块 8（收敛数据用于 stability 计算）。严格按此顺序执行。

## 核心变更文件清单

### 修改文件（必须精确定位行号后再改）

| 文件 | 变更描述 |
|------|---------|
| `src/environments/mec_v3/game_theory_env.py` | `r_imm` 权重：latency 0.55→0.40, energy 0.10→0.25；`_communication_cost` 权重：latency 0.60→0.45, energy 0.10→0.25, deadline 0.15→0.10 |
| `src/trainer/base_trainer.py` | `comm_score` 公式 `0.1 * e2e_energy_mean` → `0.3 * e2e_energy_mean` |
| `scripts/benchmark.py` | 同步 comm_score 公式；`benchmark_single()` 收集 convergence 数据；`run_benchmark()` 汇总 convergence + composite_score |
| `scripts/evaluate.py` | 同步 comm_score 公式 |
| `scripts/plot_results.py` | 新增 4 个绘图函数：convergence_curves, composite_ranking, radar_chart, weight_sensitivity |
| `scripts/generate_report.py` | 新增综合评分排名章节 |

### 新增文件

| 文件 | 描述 |
|------|------|
| `src/utils/composite_score.py` | CompositeScorer 类，加权归一化 + 多 profile 评分 |
| `configs/scoring_profiles.yaml` | 3 组权重 profile 配置 |
| `tests/test_reward_weights.py` | reward 权重变更验证 |
| `tests/test_convergence_plot.py` | 收敛曲线绘图验证 |
| `tests/test_composite_score.py` | 综合评分逻辑验证 |

## 执行规则

- 从模块 7 Step 1 开始，按 `docs/plan-patch.md` 逐步执行。
- 每个步骤完成后运行该步骤指定的验证命令。
- **验证通过 → 直接执行下一步骤，不要停下来问我。**
- **验证失败 → 自行诊断修复，最多重试 2 次；仍失败则记录到 `docs/issues.md` 并停下报告。**
- 每完成一个模块后，更新 `docs/progress.md`，在模块 6 之后追加新模块的进度。

## 关键约束

### comm_score 公式一致性

修改 comm_score 公式时，必须确保以下 3 处完全同步：
1. `src/trainer/base_trainer.py:427` — `_evaluate()` 中
2. `scripts/benchmark.py:582` — `benchmark_heuristic()` 中
3. `scripts/evaluate.py:314` — `evaluate_model()` 中

修改后运行：
```bash
grep -rn "0\.1 \* e2e_energy\|0\.1 \* energy_per_task" src/ scripts/
```
期望输出为空。如果不为空，说明有遗漏。

### reward 权重总和

`r_imm` 中的 latency_ratio + queue_ratio + deadline_miss_ratio + energy_ratio 系数总和必须恒等于 1.0（non_nearest_penalty 是条件项不算）。同理 `_communication_cost` 中 5 项系数总和 = 1.0。

### 不改动范围

- 模块 1-6 的全部文件不动
- `.vscode/launch.json`、`.vscode/tasks.json` 不动
- `src/experiment/` 目录不动
- 算法网络结构（`rl_algorithms/` 下所有 agent 类）不动
- 环境物理模型（channel、server、task 生成逻辑）不动

### 新增依赖

本 patch **不引入任何新第三方依赖**。`composite_score.py` 仅使用 `numpy`（已有）和 `yaml`（标准库 or `PyYAML` 已在 `requirements.txt`）。

## 禁止行为

- 不要修改模块 1-6 已完成的代码，除非是本 patch 明确列出的文件。
- 不要在 `composite_score.py` 中 import torch 或任何训练相关模块。
- 不要在修改 reward 权重时改变 `adaptive_weights()` 的逻辑。
- 不要重命名现有的 JSON 字段名（如 `final_reward_mean`），只追加新字段。
- 不要运行完整 benchmark 作为验证——用 1 算法 1000 步 dry-run 即可。

## 完成后

1. 更新 `docs/progress.md`，追加模块 7/8/9 的完成状态。
2. 运行最终验收命令（plan-patch.md 模块 9 Step 7 中列出的 5 条命令）。
3. **不要自行重跑完整 benchmark**。告知用户模块 7-9 代码已就绪，需手动运行：
   ```bash
   python scripts/benchmark.py --all --timesteps 100000 --device auto
   ```
4. 停下报告完成状态。
