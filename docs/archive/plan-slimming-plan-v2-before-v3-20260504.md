# docs/inbox/plan.md：paper2 收敛验证调试方案

## 元信息

- 项目：`paper2`
- 计划版本：`slimming-plan-v2`
- 变更类型：`patch`
- 基于版本：`slimming-plan-v1`
- GitHub 仓库：`w2030298-art/paper2`
- 创建日期：2026-05-03
- 最后更新：2026-05-03
- 当前目标：在不直接重跑 full 17 的前提下，对未收敛算法建立可复现、分阶段、有门禁的验证调试闭环。
- 总模块数：3
- 预计步骤总数：25
- 已完成步骤：14
- 新增步骤：11
- 建议开发顺序：模块 12-13 保持 `[DONE]` → 模块 14 Step 1-11

### 变更记录

| 版本 | 日期 | 变更摘要 |
|------|------|---------|
| `slimming-plan-v1` | 2026-05-02 | paper2 仓库瘦身 Phase 1-2：repo hygiene、旧入口删除、docs_paper 外迁、生成产物 Git tracking 移除。 |
| `slimming-plan-v2` | 2026-05-03 | 新增模块 14：收敛验证调试方案；建立 artifact 审计、50k targeted rerun、单变量稳定化、100k/200k 扩展门禁与论文图判定流程。 |

---

## Status

> 执行端读到此区块即可恢复上下文。

- 当前阶段：模块 14 Step 1
- 当前模块：模块 14：收敛验证与 targeted debugging
- 整体进度：`14 / 25`
- 状态：`变更后待执行`
- 阻塞项：
  - `C:\Users\22003\paper2\rl-mec-dashboard` 本机不存在，外部 dashboard 兼容性仍需在目标环境复核。
  - 收敛验证依赖本地实验产物；若 `experiments/`、`results/`、`figures/` 均无可读产物，模块 14 Step 1 必须标记 `BLOCKED: missing local experiment artifacts`。
- 重要说明：
  - 本轮不是 full 17 重跑。
  - 本轮不是直接启用稳定化调参。
  - 本轮先验证异常是否可复现，再决定是否扩大训练长度或启用单变量 override。
  - `configs/stability_overrides.yaml` 仍保持 `enabled: false`，不得全局启用。
  - 不提交 `experiments/`、`results/`、`figures/` 生成产物；只提交脚本、配置、测试、docs 记录。

### Last Iteration Summary

- `slimming-plan-v1` 已执行完成，仓库瘦身通过 `pytest -q`，报告为 `166 passed`，四个主入口 `--help` 通过。
- `docs/convergence_failure_analysis.md` 已生成，当前异常算法包括：
  - `diverging`：`A3C`、`MATD3`
  - `catastrophic_outlier`：`IPPO`、`IQL`、`VDN`
  - `bad_plateau`：`COMA`、`MADDPG`、`MAPPO`、`SAC`
  - `overall bad_plateau` 但 reward 接近 `converged_good`：`GRPO`、`TRPO`
- 当前证据只支持“已诊断 + 已生成稳定化候选”，不支持“无法收敛算法都已修复”。

### Pending Decisions

1. 若 50k baseline targeted rerun 后 `IQL/IPPO/VDN` 仍出现 `catastrophic_outlier`，是否暂停调参并进入算法/环境 bug 级 Review。
2. 若某算法 50k 仍为 `improving` 但未 `converged_good`，是否进入 100k 扩展，而不是判失败。
3. 若 `bad_plateau` 只在 reward 存在但 latency/energy/comm_score 已优，是否按论文指标权重接受。
4. dashboard 兼容性 blocked 与模块 14 无直接依赖，不阻塞收敛验证；但最终提交前仍需处理。

---

## 全局执行边界

### 本轮必须做

- 使用现有 `scripts/analyze_convergence_failures.py` 和 `scripts/plot_results.py` 作为诊断基础。
- 新增 targeted validation matrix，避免人工散跑。
- 对异常算法先跑 baseline 50k，不带 override。
- 对 `catastrophic_outlier` 先做事件审计，不能直接调参掩盖。
- 对稳定化方案采用单变量/单算法族验证，不能一次性启用所有候选。
- 每一阶段都生成 Markdown 审计报告，能解释为什么进入下一阶段或停止。
- 所有 Step 都要能通过测试或命令验证。

### 本轮禁止做

- 禁止直接重跑 full 17。
- 禁止把 `configs/stability_overrides.yaml` 改成默认启用。
- 禁止硬编码排除 `IQL`、`VDN`、`IPPO` 或任何异常算法。
- 禁止只看 clean plot 判定算法修复。
- 禁止把失败 seed 从报告中删除。
- 禁止提交本地实验数据、图像产物、checkpoint。
- 禁止改 GameTheory MEC 环境逻辑，除非 Step 2 明确证明异常来自环境 bug 并标记 `NEEDS_ESCALATION`。

### 目标算法集合

```text
A3C
COMA
GRPO
IPPO
IQL
MADDPG
MAPPO
MATD3
SAC
TRPO
VDN
```

### 阶段门禁

| 阶段 | 进入条件 | 退出条件 | 不通过处理 |
|------|----------|----------|------------|
| Phase 0 artifact audit | 本地有旧实验产物或质量报告 | 产物来源、缺失项、异常证据链完整 | 标记 `BLOCKED: missing local experiment artifacts` |
| Phase 1 baseline 50k | Phase 0 完成 | 目标算法均有 baseline 50k 结果或明确失败原因 | 失败算法写入 event audit，不进入 override |
| Phase 2 single-variable override | baseline 50k 证明异常可复现且非环境 bug | 单算法族 override 改善至少一个主指标且无新 catastrophic | 回滚候选 override，写入 no-go |
| Phase 3 100k/200k expansion | 50k 表现为 improving 或 bad_plateau 可改善 | 100k/200k 满足 acceptance threshold | 不进入论文图候选 |
| Phase 4 publication gate | Phase 3 通过 | raw/clean/report 三者一致，可复现 | 保留为诊断结果，不纳入论文结论 |

### Acceptance Thresholds

#### 通用

- `failed_seed_count == 0`
- `extreme_outlier_count == 0`
- `nan_or_inf_count == 0` 或全部可解释且不进入 clean plot
- `convergence_quality_report.json` 与 `convergence_failure_matrix*.json` 对同一算法状态不冲突
- raw diagnostic plot 中不能隐藏异常事件

#### Reward

- `catastrophic_outlier`：必须消失，且没有 `raw_min <= -100.0`
- `diverging`：必须至少改善为 `improving` 或 `bad_plateau`
- `bad_plateau`：`best_tail_gap <= 0.15`，或论文主指标加权后证明 tail 可接受
- `converged_good`：尾部稳定且接近历史最佳

#### Latency / Energy

- 越低越好，禁止沿用 reward 的越高越好逻辑。
- 若 reward 改善但 latency/energy 明显恶化，标记 `tradeoff_regression`，不直接判通过。

#### Comm Score

- 越高越好。
- 若 reward 与 comm_score 方向冲突，必须在报告中列出 composite score 判断依据。

---

# 模块 12：无争议仓库卫生清理 `[DONE]`

## Step 1：生成执行前审计快照 `[DONE]`

- **scope: auto**
- 产物：`docs/slimming_audit_phase2.md`
- 状态：已完成。
- 验证：已纳入 `pytest -q` 与 repo hygiene 检查。

## Step 2：从 Git tracking 移除生成产物 `[DONE]`

- **scope: auto**
- 对象：`experiments/`、`results/`、`figures/`、`logs/`、`checkpoints/`
- 状态：已完成；本地数据未删除，只处理 Git tracking / ignore policy。
- 验证：`.gitignore` 与 repo hygiene 测试。

## Step 3：删除废弃坏入口 `src/trainer/benchmark.py` `[DONE]`

- **scope: auto**
- 删除：`src/trainer/benchmark.py`
- 保留：`scripts/benchmark.py`
- 状态：已完成。
- 验证：`python scripts/benchmark.py --help`

## Step 4：统一 docs 归档并清理 graphify cache `[DONE]`

- **scope: auto**
- 对象：`docs/archive/`、`graphify-out/cache/`
- 状态：已完成。
- 验证：repo hygiene 测试。

## Step 5：移除未使用 dashboard 依赖 `[DONE]`

- **scope: auto**
- 对象：`requirements.txt`
- 状态：已完成。
- 验证：`pytest -q`

## Step 6：新增仓库卫生测试 `[DONE]`

- **scope: auto**
- 文件：`tests/test_repo_hygiene.py`、`tests/test_active_entrypoints.py`
- 状态：已完成。
- 验证：`pytest -q`

---

# 模块 13：已确认非核心功能删除 `[DONE / NEEDS_REVIEW]`

## Step 1：迁移 `docs_paper/` 到外部写作资料目录 `[DONE / NEEDS_REVIEW]`

- **scope: review**
- 源：`docs_paper/`
- 目标：`C:\Users\22003\paper2\writing_ref\docs_paper`
- 状态：已执行，等待人工审核。
- 验证：迁移记录存在，关键文件已校验。

## Step 2：删除 `scripts/evaluate.py` `[DONE]`

- **scope: auto**
- 状态：已完成。
- 验证：主入口 help 与测试通过。

## Step 3：删除 `scripts/generate_report.py` `[DONE]`

- **scope: auto**
- 状态：已完成。
- 替代链路：`scripts/plot_results.py` + `scripts/analyze_convergence_failures.py`

## Step 4：删除 callback 扩展机制 `[DONE / NEEDS_REVIEW]`

- **scope: review**
- 删除：`src/trainer/callbacks.py`
- 修改：`src/trainer/base_trainer.py`
- 状态：已执行，等待人工审核。
- 验证：`pytest -q`

## Step 5：删除旧 utils 工具 `[DONE / NEEDS_REVIEW]`

- **scope: review**
- 删除：
  - `src/utils/logger.py`
  - `src/utils/config.py`
  - `src/utils/action_utils.py`
- 状态：已执行，等待人工审核。
- 验证：`pytest -q`

## Step 6：buffer canonical owner 收敛 `[DONE / NEEDS_REVIEW]`

- **scope: review**
- 保留：`src/utils/buffer.py`
- 删除或迁移：`rl_algorithms/utils/buffers.py`
- 状态：已执行，等待人工审核。
- 验证：`pytest -q`

## Step 7：更新 README 与 docs 契约 `[DONE]`

- **scope: auto**
- 状态：已完成。
- 验证：文档引用检查 + 主入口 help。

## Step 8：更新执行报告与进度 `[DONE]`

- **scope: auto**
- 文件：`docs/report.md`、`docs/progress.md`、`docs/issues.md`
- 状态：已完成。
- 验证：docs contract 测试。

---

# 模块 14：收敛验证与 targeted debugging

## 概述

- 职责：对未收敛算法建立从旧产物审计、短程复现、单变量稳定化、扩展训练到论文图候选的完整验证闭环。
- 前置依赖：
  - `scripts/analyze_convergence_failures.py`
  - `scripts/plot_results.py`
  - `configs/stability_overrides.yaml`
  - 本地 `experiments/`、`results/`、`figures/` 中至少一种可读产物
- 输出：
  - `docs/convergence_validation_manifest.md`
  - `docs/convergence_event_audit.md`
  - `configs/convergence_validation_matrix.yaml`
  - `scripts/run_convergence_validation.py`
  - `tests/test_convergence_validation_runner.py`
  - `results/convergence_failure_matrix_baseline_50k.json`
  - `docs/convergence_validation_baseline_50k.md`
  - `docs/convergence_validation_gate_50k.md`
  - `results/convergence_validation_override_matrix.json`
  - `docs/convergence_validation_final.md`

---

## Step 1：冻结当前诊断基线与数据来源清单

- **scope: auto**
- 新增：`docs/convergence_validation_manifest.md`
- 修改：无训练代码修改。
- 操作：
  1. 检查并记录以下文件是否存在：
     ```text
     results/benchmark.json
     results/benchmark_paper2_full_17_vscode.json
     results/benchmark_full_*.json
     figures/convergence_quality_report.json
     figures/convergence_quality_report.md
     experiments/paper2_full_17_vscode/run.json
     experiments/paper2_full_17_vscode/state.json
     experiments/paper2_full_17_vscode/artifacts/*/result.json
     experiments/paper2_full_17_vscode/artifacts/*/stdout.log
     experiments/paper2_full_17_vscode/artifacts/*/stderr.log
     experiments/paper2_full_17_vscode/artifacts/*/checkpoints/train_logs.json
     ```
  2. 重新运行现有诊断脚本，不跑实验：
     ```bash
     python scripts/analyze_convergence_failures.py \
       --algorithms A3C COMA GRPO IPPO IQL MADDPG MAPPO MATD3 SAC TRPO VDN \
       --output-json results/convergence_failure_matrix_pre_validation.json \
       --output-md docs/convergence_failure_analysis_pre_validation.md
     ```
  3. 在 `docs/convergence_validation_manifest.md` 写入：
     - 选中的 benchmark JSON
     - 选中的 quality report
     - 每个目标算法是否有 `result.json/stdout.log/stderr.log/train_logs.json`
     - 缺失文件清单
     - 当前状态快照
- 验证：
  ```bash
  test -f docs/convergence_validation_manifest.md
  test -f results/convergence_failure_matrix_pre_validation.json
  test -f docs/convergence_failure_analysis_pre_validation.md
  python scripts/analyze_convergence_failures.py --help
  ```

---

## Step 2：审计 catastrophic outlier 算法的事件证据

- **scope: review**
- 新增：`docs/convergence_event_audit.md`
- 目标算法：
  ```text
  IPPO IQL VDN
  ```
- 操作：
  1. 逐算法读取：
     ```text
     experiments/**/artifacts/<ALGO>/result.json
     experiments/**/artifacts/<ALGO>/stdout.log
     experiments/**/artifacts/<ALGO>/stderr.log
     experiments/**/artifacts/<ALGO>/checkpoints/train_logs.json
     figures/convergence_quality_report.json
     results/convergence_failure_matrix_pre_validation.json
     ```
  2. 定位导致 `raw_min <= -100.0` 或 `extreme_outlier_count > 0` 的 episode / timestep / seed。
  3. 给每个算法输出以下字段：
     ```text
     algorithm
     seed
     first_bad_timestep
     raw_min
     previous_value
     next_value
     stderr_summary
     stdout_summary
     suspected_layer: env | reward | action_mask | replay | optimizer | logging | unknown
     decision: rerun_baseline | needs_code_review | insufficient_data
     ```
  4. 若发现异常来自环境/reward/action/logging bug，而非训练随机性，停止后续 override，更新 `docs/report.md` 为：
     ```text
     STATUS: NEEDS_ESCALATION
     ```
- 验证：
  ```bash
  test -f docs/convergence_event_audit.md
  grep -E "IPPO|IQL|VDN" docs/convergence_event_audit.md
  grep -E "suspected_layer|decision" docs/convergence_event_audit.md
  ```

---

## Step 3：新增 targeted validation matrix 配置

- **scope: auto**
- 新增：`configs/convergence_validation_matrix.yaml`
- 操作：
  写入以下结构，不要修改 `configs/stability_overrides.yaml` 的 `enabled: false`：
  ```yaml
  schema_version: 1
  default:
    seeds: [42, 43, 44]
    eval_interval: 1000
    eval_episodes: 10
    output_root: experiments/convergence_validation
    no_full_17: true

  phases:
    baseline_50k:
      steps: 50000
      use_stability_overrides: false
      algorithms: [A3C, COMA, GRPO, IPPO, IQL, MADDPG, MAPPO, MATD3, SAC, TRPO, VDN]

    override_50k:
      steps: 50000
      use_stability_overrides: true
      mode: single_family_only
      families:
        on_policy_high_variance: [A3C, COMA, IPPO, MAPPO]
        continuous_actor_critic: [SAC, MADDPG, MATD3]
        value_decomposition: [VDN, IQL]
        conservative_policy_update: [GRPO, TRPO]

    expansion_100k:
      steps: 100000
      enter_only_if: [improving, bad_plateau_with_small_gap, converged_good]
      use_best_validated_override: true

    expansion_200k:
      steps: 200000
      enter_only_if: [improving_after_100k]
      use_best_validated_override: true
  ```
- 验证：
  ```bash
  python - <<'PY'
  import yaml
  p = yaml.safe_load(open("configs/convergence_validation_matrix.yaml", encoding="utf-8"))
  assert p["default"]["no_full_17"] is True
  assert len(p["phases"]["baseline_50k"]["algorithms"]) == 11
  assert p["phases"]["baseline_50k"]["use_stability_overrides"] is False
  PY
  ```

---

## Step 4：新增 targeted validation runner

- **scope: review**
- 新增：
  - `scripts/run_convergence_validation.py`
  - `tests/test_convergence_validation_runner.py`
- 操作：
  1. 实现 CLI：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase baseline_50k \
       --dry-run
     ```
  2. CLI 参数：
     ```text
     --matrix
     --phase baseline_50k|override_50k|expansion_100k|expansion_200k
     --algorithms optional list
     --seeds optional list
     --dry-run
     --output-root
     --no-submit
     ```
  3. dry-run 必须输出将执行的算法、seed、steps、override 状态、结果路径，不启动训练。
  4. 非 dry-run 模式通过现有入口调度：
     - 优先复用 `scripts/benchmark.py`
     - 若 `scripts/benchmark.py` 不支持单算法/单 seed/steps，则只生成 command manifest，不强行改 benchmark 行为
  5. 输出：
     ```text
     experiments/convergence_validation/<phase>/<timestamp>/manifest.json
     experiments/convergence_validation/<phase>/<timestamp>/commands.txt
     ```
- 验证：
  ```bash
  python scripts/run_convergence_validation.py --help
  python scripts/run_convergence_validation.py \
    --matrix configs/convergence_validation_matrix.yaml \
    --phase baseline_50k \
    --dry-run
  pytest tests/test_convergence_validation_runner.py -q
  ```

---

## Step 5：补齐验证调试单元测试与契约测试

- **scope: auto**
- 新增或修改：
  - `tests/test_convergence_validation_runner.py`
  - `tests/test_analyze_convergence_failures.py`
  - `tests/test_convergence_plot.py`
  - `tests/test_docs_contract.py`
- 覆盖：
  1. `configs/convergence_validation_matrix.yaml` 可解析。
  2. baseline phase 不启用 override。
  3. override phase 只能单 family 启用，不能全局启用。
  4. target algorithms 不等于 full 17。
  5. catastrophic outlier 状态不能被 clean plot 隐藏。
  6. `docs/convergence_event_audit.md` 缺失时，override phase 不允许执行。
  7. `configs/stability_overrides.yaml` 默认仍为 `enabled: false`。
- 验证：
  ```bash
  pytest \
    tests/test_convergence_validation_runner.py \
    tests/test_analyze_convergence_failures.py \
    tests/test_convergence_plot.py \
    tests/test_docs_contract.py \
    -q
  ```

---

## Step 6：执行 baseline 50k targeted rerun

- **scope: review**
- 前置：
  - Step 1-5 全部通过。
  - Step 2 未标记 `NEEDS_ESCALATION`。
- 目标算法：
  ```text
  A3C COMA GRPO IPPO IQL MADDPG MAPPO MATD3 SAC TRPO VDN
  ```
- 操作：
  1. 先 dry-run：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase baseline_50k \
       --dry-run
     ```
  2. 确认 manifest 中：
     - `steps == 50000`
     - `use_stability_overrides == false`
     - 只包含 11 个目标算法
     - seeds 为 `[42, 43, 44]`
  3. 再执行 baseline：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase baseline_50k
     ```
  4. 生成 baseline 诊断：
     ```bash
     python scripts/analyze_convergence_failures.py \
       --input results/convergence_validation_baseline_50k.json \
       --quality-report figures/convergence_quality_report.json \
       --algorithms A3C COMA GRPO IPPO IQL MADDPG MAPPO MATD3 SAC TRPO VDN \
       --output-json results/convergence_failure_matrix_baseline_50k.json \
       --output-md docs/convergence_validation_baseline_50k.md
     ```
- 验证：
  ```bash
  test -f results/convergence_failure_matrix_baseline_50k.json
  test -f docs/convergence_validation_baseline_50k.md
  grep -E "A3C|IQL|VDN|MATD3" docs/convergence_validation_baseline_50k.md
  ```

---

## Step 7：baseline 50k 门禁判定

- **scope: review**
- 新增：`docs/convergence_validation_gate_50k.md`
- 输入：
  - `docs/convergence_failure_analysis_pre_validation.md`
  - `docs/convergence_event_audit.md`
  - `docs/convergence_validation_baseline_50k.md`
  - `results/convergence_failure_matrix_pre_validation.json`
  - `results/convergence_failure_matrix_baseline_50k.json`
- 操作：
  1. 对每个算法生成判定：
     ```text
     algorithm
     old_status
     baseline_50k_status
     status_delta: improved | unchanged | worsened | inconclusive
     outlier_delta
     best_tail_gap_delta
     decision: accept_baseline | run_override_50k | expand_100k | needs_code_review | stop
     reason
     ```
  2. 门禁规则：
     - `catastrophic_outlier` 复现：`needs_code_review`，不进入 override。
     - `diverging` 改善为 `improving/bad_plateau/converged_good`：可进入 100k 或 override。
     - `bad_plateau` 且 `best_tail_gap <= 0.15`：可进入 100k。
     - `bad_plateau` 且 `best_tail_gap > 0.15`：进入对应 family override。
     - `converged_good` 且无其他指标退化：进入 publication candidate。
  3. 若任何算法出现训练崩溃或日志 schema 变化，更新 `docs/issues.md`。
- 验证：
  ```bash
  test -f docs/convergence_validation_gate_50k.md
  grep -E "decision|status_delta|best_tail_gap_delta" docs/convergence_validation_gate_50k.md
  ```

---

## Step 8：执行单变量稳定化 override 50k

- **scope: review**
- 前置：
  - Step 7 明确哪些算法进入 `run_override_50k`
  - `docs/convergence_event_audit.md` 不存在 `needs_code_review`
- 目标：
  - 只对 Step 7 判定需要 override 的算法执行。
  - 每次只启用一个算法族 override。
  - 不修改 `configs/stability_overrides.yaml` 的默认 `enabled: false`。
- 操作：
  1. 对每个 family dry-run：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase override_50k \
       --algorithms <ALGO_LIST_FROM_GATE> \
       --dry-run
     ```
  2. 执行对应 family 的 override：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase override_50k \
       --algorithms <ALGO_LIST_FROM_GATE>
     ```
  3. 汇总：
     ```bash
     python scripts/analyze_convergence_failures.py \
       --input results/convergence_validation_override_50k.json \
       --quality-report figures/convergence_quality_report.json \
       --algorithms A3C COMA GRPO IPPO IQL MADDPG MAPPO MATD3 SAC TRPO VDN \
       --output-json results/convergence_validation_override_matrix.json \
       --output-md docs/convergence_validation_override_50k.md
     ```
- 验证：
  ```bash
  test -f results/convergence_validation_override_matrix.json
  test -f docs/convergence_validation_override_50k.md
  python - <<'PY'
  import yaml
  p = yaml.safe_load(open("configs/stability_overrides.yaml", encoding="utf-8"))
  assert p["enabled"] is False
  PY
  ```

---

## Step 9：100k / 200k 扩展复跑门禁

- **scope: review**
- 新增：`docs/convergence_validation_expansion_gate.md`
- 输入：
  - `docs/convergence_validation_gate_50k.md`
  - `docs/convergence_validation_override_50k.md`
  - `results/convergence_failure_matrix_baseline_50k.json`
  - `results/convergence_validation_override_matrix.json`
- 操作：
  1. 进入 100k 的条件：
     ```text
     status in [improving, bad_plateau, converged_good]
     catastrophic_outlier == false
     failed_seed_count == 0
     best_tail_gap improved or <= 0.15
     ```
  2. 执行 100k：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase expansion_100k
     ```
  3. 若 100k 仍为 `improving` 且无异常，才进入 200k：
     ```bash
     python scripts/run_convergence_validation.py \
       --matrix configs/convergence_validation_matrix.yaml \
       --phase expansion_200k
     ```
  4. 生成扩展报告：
     ```bash
     python scripts/analyze_convergence_failures.py \
       --input results/convergence_validation_expansion_latest.json \
       --quality-report figures/convergence_quality_report.json \
       --algorithms A3C COMA GRPO IPPO IQL MADDPG MAPPO MATD3 SAC TRPO VDN \
       --output-json results/convergence_validation_expansion_matrix.json \
       --output-md docs/convergence_validation_expansion_report.md
     ```
- 验证：
  ```bash
  test -f docs/convergence_validation_expansion_gate.md
  test -f results/convergence_validation_expansion_matrix.json
  test -f docs/convergence_validation_expansion_report.md
  ```

---

## Step 10：重新生成收敛图与质量报告

- **scope: auto**
- 修改：无算法代码修改。
- 输出：
  - `figures/convergence_curves_raw_all.png`
  - `figures/convergence_curves_clean_all.png`
  - `figures/convergence_quality_report.json`
  - `figures/convergence_quality_report.md`
  - `docs/convergence_validation_final.md`
- 操作：
  1. 使用最终 accepted 的 validation result 重新绘图：
     ```bash
     python scripts/plot_results.py \
       --input results/convergence_validation_expansion_latest.json \
       --output-dir figures \
       --format png
     ```
  2. 生成 final Markdown：
     ```text
     docs/convergence_validation_final.md
     ```
     必须包含：
     - 每个目标算法最终状态
     - raw plot 是否有异常
     - clean plot 是否可用于论文
     - 未通过算法及原因
     - 是否需要 full 17
  3. 判定 full 17：
     - 只有当 11 个目标算法通过 Phase 4，且没有新增训练逻辑变更，才允许考虑 full 17。
     - 若仍有 `catastrophic_outlier/diverging`，禁止 full 17。
- 验证：
  ```bash
  test -f figures/convergence_curves_raw_all.png
  test -f figures/convergence_curves_clean_all.png
  test -f figures/convergence_quality_report.json
  test -f figures/convergence_quality_report.md
  test -f docs/convergence_validation_final.md
  grep -E "full 17|catastrophic_outlier|diverging|publication" docs/convergence_validation_final.md
  ```

---

## Step 11：更新 docs 状态并归档本轮结论

- **scope: auto**
- 修改：
  - `docs/report.md`
  - `docs/progress.md`
  - `docs/issues.md`
  - `README.md` 或 `docs/README.md` 如需补充验证入口说明
- 操作：
  1. 若 Phase 4 通过：
     ```text
     docs/report.md STATUS: READY_FOR_REVIEW
     ```
  2. 若仍有算法无法解释：
     ```text
     docs/report.md STATUS: NEEDS_REVIEW
     ```
  3. 若发现环境/reward/action/logging bug：
     ```text
     docs/report.md STATUS: NEEDS_ESCALATION
     ```
  4. `docs/progress.md` 记录模块 14 Step 1-11 状态。
  5. `docs/issues.md` 追加未通过算法、失败 seed、阻塞路径、下一步建议。
- 验证：
  ```bash
  pytest -q
  python scripts/benchmark.py --help
  python scripts/train.py --help
  python scripts/experiment_manager.py --help
  python scripts/plot_results.py --help
  python scripts/analyze_convergence_failures.py --help
  python scripts/run_convergence_validation.py --help
  ```

---

# 总体验收

## 必跑命令

```bash
pytest -q
python scripts/benchmark.py --help
python scripts/train.py --help
python scripts/experiment_manager.py --help
python scripts/plot_results.py --help
python scripts/analyze_convergence_failures.py --help
python scripts/run_convergence_validation.py --help
```

## 新增文件验收

```bash
test -f docs/convergence_validation_manifest.md
test -f docs/convergence_event_audit.md
test -f configs/convergence_validation_matrix.yaml
test -f scripts/run_convergence_validation.py
test -f tests/test_convergence_validation_runner.py
test -f docs/convergence_validation_final.md
```

## 禁止项检查

```bash
python - <<'PY'
import yaml
p = yaml.safe_load(open("configs/stability_overrides.yaml", encoding="utf-8"))
assert p["enabled"] is False
PY

git status --short experiments results figures logs checkpoints
```

`git status --short experiments results figures logs checkpoints` 若出现待提交生成产物，必须从 Git tracking 中移除，不得提交。

## full 17 决策规则

只有满足以下条件，才允许下一轮考虑 full 17：

1. 目标 11 个算法均无 `catastrophic_outlier`。
2. 没有算法仍为 `diverging`。
3. `bad_plateau` 算法均有明确论文指标接受理由，或已通过 override/100k 改善。
4. raw diagnostic plot 与 clean publication plot 没有矛盾。
5. `convergence_quality_report.json`、`convergence_failure_matrix*.json`、`docs/convergence_validation_final.md` 三者一致。
6. `pytest -q` 与所有主入口 `--help` 通过。
7. 没有新增 `NEEDS_ESCALATION`。

---

# 任务派发

## 类型:patch

`docs/inbox/` 有新版 plan.md (`slimming-plan-v2`)，先 merge-back，再执行模块 14。

本次范围：
- 保留模块 12-13 已完成状态，不回滚瘦身结果。
- 从模块 14 Step 1 开始，建立收敛验证调试闭环。
- 先做 artifact manifest 与 catastrophic outlier event audit。
- 再做 baseline 50k targeted rerun，不启用 stability overrides。
- 只有 baseline 门禁通过后，才允许单算法族 override 50k。
- 100k/200k 只对通过门禁的算法扩展，不跑 full 17。
- `configs/stability_overrides.yaml` 必须保持 `enabled: false`。
- 不提交 `experiments/`、`results/`、`figures/` 生成产物。
- 若发现环境/reward/action/logging bug，立即标记 `NEEDS_ESCALATION`。
