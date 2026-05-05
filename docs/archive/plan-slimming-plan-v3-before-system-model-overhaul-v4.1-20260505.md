# docs/inbox/plan.md：paper2 正式收敛验证协议

## 元信息

- 项目：`paper2`
- 计划版本：`slimming-plan-v3`
- 变更类型：`patch`
- 当前状态：**新增模块 14：formal convergence verification protocol，等待执行端 merge-back**
- GitHub 仓库：`w2030298-art/paper2`
- 外部关联仓库：`w2030298-art/rl-mec-dashboard`
- 创建日期：2026-05-02
- 最后更新：2026-05-03
- 本版目标：在工程协议下直接验证算法改进是否达到 `verified_converged_under_protocol`。
- 明确不做：数学/理论意义的“保证收敛”证明。

### 变更记录

| 版本 | 日期 | 变更摘要 |
|------|------|---------|
| slimming-plan-v1 | 2026-05-02 | paper2 项目瘦身 Phase 1-2：repo hygiene、生成产物 tracking 移除、旧入口/旧工具删除 |
| slimming-plan-v2 | 2026-05-03 | 追加 targeted debugging 方案；50k 只作为预验证 |
| slimming-plan-v3 | 2026-05-03 | 重构为正式工程收敛验证协议：多 seed、100k/200k 门禁、raw/clean/quality report 绑定、禁止把 50k 单 seed 写成收敛证明 |

---

## Status

> 执行端读到此区块即可恢复上下文。

- 当前阶段：模块 14 Step 1 待执行
- 当前模块：模块 14：formal convergence verification protocol
- 整体进度：`14 / 27`
- 状态：`变更后待执行`
- 阻塞项：
  - `C:\Users\22003\paper2\rl-mec-dashboard` 本机不存在，外部 dashboard 兼容性仍需独立复核。
  - 当前 50k 单 seed 测试实验只能作为 L1 预筛，不能证明收敛。
- 当前已知证据：
  - `convergence_validation_baseline_50k.json`：11 个目标算法，`seed=42`，约 50k steps，全部 `run_status=success`。
  - 该数据能证明 benchmark 管线跑通，不能证明算法改进已经保证收敛。
  - 当前仍需关注：
    - catastrophic/outlier：`IQL`、`VDN`、`IPPO`、`MADDPG`
    - 非最终通过：`A3C`、`MATD3`、`SAC`、`GRPO`
    - 仅 L1 候选：`COMA`、`MAPPO`、`TRPO`
- 重要说明：
  - 模块 12-13 保持已完成状态，不重复执行。
  - 本轮只新增模块 14。
  - 不直接 full 17 重跑。
  - 不默认启用 `configs/stability_overrides.yaml`。
  - 不允许在 `docs/` 中写“已保证收敛”，只能写协议限定结论：`verified_converged_under_protocol`。
  - 如果发现 reward/metric/env 语义错误，立即停止并标记 `NEEDS_ESCALATION`。

### Last Iteration Summary

- 已完成 paper2 仓库瘦身 Phase 1-2。
- 已生成收敛诊断报告与 50k baseline 测试数据。
- 用户确认本轮采用“工程协议下验证收敛”，不追加数学证明模块。
- 目标是让下一步工作能够直接判定：算法改进是否通过正式工程收敛门禁。

### Pending Decisions

- dashboard 兼容性是否单独复核。
- 哪些算法通过 L1 后进入 L2/L3。
- 是否接受未通过 L3 的算法仅作为 ablation/debug 结果，而非论文主图。

---

## 术语与证据等级

| 等级 | 名称 | 输入 | 可写结论 |
|------|------|------|---------|
| L0 | offline diagnosis | 历史日志/历史 benchmark JSON | 只能定位问题，不证明改进 |
| L1 | 50k single-seed test | `seed=42`、约 50k steps | 只能写 `candidate` / `failed_l1` |
| L2 | 100k multi-seed candidate validation | `seeds=[42,43,44]`、100k steps | 可写 `candidate_converged_under_protocol` |
| L3 | 200k multi-seed formal validation | `seeds=[42,43,44,45,46]`、200k steps | 可写 `verified_converged_under_protocol` |

## 工程收敛判定口径

算法只有同时满足以下条件，才可标记 `verified_converged_under_protocol`：

1. `run_status=success`，所有 seeds 无失败。
2. `catastrophic_outlier_count=0`。
3. reward:
   - tail window 稳定；
   - `best_tail_gap <= 0.10`；
   - tail relative change 不显示持续恶化。
4. latency / energy:
   - 相比 L1 baseline 无超过 10% 的反向退化；
   - deadline miss rate 不显著升高。
5. comm_score:
   - 无明显退化；
   - 若 reward 改善但 comm_score 退化，必须进入 review。
6. raw diagnostic 图、clean publication 图、quality report 三者一致。
7. `seeds=[42,43,44,45,46]` 的 median 与 q25-q75 区间稳定。
8. 结果报告中必须标注 config hash、run id、override id、数据路径。

---

# 模块 12：无争议仓库卫生清理 `[DONE]`

## Step 1：生成执行前审计快照 `[DONE]`

- **scope: auto**
- 已完成：`docs/slimming_audit_phase2.md`
- 验证：已由模块 13 Step 8 汇总进 `docs/report.md`

## Step 2：从 Git tracking 移除生成产物 `[DONE]`

- **scope: auto**
- 已完成：`experiments/`、`results/`、`figures/`、`logs/`、`checkpoints/` 从 Git tracking 移除。
- 验证：`.gitignore` 已覆盖生成产物目录。

## Step 3：删除废弃坏入口 `src/trainer/benchmark.py` `[DONE]`

- **scope: auto**
- 已完成：删除 `src/trainer/benchmark.py`
- 验证：`python scripts/benchmark.py --help`

## Step 4：统一 docs 归档并清理 graphify cache `[DONE]`

- **scope: auto**
- 已完成：`docs/.archive/` 迁移/清理，`graphify-out/cache/` 清理。
- 验证：repo hygiene 测试覆盖。

## Step 5：移除未使用 dashboard 依赖 `[DONE]`

- **scope: auto**
- 已完成：从依赖中移除未使用 dashboard 相关依赖。
- 验证：全量测试通过。

## Step 6：新增仓库卫生测试 `[DONE]`

- **scope: auto**
- 已完成：`tests/test_repo_hygiene.py`、`tests/test_active_entrypoints.py`
- 验证：`.venv\Scripts\python.exe -m pytest -q`

---

# 模块 13：已确认非核心功能删除 `[DONE / NEEDS_REVIEW ITEMS]`

## Step 1：迁移 `docs_paper/` 到外部写作资料目录 `[DONE / review]`

- **scope: review**
- 已完成：复制到 `C:\Users\22003\paper2\writing_ref\docs_paper` 并从仓库移除。
- 待审核：写作资料外迁后仓库删除是否符合预期。
- 验证：`test -f docs/references/writing_ref_migration.md`

## Step 2：删除 `scripts/evaluate.py` `[DONE]`

- **scope: auto**
- 已完成：删除旧离线评估入口。
- 保留：`BaseTrainer.evaluate()`
- 验证：`test ! -f scripts/evaluate.py`

## Step 3：删除 `scripts/generate_report.py` `[DONE]`

- **scope: auto**
- 已完成：删除旧报告入口。
- 当前报告职责：
  - `scripts/plot_results.py`
  - `scripts/analyze_convergence_failures.py`
- 验证：`test ! -f scripts/generate_report.py`

## Step 4：删除 callback 扩展机制 `[DONE / review]`

- **scope: review**
- 已完成：删除 `src/trainer/callbacks.py` 与旧 callback 调用链。
- 待审核：dashboard 路径缺失导致外部引用未能完整验证。
- 验证：`test ! -f src/trainer/callbacks.py`

## Step 5：删除旧 utils 工具 `[DONE / review]`

- **scope: review**
- 已完成：删除 `src/utils/logger.py`、`src/utils/config.py`、`src/utils/action_utils.py`
- 待审核：`omegaconf` 删除影响。
- 验证：`pytest -q`

## Step 6：buffer canonical owner 收敛 `[DONE / review]`

- **scope: review**
- 保留：`src/utils/buffer.py`
- 已完成：删除或迁移旧 wrapper 引用。
- 待审核：`rl_algorithms/utils/buffers.py` wrapper 删除影响。
- 验证：`pytest -q`

## Step 7：更新 README 与 docs 契约 `[DONE]`

- **scope: auto**
- 已完成：README/docs 更新为当前核心链路。
- 验证：`pytest tests/test_docs_contract.py -q`

## Step 8：更新执行报告与进度 `[DONE]`

- **scope: auto**
- 已完成：`docs/report.md`、`docs/progress.md`、`docs/issues.md`
- 状态：`NEEDS_REVIEW`
- 验证：`.venv\Scripts\python.exe -m pytest -q`

---

# 模块 14：formal convergence verification protocol

## 概述

- 职责：把算法改进从“测试实验可跑”提升为“工程协议下可验证收敛”。
- 输入：
  - `convergence_validation_baseline_50k.json`
  - `benchmark_20260503_005009.log`
  - `benchmark_20260503_005009.err.log`
  - `docs/convergence_failure_analysis.md`
  - `configs/stability_overrides.yaml`
- 输出：
  - `docs/formal_convergence_protocol.md`
  - `docs/l1_baseline_convergence_assessment.md`
  - `configs/formal_convergence_matrix.yaml`
  - `docs/l2_candidate_convergence_report.md`
  - `docs/l3_verified_convergence_report.md`
  - `docs/convergence_publication_gate.md`
- 预计步骤数：13

## Step 1：固化正式收敛验证协议

- **scope: auto**
- 新增：`docs/formal_convergence_protocol.md`
- 内容必须包含：
  - L0/L1/L2/L3 证据等级。
  - `verified_converged_under_protocol` 的定义。
  - 50k 单 seed 不能证明收敛的边界。
  - reward、latency、energy、comm_score、deadline miss rate 的通过阈值。
  - raw/clean/quality report 三件套绑定要求。
  - 禁止语句：`guaranteed convergence`、`算法已保证收敛`、`verified_converged` 出现在 L1 报告中。
- 验证：
  ```bash
  test -f docs/formal_convergence_protocol.md
  grep -n "verified_converged_under_protocol" docs/formal_convergence_protocol.md
  grep -n "L1" docs/formal_convergence_protocol.md
  ```

## Step 2：把 50k baseline 转换为 L1 评估报告

- **scope: auto**
- 输入：
  - `convergence_validation_baseline_50k.json`
  - `benchmark_20260503_005009.log`
  - `benchmark_20260503_005009.err.log`
- 新增：`docs/l1_baseline_convergence_assessment.md`
- 新增：`results/l1_baseline_convergence_assessment.json`（生成产物，不纳入 Git）
- 操作：
  - 逐算法计算：
    - `tail_relative_change`
    - `tail_slope`
    - `tail_volatility`
    - `best_tail_gap`
    - `catastrophic_outlier_count`
    - `latency_regression`
    - `energy_regression`
    - `comm_score_regression`
  - 标记：
    - `failed_l1`
    - `l1_candidate`
    - `needs_event_audit`
    - `needs_single_variable_fix`
- 不允许：
  - 把任何算法标记为 `verified_converged_under_protocol`。
- 验证：
  ```bash
  test -f docs/l1_baseline_convergence_assessment.md
  python -m json.tool results/l1_baseline_convergence_assessment.json > NUL
  grep -R "verified_converged" docs/l1_baseline_convergence_assessment.md && exit 1 || true
  ```

## Step 3：补强收敛判定器

- **scope: review**
- 修改：`scripts/analyze_convergence_failures.py`
- 新增函数：
  - `compute_tail_metrics(series, higher_is_better: bool) -> dict`
  - `classify_evidence_level(run_metadata: dict) -> str`
  - `classify_formal_convergence(record: dict, protocol: dict) -> dict`
  - `detect_metric_regression(current: dict, baseline: dict) -> dict`
- 新增 CLI 参数：
  ```bash
  --protocol docs/formal_convergence_protocol.md
  --evidence-level L1|L2|L3
  --baseline-json results/l1_baseline_convergence_assessment.json
  --formal-output results/formal_convergence_decision.json
  ```
- 新增测试：`tests/test_formal_convergence_protocol.py`
- 验证：
  ```bash
  pytest tests/test_analyze_convergence_failures.py tests/test_formal_convergence_protocol.py -q
  python scripts/analyze_convergence_failures.py --help
  ```

## Step 4：建立正式实验矩阵配置

- **scope: auto**
- 新增：`configs/formal_convergence_matrix.yaml`
- 内容：
  ```yaml
  evidence_levels:
    L1:
      steps: 50000
      seeds: [42]
      purpose: screening_only
    L2:
      steps: 100000
      seeds: [42, 43, 44]
      purpose: candidate_validation
    L3:
      steps: 200000
      seeds: [42, 43, 44, 45, 46]
      purpose: formal_verification
  target_algorithms:
    catastrophic_or_unstable: [IQL, VDN, IPPO, MADDPG]
    uncertain_or_plateau: [A3C, MATD3, SAC, GRPO]
    l1_candidates: [COMA, MAPPO, TRPO]
  gates:
    reward_best_tail_gap_max: 0.10
    metric_regression_max: 0.10
    catastrophic_outlier_count: 0
    failed_seed_count: 0
  ```
- 验证：
  ```bash
  python - <<'PY'
  import yaml
  p = yaml.safe_load(open("configs/formal_convergence_matrix.yaml", encoding="utf-8"))
  assert p["evidence_levels"]["L3"]["seeds"] == [42, 43, 44, 45, 46]
  assert p["gates"]["catastrophic_outlier_count"] == 0
  PY
  ```

## Step 5：异常事件审计 gate

- **scope: review**
- 目标：`IQL`、`VDN`、`IPPO`、`MADDPG`
- 修改：`scripts/analyze_convergence_failures.py`
- 新增能力：
  - 定位 min reward 对应 eval index / timestep。
  - 读取对应算法 artifact：
    - `result.json`
    - `stdout.log`
    - `stderr.log`
    - `checkpoints/train_logs.json`
  - 判断来源：
    - `environment_metric_bug`
    - `training_instability`
    - `evaluation_noise`
    - `unknown`
- 新增：`docs/convergence_event_audit.md`
- 新增：`results/convergence_event_audit.json`
- 停止条件：
  - 若来源是 `environment_metric_bug` 或 `unknown`，停止 L2/L3，写 `NEEDS_ESCALATION`。
  - 只有来源是 `training_instability` 或 `evaluation_noise`，才允许进入 Step 6/7。
- 验证：
  ```bash
  python scripts/analyze_convergence_failures.py --algorithms IQL VDN IPPO MADDPG --formal-output results/convergence_event_audit.json
  test -f docs/convergence_event_audit.md
  ```

## Step 6：单变量修复矩阵，不污染默认配置

- **scope: review**
- 新增：`configs/formal_single_variable_fixes.yaml`
- 规则：
  - `configs/stability_overrides.yaml` 必须保持 `enabled: false`。
  - 每次只启用一个变量族。
  - 每次修复必须生成 `override_id`。
  - 不允许一次性叠加多个 override 后宣称有效。
- 推荐矩阵：
  ```yaml
  value_decomposition:
    algorithms: [IQL, VDN]
    fixes:
      - {override_id: vd_lr_half, lr_scale: 0.5}
      - {override_id: vd_grad_clip_1_0, gradient_clip_norm: 1.0}
      - {override_id: vd_target_update_slow, target_update_interval_scale: 1.5}
  on_policy_marl:
    algorithms: [IPPO, COMA, MAPPO]
    fixes:
      - {override_id: marl_lr_half, actor_lr_scale: 0.5, critic_lr_scale: 0.5}
      - {override_id: marl_grad_clip_0_5, gradient_clip_norm: 0.5}
      - {override_id: marl_eval_episodes_20, eval_episodes: 20}
  continuous_actor_critic:
    algorithms: [MADDPG, MATD3, SAC]
    fixes:
      - {override_id: cac_lr_half, actor_lr_scale: 0.5, critic_lr_scale: 0.5}
      - {override_id: cac_target_tau_half, target_tau_scale: 0.5}
      - {override_id: cac_replay_warmup_10k, replay_warmup_steps: 10000}
  conservative_policy:
    algorithms: [GRPO, TRPO]
    fixes:
      - {override_id: cp_lr_0_75, policy_lr_scale: 0.75, value_lr_scale: 0.75}
  ```
- 验证：
  ```bash
  test -f configs/formal_single_variable_fixes.yaml
  grep -R "enabled: true" configs/stability_overrides.yaml && exit 1 || true
  ```

## Step 7：执行 L2 candidate validation

- **scope: review**
- 前置：
  - Step 5 无 `environment_metric_bug` 或 `unknown`。
  - Step 6 已生成单变量候选配置。
- 目标：
  - L1 候选：`COMA`、`MAPPO`、`TRPO`
  - 修复候选：按 Step 6 的最小有效 override 进入
  - 不稳定算法：`IQL`、`VDN`、`IPPO`、`MADDPG` 必须先通过事件审计
- 实验参数：
  ```yaml
  evidence_level: L2
  steps: 100000
  seeds: [42, 43, 44]
  ```
- 输出：
  - `docs/l2_candidate_convergence_report.md`
  - `results/l2_candidate_convergence_report.json`
  - `figures/l2_candidate_convergence/`
- 通过标准：
  - all seeds success
  - no catastrophic outlier
  - reward best-tail gap <= 0.10
  - latency/energy/comm_score regression <= 0.10
  - quality report 无 severe unexplained warning
- 验证：
  ```bash
  test -f docs/l2_candidate_convergence_report.md
  python -m json.tool results/l2_candidate_convergence_report.json > NUL
  ```

## Step 8：L2 失败分流

- **scope: auto**
- 新增：`docs/l2_failure_triage.md`
- 操作：
  - 对 L2 失败算法按原因分流：
    - `event_bug`
    - `training_instability`
    - `metric_tradeoff`
    - `insufficient_steps`
    - `seed_sensitive`
  - 写入每个算法下一步：
    - retry same override
    - try next single-variable override
    - escalate env/metric
    - exclude from formal convergence claim
- 验证：
  ```bash
  test -f docs/l2_failure_triage.md
  grep -n "exclude from formal convergence claim" docs/l2_failure_triage.md
  ```

## Step 9：执行 L3 formal verification

- **scope: review**
- 前置：
  - 仅 L2 通过的算法可进入。
  - 每个算法只有一个最终配置进入 L3。
- 实验参数：
  ```yaml
  evidence_level: L3
  steps: 200000
  seeds: [42, 43, 44, 45, 46]
  ```
- 输出：
  - `docs/l3_verified_convergence_report.md`
  - `results/l3_verified_convergence_report.json`
  - `figures/l3_verified_convergence/`
- 通过标准：
  - 所有 L3 seeds 成功。
  - `catastrophic_outlier_count=0`。
  - reward、latency、energy、comm_score 全部通过协议阈值。
  - q25-q75 区间稳定，无单 seed 主导结论。
  - raw diagnostic 图和 clean publication 图结论一致。
- 验证：
  ```bash
  test -f docs/l3_verified_convergence_report.md
  python -m json.tool results/l3_verified_convergence_report.json > NUL
  grep -n "verified_converged_under_protocol" docs/l3_verified_convergence_report.md
  ```

## Step 10：生成 publication gate

- **scope: auto**
- 新增：`docs/convergence_publication_gate.md`
- 内容：
  - 哪些算法可进入论文主图。
  - 哪些只能进入 appendix/debug。
  - 哪些必须排除 formal convergence claim。
  - 每个算法的 evidence level。
- 规则：
  - 只有 L3 通过算法可进入 main convergence figure。
  - L2 通过但 L3 未跑算法只能写 candidate，不进主结论。
  - L1 通过算法不能写入论文“收敛性验证”段落。
- 验证：
  ```bash
  test -f docs/convergence_publication_gate.md
  grep -R "L1" docs/convergence_publication_gate.md
  ```

## Step 11：更新绘图和质量报告绑定

- **scope: review**
- 修改：`scripts/plot_results.py`
- 要求：
  - 输出图文件名包含 evidence level：
    - `l2_convergence_curves_raw_all.png`
    - `l2_convergence_curves_clean_all.png`
    - `l3_convergence_curves_raw_all.png`
    - `l3_convergence_curves_clean_all.png`
  - quality report 中写入：
    - `evidence_level`
    - `run_id`
    - `seed_set`
    - `config_hash`
    - `override_id`
  - clean 图不得删除 raw 异常证据。
- 验证：
  ```bash
  pytest tests/test_convergence_plot.py -q
  python scripts/plot_results.py --help
  ```

## Step 12：更新 docs 状态

- **scope: auto**
- 修改：
  - `docs/report.md`
  - `docs/progress.md`
  - `docs/issues.md`
  - `docs/convergence_failure_analysis.md`
- `docs/report.md` 状态规则：
  - L2 未完成：`IN_PROGRESS`
  - 事件审计发现 env/metric bug：`NEEDS_ESCALATION`
  - L2 完成但 L3 未完成：`NEEDS_REVIEW`
  - L3 完成且 dashboard 未验证：`NEEDS_REVIEW`
  - L3 完成且 dashboard 验证完成：`READY_FOR_REVIEW`
- 验证：
  ```bash
  pytest tests/test_docs_contract.py -q
  grep -n "evidence level" docs/report.md
  ```

## Step 13：最终验收与防误报检查

- **scope: auto**
- 操作：
  - 全量运行协议相关测试。
  - 检查 docs 中是否有越级结论。
  - 检查生成产物是否仍未纳入 Git tracking。
- 必跑命令：
  ```bash
  pytest tests/test_convergence_plot.py tests/test_analyze_convergence_failures.py tests/test_formal_convergence_protocol.py tests/test_docs_contract.py -q
  python scripts/analyze_convergence_failures.py --help
  python scripts/plot_results.py --help
  ```
- 禁止项检查：
  ```bash
  grep -R "guaranteed convergence\|保证收敛" docs/ && exit 1 || true
  grep -R "verified_converged_under_protocol" docs/l1_* docs/l2_* && exit 1 || true
  git status --short experiments results figures | grep -E "^[AM]" && exit 1 || true
  grep -R "enabled: true" configs/stability_overrides.yaml && exit 1 || true
  ```
- 完成条件：
  - L1/L2/L3 证据等级清晰。
  - 所有算法有明确状态：
    - `failed_l1`
    - `needs_event_audit`
    - `candidate_converged_under_protocol`
    - `verified_converged_under_protocol`
    - `excluded_from_formal_claim`
  - 没有把 50k 单 seed 写成正式收敛证明。
  - L3 通过算法拥有 raw/clean/quality report 三件套。
  - 未通过 L3 的算法不得进入论文主结论。

---

# 模块 14 总体验收

## 必跑命令

```bash
pytest tests/test_convergence_plot.py tests/test_analyze_convergence_failures.py tests/test_formal_convergence_protocol.py tests/test_docs_contract.py -q
python scripts/analyze_convergence_failures.py --help
python scripts/plot_results.py --help
```

## L2 实验命令模板

执行端需按当前 CLI 实际参数适配；若 `scripts/benchmark.py` 不支持，先补 wrapper，不改训练语义。

```bash
python scripts/benchmark.py --algorithms COMA MAPPO TRPO --total-steps 100000 --seeds 42 43 44 --output-dir experiments/l2_candidate_convergence
python scripts/benchmark.py --algorithms IQL VDN IPPO MADDPG --total-steps 100000 --seeds 42 43 44 --output-dir experiments/l2_event_checked_convergence
python scripts/benchmark.py --algorithms A3C MATD3 SAC GRPO --total-steps 100000 --seeds 42 43 44 --output-dir experiments/l2_uncertain_convergence
```

## L3 实验命令模板

```bash
python scripts/benchmark.py --algorithms <L2_PASSED_ALGOS> --total-steps 200000 --seeds 42 43 44 45 46 --output-dir experiments/l3_verified_convergence
```

## 必须产物

```text
docs/formal_convergence_protocol.md
docs/l1_baseline_convergence_assessment.md
docs/convergence_event_audit.md
docs/l2_candidate_convergence_report.md
docs/l2_failure_triage.md
docs/l3_verified_convergence_report.md
docs/convergence_publication_gate.md
configs/formal_convergence_matrix.yaml
configs/formal_single_variable_fixes.yaml
results/l1_baseline_convergence_assessment.json
results/l2_candidate_convergence_report.json
results/l3_verified_convergence_report.json
figures/l2_candidate_convergence/
figures/l3_verified_convergence/
```

## 成功判定

- 能直接回答每个算法是否通过 `verified_converged_under_protocol`。
- 不能回答“数学意义保证收敛”，除非后续新增理论证明模块。
- 任何算法未通过 L3，不得写入论文主图或主结论。
- L3 通过算法必须具备 raw 图、clean 图、quality report、config hash、run id、seed set。
