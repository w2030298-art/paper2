# docs/inbox/plan.md：paper2 对比算法实验边界修正计划 v4.2

## 元信息

- 项目：`paper2`
- 计划版本：`system-model-overhaul-v4.2`
- 变更类型：`patch`
- 创建日期：2026-05-06
- 最后更新：2026-05-06
- 上一版本：`system-model-overhaul-v4.1`
- GitHub 仓库：`w2030298-art/paper2`
- 项目边界：`paper2` 是对比算法实验 / 仿真实验项目，不承载论文正文改写、论文结构重写、投稿模板适配、论文主结论写作或 `writing_ref/` 写作资产维护。
- 本版目标：修正 v4.1 中“论文写作资产属于 paper2 执行链”的边界错误；保留 Mainline-A 对比算法实验链，剥离论文相关模块与状态表述。
- 保留范围：
  - MEC 系统模型实验基座。
  - 动态 Stackelberg 定价实验机制。
  - game-aware critic / primal-dual MARL 对比算法。
  - N0 smoke、N1 oracle、N2 deterministic controlled probe、N3 OOD formal execution 实验链。
  - 对比算法实验 runner、benchmark 配置、oracle/OOD/ablation 结果校验、plot/report helper。
- 明确不做：
  - 不维护论文正文。
  - 不生成论文 revision manifest。
  - 不维护 `writing_ref/`。
  - 不把实验结果直接表述为论文主结论。
  - 不把 `results/`、`experiments/`、`figures/`、`logs/`、`checkpoints/` 重新纳入 Git tracking。
  - 不恢复已移除的 `docs_paper/`、`scripts/evaluate.py`、`scripts/generate_report.py`。
  - 不声明数学意义的全局收敛保证。
- 实验解释边界：
  - N0 只能作为 smoke evidence。
  - N1 只能作为 small-scale oracle evidence。
  - N2 只能作为 deterministic controlled probe，不是 training-grade / publication-grade ablation。
  - N3 只能作为 OOD formal execution evidence。
  - 所有实验结论只在本项目中作为“对比算法实验结果 / benchmark evidence”使用。

### 变更记录

| 版本 | 日期 | 变更摘要 |
|------|------|---------|
| slimming-plan-v1 | 2026-05-02 | repo hygiene、生成产物 tracking 移除、旧入口/旧工具删除 |
| slimming-plan-v2 | 2026-05-03 | 追加 targeted debugging 方案；50k 只作为预验证 |
| slimming-plan-v3 | 2026-05-03 | 正式工程收敛验证协议：L0/L1/L2/L3、raw/clean/quality report、禁止越级收敛结论 |
| system-model-overhaul-v4 | 2026-05-04 | 初版系统模型大换血计划；误拆 paper2 与仿真实验为两个项目 |
| system-model-overhaul-v4.1 | 2026-05-04 | 修正 paper2 与仿真实验合并；但仍错误保留论文写作资产模块 |
| system-model-overhaul-v4.2 | 2026-05-06 | 修正项目边界：paper2 只作为对比算法实验项目；删除/归档论文写作模块与相关状态 |

---

## Status

> 执行端读到此区块即可恢复上下文。

- 当前阶段：模块 22 project-boundary-cleanup completed；模块 23 Mainline-A experiment final review pending。
- 当前模块：模块 23 Mainline-A experiment final review。
- 整体进度：模块 14R-20 已有实现资产并进入 `DONE_PENDING_FINAL_REVIEW`；模块 21 已判定为 `REMOVED_OUT_OF_SCOPE`；模块 22 completed。
- 状态：`NEEDS_REVIEW`
- 当前背景：
  - `paper2` 项目边界已修正为对比算法实验 / 仿真实验项目。
  - `writing_ref/paper2_mainline_a_revision/` 已从 Git tracking 删除。
  - 论文相关更改（论文正文改写、论文主结论写作、写作资产维护）不再属于本项目。
  - Mainline-A N0/N1/N2/N3 artifact-level gate review 已完成，但 final review scope 仍未关闭。
- 阻塞项：
  - dashboard 兼容性仍是外部复核项，不阻塞 final review。
- 本版总原则：
  - 本项目只负责对比算法实验资产。
  - 不改 Mainline-A 核心代码。
  - 不运行新的正式训练。

### Last Iteration Summary

- 模块 12-13 已完成仓库瘦身与非核心功能删除。
- 模块 14 formal convergence 已降级为 pre-overhaul legacy baseline。
- 模块 15-19 已实现 Mainline-A 实验所需的系统模型、动态定价、game-aware MARL、理论/数值验证 helper。
- 模块 20 已完成 N0/N1/N2/N3 对比实验链与 artifact-level gate review。
- 模块 21 原计划为论文写作资产，现根据用户确认移出 paper2 项目范围。
- 模块 22 已完成 project-boundary-cleanup：删除 `writing_ref/paper2_mainline_a_revision/` 4 个 tracked 文件，同步 progress/report/issues 到 v4.2 边界。

### Pending Decisions

- Mainline-A final review 是否关闭。
- 是否追加更大规模正式 benchmark。
- dashboard 兼容性是否在外部 dashboard 仓库环境复核。
- 是否需要另建独立论文项目；该决策不属于 paper2 执行范围。

---

# 历史模块冻结区

## 模块 1-13：历史已完成模块 `[DONE]`

- **scope: auto**
- 状态：保持完成，不重复执行，不回滚瘦身决策。
- 约束：
  - 不恢复旧入口。
  - 不恢复 `docs_paper/` 入仓。
  - 不恢复 callback 扩展机制。
  - 不把生成产物纳入 Git tracking。
- 验证：
  ```bash
  test ! -f scripts/evaluate.py
  test ! -f scripts/generate_report.py
  test ! -d docs_paper
  git status --short experiments results figures logs checkpoints | grep -E "^[AM]" && exit 1 || true
  pytest -q
  ```

## 模块 14：formal convergence verification protocol `[SUPERSEDED_FOR_MAINLINE_A_EXPERIMENT]`

- **scope: review**
- 状态：不删除历史产物，但不再作为 Mainline-A 新系统模型主门禁。
- 处理方式：
  - 归档为 legacy baseline。
  - 禁止把旧 L2/L3 结果写成 Mainline-A 对比算法正式实验结论。
  - 保留旧协议文档用于说明 pre-overhaul convergence baseline。
- 验证：
  ```bash
  grep -R "legacy_pre_overhaul" docs/ || true
  grep -R "verified_converged_under_protocol" docs/ | grep -v "legacy" | grep -v "protocol" && exit 1 || true
  ```

---

# 模块 14R：legacy convergence retirement `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：停止旧 L2/L3 对后续开发的阻塞作用，把旧收敛验证降级为 legacy baseline，并建立 Mainline-A 实验门禁。
- 状态：已实现，保持 review pending。
- 边界变更：所有“新论文主结论”表述改为“Mainline-A 对比算法实验正式结论”。
- 验证：
  ```bash
  test -f docs/legacy_convergence_retirement.md
  grep -n "legacy baseline" docs/legacy_convergence_retirement.md
  grep -n "must not be used as main claim" docs/legacy_convergence_retirement.md || true
  ```

---

# 模块 15：MEC 系统模型模块化基座 `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：提供对比算法实验所需的 MEC 系统模型层，覆盖任务模型、队列模型、异构协作边缘、能耗模型、移动性建模、通信双轨模型。
- 状态：已实现，保持 review pending。
- 关键产物：
  - `src/mec_model/`
  - `configs/system_model_mainline_a.yaml`
  - `tests/test_mec_model_*.py`
  - `docs/references/ref-mainline-a-system-model.md`
- 验证：
  ```bash
  pytest tests/test_mec_model_tasks.py tests/test_mec_model_queues.py tests/test_mec_model_edge_topology.py tests/test_mec_model_energy.py tests/test_mec_model_mobility.py tests/test_mec_model_channel.py -q
  ```

---

# 模块 16：系统模型与现有环境兼容接入 `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：把系统模型接入现有 env、reward、benchmark 与配置系统，同时保持 legacy 默认可运行。
- 状态：已实现，保持 review pending。
- 关键产物：
  - `src/mec_model/adapters.py`
  - `game_theory_env.py` Mainline-A optional path
  - `scripts/benchmark.py` Mainline-A 参数
  - `configs/benchmark_mainline_a_smoke.yaml`
  - `docs/mainline_a_compatibility_report.md`
- 验证：
  ```bash
  python scripts/benchmark.py --help
  python scripts/benchmark.py --enable-mainline-a --dry-run
  pytest tests/test_env_legacy_compat.py tests/test_env_mainline_a_state.py tests/test_reward_components_mainline_a.py -q
  ```

---

# 模块 17：状态依赖 Stackelberg 动态定价 `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：把静态价格升级为队列/信道/迁移状态依赖的动态定价，并输出可测试性质。
- 状态：已实现，保持 review pending。
- 关键产物：
  - `src/game_pricing/`
  - `configs/pricing_dynamic_mainline_a.yaml`
  - `tests/test_dynamic_pricing.py`
  - `tests/test_follower_response.py`
  - `tests/test_leader_objective.py`
  - `docs/theory/dynamic_stackelberg_pricing.md`
- 验证：
  ```bash
  pytest tests/test_dynamic_pricing.py tests/test_follower_response.py tests/test_leader_objective.py -q
  ```

---

# 模块 18：game-aware critic 与 primal-dual 多智能体更新 `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：实现对比算法实验中的 game-aware critic features、constraint-aware reward、primal-dual updater、trainer hooks。
- 状态：已实现，保持 review pending。
- 关键产物：
  - `src/rl_algorithms/game_aware/`
  - game-aware trainer hook
  - primal-dual updater
  - Mainline-A algorithm config
  - 行为级测试
- 验证：
  ```bash
  pytest tests/test_game_aware_critic.py tests/test_primal_dual_updater.py tests/test_game_aware_trainer_hook.py -q
  ```

---

# 模块 19：理论性质与数值验证 helper `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：为对比算法实验提供可复核的理论性质检查、pricing validation helper、constraint validation helper。
- 状态：已实现，保持 review pending。
- 边界变更：
  - 本模块只服务实验解释与 sanity check。
  - 不输出论文正文，不生成投稿文本。
- 验证：
  ```bash
  pytest tests/test_theory_validation_helpers.py tests/test_pricing_theory_checks.py -q
  ```

---

# 模块 20：paper2 内置 Mainline-A 对比算法实验矩阵 `[DONE_PENDING_FINAL_REVIEW]`

## 概述

- 职责：提供 paper2 内置对比算法实验链，覆盖 N0/N1/N2/N3。
- 状态：已实现，artifact-level gate review 已完成，等待 final review。
- 关键产物：
  - `configs/experiments/mainline_a_n0/*.yaml`
  - `configs/experiments/mainline_a_n1/*.yaml`
  - `configs/experiments/mainline_a_n2/*.yaml`
  - `configs/experiments/mainline_a_n3/*.yaml`
  - experiment runner
  - oracle/OOD/ablation validators
  - `docs/mainline_a_experiment_gate_review.md`
- 证据边界：
  - N0：smoke evidence only。
  - N1：small-scale oracle evidence。
  - N2：deterministic controlled probe only。
  - N3：OOD formal execution evidence。
- 验证：
  ```bash
  python scripts/benchmark.py --config configs/benchmark_mainline_a_smoke.yaml --dry-run
  pytest tests/test_mainline_a_experiment_runner.py tests/test_mainline_a_n2_ablation.py tests/test_mainline_a_n3_ood.py -q
  test -f docs/mainline_a_experiment_gate_review.md
  ```

---

# 模块 21：论文写作资产与差异化改写接口 `[REMOVED_OUT_OF_SCOPE]`

## 概述

- 职责：无；该模块不属于 paper2。
- 变更说明：
  - v4.1 中把论文写作资产放入 paper2 执行链是边界错误。
  - 本模块从 v4.2 起移出项目范围。
  - 不再维护 `writing_ref/paper2_mainline_a_revision/`。
  - 不再维护 `docs/paper_revision_manifest.md`。
  - 不再在 `docs/report.md` / `docs/progress.md` 中把论文正文或论文主结论作为 paper2 交付项。
- 执行规则：
  - 若上述路径在 Git tracking 中存在，删除。
  - 若只存在于本地 ignored / untracked，不纳入 Git，不迁移进 paper2。
- 验证：
  ```bash
  git ls-files | grep -E '(^writing_ref/|paper_revision_manifest|paper2_mainline_a_revision)' && exit 1 || true
  grep -R "论文正文\|论文改写\|paper_revision_manifest\|writing_ref/paper2_mainline_a_revision" docs/ --exclude-dir=archive --exclude=plan.md && exit 1 || true
  ```

---

# 模块 22：project-boundary-cleanup

## 概述

- 职责：把 paper2 从“实验 + 论文写作资产”修正为“对比算法实验项目”，清理 v4.1 文档与状态中的论文边界错误。
- 前置依赖：用户确认项目边界。
- 输出：
  - 更新后的 `docs/plan.md`
  - 更新后的 `docs/progress.md`
  - 更新后的 `docs/report.md`
  - 更新后的 `docs/issues.md`
  - 必要时删除 tracked `writing_ref/` / `docs/paper_revision_manifest.md`
- 预计步骤数：6

## Step 1：审计论文相关入仓痕迹

- **scope: review**
- 操作：
  - 搜索 Git tracking 中所有论文写作、revision、manifest、`writing_ref` 相关路径。
  - 区分三类：
    1. 应删除的论文写作资产。
    2. 可保留但需改写措辞的实验解释文档。
    3. archive 中历史记录，可保留。
- 命令：
  ```bash
  git ls-files | grep -E '(^writing_ref/|paper_revision_manifest|paper2_mainline_a_revision|docs_paper)' || true
  grep -RIn "论文正文\|论文改写\|论文主结论\|paper_revision_manifest\|writing_ref\|manuscript\|revision manifest" docs/ src/ scripts/ configs/ --exclude-dir=archive || true
  ```
- 验证：
  ```bash
  test -f docs/plan.md
  test -f docs/progress.md
  test -f docs/report.md
  ```

## Step 2：更新计划元信息与 Status

- **scope: auto**
- 修改：`docs/plan.md`
- 要求：
  - 版本改为 `system-model-overhaul-v4.2`。
  - 项目边界改为“对比算法实验 / 仿真实验项目”。
  - 删除“论文改写不是本 plan 的执行主线”等模糊表述，改为“论文相关更改不属于本项目”。
  - 模块 21 标记为 `[REMOVED_OUT_OF_SCOPE]`。
  - 新增模块 22。
  - Status 当前阶段指向模块 22。
- 验证：
  ```bash
  grep -n "system-model-overhaul-v4.2" docs/plan.md
  grep -n "对比算法实验" docs/plan.md
  grep -n "REMOVED_OUT_OF_SCOPE" docs/plan.md
  grep -n "project-boundary-cleanup" docs/plan.md
  ```

## Step 3：删除 tracked 论文写作资产

- **scope: auto**
- 操作：
  - 若存在 tracked `writing_ref/paper2_mainline_a_revision/`，删除。
  - 若存在 tracked `docs/paper_revision_manifest.md`，删除。
  - 若存在 tracked `docs_paper/`，删除。
  - 不处理本地 untracked / ignored 写作文件。
- 命令：
  ```bash
  git ls-files | grep -E '^writing_ref/paper2_mainline_a_revision/' | xargs -r git rm
  git ls-files | grep -E '^docs/paper_revision_manifest\.md$' | xargs -r git rm
  git ls-files | grep -E '^docs_paper/' | xargs -r git rm
  ```
- 验证：
  ```bash
  git ls-files | grep -E '(^writing_ref/paper2_mainline_a_revision/|^docs/paper_revision_manifest\.md$|^docs_paper/)' && exit 1 || true
  ```

## Step 4：同步 progress/report/issues 状态

- **scope: auto**
- 修改：
  - `docs/progress.md`
  - `docs/report.md`
  - `docs/issues.md`
- 要求：
  - `progress.md` 中模块 21 改为 `REMOVED_OUT_OF_SCOPE`，不得显示为已完成交付。
  - `report.md` 状态保持 `NEEDS_REVIEW` 或 `CHANGE_PENDING_EXECUTION`，但说明 review scope 只针对实验链。
  - `issues.md` 追加 `[Fixed] Boundary-v4.2-paper-scope`：论文相关更改移出 paper2。
  - 删除“论文主结论/论文正文改写/论文源文件”作为当前阻塞项的表述。
- 验证：
  ```bash
  grep -n "REMOVED_OUT_OF_SCOPE" docs/progress.md
  grep -n "Boundary-v4.2-paper-scope" docs/issues.md
  grep -n "对比算法实验" docs/report.md
  grep -RIn "论文源文件\|论文正文改写\|论文主结论" docs/report.md docs/progress.md docs/issues.md && exit 1 || true
  ```

## Step 5：改写实验 gate 文档中的论文措辞

- **scope: review**
- 修改：
  - `docs/mainline_a_experiment_gate_review.md`
  - `docs/convergence_publication_gate.md` 如仍存在该文件且包含论文口径
  - `docs/formal_convergence_protocol.md` 如仍存在论文口径
  - `docs/references/ref-mainline-a-overhaul-v4_1.md` 如仍包含论文写作资产口径
- 要求：
  - 把“不得写作论文主结论”改成“不得升级为正式 benchmark 结论 / 不得过度解释实验结果”。
  - 保留 evidence-level 边界。
  - 不把 gate 文档改成论文写作指南。
- 验证：
  ```bash
  grep -n "deterministic controlled probe" docs/mainline_a_experiment_gate_review.md
  grep -n "benchmark" docs/mainline_a_experiment_gate_review.md
  grep -RIn "论文主结论\|论文正文\|论文改写" docs/mainline_a_experiment_gate_review.md docs/convergence_publication_gate.md docs/formal_convergence_protocol.md docs/references/ref-mainline-a-overhaul-v4_1.md 2>/dev/null && exit 1 || true
  ```

## Step 6：最终边界验证

- **scope: auto**
- 操作：
  - 确认 tracked 文件中没有论文写作资产。
  - 确认 docs 当前状态只指向实验链 final review。
  - 确认 generated artifacts 仍 ignored。
  - 跑关键测试，不启动正式训练。
- 验证：
  ```bash
  git ls-files | grep -E '(^writing_ref/|paper_revision_manifest|docs_paper)' && exit 1 || true
  grep -RIn "论文正文\|论文改写\|论文主结论\|paper_revision_manifest\|writing_ref/paper2_mainline_a_revision" docs/ --exclude-dir=archive --exclude=plan.md && exit 1 || true
  git status --short experiments results figures logs checkpoints | grep -E "^[AM]" && exit 1 || true
  pytest -q
  ```

---

# 后续模块：Mainline-A final review

## 模块 23：Mainline-A experiment final review `[PENDING_AFTER_MODULE_22]`

## 概述

- 职责：在项目边界清理后，只按对比算法实验项目标准关闭 final review。
- 前置依赖：模块 22 完成。
- 输出：
  - final review decision
  - updated `docs/report.md`
  - updated `docs/progress.md`
- 预计步骤数：3

## Step 1：复核实验链 gate 文档

- **scope: review**
- 操作：
  - 复核 `docs/mainline_a_experiment_gate_review.md`。
  - 确认 N0/N1/N2/N3 证据等级没有被过度解释。
  - 确认 N2 仍为 deterministic controlled probe only。
- 验证：
  ```bash
  grep -n "N0" docs/mainline_a_experiment_gate_review.md
  grep -n "N3" docs/mainline_a_experiment_gate_review.md
  grep -n "deterministic controlled probe" docs/mainline_a_experiment_gate_review.md
  ```

## Step 2：决定是否关闭 final review

- **scope: escalate**
- 操作：
  - 由用户/Web 决定是否把 Mainline-A N0/N1/N2/N3 当前 artifact-level evidence 标记为 final-review accepted。
  - 如关闭，则更新 `docs/report.md` 状态为 `REVIEW_CLOSED`。
  - 如不关闭，列出阻塞项与下一轮实验要求。
- 验证：
  ```bash
  grep -n "REVIEW_CLOSED\|NEEDS_REVIEW" docs/report.md
  ```

## Step 3：若关闭，派发后续实验扩展计划

- **scope: review**
- 操作：
  - 只在 Step 2 明确关闭后执行。
  - 若用户要求更大规模 benchmark，新增后续实验模块。
  - 若不追加实验，保持项目进入 stable review-complete 状态。
- 验证：
  ```bash
  grep -n "Mainline-A" docs/progress.md
  ```

---

# 执行纪律

- 本 patch 不启动 N0/N1/N2/N3 正式训练。
- 本 patch 不改论文、不生成论文写作文件、不维护 `writing_ref/`。
- 本 patch 只清理项目边界和当前 docs/status。
- `results/` 仍 ignored，不能纳入 Git tracking。
- 若发现 tracked 论文写作资产，删除；若只是 archive 历史记录，不要求改写历史。
