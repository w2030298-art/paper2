# 开发进度

## 当前状态

- 当前计划版本: `system-model-overhaul-v4.1`
- 最后更新: 2026-05-06
- 状态: NEEDS_REVIEW
- 当前阶段: Mainline-A experiment artifact-level gate review completed; N0/N1/N2/N3 均为 DONE_PENDING_FINAL_REVIEW，review scope 仍等待用户/Web final review。
- 执行边界: 本轮只做 GitHub tracked docs 修复；未改本地工作区，未把 `results/` 纳入 Git tracking；N2 继续只作为 deterministic controlled probe，不升级为 training-grade / publication-grade ablation evidence。

## 模块进度

### 模块 1-13: 历史已完成模块

- [x] 保持完成状态；未恢复旧入口、`docs_paper/` 或 generated artifact tracking。✅ 2026-05-05

### 模块 14: formal convergence verification protocol

- [x] 已降级为 pre-overhaul legacy baseline。✅ 2026-05-05
- [x] C-2 fixed: formal convergence 单测改为 fixtures / 显式参数，不依赖 ignored `results/` 产物。✅ 2026-05-05

### 模块 14R: legacy convergence retirement

- [x] Step 1-5: legacy convergence retirement docs/status 已完成。✅ 2026-05-05 [DONE_PENDING_REVIEW]

### 模块 15: MEC 系统模型模块化基底

- [x] Step 1-10: `src/mec_model/`、配置、测试和参考资产已实现。✅ 2026-05-05 [DONE_PENDING_REVIEW]

### 模块 16: 系统模型与现有环境兼容接入

- [x] Step 1-8: env adapter、Mainline-A state/reward/benchmark dry-run 接入已实现。✅ 2026-05-05 [DONE_PENDING_REVIEW]
- [x] H-3/H-4 fixed: adapter decision 可写入 env 价格路径，reward components 使用 step 行为指标，行为级测试已补充。✅ 2026-05-05

### 模块 17: 状态依赖 Stackelberg 动态定价

- [x] Step 1-8: `src/game_pricing/`、dynamic pricing、follower response、theory checks 和 env pricing path 已实现。✅ 2026-05-05 [DONE_PENDING_REVIEW]
- [x] H-2/H-4 fixed: Mainline-A schema 统一为 `queue_model` 与 `channel_model.{theory,simulation}`，旧 `queue/channel` 字段会被拒绝。✅ 2026-05-05

### 模块 18: game-aware critic 与 primal-dual 多智能体更新

- [x] Step 1-8: `src/rl_algorithms/game_aware/`、critic features、primal-dual updater、reward design、trainer hooks、配置和参考已实现。✅ 2026-05-05 [DONE_PENDING_REVIEW]
- [x] H-3/H-4 fixed: BaseTrainer game-aware hook 现在调用 `PrimalDualUpdater.update_dual_variables()` 并暴露 dual state 行为测试。✅ 2026-05-05

### 模块 19: 理论证明与数值验证资产

- [x] Step 1-6: theory appendix、pricing/theory validation helpers、测试和报告模板已实现。✅ 2026-05-05 [DONE_PENDING_REVIEW]

### 模块 20: paper2 内置新模型实验矩阵

- [x] 20A experiment infrastructure: N0/N1/N2/N3 configs、small-scale oracle、experiment runner、plot helpers、publication gate 和 dry-run 验收已实现。✅ 2026-05-05 [DONE_PENDING_REVIEW]
- [x] C-1 fixed: `configs/experiments/mainline_a_n0/n1/n2/n3*.yaml` 进入可跟踪配置路径，runner 默认路径 clean checkout 可解析。✅ 2026-05-05
- [x] 20B N0 formal smoke: DONE_PENDING_FINAL_REVIEW。✅ 2026-05-06
- [x] 20B N1 small-scale oracle validation: DONE_PENDING_FINAL_REVIEW。✅ 2026-05-06
- [x] 20B N2 deterministic controlled probe: DONE_PENDING_FINAL_REVIEW。✅ 2026-05-06
- [x] 20B N2 review fix H-1/M-1/M-2/L-1: DONE_PENDING_FINAL_REVIEW。✅ 2026-05-06
- [x] 20B N3 OOD formal execution: DONE_PENDING_FINAL_REVIEW。✅ 2026-05-06
- [x] 20B experiment gate review: `docs/mainline_a_experiment_gate_review.md` generated from uploaded ignored results artifacts。✅ 2026-05-06 [DONE_PENDING_FINAL_REVIEW]

### 模块 21: 论文写作资产与差异化改写接口

- [x] Step 1-5: `writing_ref/paper2_mainline_a_revision/`、pending questions 和 revision manifest 已生成；未改论文正文。✅ 2026-05-05 [DONE_PENDING_REVIEW]

## 已知问题

- 外部 dashboard 兼容性仍需在 `C:\Users\22003\paper2\rl-mec-dashboard` 可用环境复核。
- Uploaded ignored `results/mainline_a/n2_ablation/summary.json` lacks the newer `evidence_level: deterministic controlled probe` field. This is documented in `docs/mainline_a_experiment_gate_review.md`; `results/` remains ignored and should be regenerated locally later if artifact metadata parity is required.
- N0/N1/N2/N3 均已完成 artifact-level gate review；这些结果在 final review scope 关闭前不得升级为论文主结论。
