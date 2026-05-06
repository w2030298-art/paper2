# 开发进度

## 当前状态

- 当前计划版本: `system-model-overhaul-v4.2`
- 最后更新: 2026-05-06
- 状态: REVIEW_CLOSED
- 当前阶段: 模块 23 Mainline-A experiment final review completed；项目进入 stable review-complete。
- 执行边界: `paper2` 只承载对比算法实验 / 仿真实验；N0 为 smoke evidence，N1 为 small-scale oracle evidence，N2 为 deterministic controlled probe，N3 为 OOD formal execution evidence；不维护论文正文、写作资产或论文主结论。

## 模块进度

### 模块 1-13: 历史已完成模块

- [x] 保持完成状态；未恢复旧入口、`docs_paper/` 或 generated artifact tracking。✅ 2026-05-05

### 模块 14: formal convergence verification protocol

- [x] 已降级为 pre-overhaul legacy baseline。✅ 2026-05-05
- [x] C-2 fixed: formal convergence 单测改为 fixtures / 显式参数，不依赖 ignored `results/` 产物。✅ 2026-05-05

### 模块 14R: legacy convergence retirement

- [x] Step 1-5: legacy convergence retirement docs/status 已完成。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]

### 模块 15: MEC 系统模型模块化基底

- [x] Step 1-10: `src/mec_model/`、配置、测试和参考资产已实现。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]

### 模块 16: 系统模型与现有环境兼容接入

- [x] Step 1-8: env adapter、Mainline-A state/reward/benchmark dry-run 接入已实现。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]
- [x] H-3/H-4 fixed: adapter decision 可写入 env 价格路径，reward components 使用 step 行为指标，行为级测试已补充。✅ 2026-05-05

### 模块 17: 状态依赖 Stackelberg 动态定价

- [x] Step 1-8: `src/game_pricing/`、dynamic pricing、follower response、theory checks 和 env pricing path 已实现。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]
- [x] H-2/H-4 fixed: Mainline-A schema 统一为 `queue_model` 与 `channel_model.{theory,simulation}`，旧 `queue/channel` 字段会被拒绝。✅ 2026-05-05

### 模块 18: game-aware critic 与 primal-dual 多智能体更新

- [x] Step 1-8: `src/rl_algorithms/game_aware/`、critic features、primal-dual updater、reward design、trainer hooks、配置和参考已实现。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]
- [x] H-3/H-4 fixed: BaseTrainer game-aware hook 现在调用 `PrimalDualUpdater.update_dual_variables()` 并暴露 dual state 行为测试。✅ 2026-05-05

### 模块 19: 理论证明与数值验证资产

- [x] Step 1-6: theory appendix、pricing/theory validation helpers、测试和报告模板已实现。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]

### 模块 20: paper2 内置新模型实验矩阵

- [x] 20A experiment infrastructure: N0/N1/N2/N3 configs、small-scale oracle、experiment runner、plot helpers、publication gate 和 dry-run 验收已实现。✅ 2026-05-05 [DONE_ACCEPTED_WITH_BOUNDARIES]
- [x] C-1 fixed: `configs/experiments/mainline_a_n0/n1/n2/n3*.yaml` 进入可跟踪配置路径，runner 默认路径 clean checkout 可解析。✅ 2026-05-05
- [x] 20B N0 formal smoke: N0_ACCEPTED_WITH_BOUNDARIES，smoke evidence only。✅ 2026-05-05
- [x] 20B N1 small-scale oracle validation: N1_ACCEPTED_WITH_BOUNDARIES，small-scale oracle evidence。✅ 2026-05-05
- [x] 20B N2 deterministic controlled probe: N2_ACCEPTED_WITH_BOUNDARIES，deterministic controlled probe only。✅ 2026-05-05
- [x] 20B N2 review fix H-1/M-1/M-2/L-1: N2_ACCEPTED_WITH_BOUNDARIES，deterministic controlled probe only。✅ 2026-05-05
- [x] 20B N3 OOD formal execution: N3_ACCEPTED_WITH_BOUNDARIES，OOD formal execution evidence。✅ 2026-05-06

### 模块 21: 论文写作资产与差异化改写接口 `[REMOVED_OUT_OF_SCOPE]`

- [x] v4.2 边界修正：该模块已从 paper2 项目范围移除。`writing_ref/paper2_mainline_a_revision/` 已从 Git tracking 删除，论文相关更改不再属于本项目。✅ 2026-05-06

### 模块 22: project-boundary-cleanup

- [x] Step 1: 审计论文相关入仓痕迹 ✅ 2026-05-06
- [x] Step 2: 更新计划元信息与 Status ✅ 2026-05-06
- [x] Step 3: 删除 tracked 论文写作资产 ✅ 2026-05-06
- [x] Step 4: 同步 progress/report/issues 状态 ✅ 2026-05-06
- [x] Step 5: 改写实验 gate 文档中的论文措辞 ✅ 2026-05-06
- [x] Step 6: 最终边界验证 ✅ 2026-05-06

### 模块 23: Mainline-A experiment final review

- [x] Step 1: 复核实验链 gate 文档；N0/N1/N2/N3 evidence level 保持边界 ✅ 2026-05-06
- [x] Step 2: final review decision = `ACCEPTED_WITH_BOUNDARIES` ✅ 2026-05-06
- [x] Step 3: 不追加训练、不跑 L2/L3、不启动更大 benchmark；项目进入 stable review-complete ✅ 2026-05-06

## 已知问题

- 外部 dashboard 兼容性仍需在 `C:\Users\22003\paper2\rl-mec-dashboard` 可用环境复核；该项为外部复核项，不阻塞 paper2 final review。
- N0/N1/N2/N3 均已按 `ACCEPTED_WITH_BOUNDARIES` 关闭 final review；证据等级仍分别为 smoke evidence、small-scale oracle evidence、deterministic controlled probe 和 OOD formal execution evidence，不得升级为正式 benchmark 结论或过度解释实验结果。
