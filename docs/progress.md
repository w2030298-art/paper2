# 开发进度

## 当前状态

- 当前计划版本：`system-model-overhaul-v4.1`
- 最后更新：2026-05-05
- 状态：NEEDS_REVIEW
- 当前阶段：模块 14R-21 已按 v4.1 执行；review scope 项等待用户/Web 审核。

## 模块进度

### 模块 1-13：历史已完成模块
- [x] 保持完成状态；未恢复旧入口、未恢复 `docs_paper/`、未恢复 generated artifact tracking。✅ 2026-05-05

### 模块 14：formal convergence verification protocol
- [x] 已降级为 pre-overhaul legacy baseline。✅ 2026-05-05
- [x] 旧 L2 run `l2_20260504_171744` 已停止；旧 L3 未启动。✅ 2026-05-05

### 模块 14R：legacy convergence retirement
- [x] Step 1: 检查并停止旧 L2 background job。✅ 2026-05-05 [review]
- [x] Step 2: 更新 publication gate，新增 `legacy_pre_overhaul`。✅ 2026-05-05 [auto]
- [x] Step 3: 在 protocol 中定义 N0/N1/N2/N3 新模型实验链。✅ 2026-05-05 [auto]
- [x] Step 4: 更新 docs 状态、issues 和 report。✅ 2026-05-05 [auto]
- [x] Step 5: 完成旧结论防误报检查。✅ 2026-05-05 [auto]

### 模块 15：MEC 系统模型模块化基座
- [x] Step 1-10: 新增 `src/mec_model/`、配置、测试和系统模型参考。✅ 2026-05-05 [review items pending]

### 模块 16：系统模型与现有环境兼容接入
- [x] Step 1-8: 新增 owner audit、adapter、可选 mainline-A state/reward、benchmark dry-run 参数和兼容性报告。✅ 2026-05-05 [review items pending]

### 模块 17：状态依赖 Stackelberg 动态定价
- [x] Step 1-8: 新增 `src/game_pricing/`、dynamic pricing、follower/leader/theory checks、env pricing path 和理论说明。✅ 2026-05-05 [review items pending]

### 模块 18：game-aware critic 与 primal-dual 多智能体更新
- [x] Step 1-8: 新增 `src/rl_algorithms/game_aware/`、critic features、primal-dual updater、reward design、trainer hooks、配置和参考。✅ 2026-05-05 [review items pending]

### 模块 19：理论证明与数值验证资产
- [x] Step 1-6: 新增 theory appendix、pricing/theory validation helpers、测试和报告模板。✅ 2026-05-05 [review items pending]

### 模块 20：paper2 内置新模型实验矩阵
- [x] Step 1-10: 新增 N0/N1/N2/N3 configs、small-scale oracle、experiment runner、plot helpers、publication gate 和 dry-run 验收。✅ 2026-05-05 [review items pending]

### 模块 21：论文写作资产与差异化改写接口
- [x] Step 1-5: 新增 `writing_ref/paper2_mainline_a_revision/`、pending questions 和 revision manifest；未改论文正文。✅ 2026-05-05 [review items pending]

## 已知问题

- 外部 dashboard 兼容性仍需在 `C:\Users\22003\paper2\rl-mec-dashboard` 可用环境复核。
- 本轮仅执行 dry-run 和单元测试；N0/N1/N2/N3 正式实验未启动。
