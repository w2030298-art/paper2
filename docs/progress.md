# paper2 progress

## 当前状态

- 当前计划版本：`system-model-overhaul-v4.3`
- 最后更新：2026-05-06
- 状态：`NEEDS_REVIEW`
- 当前阶段：模块 25 Mainline-A full-17 default migration completed；scope:review 结果待审核。模块 26 execution gate pending。
- 主线：Mainline-A。
- fallback：legacy 仅显式 fallback，不允许 silent fallback。
- 外部阻塞：dashboard 兼容性仍需外部环境复核；不阻塞本轮迁移。

## 压缩历史状态

| 范围 | 状态 |
|---|---|
| 模块 1-13 | 历史完成；不再展开到活跃 docs |
| 模块 14 | legacy convergence baseline；不作为 Mainline-A 主门禁 |
| 模块 14R-20 | Mainline-A 实验链已实现并按边界关闭 review |
| 模块 21 | `REMOVED_OUT_OF_SCOPE`，论文写作资产不属于 paper2 |
| 模块 22 | project-boundary-cleanup completed |
| 模块 23 | Mainline-A experiment final review `ACCEPTED_WITH_BOUNDARIES` |
| 模块 24 | completed |
| 模块 25 | completed；待审核 |
| 模块 26 | pending |

## 当前模块进度

### 模块 24：docs directory slimming

- [x] Step 1：替换 docs 目录 `scope:auto` ✅ 2026-05-06
- [x] Step 2：验证 docs 轻量化边界 `scope:auto` ✅ 2026-05-06
- [x] Step 3：验证状态源一致 `scope:auto` ✅ 2026-05-06
- [x] Step 4：提交 docs reset `scope:auto` ✅ 2026-05-06

### 模块 25：Mainline-A environment profile 与 full-17 默认迁移

- [x] Step 1：新增 `src/experiment/environment_profiles.py` `scope:review` ✅ 2026-05-06
- [x] Step 2：扩展 `scripts/train.py` 支持 profile `scope:review` ✅ 2026-05-06
- [x] Step 3：扩展 `src/experiment/registry.py` / `models.py` 注入 profile `scope:review` ✅ 2026-05-06
- [x] Step 4：扩展 `scripts/experiment_manager.py` CLI `scope:review` ✅ 2026-05-06
- [x] Step 5：重写 `src/experiment/presets.py` 默认 profile `scope:auto` ✅ 2026-05-06
- [x] Step 6：统一 `scripts/benchmark.py` direct benchmark profile `scope:review` ✅ 2026-05-06
- [x] Step 7：重写 `.vscode/launch.json` `scope:auto` ✅ 2026-05-06
- [x] Step 8：回归测试与边界检查 `scope:auto` ✅ 2026-05-06

### 模块 26：Mainline-A full-17 benchmark execution gate

- [ ] Step 1：为 experiment_manager 增加 dry-run `scope:review`
- [ ] Step 2：Mainline-A full-17 preflight `scope:auto`
- [ ] Step 3：Mainline-A smoke benchmark `scope:review`
- [ ] Step 4：Mainline-A full-17 正式运行 `scope:review`
- [ ] Step 5：legacy fallback 只做对照复现 `scope:review`

## 证据边界

- N0：smoke evidence only。
- N1：small-scale oracle evidence。
- N2：deterministic controlled probe only。
- N3：OOD formal execution evidence。
- v4.3 full-17 正式 benchmark 产生后，才能作为 Mainline-A full-17 算法筛选依据。
