# 任务派发

## 类型:patch

先用本包整体替换当前 `docs/`，再按 `docs/plan.md` 从模块 24 Step 1 执行。

范围:
- 本轮先做 docs slimming replacement，旧长 docs 不复制回新目录。
- 完成模块 24 后进入模块 25：Mainline-A environment profile + full-17 默认迁移。
- Mainline-A 是默认训练环境；legacy 只允许显式 fallback。
- `.vscode/launch.json` 收敛到不超过 8 个入口。
- full-17 正式训练前必须先 dry-run，再 quick smoke。

关键提醒:
- 不做 silent fallback。
- 不把 generated artifacts 纳入 Git tracking。
- 验证通过直接下一步；scope:review 完成后记录证据再继续。
