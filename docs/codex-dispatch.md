# Codex 任务派发

进入连续执行模式。按以下流程工作：从 **模块 1：现状锁定与 Quick 入口 bug 复现 Step 1** 开始，严格按 `docs/plan.md` 执行到所有模块完成。

## 启动

1. 读取 `docs/progress.md`，定位当前未完成的第一个步骤。
2. 读取 `docs/plan.md`，加载完整开发计划。
3. 本项目为简单路线，本次交付没有 `docs/references/`；不要等待参考资料。
4. 确认当前仓库为 `paper2`，核心目标是让 VSCode Run and Debug / Tasks 覆盖实验系统全部核心能力，并实现 Full 17 algorithms benchmark 一键运行。

## 计划模块

- 模块 1：现状锁定与 Quick 入口 bug 复现（5 steps）
- 模块 2：实验预设与 CLI 稳定性增强（8 steps）
- 模块 3：VSCode `launch.json` 全入口改造（7 steps）
- 模块 4：VSCode `tasks.json` 任务入口补齐（5 steps）
- 模块 5：使用文档更新（4 steps）
- 模块 6：自动化测试与最终验收（4 steps）

## 执行规则

- 从当前进度点开始，按 `docs/plan.md` 的模块顺序和步骤顺序逐步执行。
- 每个步骤完成后立即运行该步骤指定的验证命令。
- **验证通过 → 直接执行下一步骤，不要停下来问我。**
- **验证失败 → 自行诊断修复，最多重试 2 次；仍失败则记录到 `docs/issues.md` 并停下报告。**
- 每完成一个完整模块后，批量更新 `docs/progress.md`。
- `docs/progress.md` 的模块名、Step 编号、Step 标题必须继续与 `docs/plan.md` 保持逐条一致。
- 如果修改 `.vscode/launch.json` 或 `.vscode/tasks.json`，必须保证它们是标准 JSON，可以通过 `python -m json.tool` 解析。
- 不要运行正式长时间训练作为常规验证；除 `docs/plan.md` 明确要求的手动验收外，自动化验证优先使用结构测试、CLI help、单元测试和短流程 smoke check。

## 仅以下情况停下

- 验证失败重试 2 次仍无法解决。
- 遇到 `docs/plan.md` 未覆盖的技术决策。
- 需要我提供外部资源、凭证、私有数据或无法自动判断的环境信息。
- 当前步骤的前置依赖未完成。
- 发现当前仓库结构与 `docs/plan.md` 假设明显不一致，且无法用现有代码安全修复。

## 禁止行为

- 不要每完成一个小步骤就停下来请求确认。
- 不要偏离 `docs/plan.md` 自行添加功能。
- 不要引入 `docs/plan.md` 未指定的第三方依赖。
- 不要修改算法训练逻辑、奖励函数、环境定义或论文实验指标。
- 不要改变实验最小恢复单元；仍以“算法完成状态”为恢复单元。
- 不要把 `torch`、trainer 或 benchmark 相关训练依赖移回 `scripts/train.py` 顶层导入。
- 不要将 `launch.json` / `tasks.json` 写成带注释 JSON。

## 关键验收目标

- VSCode Run and Debug 面板可以点击启动 Quick 实验。
- VSCode Run and Debug 面板可以点击启动 `paper2_full_17_vscode` 的 Full 17 algorithms benchmark。
- Full 17 benchmark 覆盖以下算法且顺序不变：`GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3`。
- 用户可以点击停止、恢复、查看状态、列出实验、导出结果、重置失败算法、重建索引。
- Quick 入口报错问题已被定位并修复；如果训练本身失败，错误信息必须指向具体 stdout/stderr 日志。
- VSCode Tasks 面板提供同等轻量任务入口。
- 用户无需手输任何命令即可完成主要实验流程。

## 完成后

输出完成报告，格式必须包含：

| 模块 | 状态 | 关键改动 | 验证结果 |
|---|---|---|---|
| 模块 1：现状锁定与 Quick 入口 bug 复现 | completed/blocked | ... | ... |
| 模块 2：实验预设与 CLI 稳定性增强 | completed/blocked | ... | ... |
| 模块 3：VSCode `launch.json` 全入口改造 | completed/blocked | ... | ... |
| 模块 4：VSCode `tasks.json` 任务入口补齐 | completed/blocked | ... | ... |
| 模块 5：使用文档更新 | completed/blocked | ... | ... |
| 模块 6：自动化测试与最终验收 | completed/blocked | ... | ... |

并列出：

- 遇到的 issues 列表；没有则写“无”。
- 需要用户后续处理的事项；没有则写“无”。
- 最终建议用户在 VSCode 中优先点击的入口：`🏁 Experiment Full 17 Start/Resume`。

现在开始执行。
