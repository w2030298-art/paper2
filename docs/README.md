# docs 目录说明

`docs/` 是 Web + Codex 双端工作流的执行文档目录。执行端恢复上下文时只读取当前有效计划与 inbox 输入，不从归档目录恢复历史计划。

## 目录契约

```text
docs/
├── README.md
├── plan.md
├── plan-patch.md
├── progress.md
├── issues.md
├── report.md
├── vscode_experiment_usage.md
├── convergence_plot_quality.md
├── inbox/
├── references/
└── archive/
```

| 路径 | 用途 | 规则 |
|------|------|------|
| `docs/README.md` | docs 目录索引和契约说明 | 当前文件 |
| `docs/plan.md` | 主开发计划 | 当前有效计划，执行端按此恢复任务 |
| `docs/plan-patch.md` | 当前已合入或待合入的增量计划基线 | 保留为 patch 基线，不归档 |
| `docs/progress.md` | 步骤完成状态 | 每完成 Step 后更新 |
| `docs/issues.md` | 执行问题记录 | 仅追加问题或本轮执行备注 |
| `docs/report.md` | 与用户/Web 的双向执行报告 | 每次任务后更新 |
| `docs/vscode_experiment_usage.md` | VSCode 实验入口使用说明 | 当前有效用户文档 |
| `docs/convergence_plot_quality.md` | 收敛曲线质量规则说明 | 模块 10 的质量管线文档 |
| `docs/inbox/` | 执行端 merge-back 输入区 | Web 投递新版计划到 `docs/inbox/plan.md` 或等价 plan 文件 |
| `docs/references/` | 实现参考和调研资料 | 只读参考材料 |
| `docs/archive/` | 历史计划、备份、过期 patch | 只归档，不作为恢复依据 |

## 恢复规则

执行端只从 `docs/plan.md` 与 `docs/inbox/plan.md` 恢复计划。`docs/archive/` 中的文件只用于保留历史证据，不参与自动恢复、patch diff 或执行起点判断。

## 归档规则

- 根目录中明显历史或 backup markdown 文件移动到 `docs/archive/`。
- 不删除历史文档。
- 不移动当前有效文件：`docs/plan.md`、`docs/plan-patch.md`、`docs/inbox/plan.md`、`docs/report.md`、`docs/progress.md`、`docs/issues.md`、`docs/convergence_plot_quality.md`。
