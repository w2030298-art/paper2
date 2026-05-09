# paper2 docs contract

## 当前定位

`paper2` 是 Mainline-A 对比算法实验 / 仿真实验项目。`docs/` 只保留执行端恢复上下文、执行 v4.3 迁移、记录当前状态所需的活跃文件。

## 活跃文件

| 文件 | 用途 |
|---|---|
| `plan.md` | 当前唯一开发计划：`system-model-overhaul-v4.3` |
| `progress.md` | 当前进度快照，不展开历史流水账 |
| `report.md` | 当前执行报告与状态 |
| `issues.md` | 当前未关闭问题与少量历史索引 |
| `mainline_a_experiment_gate_review.md` | Mainline-A N0/N1/N2/N3 evidence 边界 |
| `references/context_snapshot_v4_2.md` | v4.2 历史状态压缩快照 |
| `references/mainline_a_migration_notes.md` | v4.3 迁移实现参考 |
| `references/legacy_boundary_index.md` | 旧环境/fallback 边界 |
| `inbox/README.md` | 后续 patch 入口说明 |
| `archive/README.md` | 旧长文档处理说明 |

## 取舍

- 不保留从模块 1 开始的长篇 plan/progress/report 流水账。
- 不保留旧 `docs/archive/*` 的历史正文。
- 旧 v4.2 以前的细节只压缩到 `references/context_snapshot_v4_2.md`。
- `results/`、`experiments/`、`figures/`、`logs/`、`checkpoints/` 仍不进入 Git tracking。

## 替换方式

```bash
# 在仓库根目录执行；替换前自行确保当前 docs 已被 Git/外部备份保留
rm -rf docs
unzip docs.zip
test -f docs/plan.md
test -f docs/report.md
```
