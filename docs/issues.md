# 问题记录

> Codex 在执行中遇到 plan.md 未覆盖的情况时记录于此。

（暂无问题）

## 2026-05-02 Patch v3

- 模块 10-11 执行未发现阻塞问题。

## 2026-05-02 Patch v4

- [Fixed] M12-1: 归档目录不一致 — 已将 `docs/.archive/` 内容迁入 `docs/archive/`，后续只保留 `docs/archive/` 作为归档目录。
- [Fixed] M12-2: 产物目录提交卫生 — 当前未发现 `experiments/`、`results/`、`figures/` 的待提交产物变更；`results/convergence_failure_matrix.json` 为 ignored 诊断产物。
- [Fixed] M12-3: 收敛判定过度依赖 tail relative change — 已新增 bad_plateau / oscillating / diverging / catastrophic_outlier 分类。

## 2026-05-02 Slimming Phase 1-2

- [Needs Review] S13-dashboard-check: `C:\Users\22003\paper2\rl-mec-dashboard` 本机不存在，无法执行外部 dashboard grep 兼容性验证。paper2 内部引用已清理，删除结果需在有 dashboard 仓库的环境复核。
- [Fixed] S12-generated-artifacts: `experiments/`、`results/`、`figures/`、`logs/`、`checkpoints/` 已从 Git tracking 移除，本地数据保留并通过 `.gitignore` 覆盖。
- [Fixed] S13-docs-paper-migration: `docs_paper/` 已复制到 `C:\Users\22003\paper2\writing_ref\docs_paper`，18 个源文件 SHA-256 校验通过后从 Git 删除。
