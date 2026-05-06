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

## 2026-05-03 Module 14 Convergence Validation

- [Needs Review] M14-Step6-baseline-50k-runtime: baseline 50k targeted rerun 已生成 dry-run manifest（`experiments/convergence_validation/baseline_50k/20260503_003320/commands.txt`），但实际运行将调用 11 个目标算法 x 3 seeds x 50000 steps。旧 100k 单 seed 日志显示部分算法为数小时级训练，当前交互回合未启动该长时训练，避免留下不可控运行进程。
- [Info] M14-event-audit: `IPPO`、`IQL`、`VDN` catastrophic outlier 审计未发现 traceback、NaN/Inf 或直接环境/reward/action/logging bug 证据；当前决策为 `rerun_baseline`，不进入 override。

## 2026-05-04 Module 14 Formal Convergence Protocol

- [In Progress] M14-v3-L2-runtime: v3 要求 L2 100k multi-seed validation（`seeds=[42,43,44]`）。当前已启动 background job `l2_20260504_171744`，等待长时训练完成。
- [Needs Review] M14-v3-L3-runtime: L3 200k multi-seed formal validation（`seeds=[42,43,44,45,46]`）依赖 L2 通过算法。本轮无算法完成 L3，因此不得进入论文主图或主结论。
- [Info] M14-v3-L1-boundary: `results/convergence_validation_baseline_50k.json` 只作为 L1 预筛；`COMA/MAPPO/TRPO` 为 `l1_candidate`，不是 formal convergence verdict。
- [Info] M14-v3-event-audit: `IQL/VDN/IPPO/MADDPG` 当前事件审计归类为 `training_instability`，未发现 reward/metric/env 语义错误证据；若后续发现语义错误，应立即标记 `NEEDS_ESCALATION`。
- [Info] M14-v3-forbidden-check: v3 plan 指令文本本身包含禁止语句示例，最终防误报 grep 需要排除 `docs/plan.md` 与 `docs/archive/` 后检查执行产物。
- [In Progress] M14-v3-L2-active-run: 已启动 L2 background job `l2_20260504_171744`，PID `26860`，manifest 为 `experiments/formal_convergence/l2/l2_20260504_171744/manifest.json`。当前运行算法为 `COMA/MAPPO/TRPO/IQL/VDN/IPPO/MADDPG`；`A3C/MATD3/SAC/GRPO` 因需 single-variable fix 候选暂未进入。

## 2026-05-05 System Model Overhaul v4.1

- [Info] M14R-legacy-convergence-retirement: 旧 L2/L3 已降级为 `legacy_pre_overhaul` baseline；旧结果不得进入新论文主图或新系统模型主结论。
- [Needs Review] M14R-L2-stop: 旧 L2 run `l2_20260504_171744` 的 PID `26860` 已停止，manifest/log 保留；旧 L3 未启动。该操作属于 review scope，需审核。
- [Info] Docs-cleanup: 旧 L1/L2/L3 结果报告与历史 convergence reassessment reference 已迁入 `docs/archive/legacy-convergence-20260505/`。
- [Needs Review] Mainline-A-review-scope: 模块 15-20 涉及 MEC model、dynamic pricing、env adapter、game-aware primal-dual、theory assets 和 N0/N1/N2/N3 experiment chain，均含 review scope 项，等待用户/Web 审核。

## 2026-05-05 Mainline-A Review Fix

- [Fixed] C-1: Mainline-A 默认实验 runner 引用未入仓配置 — 已补齐 `configs/experiments/mainline_a_n0/n1/n2/n3*.yaml` 并调整 ignore 规则，默认 dry-run 可解析。
- [Fixed] C-2: formal convergence 单测依赖 ignored `results/` 产物 — 已改为 fixtures / 显式参数，缺失 runtime artifacts 时 dry-run 不再阻断单测。
- [Fixed] H-1: `docs/plan.md`、`docs/progress.md`、`docs/report.md` 状态不一致 — 已同步为 C/H fix completed、NEEDS_REVIEW、正式实验 NOT_STARTED。
- [Fixed] H-2: Mainline-A config schema 不一致 — 已统一为 `queue_model` 与 `channel_model.{theory,simulation}`，旧 `queue/channel` 字段会被拒绝。
- [Fixed] H-3: env/pricing/reward/primal-dual 未形成可测试闭环 — 已让 adapter decision 写入 env price path，reward components 读取 step 行为指标，trainer hook 更新 dual state。
- [Fixed] H-4: 行为级测试不足 — 已补 runner dry-run、dynamic pricing reward path、reward components、primal-dual hook 和 active entrypoint 测试。
- [Fixed] H-1-docs-contract: clean checkout 验证发现 `docs/README.md` 与 `docs/slimming_audit_phase2.md` 缺失 — 已补回 docs contract 与 dashboard audit 最小文档。
- [Fixed] H-1-stale-l2-processes: 旧 L2 benchmark 子进程 PID 26016 / 21904 仍占用 `logs/` — 已停止，避免与本轮“不启动正式训练”边界冲突。
