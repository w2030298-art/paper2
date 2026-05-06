# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新: 2026-05-05 | plan.md 版本: system-model-overhaul-v4.1

## Last Execution

- 来源: dispatch:patch
- 摘要: 完成 N1 small-scale oracle validation，确认 `configs/experiments/mainline_a_n1_oracle.yaml` 的 users/edges 均在 oracle 支持范围内。已仅运行 N1 oracle 对比，结果写入 `results/mainline_a/n1_oracle/`；未启动 N2/N3，未跑 full 17，也未覆盖 `results/benchmark.json`。
- 当前阶段: N1_DONE_PENDING_REVIEW；总 review scope 仍未关闭，因此状态保留 NEEDS_REVIEW。

## Completed

- [x] C-1: 补齐并解除 ignore `configs/experiments/mainline_a_n0/n1/n2/n3*.yaml`，`run_mainline_a_experiments.py --stage all --dry-run` 可解析默认配置。已推送到 main
- [x] C-2: formal convergence runner 单测改为 fixtures / 显式参数，库函数不再隐式依赖 ignored `results/` 产物。已推送到 main
- [x] H-1: 同步 `docs/plan.md`、`docs/progress.md`、`docs/report.md` 的当前执行状态。已推送到 main
- [x] H-2: 统一 Mainline-A config schema，拒绝旧 `queue/channel` 字段静默通过。已推送到 main
- [x] H-3/H-4: 补 env dynamic pricing、reward components、adapter decision、primal-dual trainer hook 的行为级闭环与测试。已推送到 main
- [x] Mainline-A C/H fix 审核结论更新为 accepted。
- [x] N0 smoke completed: seed 42, configured/executed steps 1000, MAPPO game-aware path, reward -34.2025, latency mean 0.4567, energy mean 2.3157。
- [x] N0 smoke report generated: `docs/mainline_a_n0_smoke_report.md`。
- [x] N1 oracle command: `.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n1_oracle.yaml --stage N1 --results-root results\mainline_a`。
- [x] N1 case matrix: seeds [42,43,44] × users [2,3,4] × edges [2,3] = 18 cases，均满足 `num_users <= 4 and num_edges <= 3`。
- [x] N1 oracle result: 54 records；mean oracle gap baseline=1.4303333333, MAPPO=0.165, game_aware_pd_marl=0.0288888889；constraint violation NaN/Inf=0。
- [x] N1 oracle report generated: `docs/mainline_a_n1_oracle_report.md`。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest -q` passed, 229 passed in 37.99s。

## In Review

- [ ] 模块 14R-21 review scope 实现项 — 待用户/Web 审核。
- [ ] 模块 20B N0 smoke — N0_DONE_PENDING_REVIEW。
- [ ] 模块 20B N1 small-scale oracle validation — N1_DONE_PENDING_REVIEW。

## Blocked

- [ ] 外部 dashboard 兼容性 — `C:\Users\22003\paper2\rl-mec-dashboard` 本机仍不可用，需在有该仓库的环境复核。
- [ ] N2/N3 正式实验 — 本轮按指令不启动。

## Discovered Issues

- N0 前置发现 `game_aware_pd_marl` 不是已注册 benchmark 算法名；已作为 N0 config label 映射到现有 MAPPO game-aware 训练路径。
- N0 前置发现默认 benchmark 会更新 `results/benchmark.json` latest alias；已新增 `--no-latest-alias` 并用于本次 N0，未覆盖主 benchmark 结果。
- N0 日志/结果未发现 traceback、NaN、Inf；reward 非全零；价格探针未发现全部 clip 到边界。
- N1 异常审计通过：oracle gap 无缺失，oracle gap/constraint violation 均无 NaN/Inf，case matrix 未越过 oracle 支持范围。
- Medium/Low 未在本轮修复范围内处理。

## Recommendations

- 审核 N1 oracle 报告后，再决定是否进入 N2；不要在 N1 未审核前启动 N2/N3。
