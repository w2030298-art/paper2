# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新: 2026-05-06 | plan.md 版本: system-model-overhaul-v4.1

## Last Execution

- 来源: dispatch:patch
- 摘要: 从模块 20B N3 OOD formal execution 开始执行，只跑 `configs/experiments/mainline_a_n3_ood.yaml` 的 N3 路径。N0/N1/N2 保持 review 状态且未重跑，N2 继续只作为 deterministic controlled probe；未跑 `--stage all`，未跑 full 17。
- 当前阶段: N3_DONE_PENDING_REVIEW；总 review scope 仍未关闭，因此状态保留 NEEDS_REVIEW。

## Completed

- [x] N3 runner: `scripts/run_mainline_a_experiments.py` 已补齐 N3 dry-run plan、preflight 和 formal execution 路径，输出限定到 `results/mainline_a/n3_ood/`。
- [x] N3 metrics audit: 已记录 train/test 分布差异，并审计 social_welfare、average_latency、p95_latency、energy、provider_revenue、constraint_violation_rate、jain_fairness、oracle_gap_small_cases。
- [x] N3 anomaly checks: NaN/Inf、schema mismatch、空结果、全同指标、OOD test 配置应用均通过；test split 实际使用 `3gpp_lite`、high mobility、parallel queue、cooperation enabled。
- [x] Benchmark alias guard: `results/benchmark.json` 未创建、未覆盖。
- [x] N3 report: `docs/mainline_a_n3_ood_report.md` 已生成。
- [x] Verification: `python -m ruff check scripts\run_mainline_a_experiments.py tests\test_mainline_a_experiment_runner.py` passed。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest tests\test_mainline_a_experiment_runner.py -q` passed, 16 passed。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest -q` passed, 239 passed in 43.15s。

## In Review

- [ ] 模块 14R-21 review scope 实现项 — 待用户/Web 审核。
- [ ] 模块 20B N0 smoke — N0_DONE_PENDING_REVIEW。
- [ ] 模块 20B N1 small-scale oracle validation — N1_DONE_PENDING_REVIEW。
- [ ] 模块 20B N2 deterministic controlled probe — N2_DONE_PENDING_REVIEW。
- [ ] 模块 20B N3 OOD formal execution — N3_DONE_PENDING_REVIEW。

## Blocked

- [ ] 外部 dashboard 兼容性 — `C:\Users\22003\paper2\rl-mec-dashboard` 本机仍不可用，需在有该仓库的环境复核。

## Discovered Issues

- `docs/inbox/` 中存在旧 `review-report-paper2-mainline-a-audit-20260505.md`，但没有 `docs/inbox/plan.md`；本轮未执行 plan merge-back，避免把 review report 误合并为 active plan。
- `rg` 在本机报 `Access is denied`；本轮改用 PowerShell `Select-String` 完成源码定位。

## Recommendations

- 审核 `docs/mainline_a_n3_ood_report.md` 和 `results/mainline_a/n3_ood/summary.json`。在 N0/N1/N2/N3 review scope 关闭前，不要把这些结果写成论文主结论。
