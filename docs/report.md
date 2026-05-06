# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新: 2026-05-05 | plan.md 版本: system-model-overhaul-v4.1

## Last Execution

- 来源: dispatch:patch
- 摘要: 完成 N2 ablation dry-run、preflight 与 controlled ablation。结果写入 `results/mainline_a/n2_ablation/`，未启动 N3，未跑 full 17，未创建或覆盖 `results/benchmark.json`。
- 当前阶段: N2_DONE_PENDING_REVIEW；总 review scope 仍未关闭，因此状态保留 NEEDS_REVIEW。

## Completed

- [x] N2 dry-run plan: `.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n2_ablation.yaml --stage N2 --dry-run`，确认 9 个 ablation、seeds `[42,43,44]`、steps `50000`、输出路径 `results/mainline_a/n2_ablation/`。
- [x] N2 ablation runner: 每个 ablation 显式映射到 `dynamic_pricing / queue_state / channel_state / migration_state / primal_dual / cooperation / channel_model / reward_ablation` 开关。
- [x] N2 preflight: `.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n2_ablation.yaml --stage N2 --preflight --preflight-steps 256 --results-root results\mainline_a`，9 records，required metrics/schema/finite checks passed。
- [x] N2 controlled ablation: `.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n2_ablation.yaml --stage N2 --results-root results\mainline_a`，27 records，required metrics/schema/finite/non-identical checks passed。
- [x] N2 report generated: `docs/mainline_a_n2_ablation_report.md`。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest tests\test_mainline_a_experiment_runner.py -q` passed, 10 passed in 0.36s。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest -q` passed, 233 passed in 46.28s。

## In Review

- [ ] 模块 14R-21 review scope 实现项 — 待用户/Web 审核。
- [ ] 模块 20B N0 smoke — N0_DONE_PENDING_REVIEW。
- [ ] 模块 20B N1 small-scale oracle validation — N1_DONE_PENDING_REVIEW。
- [ ] 模块 20B N2 controlled ablation — N2_DONE_PENDING_REVIEW。

## Blocked

- [ ] 外部 dashboard 兼容性 — `C:\Users\22003\paper2\rl-mec-dashboard` 本机仍不可用，需在有该仓库的环境复核。
- [ ] N3 OOD formal execution — 本轮按指令不启动。

## Discovered Issues

- N2 anomaly audit passed: no label-only ablation, no missing required metrics, no schema mismatch, no NaN/Inf, no all-identical metrics, and no non-full ablation identical to `full_model`。
- Optional ruff check was unavailable in `.venv` because `ruff` is not installed; pytest verification passed.

## Recommendations

- 先审核 `docs/mainline_a_n2_ablation_report.md`。N2 本地检查已通过，但 N3 仍保持 `NOT_STARTED`，不要在 N2 审核前启动 N3。
