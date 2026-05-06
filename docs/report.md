# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新: 2026-05-05 | plan.md 版本: system-model-overhaul-v4.1

## Last Execution

- 来源: dispatch:fix
- 摘要: 完成 N2 审查 fix，将 N2 明确降级为 deterministic controlled probe，不再表述为 training-grade / publication-grade ablation。补充 `results/benchmark.json` 缺失时不得创建、重复 ablation label 必须拒绝的回归测试；未启动 N3，未跑 full 17。
- 当前阶段: N2_DONE_PENDING_REVIEW；总 review scope 仍未关闭，因此状态保留 NEEDS_REVIEW。

## Completed

- [x] H-1: `scripts/run_mainline_a_experiments.py` 和 `docs/mainline_a_n2_ablation_report.md` 已明确 N2 evidence level 为 `deterministic controlled probe`，并声明不是 training-grade / publication-grade ablation。
- [x] M-1: 补充测试，确认 `results/benchmark.json` 不存在时 N2 deterministic controlled probe 不会创建该 alias。
- [x] M-2: 补充测试并实现校验，重复 ablation label 会被 `validate_n2_ablation_config()` 拒绝。
- [x] L-1: `docs/mainline_a_n2_ablation_report.md`、`docs/report.md`、`docs/progress.md` 已同步降级表述，状态保持 `N2_DONE_PENDING_REVIEW`，N3 仍 `NOT_STARTED`。
- [x] Ruff: `python -m ruff check scripts\run_mainline_a_experiments.py tests\test_mainline_a_experiment_runner.py` passed；系统 Python ruff 可用，项目 `.venv` 未安装 ruff。
- [x] Verification: `python -m pytest tests\test_mainline_a_experiment_runner.py -q` passed, 12 passed in 0.47s。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest tests\test_docs_contract.py tests\test_repo_hygiene.py -q` passed, 9 passed in 0.05s。
- [x] Verification: `.\.venv\Scripts\python.exe -m pytest -q` passed, 235 passed in 40.97s。

## In Review

- [ ] 模块 14R-21 review scope 实现项 — 待用户/Web 审核。
- [ ] 模块 20B N0 smoke — N0_DONE_PENDING_REVIEW。
- [ ] 模块 20B N1 small-scale oracle validation — N1_DONE_PENDING_REVIEW。
- [ ] 模块 20B N2 deterministic controlled probe — N2_DONE_PENDING_REVIEW。

## Blocked

- [ ] 外部 dashboard 兼容性 — `C:\Users\22003\paper2\rl-mec-dashboard` 本机仍不可用，需在有该仓库的环境复核。
- [ ] N3 OOD formal execution — 本轮按指令不启动。

## Discovered Issues

- 指定审查报告路径 `docs/inbox/review-report-paper2-mainline-a-n2-ablation-20260505.md` 当前不存在；本轮按用户派发的 H-1/M-1/M-2/L-1 清单修复。
- `python -m pytest -q` 使用系统 Python 3.14 时 collection 失败，缺少 `tqdm`；这是环境依赖问题，不是本轮 N2 fix 回归。项目 `.venv` 全量 pytest 已通过。

## Recommendations

- 先审核 `docs/mainline_a_n2_ablation_report.md` 的 deterministic controlled probe 表述。N3 仍保持 `NOT_STARTED`，不要在 N2 审核前启动 N3。
