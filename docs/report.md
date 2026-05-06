# Execution Report

## STATUS: NEEDS_REVIEW

> 上次更新: 2026-05-06 | plan.md 版本: system-model-overhaul-v4.1

## Last Execution

- 来源: direct GitHub fix
- 摘要: 根据用户上传的 ignored `results/` JSON 产物完成 Mainline-A N0/N1/N2/N3 artifact-level gate review，并新增 `docs/mainline_a_experiment_gate_review.md`。未改本地工作区，未把 `results/` 纳入 Git tracking。
- 当前阶段: Mainline-A experiment gate review completed; N0/N1/N2/N3 进入 DONE_PENDING_FINAL_REVIEW；总 review scope 仍未关闭，因此状态保留 NEEDS_REVIEW。

## Completed

- [x] N0 artifact review: 已复核上传的 `results/mainline_a/n0_smoke/benchmark.json`，仅接受为 smoke evidence。
- [x] N1 artifact review: 已复核上传的 `results/mainline_a/n1_oracle/summary.json` 与 `oracle_gap_report.json`，接受为 small-scale oracle evidence。
- [x] N2 artifact review: 已复核上传的 `results/mainline_a/n2_ablation/summary.json`、`ablation_records.json`、`metric_deltas.json`，只接受为 deterministic controlled probe evidence。
- [x] N3 artifact review: 已复核上传的 `results/mainline_a/n3_ood/summary.json`、`ood_records.json`、`metric_summary.json`、`distribution_shift.json` 及 preflight artifacts，接受为 OOD formal execution evidence。
- [x] Gate review doc: 新增 `docs/mainline_a_experiment_gate_review.md`。
- [x] Boundary preserved: N2 未升级为 training-grade / publication-grade ablation evidence。
- [x] Boundary preserved: 未运行 N3、未运行 full 17、未运行 `--stage all`。
- [x] Boundary preserved: `results/` 仍 ignored，未纳入 Git tracking。

## In Review

- [ ] 模块 14R-21 review scope 实现项 — 待用户/Web final review。
- [ ] 模块 20B N0 smoke — DONE_PENDING_FINAL_REVIEW。
- [ ] 模块 20B N1 small-scale oracle validation — DONE_PENDING_FINAL_REVIEW。
- [ ] 模块 20B N2 deterministic controlled probe — DONE_PENDING_FINAL_REVIEW。
- [ ] 模块 20B N3 OOD formal execution — DONE_PENDING_FINAL_REVIEW。

## Blocked

- [ ] 外部 dashboard 兼容性 — `C:\Users\22003\paper2\rl-mec-dashboard` 本机仍不可用，需在有该仓库的环境复核。

## Discovered Issues

- Uploaded ignored `results/mainline_a/n2_ablation/summary.json` lacks the newer `evidence_level: deterministic controlled probe` metadata field. This is documented as a non-blocking metadata caveat because tracked runner/docs already enforce the correct N2 evidence level, and `results/` must remain untracked.
- GitHub-side fix cannot mutate local ignored results artifacts. Future local regeneration of N2 results should include the `evidence_level` field.

## Recommendations

- Proceed to final review of `docs/mainline_a_experiment_gate_review.md`.
- Do not write paper main conclusions until final review explicitly closes the review scope.
- Do not add generated `results/` artifacts to Git tracking.
