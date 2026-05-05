# docs Directory Contract

`docs/` is the active Web + Codex execution document surface. Active state is restored from `docs/plan.md`, `docs/progress.md`, `docs/report.md`, and incoming `docs/inbox/plan.md`-equivalent files. Historical files under `docs/archive/` are preserved as evidence only and are not used as an automatic restore source.

## Active Files

| Path | Purpose | Rule |
|------|---------|------|
| `docs/plan.md` | active plan | current execution authority |
| `docs/plan-patch.md` | retained patch baseline | do not archive unless explicitly requested |
| `docs/progress.md` | step status | update after task execution |
| `docs/issues.md` | execution issues and notes | append only |
| `docs/report.md` | user/Web execution report | update after each task |
| `docs/convergence_plot_quality.md` | plotting quality contract | active reference |
| `docs/convergence_event_audit.md` | legacy convergence audit gate | retained for old runner compatibility |
| `docs/convergence_publication_gate.md` | publication evidence gate | active |
| `docs/formal_convergence_protocol.md` | legacy and mainline-A evidence protocol | active |
| `docs/legacy_convergence_retirement.md` | old L2/L3 retirement record | active |
| `docs/mainline_a_compatibility_report.md` | legacy/default compatibility report | active |
| `docs/mainline_a_experiment_protocol.md` | N0/N1/N2/N3 experiment protocol | active |
| `docs/mainline_a_publication_gate.md` | new model publication gate | active |
| `docs/paper_revision_pending_questions.md` | paper rewrite questions | active |
| `docs/paper_revision_manifest.md` | writing asset manifest | active |

## Directories

| Path | Purpose | Rule |
|------|---------|------|
| `docs/inbox/` | merge-back input | clear after successful merge |
| `docs/references/` | current technical references | read-mostly, keep only current references |
| `docs/theory/` | theory assets for mainline-A | active docs |
| `docs/archive/` | historical plans, backups, legacy outputs | archive only; do not restore from here automatically |

## Archive Rules

- Use `docs/archive/`, not `docs/.archive/`.
- Preserve historical files instead of deleting them.
- Keep old L1/L2/L3 reports as legacy baseline only.
- Do not restore removed legacy entrypoints or `docs_paper/`.
- Generated experiment outputs under `experiments/`, `results/`, `figures/`, `logs/`, and `checkpoints/` remain outside Git tracking.

## Writing Boundary

Paper writing assets live under `writing_ref/paper2_mainline_a_revision/` and are indexed by `docs/paper_revision_manifest.md`. This plan does not directly edit paper body text.
