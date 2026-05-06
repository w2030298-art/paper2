# docs contract

This directory holds active execution state for the paper2 workspace.

- `docs/plan.md`: active plan and current status.
- `docs/progress.md`: execution progress and review state.
- `docs/report.md`: latest execution report for the user/Web handoff.
- `docs/issues.md`: issues, fixed review items, and escalation notes.
- `docs/inbox/plan.md`: Web-produced replacement plan input. Merge-back moves an accepted inbox plan into `docs/plan.md`.
- `docs/inbox/`: review reports and incoming Web artifacts that have not been merged or archived yet.
- `docs/references/`: read-only implementation references and audit notes.
- `docs/archive/`: historical plans, old review reports, and retired generated-analysis notes.

Generated experiment outputs belong outside active docs. Do not place backup markdown files in the docs root; archive them under `docs/archive/`.
