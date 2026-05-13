# Paper2 Stage-2 Residuals Closeout

This note is reference material only. `.ai/ledger.json` remains the single machine state source.

## Closed In v4.7

- PPO and COMA defaults were synchronized with the Stage-1 best artifacts.
- CL-PPO and GAM-COMA agent/config registrations were added and validated through synthetic update and benchmark-registration tests.
- Stage-1 regression checks, Stage-2 registration tests, py_compile, and full pytest passed in the repo-local stage1 venv during the v4.7 closeout.

## Carried Into v4.8

- Main experiment design needs explicit L1/L2/L3 evidence levels and a dry-run-safe runner.
- Scenario coverage needs ID, N1 oracle, N2 ablation, and N3 OOD stress matrices.
- Multi-seed statistical validation needs a deterministic pipeline that distinguishes dry-run manifests from real results.
- Final figures, export packaging, delivery manifest, and adaptation checklist need mechanical dry-run validation.
- VSCode launch entries need to expose matrix, statistics, figure, and export workflows.

## External / Not Locally Closable

- `DASHBOARD-EXTERNAL-COMPAT` remains open because the external dashboard repository/environment was not validated in this workspace.

## Do Not Reopen

- Do not reopen PPO/COMA/CL-PPO/GAM-COMA core algorithm logic in this patch without a failing test that proves a narrow compatibility defect.
- Do not modify frozen Mainline-A core files for experiment design, statistics, figure, or export work.
- Do not treat L1 dry-run or manifest artifacts as paper-level statistical evidence.
