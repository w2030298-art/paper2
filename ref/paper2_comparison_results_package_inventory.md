# paper2 Comparison Results Package Inventory

Generated: 2026-05-20 14:02:28 +08:00

Package: `paper2_comparison_experiment_results_20260520.zip`

Size bytes: `1790712`

SHA256: `28612371f5e445eeea71a0b988fcc3bad0faa94c768c075d13a5e6c2e797a407`

Archive entry count: `49`

The archive has a single top-level directory:

```text
paper2_comparison_experiment_results_20260520/
```

Included evidence:

- `paper2_full_17_mainline_a_export.zip`: tracked full17 Mainline-A comparison export package.
- `results/`: corrected single-policy 3-user raw JSON, summary CSV, and final algorithm screening decisions.
- `figures/`: corrected single-policy 3-user figures and figure manifest.
- `outputs/stage1/`: PPO/COMA Stage-1 comparison/tuning trial outputs.
- `logs/`: official GPU benchmark logs plus quarantined foreground pipe-failure audit logs.
- `ref/`: regenerated report, decision, v5 GPU artifact inventory, and CL-PPO freeze note.
- `configs/`: benchmark config used for the corrected GPU run.
- `.ai/`: ledger and checkpoint snapshots at packaging time.

Validation:

```text
tar -tf paper2_comparison_experiment_results_20260520.zip
Get-FileHash -Algorithm SHA256 paper2_comparison_experiment_results_20260520.zip
```
