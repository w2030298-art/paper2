# Paper2 Final Delivery Manifest

This manifest is review reference material. `.ai/ledger.json` remains the machine state source.

## Required Deliverables

| Deliverable | Source Path | Generation Command | Required Input Data | Claim Status |
|---|---|---|---|---|
| Main experiment matrix | `configs/paper2_main_experiment_matrix.yaml` | Manual config from v4.8 inbox plan | Stage-2 roadmap, final adaptation record, formal matrix | Design ready; requires human review |
| Scenario matrix | `configs/paper2_scenario_matrix.yaml` | Manual config from v4.8 inbox plan | N0/N1/N2/N3 references | Design ready; requires human review |
| L1 dry-run manifest | `results/paper2_main_matrix/dry_run_manifest.json` | `python scripts/run_paper2_main_matrix.py --matrix configs/paper2_main_experiment_matrix.yaml --scenario-matrix configs/paper2_scenario_matrix.yaml --level L1_screening --dry-run --output results/paper2_main_matrix/dry_run_manifest.json` | Matrix configs only | Planned evidence only |
| Statistics plan | `results/paper2_statistics/stats_plan.json` | `python scripts/analyze_paper2_statistics.py --input results/paper2_main_matrix/dry_run_manifest.json --output-dir results/paper2_statistics --dry-run` | Dry-run manifest | Planned evidence only |
| Real statistics exports | `results/paper2_statistics/*.csv`, `statistics_report.json` | `python scripts/analyze_paper2_statistics.py --input results/paper2_main_matrix/results.json --output-dir results/paper2_statistics` | Real complete multi-seed L2/L3 results | Not claimed until real data exists |
| Figure plan/final figures | `figures/paper2_final/` | `python scripts/generate_paper2_figures.py --results results/paper2_main_matrix/results.json --statistics results/paper2_statistics/statistics_report.json --output-dir figures/paper2_final` | Real result and statistics exports | Dry-run plan only until real data exists |
| Final package manifest/zip | `paper2_final_delivery_package.zip` | `python scripts/export_paper2_final_package.py --output paper2_final_delivery_package.zip` | Present configs, refs, scripts, results, figures | Zip is only final when real outputs are included |

## Source Path

Primary source paths are `configs/`, `scripts/`, `results/paper2_main_matrix/`, `results/paper2_statistics/`,
`figures/paper2_final/`, and `ref/`.

## Generation Command

Use the commands in the table as the mechanical contract. Dry-run commands validate wiring; non-dry-run
commands require operator approval for L2/L3 workloads.

## Required Input Data

Paper-level claims require complete L3 formal-verification result records for the selected algorithms,
scenarios, and seeds. Dry-run manifests are not result data.

## Claim Status

This v4.8 patch closes tooling and delivery readiness. It does not claim statistical superiority until
real multi-seed data is analyzed and reviewed.
