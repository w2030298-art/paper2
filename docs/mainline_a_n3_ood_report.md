# Mainline-A N3 OOD Formal Execution Report

## Status

- Date: 2026-05-06
- Status: N3_DONE_PENDING_REVIEW
- Scope: N3 OOD formal execution only; N0/N1/N2 were not rerun and full 17 was not run.
- N2 boundary: N2 remains a deterministic controlled probe only; it is not upgraded to training-grade or publication-grade ablation evidence.
- Config: `configs/experiments/mainline_a_n3_ood.yaml`
- Output directory: `results/mainline_a/n3_ood/`
- Benchmark alias: `results/benchmark.json` was not created or overwritten.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n3_ood.yaml --stage N3 --dry-run --results-root results\mainline_a
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n3_ood.yaml --stage N3 --preflight --preflight-steps 256 --results-root results\mainline_a
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n3_ood.yaml --stage N3 --results-root results\mainline_a
```

## Train/Test Distribution

| Field | Train | Test | Changed |
|---|---:|---:|---|
| users | 20 | 40 | yes |
| edges | 4 | 6 | yes |
| mobility | medium | high | yes |
| channel | analytic | 3gpp_lite | yes |
| queue_model | parallel | parallel | no |
| cooperation_enabled | false | true | yes |

The OOD test split used `3gpp_lite` channel, high mobility, parallel queue, and cooperation enabled.

## Required Metrics

All required metrics were produced for train and OOD test records:

- social_welfare
- average_latency
- p95_latency
- energy
- provider_revenue
- constraint_violation_rate
- jain_fairness
- oracle_gap_small_cases

## Formal Metric Means

| Metric | Train Mean | OOD Test Mean | Test - Train |
|---|---:|---:|---:|
| social_welfare | 1.059990 | 0.888154 | -0.171836 |
| average_latency | 0.150080 | 0.220540 | 0.070460 |
| p95_latency | 0.167084 | 0.244796 | 0.077712 |
| energy | 1.402500 | 1.774100 | 0.371600 |
| provider_revenue | 1.041836 | 1.062579 | 0.020743 |
| constraint_violation_rate | 0.042500 | 0.091033 | 0.048533 |
| jain_fairness | 0.999832 | 0.999577 | -0.000255 |
| oracle_gap_small_cases | 0.076533 | 0.113013 | 0.036480 |

## Audit

- Empty results: no; formal run wrote 6 records.
- Required metrics missing: no.
- NaN/Inf: no.
- Schema mismatch: no.
- All metrics identical: no.
- Train/test distribution recorded: yes.
- OOD test distribution applied: yes.
- `results/benchmark.json` alias creation/overwrite: no.

## Artifacts

- `results/mainline_a/n3_ood/distribution_shift.json`
- `results/mainline_a/n3_ood/ood_records.json`
- `results/mainline_a/n3_ood/metric_summary.json`
- `results/mainline_a/n3_ood/summary.json`
- `results/mainline_a/n3_ood/preflight/distribution_shift.json`
- `results/mainline_a/n3_ood/preflight/ood_records.json`
- `results/mainline_a/n3_ood/preflight/metric_summary.json`
- `results/mainline_a/n3_ood/preflight/summary.json`
