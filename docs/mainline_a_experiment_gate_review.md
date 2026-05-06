# Mainline-A Experiment Gate Review

## Status

- Date: 2026-05-06
- Status: REVIEW_CLOSED
- Decision: ACCEPTED_WITH_BOUNDARIES
- Scope: N0/N1/N2/N3 Mainline-A experiment-chain evidence review.
- Review input: tracked docs/code on GitHub plus user-uploaded ignored `results/` JSON artifacts.
- Generated result tracking: unchanged. `results/` remains ignored and must not be added to Git.

## Evidence Level Summary

| Stage | Evidence Level | Status | Gate Decision |
|---|---|---|---|
| N0 | smoke evidence | ACCEPTED_WITH_BOUNDARIES | Accepted as smoke only |
| N1 | small-scale oracle evidence | ACCEPTED_WITH_BOUNDARIES | Accepted |
| N2 | deterministic controlled probe only | ACCEPTED_WITH_BOUNDARIES | Accepted as probe only |
| N3 | OOD formal execution evidence | ACCEPTED_WITH_BOUNDARIES | Accepted |

## Uploaded Results Review

### N0 Smoke

- Artifact reviewed: `results/mainline_a/n0_smoke/benchmark.json`.
- Status: ok.
- Scope: single-seed MAPPO smoke path; not full benchmark evidence.
- Key observed metrics: final reward, latency, energy, deadline miss rate, throughput, convergence metadata.

### N1 Small-Scale Oracle

- Artifacts reviewed: `results/mainline_a/n1_oracle/summary.json`, `oracle_gap_report.json`.
- Status: ok.
- Case matrix: 18 cases.
- Record count: 54.
- Seeds: `[42, 43, 44]`.
- Users: `[2, 3, 4]`.
- Edges: `[2, 3]`.
- Result: `game_aware_pd_marl` mean oracle gap is lower than MAPPO and baseline in the uploaded summary.

### N2 Deterministic Controlled Probe

- Artifacts reviewed: `results/mainline_a/n2_ablation/summary.json`, `ablation_records.json`, `metric_deltas.json`.
- Status: accepted as deterministic controlled probe only.
- Record count: 27.
- Ablation count: 9.
- Seeds: `[42, 43, 44]`.
- Steps: 50000.
- Checks: required metrics present, schema consistent, finite metrics, explicit switches, non-identical metrics, non-full ablations differ from `full_model`.
- Boundary: N2 is not training-grade or publication-grade ablation evidence.
- Metadata caveat: the uploaded ignored `results/mainline_a/n2_ablation/summary.json` predates the `evidence_level` field and does not include `evidence_level: deterministic controlled probe`. This is not fixed in Git because `results/` is intentionally ignored. The tracked runner and docs already enforce the correct evidence-level wording, and a future local regeneration of ignored results should include the field.

### N3 OOD Formal Execution

- Artifacts reviewed: `results/mainline_a/n3_ood/summary.json`, `ood_records.json`, `metric_summary.json`, `distribution_shift.json`, plus preflight artifacts.
- Status: ok.
- Evidence level: OOD formal execution.
- Formal record count: 6.
- Preflight record count: 2.
- Seeds: `[42, 43, 44]`.
- Steps: 50000.
- Required metrics: `social_welfare`, `average_latency`, `p95_latency`, `energy`, `provider_revenue`, `constraint_violation_rate`, `jain_fairness`, `oracle_gap_small_cases`.
- OOD test distribution applied: users 40, edges 6, high mobility, `3gpp_lite` channel, parallel queue, cooperation enabled.
- Audit: empty results no; required metrics present; schema consistent; finite metrics; metrics not all identical; OOD distribution applied; benchmark alias overwrite false.

## Gate Decision

Mainline-A experiment chain passes artifact-level review with one non-blocking metadata caveat: the uploaded ignored N2 summary lacks the newer `evidence_level` field. This does not change N2's allowed interpretation because the tracked runner and tracked docs already define N2 as deterministic controlled probe only.

Final review is closed as artifact-level acceptance with boundaries. The evidence levels remain N0 smoke evidence, N1 small-scale oracle evidence, N2 deterministic controlled probe only, and N3 OOD formal execution evidence. These results must not be promoted to formal benchmark conclusions or over-interpreted beyond those levels.

## Constraints Still Active

- Do not add `results/` to Git tracking.
- Do not upgrade N2 to training-grade or publication-grade ablation evidence.
- Do not claim global convergence.
- Do not promote artifact-level evidence into stronger formal benchmark conclusions.
- External dashboard compatibility remains unresolved and must be handled in the dashboard environment.
