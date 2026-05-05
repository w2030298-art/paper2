# Formal Convergence And Mainline-A Evidence Protocol

> plan.md version: system-model-overhaul-v4.1 | scope: module 14R and module 20

## Legacy L0-L3 Boundary

Old L0/L1/L2/L3 convergence artifacts belong to the pre-overhaul model. They remain useful as a legacy baseline and regression reference, but their conclusions do not carry into the new system model.

| Level | Name | Required Input | Allowed Conclusion |
|-------|------|----------------|--------------------|
| L0 | offline diagnosis | Historical logs, benchmark JSON, quality reports | Locate issues only |
| L1 | 50k single-seed screening | `seed=42`, about 50k steps | pre-overhaul candidate or failure screening |
| L2 | 100k multi-seed candidate validation | `seeds=[42,43,44]`, 100k steps | legacy candidate only |
| L3 | 200k multi-seed formal validation | `seeds=[42,43,44,45,46]`, 200k steps | legacy formal evidence only |

## New Mainline-A Evidence Chain

| Level | Name | Required Input | Allowed Conclusion |
|-------|------|----------------|--------------------|
| N0 | model smoke check | New model enabled, small deterministic run | pipeline correctness only |
| N1 | small-scale oracle comparison | Enumerated oracle for small cases | small-scale optimality comparison |
| N2 | ablation validation | no_price/no_queue/no_channel/no_migration/no_dual/no_cooperation | controlled component evidence |
| N3 | OOD generalization validation | user, edge, mobility, channel, queue shifts | robustness/generalization evidence |

N0-N3 are the new model experiment chain. They do not inherit old L1-L3 conclusions.

## Mainline-A Publication Rule

Main paper claims for the new system model must come from N1/N2/N3. N0 only proves the model path runs. Old `verified_converged_under_protocol` wording is retained only inside legacy/protocol documents and must not be used as a new model result.

## Metric Directions

| Metric | Direction | Gate |
|--------|-----------|------|
| social welfare | higher is better | no material regression across N2/N3 |
| latency | lower is better | compare average and p95 latency |
| energy | lower is better | compare mean energy |
| provider revenue | higher is better | report with fairness and constraints |
| constraint violation rate | lower is better | N3 must not show uncontrolled increase |
| oracle gap | lower is better | N1 small cases only |
