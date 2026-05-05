# L2 Candidate Convergence Report

> evidence level: `L2` | status: not executed | plan.md version: slimming-plan-v3

## Scope

L2 requires 100k steps with `seeds=[42,43,44]`. It is the first multi-seed candidate gate and must bind raw plots, clean plots, and a quality report to the same run metadata.

No L2 training was started in this patch. The current evidence remains L1-only, based on `results/convergence_validation_baseline_50k.json` with `seed=42`.

## Inputs From L1

| Group | Algorithms | L2 Entry Condition |
|-------|------------|--------------------|
| L1 candidates | COMA, MAPPO, TRPO | May enter L2 without override, but still needs 100k multi-seed evidence |
| Needs event audit | IQL, VDN, IPPO, MADDPG | May enter L2 only because event audit classified current failures as training instability |
| Needs single-variable fix | A3C, MATD3, SAC, GRPO | Requires a specific `override_id` from `configs/formal_single_variable_fixes.yaml` before L2 |

## Required Run Metadata

Each L2 run must record:

- `evidence_level=L2`
- `run_id`
- `seed_set=[42,43,44]`
- `config_hash`
- `override_id` or `none`
- raw plot path
- clean plot path
- quality report path

## Current Decision

No algorithm is currently eligible for the paper main convergence claim from L2 data. Passing L2 can only create `candidate_converged_under_protocol`; L3 remains required before any main conclusion.
