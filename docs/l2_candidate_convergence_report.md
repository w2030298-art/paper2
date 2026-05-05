# L2 Candidate Convergence Report

> evidence level: `L2` | status: running | plan.md version: slimming-plan-v3 | run id: `l2_20260504_171744`

## Scope

L2 requires 100k steps with `seeds=[42,43,44]`. It is the first multi-seed candidate gate and must bind raw plots, clean plots, and a quality report to the same run metadata.

L2 training has been started as a long-running background job. Until `results/l2_candidate_convergence_report.json` is replaced by completed L2 decisions, the current formal evidence remains L1-only.

## Active Run

| Field | Value |
|-------|-------|
| process id | `26860` |
| run id | `l2_20260504_171744` |
| algorithms | `COMA`, `MAPPO`, `TRPO`, `IQL`, `VDN`, `IPPO`, `MADDPG` |
| deferred algorithms | `A3C`, `MATD3`, `SAC`, `GRPO` |
| steps | `100000` |
| seeds | `42`, `43`, `44` |
| manifest | `experiments/formal_convergence/l2/l2_20260504_171744/manifest.json` |
| stdout | `logs/formal_convergence/l2_20260504_171744.out.log` |
| stderr | `logs/formal_convergence/l2_20260504_171744.err.log` |

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

No algorithm is currently eligible for the paper main convergence claim from L2 data. Passing L2 can only create `candidate_converged_under_protocol`; L3 remains required before any main conclusion. The active runner was launched with `--auto-l3`, so only L2-passing algorithms may enter L3 after L2 completes.
