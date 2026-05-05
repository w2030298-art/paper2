# Convergence Publication Gate

> plan.md version: system-model-overhaul-v4.1 | evidence policy: new system model gates only

## Evidence Levels

| Level | Source | Allowed Placement |
|-------|--------|-------------------|
| legacy_pre_overhaul | Old L1/L2/L3 convergence artifacts from the pre-overhaul model | appendix, debug note, or regression reference only |
| N0 | Mainline-A model smoke check | pipeline correctness only |
| N1 | Mainline-A small-scale oracle comparison | small-scale optimality comparison |
| N2 | Mainline-A controlled ablation | ablation evidence |
| N3 | Mainline-A OOD generalization | robustness/generalization evidence |

## Gate Rule

Old L1/L2/L3 results are now `legacy_pre_overhaul`. They may document the pre-overhaul convergence baseline, but they must not be used as a new system model main claim. New paper main figures and main conclusions must come from module 20 mainline-A experiments under the new system model.

## Current Decision

| Artifact Family | Current Evidence | Publication Placement | Reason |
|-----------------|------------------|-----------------------|--------|
| Old L1/L2/L3 convergence | legacy_pre_overhaul | appendix/debug/regression reference only | system model overhaul changes observation, reward, pricing, and experiment gates |
| Mainline-A N0 | pending | not a main conclusion | smoke correctness only |
| Mainline-A N1 | pending | pending | oracle comparison not run |
| Mainline-A N2 | pending | pending | ablation not run |
| Mainline-A N3 | pending | pending | OOD generalization not run |

## Main Figure Eligibility

No new system model result is eligible for a main figure until the module 20 N1/N2/N3 chain has produced reviewed evidence.
