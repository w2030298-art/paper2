# Mainline-A Experiment Gate Review

## Status

- Date: 2026-05-06
- Status: REVIEW_CLOSED
- Decision: ACCEPTED_WITH_BOUNDARIES
- Scope: N0/N1/N2/N3 Mainline-A experiment-chain evidence review.
- Generated result tracking: unchanged. `results/` remains ignored and must not be added to Git.

## Evidence Level Summary

| Stage | Evidence Level | Status | Allowed Interpretation |
|---|---|---|---|
| N0 | smoke evidence | ACCEPTED_WITH_BOUNDARIES | Smoke path works |
| N1 | small-scale oracle evidence | ACCEPTED_WITH_BOUNDARIES | Small-scale oracle comparison |
| N2 | deterministic controlled probe only | ACCEPTED_WITH_BOUNDARIES | Controlled probe, not training-grade ablation |
| N3 | OOD formal execution evidence | ACCEPTED_WITH_BOUNDARIES | Formal OOD execution evidence |

## Constraints Still Active

- Do not add `results/` to Git tracking.
- Do not upgrade N2 to training-grade or publication-grade ablation evidence.
- Do not claim global convergence.
- Do not promote artifact-level evidence into stronger formal benchmark conclusions.
- v4.3 full-17 Mainline-A benchmark must produce its own result file before algorithm ranking is used.
- External dashboard compatibility remains unresolved and must be handled in the dashboard environment.
