# Paper2 Final Adaptation Checklist

This checklist is reference material for final human review.

- Algorithm registration: PPO, COMA, CL-PPO, GAM-COMA, baselines.
- Config coverage: algorithm configs, main matrix, scenario matrix, formal convergence matrix.
- Experiment execution: L1/L2/L3 commands and outputs.
- Statistical verification: seed completeness, CI, p-value/effect-size outputs, multiple-comparison correction.
- Figure package: convergence, final comparison, ablation, OOD, statistics.
- State protocol: ledger/checkpoint only, inbox consumed, no README dashboard.

## Review Notes

- L1 dry-runs validate command and schema readiness only.
- L2/L3 runs remain explicit operator actions and must not be inferred from manifests.
- `DASHBOARD-EXTERNAL-COMPAT` remains external until validated in the dashboard environment.
- Frozen Mainline-A core files are not part of this closeout patch.
