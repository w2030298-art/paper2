# Paper2 v4.9 Final Claim Boundary

This file records the claim boundary for the current v4.9 execution attempt.

## Supported Claims

- The v4.9 runner now expands N2 CL-PPO/GAM-COMA ablations as real matrix dimensions.
- The runner manifest includes execution metadata needed for strict collection and review.
- CL-PPO ablation switches for constraint signal, risk critic, and safety layer are reflected in generated configs.
- GAM-COMA ablation switches for graph attention, feasible action masking, warm start, and Shapley credit are reflected in generated configs.
- `GAMCOMAAgent(graph_attention.enabled=false)` uses a non-attention centralized critic fallback.
- The collector rejects dry-run manifests, missing outputs, failed runs, and incomplete seed grids in strict mode.
- The statistics analyzer supports multi-metric real-result analysis and continues to reject inferential statistics from dry-run manifests.

## Unsupported Claims

- No L2 candidate-validation performance ranking is supported because real L2 outputs were not produced and strictly collected.
- No L3 formal-verification statistical result is supported because L3 was not executed.
- No p-value, confidence interval, or effect-size claim is supported from the dry-run manifest.
- No N1 oracle-gap claim is supported because local N1 records are marked `benchmark_diagnostic`, not oracle outputs.
- `DASHBOARD-EXTERNAL-COMPAT` remains outside the repo-level claim set.

## Boundary Reason

Formal L2/L3 execution is blocked in this workspace by:

- Pre-existing dirty frozen-core file: `src/environments/mec_v3/game_theory_env.py`
- CPU-only runtime for 216 L2 runs and 360 L3 runs
- Absence of complete real multi-seed L2/L3 result records

The ledger should remain `NEEDS_WEB` until the operator resolves or accepts these blockers.
