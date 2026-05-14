# Paper2 v4.9 L2 Candidate Validation Report

This report is reference material only. `.ai/ledger.json` remains the state source.

## Status

- L2 candidate validation was not executed in this workspace.
- Strict L2 collection was not attempted against real per-run outputs because the preflight boundary failed before formal evidence execution.

## Blockers

1. Frozen Mainline-A core is not clean before this patch:
   - `src/environments/mec_v3/game_theory_env.py`
   - Diff stat: `1570 insertions(+), 1570 deletions(-)`
   - This patch did not modify or revert that file.
2. Local execution is CPU-only for this run:
   - `torch.cuda.is_available() == false`
   - PyTorch reports the NVIDIA driver is too old for the installed CUDA build.
3. The repaired L2 matrix has 216 planned runs and 21600000 total environment steps.

## Evidence Generated Before Stop

- `results/paper2_main_matrix/v49_l1_contract_manifest.json`
  - `run_count`: 72
  - N2 ablations expanded as real dimensions.
  - Required run metadata present.
- `results/paper2_statistics/v49_plan/stats_plan.json`
  - Dry-run statistics plan only.
  - No p-values, confidence intervals, or effect sizes generated.

## L3 Gate

L3 must not proceed from this workspace state. The correct state is `NEEDS_WEB` until an operator either:

- Accepts/restores the frozen-core boundary, and
- Provides sufficient execution resources or precomputed complete L2 outputs for strict collection.
