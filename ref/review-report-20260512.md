# Review Report — 2026-05-12

## Scope
- Reviewed repository: `w2030298-art/paper2`
- Reviewed commit: `d6aad93e2f5c9f7a9d0e41214c22efdbaf74668b`
- Latest change focus:
  - Mainline-A resolved runtime config repair
  - PPO/COMA boundary alignment
  - COMA externalized hyperparameters
  - Stage-1 PPO/COMA search configs and tuner
  - `.ai` v9 state protocol compliance
  - Generated `outputs/stage1/*` artifacts

## Verdict
NEEDS_WORK

## Findings

| Severity | Area | Finding | Evidence | Recommendation |
|---|---|---|---|---|
| High | Stage-1 artifact correctness | `ppo_best_config.yaml` and `coma_best_config.yaml` are written even when every starter trial failed. The files contain trial `0000` recommended-start configs, but no successful trial exists, so calling them `best_config` is materially misleading. | `outputs/stage1/ppo_trials.csv` and `outputs/stage1/coma_trials.csv` show all four trials failed with `ModuleNotFoundError: No module named 'torch'`; Pareto CSVs contain only headers; best config files still contain PPO-B / COMA-L trial `0000` configs. | Treat current best files as `recommended_start_config` or diagnostic artifacts until at least one successful trial exists. A no-success run should emit `status: no_successful_trial` or withhold `*_best_config.yaml`. |
| High | Validation gate | The task acceptance is not fully satisfied because actual training and full pytest were not executed successfully. The ledger correctly records `PARTIAL`, but the project should not be considered all-clear. | `.ai/ledger.json` records focused tests passed, but `train.py` and full `pytest -q` failed due missing `torch/gymnasium`. | Keep state as `NEEDS_REVIEW` or `BLOCKED`, not `ALL_CLEAR`; rerun starter pilot and full pytest in an environment with project dependencies installed before promoting tuned configs. |
| Medium | Runtime audit completeness | `search_audit.json` has `action_space: null` and `agent_runtime: null` for both PPO and COMA because no training run produced `resolved_runtime_config`. This is expected under dependency failure, but it means the key runtime-profile repair has not been exercised end-to-end. | `outputs/stage1/search_audit.json` records null runtime details for both algorithms. | Require at least one successful smoke/starter trial per algorithm before treating the runtime audit path as validated. |
| Medium | Status protocol consistency | `AGENTS.md` was partly migrated to `.ai/ledger.json`, but stale references to `docs/progress.md`, `docs/plan.md`, and `docs/issues.md` remain in later sections. This can cause future execution agents to update obsolete status files. | Top-level project metadata now names `.ai/ledger.json` as state source, but later sections still instruct progress updates to docs files. | Normalize `AGENTS.md` so all state/progress/problem reporting points only to `.ai/ledger.json`; `.ai/inbox/plan.md` remains the transient execution packet. |
| Medium | Search method semantics | The tuner labels sampling as Optuna TPE when Optuna exists, but it does not feed measured objective values back into a persistent study and does not implement Hyperband/ASHA pruning. This is acceptable for starter config generation, but not for claiming adaptive BO/pruning. | `scripts/tune_mainline_a_stage1.py` creates a fresh study in `_sample_with_optuna`, uses synthetic `study.tell(..., trial_index)`, and has no pruner/resource ladder. | Rename current behavior as `starter/random+recommended` or implement persistent Optuna study with actual `j_phys` feedback and pruning before broad/narrow search. |
| Low | Scope control | No repository hits were found for Stage-2 terms such as `CL-PPO`, `GAM-COMA`, or `stage2`. This matches the requested scope boundary. | Repository search returned no Stage-2 artifacts. | Keep this boundary; do not introduce later-stage algorithm work in this task. |

## Architecture Notes
- The runtime metadata path is structurally sound: `runtime_config.py` captures config hashes, CLI overrides, environment profile, env overrides, game-theory config, trainer kwargs, action/observation spaces, and agent runtime metadata.
- `train.py` embeds `resolved_runtime_config` into result JSON after successful training. This is the right integration point, but it remains unproven in the current environment because `torch` import fails before execution reaches the training/evaluation path.
- COMA parameter externalization is coherent: YAML exposes `policy_clip_low/high`, `grad_clip`, `critic_loss_coeff`, and `entropy_coeff`; `benchmark.py::create_agent()` passes these into `COMAAgent`; `COMAAgent.update()` uses them in surrogate clipping, gradient clipping, value loss, and entropy regularization.
- PPO/COMA config boundaries now both align to `100000` timesteps, matching the Mainline-A full17 boundary.
- The tuner correctly materializes trial configs and marks failed trials as infeasible / non-Pareto. The weak point is downstream artifact naming and best-config behavior when the result set has zero successful rows.

## Validation Notes
- Accepted evidence:
  - Focused tests: `14 passed`
  - Static config assertions: passed
  - Tuner dry-run: passed
  - Artifact validation: passed
  - `py_compile`: passed
- Not accepted as complete:
  - `train.py` execution: failed due missing `torch`
  - PPO starter pilot: all trial subprocesses failed due missing `torch`
  - COMA starter pilot: all trial subprocesses failed due missing `torch`
  - Full `pytest -q`: failed during collection due missing `torch` and `gymnasium`
- Current `outputs/stage1/*` are diagnostic artifacts, not empirical tuning results.

## Ledger Recommendation
```json
{
  "state": "NEEDS_REVIEW",
  "validation": {
    "status": "PARTIAL",
    "summary": "Focused tests and static artifact checks passed, but training execution and full pytest remain unvalidated because the current environment lacks torch/gymnasium."
  },
  "open_items": [
    "STAGE1-RERUN: rerun PPO/COMA starter pilot in an environment with torch/gymnasium installed; require at least one successful result_json per algorithm before accepting runtime audit and best configs.",
    "STAGE1-BEST-CONFIG-GATE: do not label recommended-start configs as best_config when all trials failed; rename or gate best-config emission.",
    "AGENTS-V9-CLEANUP: remove stale docs/progress.md, docs/plan.md, and docs/issues.md instructions from AGENTS.md.",
    "TUNER-SEMANTICS: either implement real persistent Optuna/Hyperband behavior for broad/narrow search or label current sampler as non-adaptive starter sampling."
  ]
}
```

## Report Backup
建议保存为: `ref/review-report-20260512.md`
