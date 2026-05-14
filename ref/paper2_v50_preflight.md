# Paper2 v5.0 Preflight

This note is reference material only. `.ai/ledger.json` remains the single machine state source.

## Frozen Mainline-A Core

Frozen files checked:

- `src/environments/mec_v3/game_theory_env.py`
- `configs/system_model_mainline_a.yaml`
- `configs/pricing_dynamic_mainline_a.yaml`

Validation:

```text
git diff --name-only -- src/environments/mec_v3/game_theory_env.py configs/system_model_mainline_a.yaml configs/pricing_dynamic_mainline_a.yaml
src/environments/mec_v3/game_theory_env.py
```

```text
git diff --ignore-space-at-eol -- src/environments/mec_v3/game_theory_env.py configs/system_model_mainline_a.yaml configs/pricing_dynamic_mainline_a.yaml >/tmp/paper2_v50_frozen_core_semantic.diff
wc -c /tmp/paper2_v50_frozen_core_semantic.diff
0 /tmp/paper2_v50_frozen_core_semantic.diff
```

Decision: proceed. The frozen core remains name-only dirty due line-ending or end-of-line whitespace differences, but there is no semantic diff under the required check. This patch does not modify the frozen core.

## Runtime

- Python executable: `.tmp/stage1-venv/bin/python`
- `torch`: `2.11.0+cu130`
- `gymnasium`: `1.3.0`
- `numpy`: `2.4.4`
- `scipy`: `1.17.1`
- CUDA availability: `false`
- CUDA device count probe: `1`
- Disk free at repo root: `195.24 GB`
- Graphify state: existing `graphify-out/graph.json` with `1218` nodes and `2620` links; query over benchmark/trainer/algorithm entrypoints completed.

## Experiment Boundary

The v5.0 experiment is allowed to proceed locally. The run remains a single-seed 100K engineering comparison, not statistical significance evidence.
