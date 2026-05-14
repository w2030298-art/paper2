# Paper2 v4.9 Execution Preflight

This note is reference material only. `.ai/ledger.json` remains the single machine state source.

## Runtime Inventory

- Timestamp (UTC): `2026-05-13T12:55:34Z`
- OS: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`
- Python executable: `.tmp/stage1-venv/bin/python`
- Python version: `3.12.3`
- Repo-local venv prefix: `/mnt/c/Users/22003/paper2/paper2/.tmp/stage1-venv`
- `torch`: `2.11.0+cu130`
- `gymnasium`: `1.3.0`
- `numpy`: `2.4.4`
- `scipy`: `1.17.1`
- CUDA availability: `false`
- CUDA device count probe: `1`
- CUDA version: `13.0`
- CUDA note: PyTorch warns that the NVIDIA driver is too old, so v4.9 local execution must use the CPU-compatible path.
- Disk free space at repo root: `211.37 GB`
- Graphify current state: `graphify-out/graph.json` present with `1218` nodes and `2620` links.

## Expected Run Volume

The v4.9 repaired matrix is expected to expand N2 ablations as executable dimensions:

| Level | Runs Per Seed | Seeds | Steps Per Run | Total Runs | Total Environment Steps |
|---|---:|---:|---:|---:|---:|
| `L1_screening` | 72 | 1 | 50000 | 72 | 3600000 |
| `L2_candidate_validation` | 72 | 3 | 100000 | 216 | 21600000 |
| `L3_formal_verification` | 72 | 5 | 200000 | 360 | 72000000 |

## Checkpoint Baseline

- Current `.ai/checkpoint.json`: `checkpoint-full17-backup-figures-20260513-pass`
- v4.8 checkpoint is still recorded in `.ai/ledger.json` under `previous_task_before_full17_backup_figures.checkpoint_id` as `checkpoint-v4.8-final-experiment-closeout-pass`.
- This means the literal Step 1 checkpoint check is not clean: the active checkpoint is a later direct-dispatch figure checkpoint, not the v4.8 checkpoint.

## Inbox Packet

- `.ai/inbox/plan.md` exists and contains the v4.9 packet with task id `paper2-v4.9-real-l2-l3-formal-closeout`.
- No stale v4.8 inbox packet was observed.

## Frozen Mainline-A Core Boundary

Frozen files from the v4.9 packet:

- `src/environments/mec_v3/game_theory_env.py`
- `configs/system_model_mainline_a.yaml`
- `configs/pricing_dynamic_mainline_a.yaml`

Current pre-patch diff check:

```text
git diff --name-only -- src/environments/mec_v3/game_theory_env.py configs/system_model_mainline_a.yaml configs/pricing_dynamic_mainline_a.yaml
src/environments/mec_v3/game_theory_env.py
```

`src/environments/mec_v3/game_theory_env.py` is already dirty before this patch, with a diff stat of
`1570 insertions(+), 1570 deletions(-)`. This patch will not modify or revert that file. Because the
formal verification plan requires a clean frozen-core boundary, real L2/L3 paper-level claims must stay
blocked unless the operator accepts this pre-existing frozen-core state or restores it outside this patch.

## Preflight Decision

- Tooling repair, collector implementation, statistics extension, and dry-run contract validation can proceed.
- Real L2/L3 execution for paper-level claims is preflight-blocked in this workspace by the dirty frozen-core file and CPU-only execution burden unless the operator explicitly accepts the current state and resources.
