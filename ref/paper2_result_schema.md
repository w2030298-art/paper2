# Paper2 Matrix Result Schema

This note documents the v4.9 collector contract. It is reference material only.

## Accepted Per-Run Inputs

`scripts/collect_paper2_matrix_results.py` reads a v4.9 manifest plus one output JSON per manifest run.
The per-run output may be:

- A benchmark list payload containing one or more result objects.
- A dictionary with `results`, `records`, or `runs` containing result objects.
- A single result object.

The collector selects the record matching the manifest `algorithm` and `seed` when possible.

## Normalized Output

The output JSON uses schema version `paper2-matrix-results/v1` and contains `results` records with:

- `scenario`
- `stage`
- `algorithm`
- `seed`
- `steps`
- `level`
- `ablation`
- `result_kind`
- `output_path`
- `metrics`

The metric map may include:

- `social_welfare_mean`
- `reward_mean`
- `e2e_latency_mean`
- `e2e_latency_p95`
- `deadline_miss_rate`
- `energy_total_mean`
- `throughput_tasks_per_step`
- `agent_reward_jain_mean`
- `constraint_violation_rate`
- `queue_wait_mean`
- `comm_score`

## Strict Mode

With `--strict`, collection fails for:

- Dry-run manifests.
- Missing per-run outputs.
- Failed per-run source status.
- Missing required metrics.
- Incomplete seed grids within a scenario/stage/ablation group.

Strict mode is required for v4.9 L2/L3 claim-path collection.
