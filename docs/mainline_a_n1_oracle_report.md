# Mainline-A N1 Oracle Report

## Status

- Date: 2026-05-05
- Status: N1_DONE_PENDING_REVIEW
- Scope: N1 small-scale oracle validation only; N2/N3 were not started.
- Config: `configs/experiments/mainline_a_n1_oracle.yaml`
- Output directory: `results/mainline_a/n1_oracle/`

## Command

```powershell
.\.venv\Scripts\python.exe scripts\run_mainline_a_experiments.py --config configs\experiments\mainline_a_n1_oracle.yaml --stage N1 --results-root results\mainline_a
```

## Case Matrix

- Seeds: 42, 43, 44
- Users: 2, 3, 4
- Edges: 2, 3
- Case count: 18
- Oracle bounds check: passed; max users=4 and max edges=3.
- Policies compared as deterministic N1 oracle labels: `baseline_static_stackelberg`, `MAPPO`, `game_aware_pd_marl`
- Record count: 54

## Artifacts

- `results/mainline_a/n1_oracle/case_matrix.json`
- `results/mainline_a/n1_oracle/oracle_gap_report.json`
- `results/mainline_a/n1_oracle/summary.json`

## Summary Metrics

| Policy label | Mean oracle gap | Max oracle gap | Mean constraint violation | Max constraint violation | Total runtime (s) |
|---|---:|---:|---:|---:|---:|
| baseline_static_stackelberg | 1.4303333333 | 1.7806666667 | 0.5 | 0.6666666667 | 0.0052845478 |
| MAPPO | 0.165 | 0.432 | 0.0 | 0.0 | 0.0052392483 |
| game_aware_pd_marl | 0.0288888889 | 0.112 | 0.0 | 0.0 | 0.006002903 |

Overall runtime: 0.0085260868 s.

## Safety Checks

- Oracle gap missing: no; all 54 records include `oracle_gap`.
- Oracle gap NaN/Inf: no.
- Constraint violation NaN/Inf: no.
- Case matrix out of bounds: no; `num_users <= 4` and `num_edges <= 3`.
- Benchmark alias overwrite: no; N1 used `run_mainline_a_experiments.py` and did not write `results/benchmark.json`.
- N2/N3/full 17: not started.

## Notes

- This report is an N1 small-scale optimality comparison artifact. It is not N2 ablation, N3 OOD, or a full benchmark ranking.
- The MAPPO and `game_aware_pd_marl` entries are deterministic N1 oracle-comparison policy labels, not full training runs.
