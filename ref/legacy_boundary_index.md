# Legacy boundary index

## Legacy Role

Legacy environment exists only for explicit fallback and historical reproducibility.

## Allowed

```bash
python scripts/experiment_manager.py start --preset full17 --environment-profile legacy --run-id paper2_full_17_legacy_fallback
python scripts/benchmark.py --all --environment-profile legacy --output results/benchmark_legacy_fallback.json
```

## Not Allowed

- Legacy as default profile.
- Silent fallback from Mainline-A to legacy.
- Mixing legacy results into Mainline-A full-17 ranking.
- Naming legacy result files as Mainline-A result files.
- Restoring old convergence L2/L3 as Mainline-A gate.
