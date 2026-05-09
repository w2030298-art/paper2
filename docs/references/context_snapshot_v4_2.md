# v4.2 context snapshot

## Source State

- `paper2` is an RL-MEC / Mainline-A comparison experiment project.
- v4.2 corrected project boundary: paper-writing assets are out of scope.
- Mainline-A N0/N1/N2/N3 experiment chain final review is closed with `ACCEPTED_WITH_BOUNDARIES`.
- `results/`, `experiments/`, `figures/`, `logs/`, `checkpoints/` remain ignored generated artifacts.

## Implemented Asset Families

| Family | Representative Paths |
|---|---|
| MEC model | `src/mec_model/`, `configs/system_model_mainline_a.yaml` |
| Dynamic pricing | `src/game_pricing/`, `configs/pricing_dynamic_mainline_a.yaml` |
| Game-aware MARL | `src/rl_algorithms/game_aware/` |
| Experiment runner/gates | `configs/experiments/mainline_a_n0/`, `mainline_a_n1/`, `mainline_a_n2/`, `mainline_a_n3/` |
| Benchmark CLI | `scripts/benchmark.py` |
| Experiment orchestration | `scripts/experiment_manager.py`, `src/experiment/*` |
| Training entry | `scripts/train.py` |
| VSCode entrypoints | `.vscode/launch.json` |

## Known Architecture Gap

The Mainline-A model assets exist, but full-17 benchmark orchestration still lacks a single environment-profile abstraction. Current full17 defaults to generic `env: auto`; v4.3 introduces `mainline-a` as explicit default and `legacy` as explicit fallback.
