# Mainline-A full-17 migration notes

## Target Architecture

```text
VSCode / CLI
   -> scripts/experiment_manager.py
      -> ExperimentManager.create_experiment(environment_profile)
         -> AlgorithmRegistry.build_specs(environment_profile)
            -> AlgorithmSpec.extra_args = ["--environment-profile", "mainline-a"]
               -> TrainCommandBuilder.build()
                  -> scripts/train.py --environment-profile mainline-a
                     -> make_env(... env_overrides)
                        -> GameTheoryMECEnv / adapters with Mainline-A enabled
```

Direct benchmark path:

```text
scripts/benchmark.py --all --environment-profile mainline-a
   -> resolve_environment_profile()
   -> make_env(... enable_mainline_a=True, system_model_config=..., dynamic_pricing_config=...)
```

## Profile Contract

### `mainline-a`

- default profile.
- injects Mainline-A system model config.
- injects dynamic pricing config.
- enables Mainline-A env path.
- fails fast on config/env errors.

### `legacy`

- explicit fallback only.
- does not inject Mainline-A configs.
- disables Mainline-A env path.
- result files must use legacy-specific run id / output name.

## Naming Rules

- Mainline-A full17 run id: `paper2_full_17_mainline_a`
- Legacy fallback run id: `paper2_full_17_legacy_fallback`
- Mainline-A exported result: `results/benchmark_paper2_full_17_mainline_a.json`
- Legacy exported result: `results/benchmark_paper2_full_17_legacy_fallback.json`

## Anti-Patterns

- Do not use `env: auto` as the only indicator of Mainline-A.
- Do not silently catch Mainline-A init failure and instantiate legacy.
- Do not create a second full17 preset branch unless there is a strong reason.
- Do not re-expand `.vscode/launch.json` with per-algorithm reset entries.
