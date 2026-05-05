# Mainline-A Compatibility Report

## Legacy Default

The legacy default remains unchanged because `configs/system_model_mainline_a.yaml` sets `system_model.enabled: false` and `compatibility.preserve_legacy_default: true`.

## Enablement

Mainline-A is enabled explicitly with `--enable-mainline-a` and `--system-model-config configs/system_model_mainline_a.yaml`. Dynamic pricing remains separately disabled unless its config enables it.

## Known Review Items

- External dashboard compatibility still needs review in `C:\Users\22003\paper2\rl-mec-dashboard`.
- Old L2/L3 convergence artifacts are legacy baseline only.

