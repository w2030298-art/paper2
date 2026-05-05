# Legacy Convergence Retirement

> plan.md version: system-model-overhaul-v4.1 | run_id: l2_20260504_171744

## Decision

The old L2/L3 convergence workflow is retired for the new system model. Existing artifacts are preserved as a legacy baseline, but they must not be used as main claim evidence for mainline-A.

## L2 Job Status

- PID: `26860`
- command line: `.venv\Scripts\python.exe scripts\run_formal_convergence_protocol.py --phase L2 --run-id l2_20260504_171744 --auto-l3`
- status observed: running at inspection time
- action: stopped on 2026-05-05 because system model overhaul invalidates old formal gate relevance
- L3 action: not started

## Preserved Artifacts

- `experiments/formal_convergence/l2/l2_20260504_171744/manifest.json`
- `experiments/formal_convergence/l2/l2_20260504_171744/commands.txt`
- `logs/formal_convergence/l2_20260504_171744.out.log`
- `logs/formal_convergence/l2_20260504_171744.err.log`
- `logs/formal_convergence/l2_20260504_171744.process.json`
- `docs/archive/legacy-convergence-20260505/`

## Downgrade Reason

The system model overhaul changes observation features, reward components, dynamic pricing, experiment gates, and the main evidence chain. The old L2/L3 results are therefore a pre-overhaul legacy baseline only.

## Publication Boundary

Old convergence artifacts may enter appendix, debug notes, or regression references. They must not be used as main claim evidence, main figures, or new system model conclusions.

