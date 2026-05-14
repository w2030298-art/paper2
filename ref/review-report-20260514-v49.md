# v4.9 Review Report - 2026-05-14

`.ai/ledger.json` remains the single machine state source. This file records the review boundary that triggered the v5.0 patch.

## Review Result

- v4.9 tooling repairs and contract tests passed, but real L2/L3 evidence was not executed.
- The frozen Mainline-A core had a pre-existing dirty file boundary at `src/environments/mec_v3/game_theory_env.py`.
- Local runtime was CPU-compatible only for the heavy L2/L3 plan because CUDA was unavailable in the active environment.
- No paper-level p-value, confidence interval, effect-size, or superiority claim is supported by the v4.9 dry-run artifacts.

## v5.0 Decision

- The previous single-agent full17 comparison used a 1-agent interface and must not remain the main research evidence for a multi-user MEC game.
- The corrected baseline is a single shared policy evaluated inside a 3-user Mainline-A environment.
- CL-PPO work is paused until the corrected single-policy 3-user report receives human review.

## Boundary

The old full17 artifacts may remain historical engineering artifacts. They are not valid evidence for claims about 3-user shared-resource competition.
