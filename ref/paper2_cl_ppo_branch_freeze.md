# CL-PPO Branch Freeze

This note freezes the CL-PPO branch pending corrected baseline review.

## Freeze Decision

CL-PPO development and paper-claim work are paused. The branch must not continue from the old PPO 1-agent full17 result.

## Reason

The PPO evidence used to motivate CL-PPO came from a 1-agent single-agent interface. That interface is not an adequate baseline for the 3-user shared-resource Mainline-A game.

## Restart Condition

CL-PPO can only be reconsidered after:

1. The v5.0 single-policy 3-user full17-equivalent run completes or is explicitly accepted despite blockers.
2. `ref/paper2_single_policy_3user_full17_report.md` is reviewed by a human.
3. `ref/paper2_single_agent_reassessment_decision.md` explicitly says PPO remains a valid CL-PPO baseline under the corrected interface.

Until then, CL-PPO must remain excluded from the v5.0 single-policy comparison and from research claims.
