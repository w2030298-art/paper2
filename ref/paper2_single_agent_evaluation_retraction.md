# Single-Agent Evaluation Retraction

This note retracts the research-evidence role of the old 1-agent single-agent full17 comparison.

## Retracted Evidence Scope

The old full17 single-agent results for the following algorithms are no longer used as research conclusions for the multi-user Mainline-A game:

- GRPO
- PPO
- SAC
- DDQN
- DDPG
- TD3
- A3C
- TRPO
- SimPO

## Reason

Those results were produced through a 1-agent evaluation interface. That interface does not exercise simultaneous 3-user offloading, server choice, resource competition, fairness, queue, and dynamic-pricing coupling in the Mainline-A environment.

## Allowed Use

The old records may be kept as historical artifacts and as implementation/debugging references. They must not be cited as evidence that an algorithm performs well or poorly in the 3-user MEC game.

## Replacement Gate

The replacement evidence path is `single_policy_multi_user`: one shared single-agent policy controls all 3 users by producing per-user actions inside the same `num_agents=3` Mainline-A step.
