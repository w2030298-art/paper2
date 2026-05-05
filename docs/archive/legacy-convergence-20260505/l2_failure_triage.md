# L2 Failure Triage

> evidence level: `L2` | status: pending L2 execution | plan.md version: slimming-plan-v3

## Triage Rules

| Failure Class | Meaning | Next Action |
|---------------|---------|-------------|
| event_bug | event audit points to environment, reward, action, or logging semantics | escalate env/metric |
| training_instability | unstable optimization, replay, or policy update behavior | try next single-variable override |
| metric_tradeoff | reward improves but latency, energy, comm_score, or deadline miss rate regresses | review weighted metric tradeoff |
| insufficient_steps | 100k remains improving without severe warnings | consider L3 only after review |
| seed_sensitive | median is acceptable but q25-q75 or one seed dominates | retry same override or exclude from formal convergence claim |

## Current Algorithm Routing

| Algorithm | L1 State | L2 Routing |
|-----------|----------|------------|
| COMA | l1_candidate | run L2 without override |
| MAPPO | l1_candidate | run L2 without override |
| TRPO | l1_candidate | run L2 without override |
| IQL | needs_event_audit | run L2 only after event audit remains non-escalating |
| VDN | needs_event_audit | run L2 only after event audit remains non-escalating |
| IPPO | needs_event_audit | run L2 only after event audit remains non-escalating |
| MADDPG | needs_event_audit | run L2 only after event audit remains non-escalating |
| A3C | needs_single_variable_fix | choose one override family and run L2 |
| MATD3 | needs_single_variable_fix | choose one override family and run L2 |
| SAC | needs_single_variable_fix | choose one override family and run L2 |
| GRPO | needs_single_variable_fix | choose one override family and run L2 |

## Publication Boundary

Any algorithm that fails L2, lacks L2 data, or lacks L3 data must exclude from formal convergence claim and stay out of the paper main convergence conclusion.
