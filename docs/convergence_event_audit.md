# Convergence Event Audit

> Last updated: 2026-05-04 | plan.md version: slimming-plan-v3 | phase: module 14 Step 5

This audit reads existing local artifacts only. It does not run training and does not enable stability overrides.

| Algorithm | Seed | First Bad Timestep | Raw Min | Previous Value | Next Value | Suspected Layer | Source | Decision |
|-----------|------|--------------------|---------|----------------|------------|-----------------|--------|----------|
| IPPO | 42 | 0.0000 | -657.7003 | NA | -111.6012 | optimizer | training_instability | allow_l2_l3_gate |
| IQL | 42 | 10000.0000 | -165.3977 | -94.4792 | -3198.0080 | replay | training_instability | allow_l2_l3_gate |
| MADDPG | 42 | 0.0000 | -4414.0544 | NA | -156.9052 | optimizer | training_instability | allow_l2_l3_gate |
| VDN | 42 | 0.0000 | -4644.7968 | NA | -31449.9511 | replay | training_instability | allow_l2_l3_gate |

## Gate Decision

- `environment_metric_bug` or `unknown` stops L2/L3 and requires `NEEDS_ESCALATION`.
- `training_instability` or `evaluation_noise` may proceed to L2/L3 gates, but still cannot enter the paper main claim without L3.
