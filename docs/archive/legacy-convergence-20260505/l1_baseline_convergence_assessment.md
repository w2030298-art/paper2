# L1 Baseline Convergence Assessment

> evidence level: `L1` | scope: engineering protocol

L1 is a 50k single-seed screening pass. It can only produce candidate, failure, event-audit, or single-variable-fix decisions.

| Algorithm | Decision | Reward Status | Failed Seeds | Catastrophic Outliers | Best-Tail Gap | Tail Change | Reason |
|-----------|----------|---------------|--------------|-----------------------|---------------|-------------|--------|
| A3C | needs_single_variable_fix | diverging | 0 | 0 | 0.1049 | -0.0568 | algorithm is not in the L1 candidate group |
| COMA | l1_candidate | bad_plateau | 0 | 0 | 0.1039 | -0.0326 | L1 screening passed; formal validation still requires L2/L3 |
| GRPO | needs_single_variable_fix | bad_plateau | 0 | 0 | 0.1124 | 0.0136 | algorithm is not in the L1 candidate group |
| IPPO | needs_event_audit | catastrophic_outlier | 0 | 1 | 0.6034 | 0.0045 | catastrophic outlier requires event audit; reward classified as catastrophic_outlier; reward best_tail_gap above protocol gate |
| IQL | needs_event_audit | catastrophic_outlier | 0 | 1 | 3.0488 | 0.9110 | catastrophic outlier requires event audit; reward classified as catastrophic_outlier; reward best_tail_gap above protocol gate |
| MADDPG | needs_event_audit | catastrophic_outlier | 0 | 1 | 1.6714 | -0.9135 | catastrophic outlier requires event audit; reward classified as catastrophic_outlier; reward best_tail_gap above protocol gate |
| MAPPO | l1_candidate | converged_good | 0 | 0 | 0.0830 | -0.0378 | L1 screening passed; formal validation still requires L2/L3 |
| MATD3 | needs_single_variable_fix | diverging | 0 | 0 | 1.4117 | -0.7995 | reward best_tail_gap above protocol gate |
| SAC | needs_single_variable_fix | bad_plateau | 0 | 0 | 1.5841 | -0.0189 | reward best_tail_gap above protocol gate |
| TRPO | l1_candidate | converged_good | 0 | 0 | 0.0330 | 0.0244 | L1 screening passed; formal validation still requires L2/L3 |
| VDN | needs_event_audit | catastrophic_outlier | 0 | 1 | 2.9147 | -0.3965 | catastrophic outlier requires event audit; reward classified as catastrophic_outlier; reward best_tail_gap above protocol gate |

## Protocol Boundaries

- 50k single-seed data is screening evidence only.
- L2 requires 100k steps across seeds 42, 43, and 44.
- L3 requires 200k steps across seeds 42, 43, 44, 45, and 46.
- Algorithms that have not passed L3 stay out of the paper main convergence figure.
