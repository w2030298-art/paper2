# Stage-2 Roadmap: CL-PPO and GAM-COMA

This reference note records the next research direction after the Stage-1 starter closeout.
It is not a machine state file; `.ai/ledger.json` remains the only state source.

## Direction

- Single-agent main algorithm: upgrade PPO toward CL-PPO.
- Multi-agent main algorithm: upgrade COMA toward GAM-COMA.
- Strong single-agent baselines: TRPO, SimPO, DDPG, TD3.
- Strong multi-agent baselines: MAPPO, MADDPG, MATD3.

## CL-PPO Work Packages

- Architecture: add a Lyapunov or constraint signal path for stability-aware policy updates.
- Risk modeling: add a CVaR or tail-risk critic for latency and deadline-risk sensitivity.
- Safety layer: add feasible action projection or masking before environment execution.
- Ablations: isolate constraint signal, risk critic, and action safety layer contributions.

## GAM-COMA Work Packages

- Architecture: replace or augment the centralized critic with graph-attention message passing.
- Feasible actions: add action masking for invalid or unsafe multi-agent decisions.
- Credit assignment: compare COMA counterfactual baseline with warm-start and Shapley-style ablations.
- Ablations: isolate graph attention, feasible action masking, warm start, and credit assignment variants.

## Experiment Matrix

- ID: in-distribution Mainline-A comparison against Stage-1 candidates and strong baselines.
- N1 oracle: oracle-assisted boundary or feasibility reference for upper-bound diagnostics.
- N2 ablation: module-level ablations for CL-PPO and GAM-COMA components.
- N3 OOD: out-of-distribution stress profiles for topology, load, channel, and deadline shifts.
- Seeds: use at least 5 seeds for final claims; starter pilots are not paper-level tuning evidence.

## Guardrails

- Do not implement `rl_algorithms/cl_ppo.py` or `rl_algorithms/gam_coma.py` without a separate reviewed Stage-2 implementation packet.
- Do not claim broad or narrow adaptive search from Stage-1 starter pilots.
- Treat the current 4-trial PPO/COMA starter artifacts as readiness evidence, not final paper-level tuning.
