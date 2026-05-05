# Formal Convergence Verification Protocol

> plan.md version: slimming-plan-v3 | scope: module 14 | updated: 2026-05-04

## Evidence Levels

| Level | Name | Required Input | Allowed Conclusion |
|-------|------|----------------|--------------------|
| L0 | offline diagnosis | Historical logs, historical benchmark JSON, quality reports | Locate issues only |
| L1 | 50k single-seed screening | `seed=42`, about 50k steps | `l1_candidate`, `failed_l1`, `needs_event_audit`, or `needs_single_variable_fix` |
| L2 | 100k multi-seed candidate validation | `seeds=[42,43,44]`, 100k steps | `candidate_converged_under_protocol` |
| L3 | 200k multi-seed formal validation | `seeds=[42,43,44,45,46]`, 200k steps | `verified_converged_under_protocol` |

## Final Protocol Verdict

An algorithm may be marked `verified_converged_under_protocol` only after L3 passes all gates below:

1. `run_status=success` for every seed.
2. `failed_seed_count == 0`.
3. `catastrophic_outlier_count == 0`.
4. Reward tail is stable, `best_tail_gap <= 0.10`, and tail relative change does not keep worsening.
5. Latency, energy, and deadline miss rate do not regress by more than 10% relative to the L1 baseline.
6. `comm_score` does not regress by more than 10%; reward/comm conflicts require review.
7. Median and q25-q75 bands are stable across the L3 seed set.
8. Raw diagnostic plot, clean publication plot, and quality report agree.
9. The report records `config_hash`, `run_id`, `override_id`, `seed_set`, and data paths.

## L1 Boundary

The current 50k single-seed data is an L1 screening artifact. It can prove the benchmark path runs and can identify candidates or failures, but it cannot prove convergence under this protocol.

L1 reports must not contain mathematical guarantee language in English or Chinese, and must not contain the `verified_converged` prefix.

## Metric Directions

| Metric | Direction | Gate |
|--------|-----------|------|
| reward | higher is better | stable tail and `best_tail_gap <= 0.10` |
| latency | lower is better | no more than 10% regression vs L1 baseline |
| energy | lower is better | no more than 10% regression vs L1 baseline |
| comm_score | higher is better | no more than 10% regression vs L1 baseline |
| deadline miss rate | lower is better | no significant increase; no more than 10% regression |

## Plot And Report Binding

L2 and L3 runs must produce a raw diagnostic plot, a clean publication plot, and a quality report. Clean plots may mask outlier scale for readability, but they must never delete raw evidence. Quality records must include `evidence_level`, `run_id`, `seed_set`, `config_hash`, and `override_id`.

## Publication Rule

Only algorithms that pass L3 can enter the main convergence figure or the main convergence conclusion. L2-only algorithms remain candidates. L1-only algorithms may appear in debugging or screening notes, not in the formal convergence claim.
