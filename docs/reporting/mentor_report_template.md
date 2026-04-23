# Mentor Progress Report Template

## 1. Goals This Cycle

- Goal 1: validate environment/trainer/algorithm compatibility from README scope.
- Goal 2: complete benchmark pipeline execution (smoke subset first, then full 100k x 3 seeds).
- Goal 3: produce visualizations and submission-ready reporting docs.

## 2. Completed Work

- Item A (tests and compatibility):
  - `tests/test_mec_envs.py` passed
  - `tests/test_trainers.py` passed
  - `tests/test_algorithms_on_envs.py` passed
- Item B (smoke training):
  - `scripts/train.py` for `A3C` and `SAC`
  - artifacts under `checkpoints/a3c/` and `checkpoints/sac/`
- Item C (benchmark and visualization):
  - full benchmark outputs in `results/benchmark.json` (16 algorithms, GRPO excluded)
  - figures generated in `figures/` via `scripts/plot_results.py`

## 3. Architecture and Flow Notes

- Architecture diagram: `docs/reporting/diagrams/system_architecture.md`
- Training flow: `docs/reporting/diagrams/training_flow.md`
- Benchmark flow: `docs/reporting/diagrams/benchmark_flow.md`
- Public API change summary: no public API changes in this cycle.

## 4. Experimental Results and Evidence

- Key setup:
  - `timesteps=100000`
  - `seeds=42 123 456`
  - `device=cpu`
  - evaluated algorithms: 16/17 (GRPO excluded)
- Key metrics:
  - benchmark entries: 16
  - failed entries: 0
  - best discrete reward mean: VDN = 57.2650
  - best continuous reward mean: MADDPG = 25.2393
  - test summary: 8 + 8 + 18 passed
- Result paths:
  - final merged: `results/benchmark.json`
  - per algorithm: `results/full_per_algo/`
  - figures: `figures/`

## 5. Issues and Fixes

- Issue 1: A3C rollout contamination and shape mismatch during training
  - Cause: deterministic evaluation path affected rollout buffer; single-agent return format mismatch
  - Fix:
    - prevent rollout writes in deterministic eval path in `rl_algorithms/a3c.py`
    - unwrap single-agent obs/reward in `src/trainer/on_policy_trainer.py` and `src/trainer/off_policy_trainer.py`
  - Effect: A3C smoke training and benchmark now run through
- Issue 2: QMIX benchmark input dimension mismatch
  - Cause: multi-agent mode flag mismatch
  - Fix: set `multi_agent_mode = "joint"` in `rl_algorithms/qmix.py`
  - Effect: QMIX full benchmark succeeds
- Issue 3: GRPO memory explosion during update stage
  - Cause: unstable tensor broadcasting in loss path (partially hardened but still unstable)
  - Fix this cycle: excluded GRPO from official comparison run
  - Effect: final report includes 16 stable algorithms; GRPO tracked as follow-up risk

## 6. Risks and Next Plan

- Current risks:
  - GRPO still unstable on memory usage
  - CPU-only full benchmark is time-expensive
- Next plan:
  - run targeted GRPO diagnostics with reduced batch/steps and strict tensor-shape checks
  - rerun GRPO after fix, then merge into 17-algorithm comparison
  - keep per-algorithm independent run outputs to reduce rerun cost
- Decision needed from mentor:
  - confirm whether "16 stable algorithms + GRPO risk note" is acceptable for this phase gate.

## 7. Deliverables Before Next Check-in

- Deliverable 1: GRPO diagnostic note (repro steps, root cause, patch direction)
- Deliverable 2: 17-algorithm benchmark result set (if GRPO fixed)
- Deliverable 3: updated report figures and revised comparative conclusions
