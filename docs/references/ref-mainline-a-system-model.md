# Mainline-A System Model Reference

## Scope

This reference defines the new MEC model layer under `src/mec_model/`.

## Task Model

Scalar legacy tasks are represented as single-node `TaskDAGSpec` instances. DAG tasks are opt-in through config and do not replace the legacy environment path by default.

## Queue Model

The queue layer supports `MM1QueueModel`, `MMCQueueModel`, `ParallelQueueApproxModel`, and `FiniteCapacityQueueModel`. Overloaded queues return bounded penalties rather than NaN or Inf.

## Edge Topology

The default topology is `bs_only`. Heterogeneous cooperation among BS, RSU, UAV, peer device, and cloud nodes is disabled unless config enables it.

## Energy Model

Energy uses a network-level DVFS abstraction for local compute, transmission, edge compute, and migration transfer energy.

## Mobility

The default mobility path is Markov cell transition. Trace mobility is an adapter option for later experiments.

## Channel Models

`AnalyticRateModel` is the theory-facing pathloss/SINR/Shannon model. `ThreeGppLiteRateModel` is the simulation-facing UMi/UMa LoS/NLoS derived model. Rayleigh and pathloss-only models support ablation.

