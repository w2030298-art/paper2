# Game-Aware Constrained MARL Reference

## Critic Feature Schema

The game-aware critic receives queue pressure, channel quality, migration risk, price vector, follower demand elasticity, and constraint residuals.

## Primal-Dual Update

The primal-dual layer maintains nonnegative dual variables for latency deadline, energy budget, queue stability, migration rate, and budget feasibility residuals. Rewards subtract dual-weighted positive violations.

## Reward Components

Reward components include delay, energy, queue, migration, deadline violation, cooperation gain, price payment, provider revenue, and constraint penalty.

## Ablation Switches

Supported switches are `no_price`, `no_queue`, `no_migration`, `no_dual`, and `no_cooperation`.

