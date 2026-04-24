# Environment Models (GameTheory MEC)

```mermaid
flowchart TD
    A["Agent action (target/offload/cpu/power)"] --> B["ParameterizedActionSpace"]
    B --> C["ConstraintProjection (penalty + barrier)"]
    C --> D["QueueingDelayModel (M/M/1)"]
    C --> E["DVFSEnergyModel (non-ideal)"]
    C --> F["Channel3GPP (LOS/NLOS + SINR)"]

    D --> G["BilevelGameSolver (Stackelberg + IBR Nash)"]
    E --> G
    F --> G

    G --> H["MonteCarloShapley (antithetic)"]
    H --> I["EFX + CP-net valuation layer"]
    I --> J["HierarchicalReward: r_imm + r_coop + r_eq + r_fair"]
    J --> K["info: game_hints/shapley/reward_terms/queue/constraint/fairness"]
```

## Notes

- External API is unchanged: same env IDs and adapter action spaces.
- `fairness_metrics` is additive info output, not a breaking change.
- EFX can be disabled via `game_theory.efx_enabled=false`.
- CP-net valuation can be disabled via `game_theory.cpnet_enabled=false`.
