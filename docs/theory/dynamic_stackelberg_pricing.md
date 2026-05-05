# Dynamic Stackelberg Pricing

## Follower Utility

Follower utility is concave in offloaded workload after price payment, latency cost, energy cost, and deadline risk are included.

## Leader Utility

Leader utility is provider revenue minus operating and congestion costs.

## Existence

Equilibrium existence follows under compact price/action sets and continuous utilities.

## Uniqueness

Uniqueness is a sufficient condition, not an unconditional claim. It is supported when follower utility is strictly concave, demand response is Lipschitz in price, and leader utility is strongly concave or contractive in price.

## Monotonicity

Demand is monotone non-increasing in price. Price is monotone non-decreasing in queue pressure and migration risk. The implementation lowers price as channel quality improves, treating better channels as lower risk for the same workload.

## Lipschitz Price Coupling

Queue, channel, and migration state enter through bounded coefficients and clipped price bounds.

