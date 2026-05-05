# Mainline-A Theory Appendix

## Problem formulation

The mainline-A problem couples MEC offloading, dynamic pricing, queue state, channel state, migration risk, and constrained MARL updates.

## State-dependent Stackelberg pricing

Prices use `p_i(t)=clip(p0_i + alpha_q Q_i(t) - alpha_h H_i(t) + alpha_m M_i(t), p_min, p_max)`.

## Existence of Stackelberg equilibrium

An equilibrium exists under compact action sets, bounded prices, and continuous follower and leader utilities.

## Uniqueness sufficient condition

Uniqueness is a sufficient condition, not an unconditional uniqueness claim. It requires follower utility to be strictly concave, demand response to be Lipschitz in price, and leader utility to satisfy strong concavity or a contraction mapping condition in price.

## Monotonicity of demand response

Demand is monotone non-increasing in price. Price is monotone non-decreasing in queue pressure and migration risk. Price is non-increasing in channel quality in this implementation because higher quality is interpreted as lower risk, not scarcity rent.

## Primal-dual update and constraint residual

Dual variables update by projected ascent on positive residuals for latency deadline, energy budget, queue stability, migration rate, and budget feasibility.

## Constraint violation probability bound

Under bounded reward, bounded gradients, and bounded stochastic noise, a conservative Lyapunov or primal-dual residual argument supports an `O(1/sqrt(T))` violation-rate discussion. This is not a global neural-policy convergence proof.

## Convergence-rate discussion under stochastic approximation assumptions

The rate statement is limited to residual trends under bounded stochastic approximation assumptions and does not claim unconditional global convergence.

