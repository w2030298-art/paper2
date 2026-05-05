"""
GameTheory MEC environment with optimization-driven pricing and RL/game-theory fusion hooks.

This module keeps the original external contract (multi-agent Dict action input and
list observation output), while upgrading internal models:
- Convex optimization pricing (OptimalPricingMechanism)
- Bilevel solver (Stackelberg outer + Nash IBR inner)
- Monte Carlo Shapley with antithetic sampling
- M/M/1 queue delay model
- Non-ideal DVFS energy model
- 3GPP-like LOS/NLOS channel + multi-user interference SINR
- Hierarchical reward with adaptive weights
- Parameterized action mapping + constraint projection
- EFX fairness repair with CP-net preference-informed valuations
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces


sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.environments.mec_v2.base_env import BaseMECEnv, MECRewardShaper


EPS = 1e-8
EFX_TRANSFER_CAP = 0.05


class OptimalPricingMechanism:
    """Convex(-like) pricing optimizer for edge-server prices."""

    def __init__(
        self,
        eta: np.ndarray,
        c: np.ndarray,
        d0: np.ndarray,
        p_min: float,
        p_max: float,
        tol: float = 1e-6,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.eta = np.asarray(eta, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.d0 = np.asarray(d0, dtype=np.float64)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.tol = float(tol)
        self._rng = rng if rng is not None else np.random.default_rng()

    def demand(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        return self.d0 * np.exp(-self.eta * prices)

    def profit_gradient(self, prices: np.ndarray) -> np.ndarray:
        d = self.demand(prices)
        dd_dp = -self.eta * d
        return d + prices * dd_dp - 2.0 * self.c * d * dd_dp

    def _profit_hessian_diag(self, prices: np.ndarray) -> np.ndarray:
        d = self.demand(prices)
        dd_dp = -self.eta * d
        d2d_dp2 = self.eta * self.eta * d
        return 2.0 * dd_dp + prices * d2d_dp2 - 2.0 * self.c * (dd_dp * dd_dp + d * d2d_dp2)

    def solve_optimal_price(self, max_iter: int = 100) -> np.ndarray:
        prices = np.full_like(self.eta, (self.p_min + self.p_max) * 0.5, dtype=np.float64)
        for _ in range(max_iter):
            grad = self.profit_gradient(prices)
            hdiag = self._profit_hessian_diag(prices)
            step = -grad / (hdiag + 1e-6)
            prices = np.clip(prices + step, self.p_min, self.p_max)
            if np.max(np.abs(grad)) < self.tol:
                break
        return prices.astype(np.float32)

    def update_with_demand(self, realized_demand: np.ndarray, momentum: float = 0.2) -> np.ndarray:
        realized_demand = np.asarray(realized_demand, dtype=np.float64)
        self.d0 = (1.0 - momentum) * self.d0 + momentum * np.maximum(realized_demand, 1e-4)
        return self.solve_optimal_price()


class BilevelGameSolver:
    """Outer Stackelberg loop + inner user Nash IBR loop."""

    def __init__(
        self,
        pricing: OptimalPricingMechanism,
        n_users: int,
        n_servers: int,
        max_outer: int = 12,
        max_inner: int = 32,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.pricing = pricing
        self.n_users = n_users
        self.n_servers = n_servers
        self.max_outer = max_outer
        self.max_inner = max_inner
        self._rng = rng if rng is not None else np.random.default_rng()

    def _best_response(
        self,
        user_id: int,
        prices: np.ndarray,
        channel_db: np.ndarray,
        queue_lengths: np.ndarray,
        task: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        if task is None:
            return np.array([-1.0, -1.0, 0.0, -1.0], dtype=np.float32)

        local_cost = 1.0 + 0.3 * float(np.mean(queue_lengths))
        costs = np.full(self.n_servers + 1, local_cost, dtype=np.float64)
        snr_row = channel_db[user_id]
        for k in range(self.n_servers):
            snr_lin = 10.0 ** (snr_row[k] / 10.0)
            channel_term = 1.0 / (snr_lin + 1e-3)
            queue_term = 0.4 * queue_lengths[k]
            costs[k + 1] = prices[k] + channel_term + queue_term

        target = int(np.argmin(costs))
        if target == 0:
            offload_ratio = 0.0
        else:
            margin = float(costs[0] - costs[target])
            offload_ratio = 1.0 / (1.0 + math.exp(-3.0 * margin))
            offload_ratio = float(np.clip(offload_ratio, 0.05, 0.95))

        deadline = float(task.get("deadline", 1.0))
        cpu_ratio = float(np.clip(1.0 - deadline / 5.0, 0.1, 1.0))
        if target > 0:
            snr_quality = float(np.clip((snr_row[target - 1] + 20.0) / 40.0, 0.0, 1.0))
        else:
            snr_quality = 0.2
        tx_ratio = float(np.clip(0.3 + 0.7 * (1.0 - snr_quality), 0.1, 1.0))

        target_selector = -1.0 + 2.0 * target / max(1, self.n_servers)
        return np.array(
            [
                target_selector,
                2.0 * offload_ratio - 1.0,
                2.0 * cpu_ratio - 1.0,
                2.0 * tx_ratio - 1.0,
            ],
            dtype=np.float32,
        )

    def _solve_nash_equilibrium(
        self,
        prices: np.ndarray,
        channel_db: np.ndarray,
        queue_lengths: np.ndarray,
        tasks: List[Optional[Dict[str, Any]]],
    ) -> np.ndarray:
        actions = np.zeros((self.n_users, 4), dtype=np.float32)
        for i in range(self.n_users):
            actions[i] = self._best_response(i, prices, channel_db, queue_lengths, tasks[i])

        for _ in range(self.max_inner):
            old = actions.copy()
            for i in range(self.n_users):
                actions[i] = self._best_response(i, prices, channel_db, queue_lengths, tasks[i])
            if np.max(np.abs(actions - old)) < 1e-4:
                break
        return actions

    def _compute_demand(self, actions: np.ndarray) -> np.ndarray:
        demand = np.zeros(self.n_servers, dtype=np.float32)
        for a in actions:
            target = int(np.clip(np.round((a[0] + 1.0) * 0.5 * self.n_servers), 0, self.n_servers))
            if target > 0:
                offload_ratio = float(np.clip((a[1] + 1.0) * 0.5, 0.0, 1.0))
                demand[target - 1] += offload_ratio
        return demand

    def solve(
        self,
        channel_db: np.ndarray,
        queue_lengths: np.ndarray,
        tasks: List[Optional[Dict[str, Any]]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        prices = self.pricing.solve_optimal_price()
        actions = self._solve_nash_equilibrium(prices, channel_db, queue_lengths, tasks)

        for _ in range(self.max_outer):
            demand = self._compute_demand(actions)
            new_prices = self.pricing.update_with_demand(demand)
            new_actions = self._solve_nash_equilibrium(new_prices, channel_db, queue_lengths, tasks)
            if np.max(np.abs(new_prices - prices)) < 1e-4 and np.max(np.abs(new_actions - actions)) < 1e-4:
                prices, actions = new_prices, new_actions
                break
            prices, actions = new_prices, new_actions

        demand = self._compute_demand(actions)
        return prices.astype(np.float32), actions.astype(np.float32), demand.astype(np.float32)


class MonteCarloShapley:
    """Monte Carlo Shapley estimator with optional antithetic sampling."""

    def __init__(
        self,
        n_agents: int,
        coalition_value_fn: Optional[Callable[[frozenset[int]], float]] = None,
        n_samples: int = 128,
        use_antithetic: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_agents = n_agents
        self.coalition_value_fn = coalition_value_fn
        self.n_samples = max(1, int(n_samples))
        self.use_antithetic = use_antithetic
        self._rng = rng if rng is not None else np.random.default_rng()

    def compute(self, coalition_value_fn: Optional[Callable[[frozenset[int]], float]] = None) -> np.ndarray:
        value_fn = coalition_value_fn if coalition_value_fn is not None else self.coalition_value_fn
        if value_fn is None:
            raise ValueError("coalition_value_fn must be provided.")

        shapley = np.zeros(self.n_agents, dtype=np.float64)
        n_perm_total = 0

        for _ in range(self.n_samples):
            perm = self._rng.permutation(self.n_agents)
            permutations = [perm]
            if self.use_antithetic:
                permutations.append(perm[::-1])

            for p in permutations:
                coalition: set[int] = set()
                prev_value = 0.0
                for agent in p:
                    coalition.add(int(agent))
                    curr = float(value_fn(frozenset(coalition)))
                    shapley[int(agent)] += curr - prev_value
                    prev_value = curr
                n_perm_total += 1

        if n_perm_total > 0:
            shapley /= float(n_perm_total)
        return shapley.astype(np.float32)


class QueueingDelayModel:
    """M/M/1 queue model with per-server arrival/service rates."""

    def __init__(self, stability_margin: float = 0.95) -> None:
        self.rho_max = float(stability_margin)

    def compute_delay(self, arrival_rates: np.ndarray, service_rates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrival_rates = np.asarray(arrival_rates, dtype=np.float64)
        service_rates = np.asarray(service_rates, dtype=np.float64)
        rho = np.clip(arrival_rates / np.maximum(service_rates, EPS), 0.0, self.rho_max)
        t_queue = rho / np.maximum(service_rates * (1.0 - rho), EPS)
        t_system = 1.0 / np.maximum(service_rates * (1.0 - rho), EPS)
        return t_queue.astype(np.float32), t_system.astype(np.float32), rho.astype(np.float32)


class DVFSEnergyModel:
    """Non-ideal DVFS model: E = k f^(alpha-1) C + P_leak C/f."""

    def __init__(
        self,
        kappa_dyn: float = 1e-27,
        alpha: float = 2.7,
        p_leak: float = 0.01,
        f_min: float = 0.5e9,
        f_max: float = 3.0e9,
    ) -> None:
        self.kappa = float(kappa_dyn)
        self.alpha = float(alpha)
        self.p_leak = float(p_leak)
        self.f_min = float(f_min)
        self.f_max = float(f_max)

    def compute_energy(self, freq: float, cpu_cycles: float) -> float:
        freq = float(np.clip(freq, self.f_min, self.f_max))
        cpu_cycles = float(max(cpu_cycles, 0.0))
        e_dyn = self.kappa * (freq ** (self.alpha - 1.0)) * cpu_cycles
        e_leak = self.p_leak * cpu_cycles / max(freq, EPS)
        return float(e_dyn + e_leak)

    def optimal_frequency(self, delay_weight: float) -> float:
        numer = self.p_leak + max(float(delay_weight), 0.0)
        denom = max((self.alpha - 1.0) * self.kappa, EPS)
        f_opt = (numer / denom) ** (1.0 / self.alpha)
        return float(np.clip(f_opt, self.f_min, self.f_max))


class Channel3GPP:
    """Compact 3GPP-like LOS/NLOS channel with SINR under interference."""

    def __init__(self, fc_ghz: float = 3.5, h_bs: float = 25.0, h_ut: float = 1.5, bandwidth: float = 20e6):
        self.fc_ghz = float(fc_ghz)
        self.h_bs = float(h_bs)
        self.h_ut = float(h_ut)
        self.bandwidth = float(bandwidth)
        self.noise_dbm = -174.0 + 10.0 * math.log10(self.bandwidth)

    def prob_los(self, d: float) -> float:
        d = max(float(d), 1.0)
        return float(min(1.0, 18.0 / d) * (1.0 - math.exp(-d / 36.0)) + math.exp(-d / 36.0))

    def path_loss_los(self, d: float) -> float:
        d = max(float(d), 1.0)
        return 32.4 + 21.0 * math.log10(d) + 20.0 * math.log10(self.fc_ghz)

    def path_loss_nlos(self, d: float) -> float:
        d = max(float(d), 1.0)
        return 35.3 * math.log10(d) + 22.4 + 21.3 * math.log10(self.fc_ghz) - 0.3 * (self.h_ut - 1.5)

    def composite_path_loss(self, d: float) -> float:
        p_los = self.prob_los(d)
        return p_los * self.path_loss_los(d) + (1.0 - p_los) * self.path_loss_nlos(d)

    def compute_sinr(
        self,
        d_target: float,
        d_interferers: np.ndarray,
        p_target_w: float,
        p_interferers_w: np.ndarray,
    ) -> float:
        pl_t = self.composite_path_loss(d_target)
        p_target_dbm = 10.0 * math.log10(max(p_target_w, 1e-12) * 1000.0)
        s_lin = 10.0 ** ((p_target_dbm - pl_t) / 10.0)

        i_lin = 0.0
        for d_j, p_j in zip(np.asarray(d_interferers), np.asarray(p_interferers_w)):
            pl_j = self.composite_path_loss(float(d_j))
            p_j_dbm = 10.0 * math.log10(max(float(p_j), 1e-12) * 1000.0)
            i_lin += 10.0 ** ((p_j_dbm - pl_j) / 10.0)

        n_lin = 10.0 ** (self.noise_dbm / 10.0)
        sinr = s_lin / max(i_lin + n_lin, EPS)
        return float(sinr)

    def rate(self, sinr: float) -> float:
        return self.bandwidth * math.log2(1.0 + max(float(sinr), 0.0))


class ParameterizedActionSpace:
    """Map normalized policy output to physical offload/cpu/power variables."""

    def __init__(
        self,
        k: int,
        f_min: float = 0.5e9,
        f_max: float = 3.0e9,
        p_min: float = 0.01,
        p_max: float = 0.5,
    ) -> None:
        self.k = int(k)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def map_action(self, raw_action: Dict[str, Any]) -> Dict[str, float]:
        target = int(raw_action["target"])
        ratio = np.asarray(raw_action["ratio"], dtype=np.float32).flatten()
        ratio = np.pad(ratio, (0, max(0, 3 - ratio.size)), constant_values=0.0)[:3]

        offload_ratio = self._sigmoid(float(ratio[0]) * 5.0)
        cpu_freq = self.f_min + (float(ratio[1]) + 1.0) * 0.5 * (self.f_max - self.f_min)
        tx_power = self.p_min + (float(ratio[2]) + 1.0) * 0.5 * (self.p_max - self.p_min)

        return {
            "target": int(np.clip(target, 0, self.k)),
            "offload_ratio": float(np.clip(offload_ratio, 0.0, 1.0)),
            "cpu_freq": float(np.clip(cpu_freq, self.f_min, self.f_max)),
            "tx_power": float(np.clip(tx_power, self.p_min, self.p_max)),
        }


@dataclass
class ProjectionResult:
    action: Dict[str, float]
    penalty: float
    barrier: float
    violation_energy: float
    violation_power: float
    violation_latency: float


@dataclass
class EFXRepairResult:
    transfers: np.ndarray
    is_efx: bool
    iterations: int
    violation: Optional[Tuple[int, int, str]]


class CPNetPreferenceModel:
    """Compact CP-net inspired conditional preference model for MEC actions."""

    def __init__(self, n_agents: int, n_servers: int) -> None:
        self.n_agents = int(n_agents)
        self.n_servers = int(n_servers)

    @staticmethod
    def _task_urgency(task: Optional[Dict[str, Any]]) -> float:
        if task is None:
            return 0.5
        deadline = float(task.get("deadline", 1.0))
        priority = float(task.get("priority", 1.0))
        cpu_cycles = float(task.get("cpu_cycles", 1.0))
        data_size = float(task.get("data_size", 1.0))
        tight_deadline = np.clip((2.5 - deadline) / 2.5, 0.0, 1.0)
        high_priority = np.clip(priority / 5.0, 0.0, 1.0)
        heavy_compute = np.clip((cpu_cycles / max(data_size, EPS)) / 2e3, 0.0, 1.0)
        return float(np.clip(0.45 * tight_deadline + 0.35 * high_priority + 0.20 * heavy_compute, 0.0, 1.0))

    @staticmethod
    def _offload_level(offload_ratio: float) -> str:
        if offload_ratio < 0.33:
            return "low"
        if offload_ratio < 0.67:
            return "mid"
        return "high"

    def build_server_utilities(
        self,
        channel_db: np.ndarray,
        queue_lengths: np.ndarray,
        prices: np.ndarray,
        tasks: List[Optional[Dict[str, Any]]],
    ) -> np.ndarray:
        # utilities[:, 0] is local processing option.
        utilities = np.zeros((self.n_agents, self.n_servers + 1), dtype=np.float32)
        q_norm = np.clip(queue_lengths / max(float(np.max(queue_lengths) + 1.0), 1.0), 0.0, 1.0)
        p_min = float(np.min(prices)) if prices.size else 0.0
        p_max = float(np.max(prices)) if prices.size else 1.0
        p_span = max(p_max - p_min, 1e-6)
        p_norm = np.clip((prices - p_min) / p_span, 0.0, 1.0)

        for i in range(self.n_agents):
            urgency = self._task_urgency(tasks[i])
            local_bias = 0.30 + 0.55 * urgency
            utilities[i, 0] = float(local_bias)
            for k in range(self.n_servers):
                snr_norm = float(np.clip((channel_db[i, k] + 20.0) / 40.0, 0.0, 1.0))
                server_score = 0.55 * snr_norm - 0.25 * q_norm[k] - 0.20 * p_norm[k]
                if urgency > 0.7:
                    server_score -= 0.10 * q_norm[k]
                utilities[i, k + 1] = float(server_score)
        return utilities

    def build_offload_utilities(self, tasks: List[Optional[Dict[str, Any]]]) -> np.ndarray:
        # columns are [low, mid, high]
        offload_util = np.zeros((self.n_agents, 3), dtype=np.float32)
        for i in range(self.n_agents):
            urgency = self._task_urgency(tasks[i])
            # CP-net style conditional preference:
            # high urgency -> local/low-offload preferred
            # low urgency/heavy tasks -> higher offload preferred
            low = 0.50 + 0.40 * urgency
            mid = 0.60 - 0.10 * abs(urgency - 0.5)
            high = 0.55 + 0.35 * (1.0 - urgency)
            offload_util[i] = np.array([low, mid, high], dtype=np.float32)
        return offload_util

    def build_allocation(self, mapped_actions: List[Dict[str, float]]) -> List[set[str]]:
        allocation: List[set[str]] = []
        for action in mapped_actions:
            target = int(action.get("target", 0))
            offload_level = self._offload_level(float(action.get("offload_ratio", 0.0)))
            bundle = {f"srv_{target}", f"off_{offload_level}"}
            allocation.append(bundle)
        return allocation

    def build_valuation_functions(
        self,
        channel_db: np.ndarray,
        queue_lengths: np.ndarray,
        prices: np.ndarray,
        tasks: List[Optional[Dict[str, Any]]],
    ) -> Tuple[List[Callable[[set[str]], float]], np.ndarray, np.ndarray]:
        server_util = self.build_server_utilities(channel_db, queue_lengths, prices, tasks)
        offload_util = self.build_offload_utilities(tasks)

        def _parse_bundle(bundle: set[str]) -> Tuple[int, int]:
            srv_idx = 0
            off_idx = 1
            for item in bundle:
                if item.startswith("srv_"):
                    try:
                        srv_idx = int(item.split("_", 1)[1])
                    except ValueError:
                        srv_idx = 0
                elif item.startswith("off_"):
                    lvl = item.split("_", 1)[1]
                    off_idx = {"low": 0, "mid": 1, "high": 2}.get(lvl, 1)
            srv_idx = int(np.clip(srv_idx, 0, self.n_servers))
            off_idx = int(np.clip(off_idx, 0, 2))
            return srv_idx, off_idx

        valuations: List[Callable[[set[str]], float]] = []
        for i in range(self.n_agents):
            def _val(bundle: set[str], idx: int = i) -> float:
                if not bundle:
                    return 0.0
                srv_idx, off_idx = _parse_bundle(set(bundle))
                return float(server_util[idx, srv_idx] + offload_util[idx, off_idx])

            valuations.append(_val)
        return valuations, server_util, offload_util


class EFXFairAllocation:
    """EFX check and transfer-payment based repair."""

    def __init__(self, n_agents: int, transfer_rate: float = 0.5, max_iters: int = 32) -> None:
        self.n_agents = int(n_agents)
        self.transfer_rate = float(np.clip(transfer_rate, 0.05, 1.0))
        self.max_iters = int(max(1, max_iters))

    def check_efx(
        self,
        allocation: List[set[str]],
        valuations: List[Callable[[set[str]], float]],
        transfers: Optional[np.ndarray] = None,
    ) -> Tuple[bool, Optional[Tuple[int, int, str]]]:
        if transfers is None:
            transfers = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i == j:
                    continue
                bundle_j = set(allocation[j])
                own_utility = float(valuations[i](allocation[i]) + transfers[i])
                if len(bundle_j) == 0:
                    continue
                for item in list(bundle_j):
                    reduced = set(bundle_j)
                    reduced.discard(item)
                    other_utility = float(valuations[i](reduced) + transfers[j])
                    if own_utility + 1e-8 < other_utility:
                        return False, (i, j, item)
        return True, None

    def repair_allocation(
        self,
        allocation: List[set[str]],
        valuations: List[Callable[[set[str]], float]],
    ) -> EFXRepairResult:
        transfers = np.zeros(self.n_agents, dtype=np.float32)
        is_efx, violation = self.check_efx(allocation, valuations, transfers)
        n_iter = 0
        while not is_efx and n_iter < self.max_iters and violation is not None:
            i, j, item = violation
            reduced = set(allocation[j])
            reduced.discard(item)
            own = float(valuations[i](allocation[i]) + transfers[i])
            other = float(valuations[i](reduced) + transfers[j])
            deficit = max(other - own, 0.0)
            delta = 0.5 * self.transfer_rate * deficit
            transfers[i] += float(delta)
            transfers[j] -= float(delta)
            transfers = np.clip(transfers, -EFX_TRANSFER_CAP, EFX_TRANSFER_CAP)
            is_efx, violation = self.check_efx(allocation, valuations, transfers)
            n_iter += 1

        transfers = np.clip(transfers, -EFX_TRANSFER_CAP, EFX_TRANSFER_CAP).astype(np.float32)
        return EFXRepairResult(
            transfers=transfers,
            is_efx=bool(is_efx),
            iterations=n_iter,
            violation=violation,
        )


class ConstraintProjection:
    """Soft+barrier constraints for action feasibility."""

    def __init__(
        self,
        e_max: float,
        t_max: float,
        p_total_max: float,
        penalty_coeff: float = 1.0,
        barrier_coeff: float = 1e-3,
    ) -> None:
        self.e_max = float(e_max)
        self.t_max = float(t_max)
        self.p_total_max = float(p_total_max)
        self.penalty_coeff = float(penalty_coeff)
        self.barrier_coeff = float(barrier_coeff)

    def project(
        self,
        action: Dict[str, float],
        est_energy: float,
        est_latency: float,
        total_power: float,
    ) -> ProjectionResult:
        off = float(np.clip(action["offload_ratio"], 0.0, 1.0))
        freq = float(action["cpu_freq"])
        power = float(action["tx_power"])

        v_energy = max(0.0, est_energy - self.e_max)
        v_latency = max(0.0, est_latency - self.t_max)
        v_power = max(0.0, total_power - self.p_total_max)

        if est_energy > self.e_max and est_energy > 0:
            scale = math.sqrt(self.e_max / est_energy)
            freq *= scale
            power *= scale
        if total_power > self.p_total_max and total_power > 0:
            power *= self.p_total_max / total_power

        penalty = self.penalty_coeff * (v_energy * v_energy + v_latency * v_latency + v_power * v_power)

        b_energy = max(self.e_max - est_energy, 1e-6)
        b_latency = max(self.t_max - est_latency, 1e-6)
        b_power = max(self.p_total_max - total_power, 1e-6)
        barrier = -self.barrier_coeff * (math.log(b_energy) + math.log(b_latency) + math.log(b_power))

        return ProjectionResult(
            action={
                "target": int(action["target"]),
                "offload_ratio": float(np.clip(off, 0.0, 1.0)),
                "cpu_freq": float(freq),
                "tx_power": float(power),
            },
            penalty=float(penalty),
            barrier=float(barrier),
            violation_energy=float(v_energy),
            violation_power=float(v_power),
            violation_latency=float(v_latency),
        )


class HierarchicalReward:
    """Reward = alpha * communication performance + beta * cooperative + gamma * equilibrium."""

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        conflict_threshold: float = 0.1,
        window: int = 50,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.conflict_threshold = float(conflict_threshold)
        self.window = max(10, int(window))
        self.history: deque[Dict[str, float]] = deque(maxlen=max(100, self.window * 2))

    def adaptive_weights(self) -> Tuple[float, float, float]:
        if len(self.history) < self.window:
            total = self.alpha + self.beta + self.gamma
            return self.alpha / total, self.beta / total, self.gamma / total

        recent = list(self.history)[-self.window :]
        r_imm = np.array([x["r_imm"] for x in recent], dtype=np.float64)
        r_coop = np.array([x["r_coop"] for x in recent], dtype=np.float64)
        r_eq = np.array([x["r_eq"] for x in recent], dtype=np.float64)

        corr_ic = np.corrcoef(r_imm, r_coop)[0, 1] if np.std(r_imm) > 1e-6 and np.std(r_coop) > 1e-6 else 0.0
        corr_ie = np.corrcoef(r_imm, r_eq)[0, 1] if np.std(r_imm) > 1e-6 and np.std(r_eq) > 1e-6 else 0.0

        beta_adj = self.beta * max(0.5, 1.0 + corr_ic)
        gamma_adj = self.gamma * max(0.5, 1.0 + corr_ie)
        total = self.alpha + beta_adj + gamma_adj
        a = self.alpha / total
        b = beta_adj / total
        g = gamma_adj / total
        if self.alpha >= 0.75 and a < 0.70:
            rem = max(b + g, EPS)
            b = 0.30 * b / rem
            g = 0.30 * g / rem
            a = 0.70
        return a, b, g

    def compute(
        self,
        latency: float,
        energy: float,
        latency_budget: float,
        energy_budget: float,
        shapley_value: float,
        cooperation_gain: float,
        action_vec: np.ndarray,
        equilibrium_action_vec: np.ndarray,
        queue_wait: float = 0.0,
        deadline: Optional[float] = None,
        nearest_server_selected: bool = True,
        penalty: float = 0.0,
        barrier: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        effective_deadline = float(deadline) if deadline is not None else float(latency_budget)
        latency_ratio = float(latency) / max(float(latency_budget), EPS)
        queue_ratio = float(queue_wait) / max(float(latency_budget), EPS)
        energy_ratio = float(energy) / max(float(energy_budget), EPS)
        deadline_miss_ratio = max(0.0, float(latency) - effective_deadline) / max(effective_deadline, EPS)
        non_nearest_penalty = 0.0 if nearest_server_selected else 0.05
        r_imm = -(
            0.40 * latency_ratio
            + 0.15 * queue_ratio
            + 0.20 * deadline_miss_ratio * deadline_miss_ratio
            + 0.25 * energy_ratio
            + non_nearest_penalty
        )
        r_coop = float(np.clip(float(shapley_value) * float(cooperation_gain), -1.0, 1.0))
        diff = np.asarray(action_vec, dtype=np.float64) - np.asarray(equilibrium_action_vec, dtype=np.float64)
        r_eq = -float(np.dot(diff, diff) / max(diff.size, 1))

        a, b, g = self.adaptive_weights()
        total = a * r_imm + b * r_coop + g * r_eq - penalty - barrier
        terms = {
            "r_imm": float(r_imm),
            "r_coop": float(r_coop),
            "r_eq": float(r_eq),
            "alpha": float(a),
            "beta": float(b),
            "gamma": float(g),
            "penalty": float(penalty),
            "barrier": float(barrier),
            "queue_wait": float(queue_wait),
            "deadline_miss_ratio": float(deadline_miss_ratio),
            "nearest_server_selected": float(1.0 if nearest_server_selected else 0.0),
        }
        self.history.append({"r_imm": float(r_imm), "r_coop": float(r_coop), "r_eq": float(r_eq)})
        return float(total), terms


class GameTheoryMECEnv(BaseMECEnv):
    """
    Multi-base-station cooperative MEC environment with game-theory modules.

    External contracts kept:
    - reset() -> (List[np.ndarray], info)
    - step(List[Dict]) -> (List[np.ndarray], List[float], terminated, truncated, info)
    """

    def __init__(
        self,
        num_agents: int = 3,
        num_edge_servers: int = 3,
        max_steps: int = 100,
        enable_game_init: bool = True,
        enable_shapley: bool = True,
        enable_efx: bool = True,
        enable_cp_nets: bool = True,
        enable_action_projection: bool = True,
        game_history_len: int = 5,
        shapley_samples: int = 128,
        shapley_antithetic: bool = True,
        reward_weights: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        efx_transfer_rate: float = 0.5,
        # Budgets and physical ranges
        energy_budget: float = 10.0,
        latency_budget: float = 2.0,
        f_min: float = 0.5e9,
        f_max: float = 3.0e9,
        p_min: float = 0.01,
        p_max: float = 0.5,
        p_total_max: float = 1.5,
        edge_cpu_freq: float = 5.0e9,
        # Queue/channel parameters
        queue_stability_margin: float = 0.95,
        channel_fc_ghz: float = 3.5,
        # Pricing model
        price_min: float = 0.1,
        price_max: float = 2.0,
        price_eta: Optional[List[float]] = None,
        price_c: Optional[List[float]] = None,
        price_d0: Optional[List[float]] = None,
        # Base station positions
        bs_positions: Optional[List[List[float]]] = None,
        ris_elements: int = 0,
        **kwargs: Any,
    ) -> None:
        self._num_edge_servers = int(num_edge_servers)
        self.num_agents = int(num_agents)
        self.enable_game_init = bool(enable_game_init)
        self.enable_shapley = bool(enable_shapley)
        self.enable_efx = bool(enable_efx)
        self.enable_cp_nets = bool(enable_cp_nets)
        self.enable_action_projection = bool(enable_action_projection)
        self.game_history_len = int(game_history_len)
        self.energy_budget = float(energy_budget)
        self.latency_budget = float(latency_budget)
        self.edge_cpu_freq = float(edge_cpu_freq)
        self.ris_elements = int(ris_elements)
        self.efx_transfer_rate = float(np.clip(efx_transfer_rate, 0.05, 1.0))
        self.max_velocity = 5.0

        if bs_positions is None:
            self.bs_positions = [[10.0 + i * 15.0, 0.0, 25.0] for i in range(self._num_edge_servers)]
        else:
            self.bs_positions = bs_positions

        # Base env init (we override reset/step, but reuse task generation and helpers).
        super().__init__(num_edge_servers=self._num_edge_servers, max_steps=max_steps, **kwargs)

        self._rng = self.np_random
        self.channel3gpp = Channel3GPP(fc_ghz=channel_fc_ghz, bandwidth=self.channel.bandwidth)
        self.dvfs_model = DVFSEnergyModel(
            kappa_dyn=1e-27,
            alpha=2.7,
            p_leak=0.01,
            f_min=f_min,
            f_max=f_max,
        )
        self.queue_model = QueueingDelayModel(stability_margin=queue_stability_margin)
        self.action_mapper = ParameterizedActionSpace(
            k=self._num_edge_servers,
            f_min=f_min,
            f_max=f_max,
            p_min=p_min,
            p_max=p_max,
        )
        self.constraint_projector = ConstraintProjection(
            e_max=self.energy_budget,
            t_max=self.latency_budget,
            p_total_max=p_total_max,
            penalty_coeff=0.05,
            barrier_coeff=1e-4,
        )

        eta = np.asarray(price_eta if price_eta is not None else [0.35] * self._num_edge_servers, dtype=np.float32)
        c = np.asarray(price_c if price_c is not None else [0.12] * self._num_edge_servers, dtype=np.float32)
        d0 = np.asarray(price_d0 if price_d0 is not None else [1.0] * self._num_edge_servers, dtype=np.float32)
        self.pricing = OptimalPricingMechanism(
            eta=eta,
            c=c,
            d0=d0,
            p_min=price_min,
            p_max=price_max,
            rng=self._rng,
        )
        self.bilevel_solver = BilevelGameSolver(
            pricing=self.pricing,
            n_users=self.num_agents,
            n_servers=self._num_edge_servers,
            rng=self._rng,
        )
        self.stackelberg = self.bilevel_solver  # compatibility alias
        self.shapley_estimator = MonteCarloShapley(
            n_agents=self.num_agents,
            n_samples=shapley_samples,
            use_antithetic=shapley_antithetic,
            rng=self._rng,
        )
        self.cpnet_model = CPNetPreferenceModel(n_agents=self.num_agents, n_servers=self._num_edge_servers)
        self.efx_allocator = EFXFairAllocation(
            n_agents=self.num_agents,
            transfer_rate=self.efx_transfer_rate,
            max_iters=32,
        )

        self.reward_model = HierarchicalReward(*reward_weights)
        self.reward_shaper = MECRewardShaper()

        # Dynamic states.
        self.queue_lengths = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.prev_queue = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.delta_q = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.rho = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.arrival_rates = np.zeros(self._num_edge_servers, dtype=np.float32)
        avg_cycles = float(np.mean([self.task_cpu_range[0], self.task_cpu_range[1]]))
        self.service_rates = np.full(
            self._num_edge_servers,
            self.edge_cpu_freq / max(avg_cycles, 1.0),
            dtype=np.float32,
        )

        self.user_mobility = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.channel_qualities = np.zeros((self.num_agents, self._num_edge_servers), dtype=np.float32)
        self.prev_channel_qualities = np.zeros((self.num_agents, self._num_edge_servers), dtype=np.float32)
        self.delta_snr = np.zeros((self.num_agents, self._num_edge_servers), dtype=np.float32)
        self.last_tx_powers = np.full(self.num_agents, p_min, dtype=np.float32)

        self.agent_tasks: List[Optional[Dict[str, Any]]] = [None] * self.num_agents
        self.action_counts = np.zeros(self._num_edge_servers + 1, dtype=np.float32)
        self.price_history: deque[np.ndarray] = deque(maxlen=self.game_history_len)
        self.shapley_history: deque[np.ndarray] = deque(maxlen=self.game_history_len)

        self.latest_equilibrium_prices = np.full(self._num_edge_servers, (price_min + price_max) * 0.5, dtype=np.float32)
        self.latest_equilibrium_actions = np.zeros((self.num_agents, 4), dtype=np.float32)
        self.latest_demands = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.latest_shapley = np.full(self.num_agents, 1.0 / max(self.num_agents, 1), dtype=np.float32)
        self.latest_shapley_credit = np.full(self.num_agents, 1.0 / max(self.num_agents, 1), dtype=np.float32)
        self.latest_efx_transfers = np.zeros(self.num_agents, dtype=np.float32)
        self.latest_cpnet_scores = np.zeros(self.num_agents, dtype=np.float32)
        self.latest_efx_satisfied = True
        self.latest_efx_iterations = 0
        self.latest_efx_violation: Optional[Tuple[int, int, str]] = None

        self.game_history = {"prices": [], "demands": [], "equilibrium_steps": []}

        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.multi_action_space = spaces.MultiDiscrete([self._num_edge_servers + 1] * self.num_agents)
        self.action_space = spaces.Dict(
            {
                "target": spaces.Discrete(self._num_edge_servers + 1),
                "ratio": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

    def _get_obs_dim(self) -> int:
        k = self._num_edge_servers
        m = self.game_history_len
        return 3 * k + 5 + m * k + (k + 1) + m + 3 * k

    def _get_battery_level(self, agent_id: int) -> float:
        return float(55.0 + 25.0 * math.sin(self.current_step * 0.07 + agent_id))

    def _update_user_mobility(self) -> None:
        for i in range(self.num_agents):
            self.user_mobility[i, 0] += float(self._rng.normal(0.0, 0.8))
            self.user_mobility[i, 1] += float(self._rng.normal(0.0, 0.8))
            dist = float(np.linalg.norm(self.user_mobility[i, :2]))
            if dist > 60.0:
                self.user_mobility[i, :2] *= 60.0 / max(dist, EPS)

    def _distance_user_to_bs(self, agent_id: int, server_idx: int) -> float:
        ux, uy = self.user_mobility[agent_id, 0], self.user_mobility[agent_id, 1]
        bx, by, _ = self.bs_positions[server_idx]
        return float(math.hypot(ux - bx, uy - by) + 1.0)

    def _nearest_server_index(self, agent_id: int) -> int:
        distances = [self._distance_user_to_bs(agent_id, k) for k in range(self._num_edge_servers)]
        return int(np.argmin(distances))

    def _empty_latency_components(self, agent_id: int, target: int) -> Dict[str, float]:
        return {
            "agent_id": float(agent_id),
            "target": float(target),
            "local_compute_time": 0.0,
            "tx_time": 0.0,
            "queue_wait_time": 0.0,
            "edge_compute_time": 0.0,
            "e2e_latency": 0.0,
            "deadline": float(self.latency_budget),
            "deadline_miss": 0.0,
            "selected_server_distance": 0.0,
            "nearest_server_selected": 1.0,
        }

    def _update_channel_qualities(self) -> None:
        old = self.channel_qualities.copy()
        snr_db = np.zeros_like(self.channel_qualities, dtype=np.float32)
        for i in range(self.num_agents):
            for k in range(self._num_edge_servers):
                d_target = self._distance_user_to_bs(i, k)
                d_int = []
                p_int = []
                for j in range(self.num_agents):
                    if j == i:
                        continue
                    d_int.append(self._distance_user_to_bs(j, k))
                    p_int.append(float(self.last_tx_powers[j]))
                sinr = self.channel3gpp.compute_sinr(
                    d_target=d_target,
                    d_interferers=np.asarray(d_int, dtype=np.float32),
                    p_target_w=float(self.last_tx_powers[i]),
                    p_interferers_w=np.asarray(p_int, dtype=np.float32),
                )
                snr_db[i, k] = float(10.0 * math.log10(max(sinr, 1e-12)))
        self.channel_qualities = snr_db
        self.delta_snr = self.channel_qualities - old
        self.prev_channel_qualities = self.channel_qualities.copy()

    def _build_history_features(self, agent_id: int) -> Tuple[np.ndarray, np.ndarray]:
        k = self._num_edge_servers
        m = self.game_history_len
        price_hist = np.zeros(m * k, dtype=np.float32)
        for t, p in enumerate(self.price_history):
            price_hist[t * k : (t + 1) * k] = p

        shap_hist = np.zeros(m, dtype=np.float32)
        for t, s in enumerate(self.shapley_history):
            shap_hist[t] = s[agent_id]

        return price_hist, shap_hist

    def _get_agent_obs(self, agent_id: int) -> np.ndarray:
        queue_obs = self.queue_lengths / max(float(np.max(self.queue_lengths) + 1.0), 1.0)
        snr_obs = np.clip((self.channel_qualities[agent_id] + 20.0) / 40.0, 0.0, 1.0)
        cpu_load = np.clip(self.rho, 0.0, 1.0)

        task = self.agent_tasks[agent_id]
        if task is None:
            task_features = np.zeros(5, dtype=np.float32)
        else:
            task_features = np.array(
                [
                    self._get_battery_level(agent_id) / 100.0,
                    float(task["deadline"]) / 5.0,
                    float(task["data_size"]) / 1e6,
                    float(task["cpu_cycles"]) / 1e9,
                    float(np.linalg.norm(self.user_mobility[agent_id, :2])) / 60.0,
                ],
                dtype=np.float32,
            )

        price_hist, shap_hist = self._build_history_features(agent_id)
        action_freq = self.action_counts / max(float(np.sum(self.action_counts)), 1.0)
        delta_snr = np.clip(self.delta_snr[agent_id] / 20.0, -1.0, 1.0)
        delta_q = np.clip(self.delta_q / 5.0, -1.0, 1.0)
        rho = np.clip(self.rho, 0.0, 1.0)

        return np.concatenate(
            [
                queue_obs,
                snr_obs,
                cpu_load,
                task_features,
                price_hist,
                action_freq,
                shap_hist,
                delta_snr,
                delta_q,
                rho,
            ]
        ).astype(np.float32)

    def _get_global_obs(self) -> np.ndarray:
        all_obs = [self._get_agent_obs(i) for i in range(self.num_agents)]
        return np.concatenate(all_obs).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return self._get_agent_obs(0)

    def _compute_game_equilibrium_hints(self) -> Dict[str, np.ndarray]:
        prices, eq_actions, demands = self.bilevel_solver.solve(
            channel_db=self.channel_qualities,
            queue_lengths=self.queue_lengths,
            tasks=self.agent_tasks,
        )
        self.latest_equilibrium_prices = prices
        self.latest_equilibrium_actions = eq_actions
        self.latest_demands = demands
        self.price_history.append(prices.copy())

        self.game_history["prices"].append(prices.copy())
        self.game_history["demands"].append(demands.copy())
        self.game_history["equilibrium_steps"].append(self.current_step)

        eq_targets = np.array(
            [int(np.clip(np.round((a[0] + 1.0) * 0.5 * self._num_edge_servers), 0, self._num_edge_servers)) for a in eq_actions],
            dtype=np.int32,
        )

        return {
            "equilibrium_prices": prices.copy(),
            "equilibrium_actions": eq_actions.copy(),
            "equilibrium_targets": eq_targets,
            "predicted_demands": demands.copy(),
        }

    def _local_baseline_components(self, task: Optional[Dict[str, Any]], agent_id: int) -> Tuple[Dict[str, float], float]:
        if task is None:
            return self._empty_latency_components(agent_id=agent_id, target=0), 0.0
        cpu_cycles = float(task["cpu_cycles"])
        freq = float(self.action_mapper.f_max)
        local_time = cpu_cycles / max(freq, EPS)
        local_energy = self.dvfs_model.compute_energy(freq, cpu_cycles)
        comp = self._empty_latency_components(agent_id=agent_id, target=0)
        comp.update(
            {
                "local_compute_time": float(local_time),
                "e2e_latency": float(local_time),
                "deadline": float(task.get("deadline", self.latency_budget)),
                "deadline_miss": float(local_time > float(task.get("deadline", self.latency_budget))),
                "nearest_server_selected": 1.0,
            }
        )
        return comp, float(local_energy)

    def _communication_cost(self, components: Dict[str, float], energy: float) -> float:
        latency = float(components.get("e2e_latency", 0.0))
        queue_wait = float(components.get("queue_wait_time", 0.0))
        deadline = float(components.get("deadline", self.latency_budget))
        deadline_miss_ratio = max(0.0, latency - deadline) / max(deadline, EPS)
        non_nearest = 0.0 if bool(components.get("nearest_server_selected", 1.0)) else 1.0
        return float(
            0.45 * latency / max(self.latency_budget, EPS)
            + 0.15 * queue_wait / max(self.latency_budget, EPS)
            + 0.10 * deadline_miss_ratio * deadline_miss_ratio
            + 0.25 * float(energy) / max(self.energy_budget, EPS)
            + 0.05 * non_nearest
        )

    def _estimate_full_offload_components(
        self,
        task: Optional[Dict[str, Any]],
        agent_id: int,
        server_idx: int,
        queue_wait: float,
    ) -> Tuple[Dict[str, float], float]:
        if task is None:
            return self._empty_latency_components(agent_id=agent_id, target=server_idx + 1), 0.0
        cpu_cycles = float(task["cpu_cycles"])
        data_size = float(task["data_size"])
        snr_lin = 10.0 ** (float(self.channel_qualities[agent_id, server_idx]) / 10.0)
        rate = self.channel3gpp.rate(snr_lin)
        tx_time = data_size / max(rate, EPS)
        edge_time = cpu_cycles / max(self.edge_cpu_freq, EPS)
        e2e_latency = tx_time + float(queue_wait) + edge_time
        tx_energy = self.energy_model.transmission_energy(self.action_mapper.p_max, tx_time)
        distance = self._distance_user_to_bs(agent_id, server_idx)
        nearest_idx = self._nearest_server_index(agent_id)
        comp = {
            "agent_id": float(agent_id),
            "target": float(server_idx + 1),
            "local_compute_time": 0.0,
            "tx_time": float(tx_time),
            "queue_wait_time": float(queue_wait),
            "edge_compute_time": float(edge_time),
            "e2e_latency": float(e2e_latency),
            "deadline": float(task.get("deadline", self.latency_budget)),
            "deadline_miss": float(e2e_latency > float(task.get("deadline", self.latency_budget))),
            "selected_server_distance": float(distance),
            "nearest_server_selected": float(server_idx == nearest_idx),
        }
        return comp, float(tx_energy)

    def _coalition_value(self, coalition: frozenset[int]) -> float:
        if len(coalition) == 0:
            return 0.0
        queue_delay, _, _ = self.queue_model.compute_delay(self.arrival_rates, self.service_rates)
        extra_arrivals = np.zeros(self._num_edge_servers, dtype=np.float64)
        baseline_cost = 0.0
        cooperative_cost = 0.0

        for i in sorted(coalition):
            task = self.agent_tasks[i]
            local_comp, local_energy = self._local_baseline_components(task, i)
            local_cost = self._communication_cost(local_comp, local_energy)
            baseline_cost += local_cost

            best_cost = local_cost
            best_server = -1
            for k in range(self._num_edge_servers):
                queue_wait = float(queue_delay[k]) + float(extra_arrivals[k]) / max(float(self.service_rates[k]), EPS)
                edge_comp, edge_energy = self._estimate_full_offload_components(task, i, k, queue_wait)
                edge_cost = self._communication_cost(edge_comp, edge_energy)
                if edge_cost < best_cost:
                    best_cost = edge_cost
                    best_server = k
            if best_server >= 0:
                extra_arrivals[best_server] += 1.0
            cooperative_cost += best_cost

        return float(baseline_cost - cooperative_cost)

    def _compute_shapley_allocation(self) -> np.ndarray:
        if not self.enable_shapley:
            phi = np.full(self.num_agents, 1.0 / max(self.num_agents, 1), dtype=np.float32)
            self.latest_shapley_credit = phi.copy()
            return phi

        raw_phi = self.shapley_estimator.compute(self._coalition_value)
        credit_denom = float(np.sum(np.abs(raw_phi)))
        if credit_denom <= EPS:
            self.latest_shapley_credit = np.zeros(self.num_agents, dtype=np.float32)
        else:
            self.latest_shapley_credit = (raw_phi / credit_denom).astype(np.float32)

        # Keep a non-negative allocation for existing trainers that use it as a multiplier.
        alloc_source = np.maximum(raw_phi, 0.0) + 1e-6
        s = float(np.sum(alloc_source))
        if s <= 0:
            phi = np.full(self.num_agents, 1.0 / max(self.num_agents, 1), dtype=np.float32)
        else:
            phi = (alloc_source / s).astype(np.float32)
        self.latest_shapley = phi
        self.shapley_history.append(phi.copy())
        return phi

    def _compute_efx_transfers(
        self,
        mapped_actions: List[Dict[str, float]],
        prices: np.ndarray,
    ) -> EFXRepairResult:
        if not self.enable_efx:
            zeros = np.zeros(self.num_agents, dtype=np.float32)
            return EFXRepairResult(transfers=zeros, is_efx=True, iterations=0, violation=None)

        if self.enable_cp_nets:
            valuations, server_util, offload_util = self.cpnet_model.build_valuation_functions(
                channel_db=self.channel_qualities,
                queue_lengths=self.queue_lengths,
                prices=prices,
                tasks=self.agent_tasks,
            )
            allocation = self.cpnet_model.build_allocation(mapped_actions)
            repair = self.efx_allocator.repair_allocation(allocation, valuations)

            cp_scores = np.zeros(self.num_agents, dtype=np.float32)
            for i, action in enumerate(mapped_actions):
                target = int(np.clip(action["target"], 0, self._num_edge_servers))
                off_level = self.cpnet_model._offload_level(float(action["offload_ratio"]))
                off_idx = {"low": 0, "mid": 1, "high": 2}[off_level]
                cp_scores[i] = float(server_util[i, target] + offload_util[i, off_idx])
            self.latest_cpnet_scores = cp_scores.astype(np.float32)
            return repair

        # Fallback valuation without CP-nets (queue + channel proxy).
        allocation = []
        valuations: List[Callable[[set[str]], float]] = []
        for i, action in enumerate(mapped_actions):
            target = int(np.clip(action["target"], 0, self._num_edge_servers))
            allocation.append({f"srv_{target}", f"off_{self.cpnet_model._offload_level(float(action['offload_ratio']))}"})

            def _v(bundle: set[str], idx: int = i) -> float:
                if not bundle:
                    return 0.0
                srv = 0
                for item in bundle:
                    if item.startswith("srv_"):
                        srv = int(np.clip(int(item.split("_", 1)[1]), 0, self._num_edge_servers))
                if srv == 0:
                    return 0.25
                snr = float(np.clip((self.channel_qualities[idx, srv - 1] + 20.0) / 40.0, 0.0, 1.0))
                q = float(np.clip(self.queue_lengths[srv - 1] / max(float(np.max(self.queue_lengths) + 1.0), 1.0), 0.0, 1.0))
                return 0.65 * snr - 0.35 * q

            valuations.append(_v)
        self.latest_cpnet_scores = np.zeros(self.num_agents, dtype=np.float32)
        return self.efx_allocator.repair_allocation(allocation, valuations)

    def _estimate_action_costs(
        self,
        task: Optional[Dict[str, Any]],
        action: Dict[str, float],
        agent_id: int,
    ) -> Tuple[float, float]:
        if task is None:
            return 0.0, 0.0
        cpu_cycles = float(task["cpu_cycles"])
        off = float(action["offload_ratio"])
        freq = float(action["cpu_freq"])
        tx_power = float(action["tx_power"])
        if int(action["target"]) == 0:
            off = 0.0
        local_cycles = max(cpu_cycles * (1.0 - off), 0.0)
        local_time = local_cycles / max(freq, EPS)
        local_energy = self.dvfs_model.compute_energy(freq, local_cycles)

        if int(action["target"]) == 0 or off <= 1e-3:
            return local_energy, local_time

        target_idx = int(action["target"]) - 1
        snr_lin = 10.0 ** (float(self.channel_qualities[agent_id, target_idx]) / 10.0)
        rate = self.channel3gpp.rate(snr_lin)
        tx_bits = float(task["data_size"]) * off
        tx_time = tx_bits / max(rate, EPS)
        edge_cycles = cpu_cycles * off
        edge_time = edge_cycles / max(self.edge_cpu_freq, EPS)
        queue_wait = float(self.queue_lengths[target_idx]) / max(float(self.service_rates[target_idx]), EPS)
        tx_energy = self.energy_model.transmission_energy(tx_power, tx_time)

        latency = max(local_time, tx_time + queue_wait + edge_time)
        energy = local_energy + tx_energy
        return float(energy), float(latency)

    def _process_action_game_theory(
        self,
        task: Optional[Dict[str, Any]],
        action: Dict[str, float],
        agent_id: int,
        queue_delay: np.ndarray,
    ) -> Tuple[float, float, Dict[str, float]]:
        if task is None:
            return 0.0, 0.0, self._empty_latency_components(agent_id=agent_id, target=int(action.get("target", 0)))

        off = float(action["offload_ratio"])
        target = int(action["target"])
        freq = float(action["cpu_freq"])
        tx_power = float(action["tx_power"])
        cpu_cycles = float(task["cpu_cycles"])
        data_size = float(task["data_size"])
        if target == 0:
            off = 0.0

        local_cycles = cpu_cycles * (1.0 - off)
        local_time = local_cycles / max(freq, EPS)
        local_energy = self.dvfs_model.compute_energy(freq, local_cycles)
        deadline = float(task.get("deadline", self.latency_budget))

        if target == 0 or off <= 1e-3:
            components = self._empty_latency_components(agent_id=agent_id, target=0)
            components.update(
                {
                    "local_compute_time": float(local_time),
                    "e2e_latency": float(local_time),
                    "deadline": deadline,
                    "deadline_miss": float(local_time > deadline),
                    "nearest_server_selected": 1.0,
                }
            )
            return float(local_time), float(local_energy), components

        target_idx = target - 1
        snr_lin = 10.0 ** (float(self.channel_qualities[agent_id, target_idx]) / 10.0)
        rate = self.channel3gpp.rate(snr_lin)
        tx_bits = data_size * off
        tx_time = tx_bits / max(rate, EPS)
        edge_cycles = cpu_cycles * off
        edge_time = edge_cycles / max(self.edge_cpu_freq, EPS)
        wait_time = float(queue_delay[target_idx])

        latency = max(local_time, tx_time + wait_time + edge_time)
        energy = local_energy + self.energy_model.transmission_energy(tx_power, tx_time)
        distance = self._distance_user_to_bs(agent_id, target_idx)
        nearest_idx = self._nearest_server_index(agent_id)
        components = {
            "agent_id": float(agent_id),
            "target": float(target),
            "local_compute_time": float(local_time),
            "tx_time": float(tx_time),
            "queue_wait_time": float(wait_time),
            "edge_compute_time": float(edge_time),
            "e2e_latency": float(latency),
            "deadline": deadline,
            "deadline_miss": float(latency > deadline),
            "selected_server_distance": float(distance),
            "nearest_server_selected": float(target_idx == nearest_idx),
        }
        return float(latency), float(energy), components

    def _build_info(
        self,
        game_hints: Dict[str, np.ndarray],
        rewards: List[float],
        reward_terms: List[Dict[str, float]],
        efx_transfers: np.ndarray,
        projection_terms: List[ProjectionResult],
        individual_latencies: List[float],
        individual_energies: List[float],
        individual_offload: List[float],
        latency_components: List[Dict[str, float]],
        eq_actions: np.ndarray,
    ) -> Dict[str, Any]:
        queue_delay, queue_system_delay, rho = self.queue_model.compute_delay(self.arrival_rates, self.service_rates)
        constraint_metrics = {
            "penalty_mean": float(np.mean([x.penalty for x in projection_terms])) if projection_terms else 0.0,
            "barrier_mean": float(np.mean([x.barrier for x in projection_terms])) if projection_terms else 0.0,
            "violation_energy_mean": float(np.mean([x.violation_energy for x in projection_terms])) if projection_terms else 0.0,
            "violation_power_mean": float(np.mean([x.violation_power for x in projection_terms])) if projection_terms else 0.0,
            "violation_latency_mean": float(np.mean([x.violation_latency for x in projection_terms])) if projection_terms else 0.0,
        }
        queue_metrics = {
            "arrival_rates": self.arrival_rates.copy(),
            "service_rates": self.service_rates.copy(),
            "queue_delay": queue_delay,
            "system_delay": queue_system_delay,
            "rho": rho,
        }
        fairness_metrics = {
            "efx_enabled": bool(self.enable_efx),
            "cpnet_enabled": bool(self.enable_cp_nets),
            "efx_satisfied": bool(self.latest_efx_satisfied),
            "efx_iterations": int(self.latest_efx_iterations),
            "efx_violation": self.latest_efx_violation,
            "efx_transfer_sum_abs": float(np.sum(np.abs(efx_transfers))),
            "efx_transfers": np.asarray(efx_transfers, dtype=np.float32).copy(),
            "cpnet_scores": self.latest_cpnet_scores.copy(),
        }
        e2e = np.asarray([c.get("e2e_latency", 0.0) for c in latency_components], dtype=np.float64)
        queue_wait = np.asarray([c.get("queue_wait_time", 0.0) for c in latency_components], dtype=np.float64)
        deadlines = np.asarray([c.get("deadline", self.latency_budget) for c in latency_components], dtype=np.float64)
        nearest = np.asarray([c.get("nearest_server_selected", 1.0) for c in latency_components], dtype=np.float64)
        deadline_miss = e2e > deadlines
        communication_metrics = {
            "e2e_latency_mean": float(np.mean(e2e)) if e2e.size else 0.0,
            "e2e_latency_max": float(np.max(e2e)) if e2e.size else 0.0,
            "queue_wait_mean": float(np.mean(queue_wait)) if queue_wait.size else 0.0,
            "queue_wait_max": float(np.max(queue_wait)) if queue_wait.size else 0.0,
            "deadline_miss_count": int(np.sum(deadline_miss)) if deadline_miss.size else 0,
            "deadline_miss_rate": float(np.mean(deadline_miss)) if deadline_miss.size else 0.0,
            "offload_ratio_mean": float(np.mean(individual_offload)) if individual_offload else 0.0,
            "non_nearest_server_rate": float(np.mean(1.0 - nearest)) if nearest.size else 0.0,
        }
        return {
            "global_obs": self._get_global_obs(),
            "agent_tasks": self.agent_tasks.copy(),
            "game_hints": game_hints,
            "eq_actions": eq_actions.copy(),
            "shapley_allocation": self.latest_shapley.copy(),
            "shapley_credit": self.latest_shapley_credit.copy(),
            "reward_terms": reward_terms,
            "queue_metrics": queue_metrics,
            "constraint_metrics": constraint_metrics,
            "fairness_metrics": fairness_metrics,
            "communication_metrics": communication_metrics,
            "latency_components": latency_components,
            "individual_latencies": individual_latencies,
            "individual_energies": individual_energies,
            "individual_offload": individual_offload,
            "queue_lengths": self.queue_lengths.copy(),
            "task_completed": int(self.task_completed),
            "avg_latency": float(self.total_latency / max(1, self.task_completed)),
            "avg_energy": float(self.total_energy / max(1, self.task_completed)),
            "episode_reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "channel_qualities": self.channel_qualities.copy(),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.total_energy = 0.0
        self.total_latency = 0.0
        self.task_completed = 0

        self.queue_lengths = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.prev_queue = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.delta_q = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.rho = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.arrival_rates = np.zeros(self._num_edge_servers, dtype=np.float32)
        self.action_counts = np.zeros(self._num_edge_servers + 1, dtype=np.float32)
        self.price_history.clear()
        self.shapley_history.clear()
        self.latest_shapley_credit = np.full(self.num_agents, 1.0 / max(self.num_agents, 1), dtype=np.float32)
        self.latest_efx_transfers = np.zeros(self.num_agents, dtype=np.float32)
        self.latest_cpnet_scores = np.zeros(self.num_agents, dtype=np.float32)
        self.latest_efx_satisfied = True
        self.latest_efx_iterations = 0
        self.latest_efx_violation = None

        self.user_mobility = np.zeros((self.num_agents, 3), dtype=np.float32)
        for i in range(self.num_agents):
            angle = float(self._rng.uniform(0, 2 * math.pi))
            radius = float(self._rng.uniform(8.0, 30.0))
            self.user_mobility[i, 0] = radius * math.cos(angle)
            self.user_mobility[i, 1] = radius * math.sin(angle)

        self.last_tx_powers = np.full(self.num_agents, self.action_mapper.p_min, dtype=np.float32)
        self._update_channel_qualities()
        self.agent_tasks = [self._generate_task() for _ in range(self.num_agents)]

        game_hints = self._compute_game_equilibrium_hints() if self.enable_game_init else {
            "equilibrium_prices": self.latest_equilibrium_prices.copy(),
            "equilibrium_actions": self.latest_equilibrium_actions.copy(),
            "equilibrium_targets": np.zeros(self.num_agents, dtype=np.int32),
            "predicted_demands": self.latest_demands.copy(),
        }

        shapley = self._compute_shapley_allocation()
        if len(self.shapley_history) == 0:
            self.shapley_history.append(shapley.copy())

        obs = [self._get_agent_obs(i) for i in range(self.num_agents)]
        info = self._build_info(
            game_hints=game_hints,
            rewards=[0.0] * self.num_agents,
            reward_terms=[{"r_imm": 0.0, "r_coop": 0.0, "r_eq": 0.0}] * self.num_agents,
            efx_transfers=self.latest_efx_transfers.copy(),
            projection_terms=[],
            individual_latencies=[0.0] * self.num_agents,
            individual_energies=[0.0] * self.num_agents,
            individual_offload=[0.0] * self.num_agents,
            latency_components=[self._empty_latency_components(agent_id=i, target=0) for i in range(self.num_agents)],
            eq_actions=game_hints["equilibrium_actions"],
        )
        return obs, info

    def step(
        self,
        actions: List[Dict[str, Any]],
    ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict[str, Any]]:
        self.current_step += 1
        self._update_user_mobility()
        self._update_channel_qualities()

        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")

        # First pass: map and estimate constraints.
        mapped_actions: List[Dict[str, float]] = []
        projection_terms: List[ProjectionResult] = []
        est_energy = []
        est_latency = []
        for i, action in enumerate(actions):
            mapped = self.action_mapper.map_action(action)
            e_est, t_est = self._estimate_action_costs(self.agent_tasks[i], mapped, i)
            mapped_actions.append(mapped)
            est_energy.append(e_est)
            est_latency.append(t_est)

        total_power = float(np.sum([a["tx_power"] for a in mapped_actions]))
        for i, mapped in enumerate(mapped_actions):
            if self.enable_action_projection:
                proj = self.constraint_projector.project(
                    action=mapped,
                    est_energy=float(est_energy[i]),
                    est_latency=float(est_latency[i]),
                    total_power=total_power,
                )
                mapped_actions[i] = proj.action
            else:
                proj = ProjectionResult(
                    action=mapped,
                    penalty=0.0,
                    barrier=0.0,
                    violation_energy=0.0,
                    violation_power=0.0,
                    violation_latency=0.0,
                )
            projection_terms.append(proj)
            self.last_tx_powers[i] = float(mapped_actions[i]["tx_power"])

        # Arrival/service/queue.
        self.arrival_rates = np.zeros(self._num_edge_servers, dtype=np.float32)
        for i, act in enumerate(mapped_actions):
            target = int(act["target"])
            if target > 0 and self.agent_tasks[i] is not None:
                self.arrival_rates[target - 1] += float(act["offload_ratio"])
        queue_delay, _, self.rho = self.queue_model.compute_delay(self.arrival_rates, self.service_rates)

        # Process actions and compute raw metrics.
        individual_latencies: List[float] = []
        individual_energies: List[float] = []
        individual_offload: List[float] = []
        latency_components: List[Dict[str, float]] = []
        for i in range(self.num_agents):
            latency, energy, components = self._process_action_game_theory(
                task=self.agent_tasks[i],
                action=mapped_actions[i],
                agent_id=i,
                queue_delay=queue_delay,
            )
            individual_latencies.append(latency)
            individual_energies.append(energy)
            individual_offload.append(float(mapped_actions[i]["offload_ratio"]) if mapped_actions[i]["target"] > 0 else 0.0)
            latency_components.append(components)

        # Update queue state.
        self.queue_lengths = np.maximum(self.queue_lengths + self.arrival_rates - self.service_rates, 0.0).astype(np.float32)
        self.delta_q = self.queue_lengths - self.prev_queue
        self.prev_queue = self.queue_lengths.copy()

        # Compute game hints and Shapley.
        game_hints = self._compute_game_equilibrium_hints() if self.enable_game_init else {
            "equilibrium_prices": self.latest_equilibrium_prices.copy(),
            "equilibrium_actions": self.latest_equilibrium_actions.copy(),
            "equilibrium_targets": np.zeros(self.num_agents, dtype=np.int32),
            "predicted_demands": self.latest_demands.copy(),
        }
        eq_actions = game_hints["equilibrium_actions"]
        shapley_alloc = self._compute_shapley_allocation()
        efx_result = self._compute_efx_transfers(mapped_actions, game_hints["equilibrium_prices"])
        self.latest_efx_transfers = efx_result.transfers.copy()
        self.latest_efx_satisfied = bool(efx_result.is_efx)
        self.latest_efx_iterations = int(efx_result.iterations)
        self.latest_efx_violation = efx_result.violation

        # Team/cooperation signal is positive only when the chosen joint behavior
        # beats each task's local independent baseline on communication cost.
        observed_costs = [
            self._communication_cost(latency_components[i], individual_energies[i])
            for i in range(self.num_agents)
        ]
        baseline_costs = []
        for i in range(self.num_agents):
            base_components, base_energy = self._local_baseline_components(self.agent_tasks[i], i)
            baseline_costs.append(self._communication_cost(base_components, base_energy))
        cooperation_gain = float(np.clip(np.mean(baseline_costs) - np.mean(observed_costs), -1.0, 1.0))
        shapley_credit = getattr(self, "latest_shapley_credit", shapley_alloc)

        rewards: List[float] = []
        reward_terms: List[Dict[str, float]] = []
        for i in range(self.num_agents):
            a_vec = np.array(
                [
                    -1.0 + 2.0 * int(mapped_actions[i]["target"]) / max(1, self._num_edge_servers),
                    2.0 * mapped_actions[i]["offload_ratio"] - 1.0,
                    2.0
                    * ((mapped_actions[i]["cpu_freq"] - self.action_mapper.f_min) / max(self.action_mapper.f_max - self.action_mapper.f_min, EPS))
                    - 1.0,
                    2.0
                    * ((mapped_actions[i]["tx_power"] - self.action_mapper.p_min) / max(self.action_mapper.p_max - self.action_mapper.p_min, EPS))
                    - 1.0,
                ],
                dtype=np.float32,
            )
            r, terms = self.reward_model.compute(
                latency=individual_latencies[i],
                energy=individual_energies[i],
                latency_budget=self.latency_budget,
                energy_budget=self.energy_budget,
                shapley_value=float(shapley_credit[i]),
                cooperation_gain=cooperation_gain,
                action_vec=a_vec,
                equilibrium_action_vec=eq_actions[i],
                queue_wait=float(latency_components[i].get("queue_wait_time", 0.0)),
                deadline=float(latency_components[i].get("deadline", self.latency_budget)),
                nearest_server_selected=bool(latency_components[i].get("nearest_server_selected", 1.0)),
                penalty=projection_terms[i].penalty,
                barrier=projection_terms[i].barrier,
            )
            fair_transfer = float(np.clip(efx_result.transfers[i], -EFX_TRANSFER_CAP, EFX_TRANSFER_CAP))
            r += fair_transfer
            terms["r_fair"] = fair_transfer
            terms["r_fair_raw"] = float(efx_result.transfers[i])
            terms["cp_pref"] = float(self.latest_cpnet_scores[i])
            terms["efx_satisfied"] = float(1.0 if efx_result.is_efx else 0.0)
            terms["efx_iterations"] = float(efx_result.iterations)
            terms["communication_cost"] = float(observed_costs[i])
            terms["cooperation_gain"] = float(cooperation_gain)
            rewards.append(r)
            reward_terms.append(terms)

            self.total_latency += individual_latencies[i]
            self.total_energy += individual_energies[i]
            self.task_completed += 1
            self.action_counts[int(mapped_actions[i]["target"])] += 1.0

        # New tasks for next step.
        self.agent_tasks = [self._generate_task() for _ in range(self.num_agents)]
        obs = [self._get_agent_obs(i) for i in range(self.num_agents)]

        terminated = False
        truncated = self.current_step >= self.max_steps

        info = self._build_info(
            game_hints=game_hints,
            rewards=rewards,
            reward_terms=reward_terms,
            efx_transfers=efx_result.transfers,
            projection_terms=projection_terms,
            individual_latencies=individual_latencies,
            individual_energies=individual_energies,
            individual_offload=individual_offload,
            latency_components=latency_components,
            eq_actions=eq_actions,
        )
        return obs, rewards, terminated, truncated, info
