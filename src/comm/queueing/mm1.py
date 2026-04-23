"""
M/M/1 and M/M/K Queue Models

排队论模型用于分析服务器队列性能
"""

import numpy as np
from typing import Tuple, Optional


class MM1Queue:
    """
    M/M/1 排队系统

    - 到达: 泊松过程, 率 λ
    - 服务: 指数分布, 率 μ
    - 服务员: 1 个

    Args:
        arrival_rate: 到达率 λ (requests/sec)
        service_rate: 服务率 μ (requests/sec)
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
    ):
        if arrival_rate >= service_rate:
            raise ValueError(f"arrival_rate ({arrival_rate}) must be < service_rate ({service_rate})")

        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate  # 利用率

    @property
    def rho(self) -> float:
        """利用率"""
        return self._rho

    @rho.setter
    def rho(self, value: float):
        self._rho = value

    def stability_condition(self) -> bool:
        """判断稳定性 (λ < μ)"""
        return self.lambda_ < self.mu

    def mean_queue_length(self) -> float:
        """
        平均队列长度 (正在排队的人数)

        L_q = ρ² / (1 - ρ)
        """
        return self.rho ** 2 / (1 - self.rho)

    def mean_system_length(self) -> float:
        """
        平均系统人数 (队列 + 服务中)

        L = ρ / (1 - ρ) = L_q + ρ
        """
        return self.rho / (1 - self.rho)

    def mean_waiting_time(self) -> float:
        """
        平均等待时间 (排队时间)

        W_q = L_q / λ = ρ / (μ - λ)
        """
        return self.mean_queue_length() / self.lambda_

    def mean_sojourn_time(self) -> float:
        """
        平均逗留时间 (等待 + 服务)

        W = W_q + 1/μ = 1 / (μ - λ)
        """
        return 1 / (self.mu - self.lambda_)

    def probability_n_in_system(self, n: int) -> float:
        """
        系统中有 n 个顾客的概率

        P_n = (1 - ρ) * ρ^n
        """
        return (1 - self.rho) * (self.rho ** n)

    def probability_wait_exceeds(self, t: float) -> float:
        """
        等待时间超过 t 的概率

        P(W_q > t) = ρ * exp(-(μ - λ) * t)
        """
        return self.rho * np.exp(-(self.mu - self.lambda_) * t)

    def throughput(self) -> float:
        """吞吐量 (等于 λ 当系统稳定)"""
        return self.lambda_


class MMCQueue:
    """
    M/M/c 排队系统 (多服务器)

    - 到达: 泊松过程, 率 λ
    - 服务: 指数分布, 率 μ
    - 服务员: c 个

    Args:
        arrival_rate: 到达率 λ
        service_rate: 服务率 μ (每个服务器)
        num_servers: 服务器数 c
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        num_servers: int,
    ):
        self.lambda_ = arrival_rate
        self.mu = service_rate
        self.c = num_servers

        # 利用率
        self.rho = arrival_rate / (num_servers * service_rate)

        if self.rho >= 1:
            raise ValueError(f"System unstable: rho = {self.rho:.2f} >= 1")

    def stability_condition(self) -> bool:
        return self.rho < 1

    def probability_all_servers_busy(self) -> float:
        """
        所有服务器都忙的概率 (Erlang C 公式的一部分)

        P_wait = (C_c(α) * α^c) / (c! * (1 - ρ) + C_c(α) * α^c)

        其中 α = λ/μ, C_c(α) 是 Erlang C 公式
        """
        alpha = self.lambda_ / self.mu

        # Erlang C 公式
        # C_c(α) = (α^c / c!) / (∑_{n=0}^{c-1} α^n/n! + α^c/(c!*(1-ρ)))
        sum_term = sum((alpha ** n) / np.math.factorial(n) for n in range(self.c))
        c_factor = (alpha ** self.c) / (np.math.factorial(self.c) * (1 - self.rho))

        Erlang_C = (alpha ** self.c) / (np.math.factorial(self.c)) / (sum_term + c_factor)

        # P_wait: 到达时需要等待的概率
        numerator = Erlang_C * (alpha ** self.c) / np.math.factorial(self.c)
        denominator = sum_term + numerator / (1 - self.rho)
        P_wait = numerator / denominator

        return P_wait

    def mean_queue_length(self) -> float:
        """
        平均队列长度
        """
        P_wait = self.probability_all_servers_busy()
        alpha = self.lambda_ / self.mu

        L_q = P_wait * (alpha / (self.c * (1 - self.rho)))

        return L_q

    def mean_system_length(self) -> float:
        """
        平均系统人数
        """
        L_q = self.mean_queue_length()
        L = L_q + self.lambda_ / self.mu
        return L

    def mean_waiting_time(self) -> float:
        """平均等待时间"""
        return self.mean_queue_length() / self.lambda_

    def mean_sojourn_time(self) -> float:
        """平均逗留时间"""
        return self.mean_waiting_time() + 1 / self.mu

    def probability_wait_exceeds(self, t: float) -> float:
        """
        等待时间超过 t 的近似概率
        """
        P_wait = self.probability_all_servers_busy()
        eff_lambda = self.lambda_ * (1 - P_wait)
        mu_eff = self.c * self.mu - self.lambda_

        if mu_eff <= 0:
            return 1.0

        return P_wait * np.exp(-mu_eff * t)


class MMcKQueue(MMCQueue):
    """
    M/M/c/K 有限容量排队系统

    当队列满时，新到达被拒绝 (用于计算阻塞概率)

    Args:
        arrival_rate: 到达率 λ
        service_rate: 服务率 μ
        num_servers: 服务器数 c
        capacity: 系统容量 K (队列+服务中最大人数)
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        num_servers: int,
        capacity: int,
    ):
        super().__init__(arrival_rate, service_rate, num_servers)
        self.K = capacity

    def probability_n_in_system(self, n: int) -> float:
        """
        系统中有 n 个顾客的概率
        """
        if n > self.K:
            return 0.0

        alpha = self.lambda_ / self.mu

        if n <= self.c:
            p_n = (alpha ** n) / np.math.factorial(n)
        else:
            p_n = (alpha ** n) / (np.math.factorial(self.c) * (self.c ** (n - self.c)))

        # 归一化常数
        sum_term = sum((alpha ** n) / np.math.factorial(n) for n in range(self.c))
        sum_term += sum(
            (alpha ** n) / (np.math.factorial(self.c) * (self.c ** (n - self.c)))
            for n in range(self.c, self.K + 1)
        )

        return p_n / sum_term

    def blocking_probability(self) -> float:
        """
        阻塞概率 (队列满的概率)
        """
        return self.probability_n_in_system(self.K)

    def effective_throughput(self) -> float:
        """
        有效吞吐量 (排除被阻塞的)
        """
        P_block = self.blocking_probability()
        return self.lambda_ * (1 - P_block)
