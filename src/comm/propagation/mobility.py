"""
Mobility Models

支持:
- 静态模型
- 随机游走模型
- 高斯-马尔可夫模型
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class MobilityModel(ABC):
    """移动性模型基类"""

    @abstractmethod
    def position(self, t: float) -> Tuple[float, float]:
        """
        返回 t 时刻的位置

        Returns:
            x, y: 位置 (m)
        """
        pass

    @abstractmethod
    def velocity(self, t: float) -> Tuple[float, float]:
        """
        返回 t 时刻的速度

        Returns:
            vx, vy: 速度 (m/s)
        """
        pass

    def distance(self, t: float, other_x: float, other_y: float) -> float:
        """计算 t 时刻与另一点的距离"""
        x, y = self.position(t)
        return np.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)


class StaticMobility(MobilityModel):
    """静态模型 (无移动)"""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x0 = x
        self.y0 = y

    def position(self, t: float) -> Tuple[float, float]:
        return self.x0, self.y0

    def velocity(self, t: float) -> Tuple[float, float]:
        return 0.0, 0.0


class RandomWalkMobility(MobilityModel):
    """
    随机游走移动模型

    每一时刻随机改变速度和方向

    Args:
        speed_mean: 平均速度 (m/s)
        speed_std: 速度标准差 (m/s)
        direction_mean: 平均方向 (rad)
        direction_std: 方向标准差 (rad)
        seed: 随机种子
    """

    def __init__(
        self,
        speed_mean: float = 1.0,
        speed_std: float = 0.5,
        direction_mean: float = 0.0,
        direction_std: float = np.pi / 4,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.direction_mean = direction_mean
        self.direction_std = direction_std

        self.x0 = initial_x
        self.y0 = initial_y

        self.rng = np.random.default_rng(seed)
        self._current_time = 0.0

        # 初始化速度和方向
        self._speed = self.rng.normal(speed_mean, speed_std)
        self._direction = self.rng.normal(direction_mean, direction_std)

        # 每步更新时间
        self.dt = 1.0  # 默认 1 秒

    def position(self, t: float) -> Tuple[float, float]:
        """返回 t 时刻的位置"""
        if t < self._current_time:
            # 重置
            self._current_time = 0.0
            self._speed = self.rng.normal(self.speed_mean, self.speed_std)
            self._direction = self.rng.normal(self.direction_mean, self.direction_std)

        # 更新位置
        while self._current_time < t:
            vx = self._speed * np.cos(self._direction)
            vy = self._speed * np.sin(self._direction)

            self.x0 += vx * self.dt
            self.y0 += vy * self.dt

            # 更新速度和方向
            self._speed = max(0, self.rng.normal(self.speed_mean, self.speed_std))
            self._direction = self.rng.normal(self.direction_mean, self.direction_std)

            self._current_time += self.dt

        return self.x0, self.y0

    def velocity(self, t: float) -> Tuple[float, float]:
        """返回 t 时刻的速度"""
        x, y = self.position(t)
        self.position(t + 0.01)
        x2, y2 = self.position(t + 0.01)
        return (x2 - x) / 0.01, (y2 - y) / 0.01


class GaussMarkovMobility(MobilityModel):
    """
    高斯-马尔可夫移动模型

    平滑随机运动, 记忆之前速度

    x(t+1) = a * x(t) + (1-a) * μ_x + sqrt((1-a²) * σ²) * w_x

    Args:
        speed_mean: 平均速度 (m/s)
        direction_mean: 平均方向 (rad)
        alpha: 平滑因子 [0, 1], 1=完全记忆, 0=完全随机
        sigma: 高斯噪声标准差
        initial_x, initial_y: 初始位置
        initial_vx, initial_vy: 初始速度
        seed: 随机种子
    """

    def __init__(
        self,
        speed_mean: float = 1.0,
        direction_mean: float = 0.0,
        alpha: float = 0.5,
        sigma: float = 0.5,
        initial_x: float = 0.0,
        initial_y: float = 0.0,
        initial_vx: Optional[float] = None,
        initial_vy: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        self.speed_mean = speed_mean
        self.direction_mean = direction_mean
        self.alpha = alpha
        self.sigma = sigma

        self.x = initial_x
        self.y = initial_y

        self.rng = np.random.default_rng(seed)

        # 初始速度
        if initial_vx is not None:
            self.vx = initial_vx
        else:
            self.vx = self.rng.normal(speed_mean, sigma)

        if initial_vy is not None:
            self.vy = initial_vy
        else:
            self.vy = self.rng.normal(speed_mean, sigma)

        self._last_time = 0.0

    def update(self, dt: float):
        """更新位置和速度"""
        # 随机噪声
        w_vx = self.rng.normal(0, 1)
        w_vy = self.rng.normal(0, 1)

        # 速度更新 (Gauss-Markov)
        vx_new = self.alpha * self.vx + (1 - self.alpha) * self.speed_mean * np.cos(self.direction_mean) \
                 + np.sqrt(1 - self.alpha ** 2) * self.sigma * w_vx

        vy_new = self.alpha * self.vy + (1 - self.alpha) * self.speed_mean * np.sin(self.direction_mean) \
                 + np.sqrt(1 - self.alpha ** 2) * self.sigma * w_vy

        self.vx = vx_new
        self.vy = vy_new

        # 位置更新
        self.x += self.vx * dt
        self.y += self.vy * dt

        self._last_time += dt

    def position(self, t: float) -> Tuple[float, float]:
        """返回 t 时刻的位置"""
        if t > self._last_time:
            dt = t - self._last_time
            self.update(dt)
        return self.x, self.y

    def velocity(self, t: float) -> Tuple[float, float]:
        """返回 t 时刻的速度"""
        self.position(t)  # 确保状态更新
        return self.vx, self.vy
