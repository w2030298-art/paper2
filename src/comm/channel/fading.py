"""
Fading Channel Models

支持:
- Rayleigh: 经典瑞利衰落 (无直视径)
- Rician: 莱斯衰落 (有直视径)
- Nakagami-m: 通用衰落模型
- Shadowing: 对数正态阴影衰落
- Jakes: 时间相关瑞利衰落
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple


class RayleighFading:
    """
    瑞利衰落信道

    h ~ CN(0, 1)
    |h|^2 ~ Exp(1) (指数分布)

    用于: 城市宏基站无直视径 (NLOS) 场景
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """
        生成瑞利衰落信道系数

        Args:
            size: 输出形状

        Returns:
            h: 复信道系数, shape=size
        """
        # 复高斯: (X + jY) / sqrt(2), X,Y ~ N(0, 1)
        h = (self.rng.standard_normal(size) + 1j * self.rng.standard_normal(size)) / np.sqrt(2)
        return h

    def channel_gain(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """返回信道增益 |h|^2 (服从指数分布)"""
        return self.rng.exponential(1.0, size=size)


class RicianFading:
    """
    莱斯衰落信道

    h = sqrt(K/(K+1)) * exp(j*phi) + sqrt(1/(K+1)) * CN(0, 1)
    |h|^2 服从非中心卡方分布

    Args:
        k_factor: 莱斯K因子 (主散射分量与散射分量的功率比), 线性尺度
        initial_phase: 直视径初始相位 (rad)
        seed: 随机种子
    """

    def __init__(self, k_factor: float = 3.0, initial_phase: float = 0.0, seed: Optional[int] = None):
        self.k = k_factor
        self.initial_phase = initial_phase
        self.rng = np.random.default_rng(seed)

    def __call__(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """
        生成莱斯衰落信道系数

        Args:
            size: 输出形状

        Returns:
            h: 复信道系数
        """
        # 散射分量
        scatter = (self.rng.standard_normal(size) + 1j * self.rng.standard_normal(size)) / np.sqrt(2)

        # 直视径分量
        sqrt_k = np.sqrt(self.k / (self.k + 1))
        los = sqrt_k * np.exp(1j * self.initial_phase)

        h = los + np.sqrt(1 / (self.k + 1)) * scatter
        return h

    def channel_gain(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """返回信道增益 |h|^2"""
        h = self(size=size)
        return np.abs(h) ** 2


class NakagamiFading:
    """
    Nakagami-m 衰落信道

    通用衰落模型:
    - m=1   -> 瑞利衰落
    - m=0.5 -> 单边高斯衰落
    - m->∞  -> 无衰落 (确定性)

    幅度分布: Nakagami-m
    功率分布: Gamma 分布

    Args:
        m: 形状参数 (m >= 0.5)
        omega: 功率参数 (平均功率)
        seed: 随机种子
    """

    def __init__(self, m: float = 1.0, omega: float = 1.0, seed: Optional[int] = None):
        if m < 0.5:
            raise ValueError(f"Nakagami m must be >= 0.5, got {m}")
        self.m = m
        self.omega = omega
        self.rng = np.random.default_rng(seed)

    def __call__(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """
        生成 Nakagami-m 衰落信道系数

        Args:
            size: 输出形状

        Returns:
            h: 复信道系数
        """
        # 幅度服从 Nakagami 分布
        # |h| ~ sqrt(Gamma(m, omega/m) / m)
        gamma_sample = self.rng.gamma(self.m, self.omega / self.m, size=size)
        amplitude = np.sqrt(gamma_sample)

        # 随机相位
        phase = self.rng.uniform(0, 2 * np.pi, size=size)

        return amplitude * np.exp(1j * phase)

    def channel_gain(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """返回信道增益 |h|^2 (Gamma 分布)"""
        return self.rng.gamma(self.m, self.omega / self.m, size=size)


class ShadowFading:
    """
    对数正态阴影衰落

    X_dB ~ N(mu, sigma^2)

    用于: 建筑物遮挡导致的快衰落
    常与路径损耗叠加: PL_total = PL + X_sigma

    Args:
        sigma: 阴影衰落标准差 (dB), 典型值 4~12 dB
        seed: 随机种子
    """

    def __init__(self, sigma_db: float = 8.0, seed: Optional[int] = None):
        self.sigma_db = sigma_db
        self.rng = np.random.default_rng(seed)

    def __call__(self, size: Tuple[int, ...] = ()) -> np.ndarray:
        """
        生成阴影衰落 (dB)

        Args:
            size: 输出形状

        Returns:
            shadow_db: 阴影衰落 (dB)
        """
        return self.rng.normal(0.0, self.sigma_db, size=size)

    def apply(self, path_loss_db: np.ndarray) -> np.ndarray:
        """
        将阴影衰落叠加到路径损耗上

        Args:
            path_loss_db: 路径损耗 (dB)

        Returns:
            total_loss_db: 含阴影的路径损耗 (dB)
        """
        shadow = self(size=path_loss_db.shape)
        return path_loss_db + shadow


class JakesFading:
    """
    Jake's 模型 — 时间相关瑞利衰落

    适用于模拟时变频率非选择性信道
    服从 Clarke's 谐波叠加模型

    h(t) = sqrt(2) * sum_{n=1}^N [cos(phi_n) * cos(w_d * t * cos(alpha_n))
                                    + j * sin(phi_n) * cos(w_d * t * sin(alpha_n))]

    Args:
        doppler_hz: 最大多普勒频移 (Hz), v * f_c / c
        num_paths: 路径数 N, 越大越接近真实瑞利
        initial_phase: 初始相位
        seed: 随机种子
    """

    def __init__(
        self,
        doppler_hz: float = 10.0,
        num_paths: int = 20,
        initial_phase: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.f_d = doppler_hz
        self.N = num_paths
        self.phi_0 = initial_phase
        self.rng = np.random.default_rng(seed)

        # 预计算角度
        self.alphas = 2 * np.pi * np.arange(1, self.N + 1) / self.N

    def __call__(self, time_s: float) -> np.ndarray:
        """
        生成 t 时刻的复信道系数

        Args:
            time_s: 时间 (秒)

        Returns:
            h: 复信道系数
        """
        w_d = 2 * np.pi * self.f_d

        # 随机初始相位 (每次实例化时固定)
        phi_n = self.phi_0 + 2 * np.pi * self.rng.uniform(0, 1, size=(self.N,))

        # 预计算 cos 和 sin
        cos_alpha = np.cos(self.alphas)
        sin_alpha = np.sin(self.alphas)
        cos_wdt = np.cos(w_d * time_s)

        # 谐波叠加
        I = np.sqrt(2 / self.N) * np.sum(cos(phi_n) * cos_wdt * cos_alpha - 1j * sin(phi_n) * cos_wdt * sin_alpha)
        Q = np.sqrt(2 / self.N) * np.sum(sin(phi_n) * cos_wdt * sin_alpha + 1j * cos(phi_n) * cos_wdt * cos_alpha)

        h = I + Q
        return h

    def generate(self, t_array: ArrayLike) -> np.ndarray:
        """
        生成多个时刻的信道系数

        Args:
            t_array: 时间数组 (秒)

        Returns:
            h: shape=(len(t_array),)
        """
        return np.array([self(t) for t in t_array])


# helpers
from numpy import cos, sin, pi
