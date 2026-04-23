"""
MIMO Channel Models

支持:
- 简单 MIMO 信道 (独立衰落)
- 3GPP Spatial Channel Model (SCM)
- MIMO 信道容量与奇异值分解
"""

import numpy as np
from typing import Optional, Tuple
from .fading import RayleighFading, RicianFading


class MIMOChannel:
    """
    简单 MIMO 信道

    H ∈ C^(N_t × N_r), 各元素独立复高斯

    用于: 理论容量分析、快速仿真

    Args:
        n_tx: 发送天线数 N_t
        n_rx: 接收天线数 N_r
        fading_model: "rayleigh" | "rician"
        k_factor: 莱斯K因子 (仅 rician 时)
        seed: 随机种子
    """

    def __init__(
        self,
        n_tx: int = 4,
        n_rx: int = 4,
        fading_model: str = "rayleigh",
        k_factor: float = 3.0,
        seed: Optional[int] = None,
    ):
        if n_tx < 1 or n_rx < 1:
            raise ValueError(f"n_tx and n_rx must be >= 1, got {n_tx}, {n_rx}")

        self.n_tx = n_tx
        self.n_rx = n_rx

        if fading_model == "rayleigh":
            self._fader = RayleighFading(seed=seed)
        elif fading_model == "rician":
            self._fader = RicianFading(k_factor=k_factor, seed=seed)
        else:
            raise ValueError(f"Unknown fading_model: {fading_model}")

    def __call__(self) -> np.ndarray:
        """生成一次 MIMO 信道实现 H"""
        return self._fader(size=(self.n_rx, self.n_tx))

    def channel_matrix(self) -> np.ndarray:
        """同 __call__"""
        return self()

    def singular_values(self, H: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 MIMO 信道的奇异值分解

        Args:
            H: 可选，指定信道矩阵

        Returns:
            s: 奇异值 (N_s = min(N_t, N_r))
            U: 左奇异矢量 (N_r × N_r)
            V: 右奇异矢量 (N_t × N_t)
        """
        if H is None:
            H = self()
        U, s, Vh = np.linalg.svd(H, full_matrices=False)
        return s, U, Vh.conj().T

    def channel_rank(self, H: Optional[np.ndarray] = None, threshold: float = 1e-6) -> int:
        """返回信道有效秩 (奇异值 > threshold)"""
        s, _, _ = self.singular_values(H)
        return int(np.sum(s > threshold))

    def channel_gain_db(self, H: Optional[np.ndarray] = None) -> float:
        """总信道增益 (dB), sum of singular values"""
        s, _, _ = self.singular_values(H)
        return 10 * np.log10(np.sum(s ** 2) + 1e-10)

    def capacity(
        self,
        snr_db: float,
        H: Optional[np.ndarray] = None,
        water_filling: bool = False,
    ) -> float:
        """
        计算 MIMO 信道容量 (bps/Hz)

        Args:
            snr_db: 接收 SNR (dB)
            H: 可选，指定信道矩阵
            water_filling: 是否使用注水功率分配

        Returns:
            capacity: 容量 (bps/Hz)
        """
        if H is None:
            H = self()

        snr_linear = 10 ** (snr_db / 10)
        s_sq, _, _ = self.singular_values(H)
        s_sq = s_sq ** 2  # 奇异值平方 = 特征值

        if water_filling:
            # 简化的注水算法
            n_s = len(s_sq)
            power = snr_linear / n_s * np.ones(n_s)
            for _ in range(50):
                # 简化解
                snr_eff = s_sq * power
                mu = np.median(snr_eff)
                power = np.maximum(1 / s_sq * (snr_eff - mu + 1e-8), 0.01)
                power = power / np.sum(power) * snr_linear
            capacity = np.sum(np.log2(1 + s_sq * power))
        else:
            # 均匀功率分配
            capacity = np.sum(np.log2(1 + snr_linear / self.n_tx * s_sq))

        return capacity

    def received_power_db(
        self,
        tx_power_dbw: float,
        H: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        计算每根接收天线的功率 (dB)

        Args:
            tx_power_dbw: 发射功率 (dBW)
            H: 可选，信道矩阵

        Returns:
            total_power_db: 总接收功率 (dB)
            per_antenna_db: 每根接收天线功率 (dB)
        """
        if H is None:
            H = self()

        tx_linear = 10 ** (tx_power_dbw / 10)
        rx_power = np.sum(np.abs(H) ** 2, axis=1) * tx_linear
        rx_power_db = 10 * np.log10(rx_power + 1e-10)
        return float(np.sum(rx_power_db)), rx_power_db


class SpatialChannelModel:
    """
    简化的 3GPP Spatial Channel Model (SCM)

    考虑:
    - 角度扩展 (Angle Spread)
    - 到达角 (AoA) / 出发角 (AoD)
    - 天线阵列响应矢量

    用于: 真实天线配置下的信道建模

    Args:
        n_tx: 发送天线数
        n_rx: 接收天线数
        carrier_freq_hz: 载波频率 (Hz)
        bandwidth_hz: 信号带宽 (Hz)
        aoa_rad: 到达角均值 (rad)
        aoa_spread_rad: 到达角扩展 (rad)
        aod_rad: 出发角均值 (rad)
        aod_spread_rad: 出发角扩展 (rad)
        num_paths: 路径数 (子路径数)
        seed: 随机种子
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        n_tx: int = 4,
        n_rx: int = 4,
        carrier_freq_hz: float = 3.5e9,
        bandwidth_hz: float = 20e6,
        aoa_rad: float = 0.0,
        aoa_spread_rad: float = np.pi / 12,
        aod_rad: float = np.pi,
        aod_spread_rad: float = np.pi / 12,
        num_paths: int = 20,
        seed: Optional[int] = None,
    ):
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.f_c = carrier_freq_hz
        self.bandwidth = bandwidth_hz
        self.lambda_c = self.SPEED_OF_LIGHT / self.f_c
        self.aoa_mean = aoa_rad
        self.aoa_spread = aoa_spread_rad
        self.aod_mean = aod_rad
        self.aod_spread = aod_spread_rad
        self.num_paths = num_paths
        self.rng = np.random.default_rng(seed)

    def _array_response(self, angle: np.ndarray, n_ant: int, d_lambda: float = 0.5) -> np.ndarray:
        """
        计算均匀线阵 (ULA) 的阵列响应矢量

        Args:
            angle: 波达方向 (rad), shape=(num_paths,)
            n_ant: 天线数
            d_lambda: 天线间距 / 波长

        Returns:
            a: 阵列响应矢量, shape=(n_ant, num_paths)
        """
        k = 2 * np.pi / self.lambda_c
        n = np.arange(n_ant)  # [0, 1, 2, ..., N-1]
        # a_n = exp(j * k * d * n * sin(angle))
        a = np.exp(1j * k * d_lambda * np.outer(n, np.sin(angle)))
        return a

    def __call__(self) -> np.ndarray:
        """
        生成一个 SCM 信道实现

        Returns:
            H: 信道矩阵, shape=(n_rx, n_tx)
        """
        # 角度采样 (拉普拉斯分布近似)
        aoa = self.aoa_mean + self.aoa_spread * self.rng.standard_normal(self.num_paths)
        aod = self.aod_mean + self.aod_spread * self.rng.standard_normal(self.num_paths)

        # 路径功率 (均匀)
        path_gains = np.exp(1j * 2 * np.pi * self.rng.uniform(0, 1, size=self.num_paths))

        # 阵列响应矢量
        a_rx = self._array_response(aoa, self.n_rx)  # (n_rx, num_paths)
        a_tx = self._array_response(aod, self.n_tx)  # (n_tx, num_paths)

        # 信道矩阵: H = a_rx @ a_tx^H * path_gains / sqrt(num_paths)
        H = a_rx @ a_tx.conj().T * path_gains / np.sqrt(self.num_paths)

        return H

    def channel_matrix(self) -> np.ndarray:
        return self()

    def generate(self, num_realizations: int) -> np.ndarray:
        """
        生成多个信道实现

        Args:
            num_realizations: 实现数

        Returns:
            H_set: shape=(num_realizations, n_rx, n_tx)
        """
        return np.array([self() for _ in range(num_realizations)])
