"""
SNR Calculator

计算各种场景下的信噪比
"""

import numpy as np
from typing import Optional, Tuple


class SNRCalculator:
    """
    SNR 计算器

    支持:
    - AWGN 信道
    - 衰落信道
    - MIMO 信道

    Args:
        noise_power_dbm: 噪声功率 (dBm)
        bandwidth_hz: 信号带宽 (Hz)
        temperature_k: 噪声温度 (K), 默认 290K
    """

    BOLTZMANN_CONSTANT = 1.38e-23  # J/K

    def __init__(
        self,
        noise_power_dbm: Optional[float] = None,
        bandwidth_hz: float = 20e6,
        temperature_k: float = 290.0,
    ):
        self.bandwidth_hz = bandwidth_hz
        self.temperature_k = temperature_k

        if noise_power_dbm is not None:
            self.noise_power_dbm = noise_power_dbm
        else:
            # 计算热噪声功率
            # N = k * T * B (W)
            noise_power_w = self.BOLTZMANN_CONSTANT * temperature_k * bandwidth_hz
            self.noise_power_dbm = 10 * np.log10(noise_power_w * 1000)

    @property
    def noise_power_dbw(self) -> float:
        """噪声功率 (dBW)"""
        return self.noise_power_dbm - 30.0

    @property
    def noise_power_watts(self) -> float:
        """噪声功率 (W)"""
        return 10 ** (self.noise_power_dbm / 10) / 1000

    def snr_linear(
        self,
        signal_power_watts: float,
    ) -> float:
        """线性 SNR"""
        return signal_power_watts / self.noise_power_watts

    def snr_db(
        self,
        signal_power_dbm: float,
    ) -> float:
        """
        dB 形式 SNR

        Args:
            signal_power_dbm: 信号功率 (dBm)

        Returns:
            snr_db: SNR (dB)
        """
        return signal_power_dbm - self.noise_power_dbm

    def snr_from_path_loss(
        self,
        tx_power_dbm: float,
        path_loss_db: float,
        tx_gain_dbi: float = 0.0,
        rx_gain_dbi: float = 0.0,
        fading_db: float = 0.0,
    ) -> float:
        """
        从路径损耗计算 SNR

        Args:
            tx_power_dbm: 发射功率 (dBm)
            path_loss_db: 路径损耗 (dB)
            tx_gain_dbi: 发射天线增益 (dBi)
            rx_gain_dbi: 接收天线增益 (dBi)
            fading_db: 衰落损耗 (dB)

        Returns:
            snr_db: SNR (dB)
        """
        rx_power_dbm = tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db - fading_db
        return self.snr_db(rx_power_dbm)

    def sinr_db(
        self,
        signal_power_dbm: float,
        interference_power_dbm: float,
    ) -> float:
        """
        SINR (信号与干扰加噪声比)

        Args:
            signal_power_dbm: 信号功率 (dBm)
            interference_power_dbm: 干扰功率 (dBm)

        Returns:
            sinr_db: SINR (dB)
        """
        # 干扰 + 噪声功率 (线性)
        i_n_power_w = 10 ** (interference_power_dbm / 10) / 1000
        i_n_power_w += self.noise_power_watts

        # 信号功率 (线性)
        s_power_w = 10 ** (signal_power_dbm / 10) / 1000

        sinr_linear = s_power_w / i_n_power_w
        return 10 * np.log10(sinr_linear)

    def capacity(
        self,
        snr_db: float,
        bandwidth_hz: Optional[float] = None,
    ) -> float:
        """
        香农容量

        Args:
            snr_db: SNR (dB)
            bandwidth_hz: 带宽 (Hz), 默认使用实例带宽

        Returns:
            capacity_bps: 容量 (bps)
        """
        if bandwidth_hz is None:
            bandwidth_hz = self.bandwidth_hz

        snr_linear = 10 ** (snr_db / 10)
        capacity = bandwidth_hz * np.log2(1 + snr_linear)
        return capacity

    def energy_per_bit(
        self,
        tx_power_dbm: float,
        data_rate_bps: float,
    ) -> float:
        """
        每比特能量

        Args:
            tx_power_dbm: 发射功率 (dBm)
            data_rate_bps: 数据率 (bps)

        Returns:
            eb_n0_db: Eb/N0 (dB)
        """
        tx_power_w = 10 ** (tx_power_dbm / 10) / 1000
        eb = tx_power_w / data_rate_bps

        # Eb/N0 = Eb / (N/B) = (P/R) / (N/B) = P * B / (R * N) = SNR * (B/R)
        snr_linear = tx_power_w / self.noise_power_watts
        eb_n0_linear = snr_linear * (self.bandwidth_hz / data_rate_bps)

        return 10 * np.log10(eb_n0_linear)
