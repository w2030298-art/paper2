"""
Noise Models

支持:
- 热噪声 (Thermal Noise)
- AWGN 信道
- 噪声系数 (Noise Figure)
"""

import numpy as np
from typing import Optional


class ThermalNoise:
    """
    热噪声模型

    N = k * T * B (W)

    Args:
        temperature_k: 噪声温度 (K), 默认 290K (室温)
        bandwidth_hz: 带宽 (Hz)
    """

    BOLTZMANN_CONSTANT = 1.38e-23  # J/K

    def __init__(
        self,
        temperature_k: float = 290.0,
        bandwidth_hz: float = 20e6,
    ):
        self.temperature_k = temperature_k
        self.bandwidth_hz = bandwidth_hz

    @property
    def power_watts(self) -> float:
        """噪声功率 (W)"""
        return self.BOLTZMANN_CONSTANT * self.temperature_k * self.bandwidth_hz

    @property
    def power_dbm(self) -> float:
        """噪声功率 (dBm)"""
        return 10 * np.log10(self.power_watts * 1000)

    @property
    def power_dbw(self) -> float:
        """噪声功率 (dBW)"""
        return 10 * np.log10(self.power_watts)

    def power(self, bandwidth_hz: Optional[float] = None) -> float:
        """计算指定带宽的噪声功率"""
        if bandwidth_hz is None:
            bandwidth_hz = self.bandwidth_hz
        return self.BOLTZMANN_CONSTANT * self.temperature_k * bandwidth_hz

    def power_db(self, bandwidth_hz: Optional[float] = None) -> float:
        """dB 形式"""
        return 10 * np.log10(self.power(bandwidth_hz))


class AWGNChannel:
    """
    加性高斯白噪声 (AWGN) 信道

    y = x + n

    Args:
        snr_db: 信噪比 (dB)
        noise_power_dbm: 噪声功率 (dBm), 与 snr_db 二选一
        bandwidth_hz: 带宽 (Hz)
    """

    def __init__(
        self,
        snr_db: Optional[float] = None,
        noise_power_dbm: Optional[float] = None,
        bandwidth_hz: float = 20e6,
    ):
        self.bandwidth_hz = bandwidth_hz

        if snr_db is not None:
            self.snr_db = snr_db
        elif noise_power_dbm is not None:
            self.noise_power_dbm = noise_power_dbm
        else:
            # 默认: 20dB SNR
            self.snr_db = 20.0

    @property
    def snr_db(self) -> float:
        return self._snr_db

    @snr_db.setter
    def snr_db(self, value: float):
        self._snr_db = value

    @property
    def noise_power_dbm(self) -> float:
        """从 SNR 推断"""
        return self._noise_power_dbm

    @noise_power_dbm.setter
    def noise_power_dbm(self, value: float):
        self._noise_power_dbm = value

    def __call__(
        self,
        x: np.ndarray,
        signal_power_dbm: Optional[float] = None,
    ) -> np.ndarray:
        """
        添加噪声

        Args:
            x: 输入信号
            signal_power_dbm: 信号功率 (dBm), 可从 x 推断

        Returns:
            y: 输出信号
        """
        # 计算信号功率
        if signal_power_dbm is None:
            signal_power_w = np.mean(np.abs(x) ** 2)
            signal_power_dbm = 10 * np.log10(signal_power_w * 1000)

        # 噪声功率
        noise_power_dbm = signal_power_dbm - self.snr_db
        noise_power_w = 10 ** (noise_power_dbm / 10) / 1000

        # 生成噪声
        noise = np.sqrt(noise_power_w / 2) * (
            np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
        )

        return x + noise

    def add_noise(
        self,
        x: np.ndarray,
        snr_db: Optional[float] = None,
    ) -> np.ndarray:
        """添加噪声"""
        if snr_db is not None:
            self.snr_db = snr_db
        return self(x)


class NoiseFigure:
    """
    噪声系数 (Noise Figure)

    F = SNR_in / SNR_out (线性)

    Args:
        noise_figure_db: 噪声系数 (dB)
        temperature_k: 参考噪声温度 (K)
    """

    def __init__(
        self,
        noise_figure_db: float = 3.0,
        temperature_k: float = 290.0,
    ):
        self.noise_figure_db = noise_figure_db
        self.noise_figure_linear = 10 ** (noise_figure_db / 10)
        self.temperature_k = temperature_k

    def noise_temperature(self) -> float:
        """等效噪声温度 (K)"""
        # T_e = (F - 1) * T_0
        return (self.noise_figure_linear - 1) * self.temperature_k

    def noise_power_dbm(
        self,
        bandwidth_hz: float,
        input_snr_db: Optional[float] = None,
    ) -> float:
        """
        输出噪声功率

        Args:
            bandwidth_hz: 带宽 (Hz)
            input_snr_db: 输入 SNR (dB), 可选

        Returns:
            noise_power_dbm: 噪声功率 (dBm)
        """
        k = 1.38e-23
        T_e = self.noise_temperature()
        n_power_w = k * (T_e + self.temperature_k) * bandwidth_hz
        return 10 * np.log10(n_power_w * 1000)

    def output_snr_db(
        self,
        input_snr_db: float,
    ) -> float:
        """
        计算输出 SNR

        Args:
            input_snr_db: 输入 SNR (dB)

        Returns:
            output_snr_db: 输出 SNR (dB)
        """
        return input_snr_db - self.noise_figure_db
