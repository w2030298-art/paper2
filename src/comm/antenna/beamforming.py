"""
Beamforming Algorithms

支持:
- MRT: Maximum Ratio Transmission
- ZF: Zero Forcing
- MMSE: Minimum Mean Square Error
- Hybrid Beamforming: 模拟/数字混合波束赋形
"""

import numpy as np
from typing import Optional, Tuple


class MRT:
    """
    最大比率传输 (Maximum Ratio Transmission)

    发送端波束赋形: w = H^H / ||H||_F

    适用于: 高 SNR, 单用户 MIMO
    """

    def __init__(self, n_tx: int, n_rx: int):
        self.n_tx = n_tx
        self.n_rx = n_rx

    def compute_weights(self, H: np.ndarray) -> np.ndarray:
        """
        计算 MRT 发送权重

        Args:
            H: 信道矩阵, shape=(n_rx, n_tx)

        Returns:
            w: 发送权重矢量, shape=(n_tx, 1)
        """
        # MRT: w ∝ H^H @ r, 取 r = [1, 0, ..., 0] 即第一列
        w = H.conj().T[:, 0:1]  # (n_tx, 1)
        w = w / np.linalg.norm(w)
        return w

    def apply(
        self,
        H: np.ndarray,
        x: np.ndarray,
        noise_power: float = 1e-10,
    ) -> Tuple[np.ndarray, float]:
        """
        应用 MRT 波束赋形

        Args:
            H: 信道矩阵
            x: 待发送信号, shape=(n_tx, M) 或 (n_tx,)
            noise_power: 噪声功率

        Returns:
            y: 接收信号
            snr: 输出 SNR
        """
        w = self.compute_weights(H)

        if x.ndim == 1:
            x = x[:, None]

        # 发送
        tx_signal = w @ x.T  # (n_tx, M)

        # 接收
        y = H @ tx_signal  # (n_rx, M)

        # SNR
        signal_power = np.mean(np.abs(y) ** 2)
        snr = signal_power / (noise_power + 1e-10)

        return y, float(snr)


class ZF:
    """
    零强制波束赋形 (Zero Forcing)

    发送端预编码: W = H^H (H H^H)^{-1}

    适用于: 多用户 MIMO, 干扰抑制
    """

    def __init__(self, n_tx: int, n_rx: int):
        self.n_tx = n_tx
        self.n_rx = n_rx

    def compute_weights(self, H: np.ndarray) -> np.ndarray:
        """
        计算 ZF 发送权重 (迫零)

        Args:
            H: 信道矩阵, shape=(n_rx, n_tx)

        Returns:
            W: 发送权重矩阵, shape=(n_tx, n_rx)
        """
        # ZF: W = H^H (H H^H)^{-1}
        # 当 n_rx <= n_tx 时有效
        HH = H @ H.conj().T  # (n_rx, n_rx)
        inv_HH = np.linalg.inv(HH + 1e-10 * np.eye(self.n_rx))
        W = H.conj().T @ inv_HH  # (n_tx, n_rx)

        # 归一化
        for i in range(self.n_rx):
            norm = np.linalg.norm(W[:, i])
            if norm > 0:
                W[:, i] = W[:, i] / norm

        return W

    def apply(
        self,
        H: np.ndarray,
        x: np.ndarray,
        noise_power: float = 1e-10,
    ) -> Tuple[np.ndarray, float]:
        """应用 ZF 波束赋形"""
        W = self.compute_weights(H)

        if x.ndim == 1:
            x = x[:, None]

        # 发送
        tx_signal = W @ x  # (n_tx, n_rx)

        # 接收
        y = H @ tx_signal  # (n_rx, n_rx)

        # SNR (ZF 完全消除干扰)
        signal_power = np.mean(np.abs(y) ** 2)
        snr = signal_power / (noise_power + 1e-10)

        return y, float(snr)


class MMSE:
    """
    最小均方误差波束赋形 (MMSE)

    发送端预编码: W = H^H (H H^H + σ^2 I)^{-1}

    适用于: 有限 SNR, 干扰与噪声平衡
    """

    def __init__(self, n_tx: int, n_rx: int):
        self.n_tx = n_tx
        self.n_rx = n_rx

    def compute_weights(
        self,
        H: np.ndarray,
        noise_power: float = 1e-10,
    ) -> np.ndarray:
        """
        计算 MMSE 发送权重

        Args:
            H: 信道矩阵, shape=(n_rx, n_tx)
            noise_power: 噪声功率 σ²

        Returns:
            W: 发送权重矩阵, shape=(n_tx, n_rx)
        """
        # MMSE: W = H^H (H H^H + σ² I)^{-1}
        HH = H @ H.conj().T  # (n_rx, n_rx)
        reg = HH + noise_power * np.eye(self.n_rx)
        inv_reg = np.linalg.inv(reg)
        W = H.conj().T @ inv_reg  # (n_tx, n_rx)

        # 归一化
        for i in range(self.n_rx):
            norm = np.linalg.norm(W[:, i])
            if norm > 0:
                W[:, i] = W[:, i] / norm

        return W

    def apply(
        self,
        H: np.ndarray,
        x: np.ndarray,
        noise_power: float = 1e-10,
    ) -> Tuple[np.ndarray, float]:
        """应用 MMSE 波束赋形"""
        W = self.compute_weights(H, noise_power)

        if x.ndim == 1:
            x = x[:, None]

        tx_signal = W @ x
        y = H @ tx_signal

        signal_power = np.mean(np.abs(y) ** 2)
        snr = signal_power / (noise_power + 1e-10)

        return y, float(snr)


class HybridBeamforming:
    """
    混合模拟/数字波束赋形

    结构: 模拟波束 (移相器) + 数字基带预编码

    适用于: 毫米波大规模 MIMO

    Args:
        n_rf: RF 链路数 (远小于天线数)
        n_ant: 天线数
        n_stream: 数据流数
    """
    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        n_rf: int,
        n_ant: int,
        n_stream: int = 1,
        carrier_freq_hz: float = 28e9,
    ):
        if n_rf > n_ant:
            raise ValueError("n_rf must be <= n_ant")
        if n_stream > n_rf:
            raise ValueError("n_stream must be <= n_rf")

        self.n_rf = n_rf
        self.n_ant = n_ant
        self.n_stream = n_stream
        self.f_c = carrier_freq_hz
        self.wavelength = self.SPEED_OF_LIGHT / carrier_freq_hz

    def analog_precoder(self, angles_rad: np.ndarray) -> np.ndarray:
        """
        生成模拟波束赋形矩阵 (移相器)

        Args:
            angles_rad: 波束方向 (rad), shape=(n_rf,)

        Returns:
            F_analog: 模拟预编码矩阵, shape=(n_ant, n_rf)
        """
        # ULA 模拟移相器: 每个 RF 链路一个波束方向
        n = np.arange(self.n_ant)
        F_analog = np.zeros((self.n_ant, self.n_rf), dtype=complex)

        for i, theta in enumerate(angles_rad):
            # 相位旋转
            phase = 2 * np.pi * self.wavelength * n * np.sin(theta)
            F_analog[:, i] = np.exp(1j * phase) / np.sqrt(self.n_ant)

        return F_analog

    def digital_precoder(self, H_eff: np.ndarray) -> np.ndarray:
        """
        数字基带预编码 (基于有效信道)

        Args:
            H_eff: 有效信道矩阵, shape=(n_rx, n_rf)

        Returns:
            F_digital: 数字预编码矩阵, shape=(n_rf, n_stream)
        """
        # 使用 ZF 作为数字基带预编码
        if H_eff.shape[0] >= H_eff.shape[1]:
            # 窄阵列: H_eff @ F_digital
            F_digital = H_eff.conj().T @ np.linalg.inv(H_eff @ H_eff.conj().T + 1e-10 * np.eye(H_eff.shape[0]))
        else:
            # 宽数列: F_digital = H_eff^H (H_eff H_eff^H)^{-1}
            F_digital = np.linalg.inv(H_eff.conj().T @ H_eff + 1e-10 * np.eye(H_eff.shape[1])) @ H_eff.conj().T

        # 归一化
        for i in range(self.n_stream):
            norm = np.linalg.norm(F_digital[:, i])
            if norm > 0:
                F_digital[:, i] = F_digital[:, i] / norm

        return F_digital

    def apply(
        self,
        H: np.ndarray,
        x: np.ndarray,
        beam_angles_rad: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用混合波束赋形

        Args:
            H: 完整信道矩阵, shape=(n_rx, n_ant)
            x: 待发送信号, shape=(n_stream,)
            beam_angles_rad: 模拟波束角度 (默认均匀分布)

        Returns:
            y: 接收信号
            F_total: 总预编码矩阵 (n_ant, n_stream)
        """
        if beam_angles_rad is None:
            # 默认: 均匀分布
            beam_angles_rad = np.linspace(-np.pi / 3, np.pi / 3, self.n_rf)

        # 模拟预编码
        F_analog = self.analog_precoder(beam_angles_rad)

        # 有效信道: H @ F_analog
        H_eff = H @ F_analog  # (n_rx, n_rf)

        # 数字预编码
        F_digital = self.digital_precoder(H_eff)  # (n_rf, n_stream)

        # 总预编码
        F_total = F_analog @ F_digital  # (n_ant, n_stream)

        # 归一化
        F_total = F_total / np.linalg.norm(F_total) * np.sqrt(self.n_stream)

        # 发送
        if x.ndim == 1:
            x = x[:, None]
        tx_signal = F_total @ x  # (n_ant, n_stream)

        # 接收
        y = H @ tx_signal  # (n_rx, n_stream)

        return y, F_total
