"""
Antenna Array Models

支持:
- ULA: 均匀线阵 (Uniform Linear Array)
- URA: 均匀矩形阵 (Uniform Rectangular Array)
- UPA: 均匀平面阵 (Uniform Planar Array)
- SteeringVector: 导向矢量
- ArrayFactor: 阵列因子计算
"""

import numpy as np
from typing import Tuple, Optional


class SteeringVector:
    """
    导向矢量计算器

    导向矢量 a(θ) 描述了信号从角度 θ 入射时，各天线单元的相位差

    Args:
        n_ant: 天线数
        d_lambda: 阵元间距 / 波长 (默认 0.5, 半波长)
        carrier_freq_hz: 载波频率 (用于计算波长)
        speed_of_light: 光速 (m/s)
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        n_ant: int,
        d_lambda: float = 0.5,
        carrier_freq_hz: Optional[float] = None,
        speed_of_light: float = 3e8,
    ):
        if n_ant < 1:
            raise ValueError(f"n_ant must be >= 1, got {n_ant}")
        self.n_ant = n_ant
        self.d_lambda = d_lambda
        if carrier_freq_hz is not None:
            self.wavelength = speed_of_light / carrier_freq_hz
        else:
            self.wavelength = None
        self.speed_of_light = speed_of_light

    def _compute_wavelength(self, carrier_freq_hz: float) -> float:
        return self.speed_of_light / carrier_freq_hz

    def ula(self, angle_rad: np.ndarray) -> np.ndarray:
        """
        计算 ULA 导向矢量

        Args:
            angle_rad: 波达/出发方向 (rad), shape=(M,)

        Returns:
            a: 导向矢量, shape=(n_ant, M)
                a[n, m] = exp(j * 2π * d * n * sin(θ_m))
        """
        if self.wavelength is None:
            raise ValueError("carrier_freq_hz must be set to compute wavelength")

        k = 2 * np.pi / self.wavelength
        n = np.arange(self.n_ant)  # [0, 1, 2, ..., N-1]
        # 导向矢量: 外积 (N, 1) @ (1, M) = (N, M)
        a = np.exp(1j * k * self.d_lambda * np.outer(n, np.sin(angle_rad)))
        return a

    def ura(self, phi_rad: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
        """
        计算 URA 导向矢量 (矩形阵列)

        Args:
            phi_rad: 方位角 (rad)
            theta_rad: 俯仰角 (rad)

        Returns:
            a: 导向矢量, shape=(n_ant, M)
        """
        # URA 是两个 ULA 的 Kronecker 积
        # 先算水平方向
        k = 2 * np.pi / self.wavelength

        # 假设为 Nx x Ny 的矩形阵，这里简化为 sqrt(N) x sqrt(N)
        n_x = int(np.round(np.sqrt(self.n_ant)))
        n_y = self.n_ant // n_x

        # 水平相位
        n_x_arr = np.arange(n_x)
        a_x = np.exp(1j * k * self.d_lambda * np.outer(n_x_arr, np.sin(phi_rad)))

        # 垂直相位
        n_y_arr = np.arange(n_y)
        a_y = np.exp(1j * k * self.d_lambda * np.outer(n_y_arr, np.cos(theta_rad)))

        # Kronecker 积
        a = np.kron(a_x, a_y)
        return a


class ArrayFactor:
    """
    阵列因子计算器

    阵列因子 AF(θ) = sum_{n=0}^{N-1} w_n * exp(j * 2π * d * n * sin(θ))
    描述了阵列的方向图

    Args:
        n_ant: 天线数
        d_lambda: 阵元间距 / 波长
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(self, n_ant: int, d_lambda: float = 0.5):
        self.n_ant = n_ant
        self.d_lambda = d_lambda

    def compute(
        self,
        angles_rad: np.ndarray,
        weights: Optional[np.ndarray] = None,
        wavelength: Optional[float] = None,
    ) -> np.ndarray:
        """
        计算阵列因子

        Args:
            angles_rad: 方向角度 (rad), shape=(M,)
            weights: 复权重, shape=(n_ant,). 默认均匀加权
            wavelength: 波长 (m). 如果提供则使用，否则用 d_lambda 归一化

        Returns:
            AF: 阵列因子幅度, shape=(M,)
        """
        if weights is None:
            weights = np.ones(self.n_ant, dtype=complex) / self.n_ant

        if wavelength is not None:
            k = 2 * np.pi / wavelength
        else:
            k = 2 * np.pi  # 归一化

        n = np.arange(self.n_ant)
        # AF = sum_n w_n * exp(j * k * d * n * sin(θ))
        # shape: (n_ant, 1) * (1, M) -> (n_ant, M) -> sum (M,)
        AF = np.sum(weights[:, None] * np.exp(1j * k * self.d_lambda * np.outer(n, np.sin(angles_rad))), axis=0)
        return np.abs(AF)

    def beam_width_3db(self, wavelength: Optional[float] = None) -> float:
        """
        估算 3dB 波束宽度 (rad)

        近似公式: θ_3dB ≈ 0.886 * λ / (N * d * cos(θ_0))

        Returns:
            beam_width_rad: 3dB 波束宽度 (rad)
        """
        if wavelength is not None:
            return 0.886 * wavelength / (self.n_ant * self.d_lambda * wavelength)
        return 0.886 / (self.n_ant * self.d_lambda)


class ULA:
    """
    均匀线阵 (Uniform Linear Array)

    天线沿直线等间距排列

    Args:
        n_ant: 天线数
        d_lambda: 阵元间距 / 波长 (默认 0.5 = 半波长)
        carrier_freq_hz: 载波频率 (Hz), 可选
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        n_ant: int,
        d_lambda: float = 0.5,
        carrier_freq_hz: Optional[float] = None,
    ):
        self.n_ant = n_ant
        self.d_lambda = d_lambda
        self.carrier_freq_hz = carrier_freq_hz

        self._steering = SteeringVector(
            n_ant=n_ant,
            d_lambda=d_lambda,
            carrier_freq_hz=carrier_freq_hz,
        )

    @property
    def wavelength(self) -> float:
        if self.carrier_freq_hz is None:
            raise ValueError("carrier_freq_hz not set")
        return self.SPEED_OF_LIGHT / self.carrier_freq_hz

    def steering_vector(self, angle_rad: np.ndarray) -> np.ndarray:
        """
        计算导向矢量

        Args:
            angle_rad: 方向 (rad), shape=(M,)

        Returns:
            a: shape=(n_ant, M)
        """
        if self.carrier_freq_hz is None:
            raise ValueError("carrier_freq_hz not set")
        return self._steering.ula(angle_rad)

    def array_factor(
        self,
        angles_rad: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        计算阵列方向图

        Args:
            angles_rad: 扫描角度 (rad)
            weights: 波束赋形权重

        Returns:
            gain_db: 阵列增益 (dB)
        """
        af = ArrayFactor(self.n_ant, self.d_lambda)
        gain_linear = af.compute(angles_rad, weights, self.wavelength)
        gain_db = 20 * np.log10(gain_linear + 1e-10)
        return gain_db

    def beam_gain_db(self, target_angle_rad: float, scan_angle_rad: float = 0.0) -> float:
        """
        计算目标方向的波束增益

        Args:
            target_angle_rad: 目标方向 (rad)
            scan_angle_rad: 扫描方向 (rad)

        Returns:
            gain_db: 波束增益 (dB)
        """
        # 扫描导向矢量
        a_scan = self.steering_vector(np.array([scan_angle_rad]))[:, 0]
        # 权重的相位补偿实现扫描
        weights = a_scan.conj()

        # 目标方向增益
        angles = np.array([target_angle_rad])
        gain_db = self.array_factor(angles, weights)[0]
        return gain_db


class URA:
    """
    均匀矩形阵 (Uniform Rectangular Array)

    Args:
        n_x: 水平天线数
        n_y: 垂直天线数
        d_lambda: 阵元间距 / 波长
        carrier_freq_hz: 载波频率
    """

    def __init__(
        self,
        n_x: int,
        n_y: int,
        d_lambda: float = 0.5,
        carrier_freq_hz: Optional[float] = None,
    ):
        self.n_x = n_x
        self.n_y = n_y
        self.n_ant = n_x * n_y
        self.d_lambda = d_lambda
        self.carrier_freq_hz = carrier_freq_hz

    @property
    def wavelength(self) -> float:
        return self.SPEED_OF_LIGHT / self.carrier_freq_hz

    def steering_vector(self, phi_rad: float, theta_rad: float) -> np.ndarray:
        """
        计算 URA 导向矢量

        Args:
            phi_rad: 方位角 (rad)
            theta_rad: 俯仰角 (rad)

        Returns:
            a: shape=(n_x * n_y,)
        """
        k = 2 * np.pi / self.wavelength

        # 水平方向
        n_x_arr = np.arange(self.n_x)
        a_x = np.exp(1j * k * self.d_lambda * n_x_arr * np.sin(phi_rad) * np.sin(theta_rad))

        # 垂直方向
        n_y_arr = np.arange(self.n_y)
        a_y = np.exp(1j * k * self.d_lambda * n_y_arr * np.cos(theta_rad))

        # Kronecker 积
        a = np.kron(a_x, a_y)
        return a


class UPA:
    """
    均匀平面阵 (Uniform Planar Array)

    同 URA, 统一命名

    Args:
        n_x: 水平天线数
        n_y: 垂直天线数
        d_lambda: 阵元间距 / 波长
        carrier_freq_hz: 载波频率
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        n_x: int,
        n_y: int,
        d_lambda: float = 0.5,
        carrier_freq_hz: Optional[float] = None,
    ):
        self.n_x = n_x
        self.n_y = n_y
        self.n_ant = n_x * n_y
        self.d_lambda = d_lambda
        self.carrier_freq_hz = carrier_freq_hz

    @property
    def wavelength(self) -> float:
        return self.SPEED_OF_LIGHT / self.carrier_freq_hz

    def steering_vector(self, phi_rad: float, theta_rad: float) -> np.ndarray:
        """同 URA"""
        k = 2 * np.pi / self.wavelength
        n_x_arr = np.arange(self.n_x)
        n_y_arr = np.arange(self.n_y)
        a_x = np.exp(1j * k * self.d_lambda * n_x_arr * np.sin(phi_rad) * np.sin(theta_rad))
        a_y = np.exp(1j * k * self.d_lambda * n_y_arr * np.cos(theta_rad))
        return np.kron(a_x, a_y)
