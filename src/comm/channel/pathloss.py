"""
Path Loss Models

支持:
- 自由空间路径损耗 (Freespace)
- 3GPP TR 38.901 LOS / NLOS 路径损耗
- 可配置载波频率、天线高度、场景类型
"""

import numpy as np
from typing import Optional


class FreeSpacePathLoss:
    """
    自由空间路径损耗

    PL(d) = 20*log10(4*pi*d*f/c)  [dB]

    Args:
        carrier_freq: 载波频率 (Hz)
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(self, carrier_freq: float = 3.5e9):
        self.carrier_freq = carrier_freq
        self.wavelength = self.SPEED_OF_LIGHT / carrier_freq

    def __call__(self, distance: float) -> float:
        """
        计算路径损耗 (dB)

        Args:
            distance: 距离 (m)

        Returns:
            path_loss_dB: 路径损耗 (dB)
        """
        if distance <= 0:
            return 0.0
        return 20 * np.log10(4 * np.pi * distance / self.wavelength)

    def compute(self, distance: float) -> float:
        """同 __call__"""
        return self(distance)

    def compute_linear(self, distance: float) -> float:
        """返回线性尺度路径损耗"""
        return 10 ** (self(distance) / 10)


class PathLoss3GPP_LOS:
    """
    3GPP TR 38.901 LOS 路径损耗模型

    适用于: UMi, UMa, RMa 等城市场景

    PL = 28.0 + 22*log10(3D_distance) + 20*log10(f_c)  [dB]
    (d < d_BP)

    PL = 28.0 + 40*log10(3D_distance) + 20*log10(f_c)
         - 9*log10(d_BP^2 + (h_BS - h_UT)^2)  [dB]
    (d >= d_BP)

    Args:
        carrier_freq: 载波频率 (Hz), 默认 3.5 GHz
        h_bs: 基站高度 (m), 默认 25m
        h_ut: 终端高度 (m), 默认 1.5m
        scenario: 场景类型 "umi" | "uma" | "rma" | "indoor"
    """

    SPEED_OF_LIGHT = 3e8

    def __init__(
        self,
        carrier_freq: float = 3.5e9,
        h_bs: float = 25.0,
        h_ut: float = 1.5,
        scenario: str = "umi",
    ):
        self.carrier_freq = carrier_freq
        self.fc_ghz = carrier_freq / 1e9
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.scenario = scenario

        # 计算断点距离
        self.d_bp = self._breakpoint_distance()

    def _breakpoint_distance(self) -> float:
        """计算 3GPP 断点距离 d_BP"""
        h_diff = self.h_bs - self.h_ut
        return 4 * self.h_bs * self.h_ut / self.fc_ghz * 1e9 / self.SPEED_OF_LIGHT

    def __call__(self, distance_2d: float) -> float:
        """
        计算 LOS 路径损耗 (dB)

        Args:
            distance_2d: 2D 水平距离 (m)

        Returns:
            path_loss_dB: 路径损耗 (dB)
        """
        if distance_2d < 1.0:
            distance_2d = 1.0

        distance_3d = np.sqrt(distance_2d ** 2 + (self.h_bs - self.h_ut) ** 2)

        if self.scenario == "umi":
            if distance_2d < self.d_bp:
                pl = 32.4 + 21 * np.log10(distance_3d) + 20 * np.log10(self.fc_ghz)
            else:
                pl = 32.4 + 40 * np.log10(distance_3d) + 20 * np.log10(self.fc_ghz) - 9.5 * np.log10(
                    self.d_bp ** 2 + (self.h_bs - self.h_ut) ** 2
                )
        elif self.scenario == "uma":
            if distance_2d < self.d_bp:
                pl = 28.0 + 22 * np.log10(distance_3d) + 20 * np.log10(self.fc_ghz)
            else:
                pl = 28.0 + 40 * np.log10(distance_3d) + 20 * np.log10(self.fc_ghz) - 9 * np.log10(
                    self.d_bp ** 2 + (self.h_bs - self.h_ut) ** 2
                )
        elif self.scenario == "rma":
            pl = 20 * np.log10(40 * np.pi * distance_3d * self.fc_ghz / 3) - (
                3.0 * np.log10(3.0) - 5.0
            ) if distance_3d <= 10 else (
                20 * np.log10(40 * np.pi * distance_3d * self.fc_ghz / 3)
                + 25.5 * np.log10(distance_3d)
                - 3.0 * np.log10(3.0) - 5.0
            )
        else:
            # 默认 UMi
            pl = 32.4 + 21 * np.log10(distance_3d) + 20 * np.log10(self.fc_ghz)

        return pl

    def compute(self, distance_2d: float) -> float:
        return self(distance_2d)


class PathLoss3GPP_NLOS:
    """
    3GPP TR 38.901 NLOS 路径损耗模型

    PL_NLOS = max(PL_LOS, PL_NLOS_specific)

    Args:
        carrier_freq: 载波频率 (Hz)
        h_bs: 基站高度 (m)
        h_ut: 终端高度 (m)
        scenario: 场景类型
    """

    def __init__(
        self,
        carrier_freq: float = 3.5e9,
        h_bs: float = 25.0,
        h_ut: float = 1.5,
        scenario: str = "umi",
    ):
        self.carrier_freq = carrier_freq
        self.fc_ghz = carrier_freq / 1e9
        self.h_bs = h_bs
        self.h_ut = h_ut
        self.scenario = scenario

        # LOS 模型用于 max 运算
        self._los_model = PathLoss3GPP_LOS(
            carrier_freq=carrier_freq, h_bs=h_bs, h_ut=h_ut, scenario=scenario
        )

    def __call__(self, distance_2d: float) -> float:
        """
        计算 NLOS 路径损耗 (dB)

        Args:
            distance_2d: 2D 水平距离 (m)

        Returns:
            path_loss_dB: 路径损耗 (dB)
        """
        if distance_2d < 1.0:
            distance_2d = 1.0

        distance_3d = np.sqrt(distance_2d ** 2 + (self.h_bs - self.h_ut) ** 2)

        # LOS 路径损耗
        pl_los = self._los_model(distance_2d)

        # NLOS 特定路径损耗
        if self.scenario == "umi":
            pl_nlos = 35.3 * np.log10(distance_3d) + 22.4 + 21.3 * np.log10(self.fc_ghz) - 0.3 * (self.h_ut - 1.5)
        elif self.scenario == "uma":
            pl_nlos = 13.54 + 39.08 * np.log10(distance_3d) + 20 * np.log10(self.fc_ghz) - 0.6 * (self.h_ut - 1.5)
        elif self.scenario == "rma":
            pl_nlos = 161.04 - 7.1 * np.log10(10) + 7.5 * np.log10(self.h_ut) - (
                24.37 - 3.7 * (self.h_ut / 10) ** 2
            ) * np.log10(self.h_bs) + (
                43.42 - 3.1 * np.log10(self.h_bs)
            ) * np.log10(distance_3d)
        else:
            pl_nlos = 35.3 * np.log10(distance_3d) + 22.4 + 21.3 * np.log10(self.fc_ghz)

        return max(pl_los, pl_nlos)

    def compute(self, distance_2d: float) -> float:
        return self(distance_2d)
