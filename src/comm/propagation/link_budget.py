"""
Link Budget Calculator

计算通信链路的总增益与损耗
"""

import numpy as np
from typing import Optional, Tuple


class LinkBudget:
    """
    链路预算计算器

    计算发射功率经路径损耗、增益损耗后的接收功率

    公式: P_rx = P_tx + G_tx + G_rx - L_tx - L_rx - PL

    Args:
        tx_power_dbm: 发射功率 (dBm)
        tx_gain_dbi: 发射天线增益 (dBi)
        rx_gain_dbi: 接收天线增益 (dBi)
        tx_loss_db: 发射馈线损耗 (dB)
        rx_loss_db: 接收馈线损耗 (dB)
    """

    def __init__(
        self,
        tx_power_dbm: float = 30.0,
        tx_gain_dbi: float = 0.0,
        rx_gain_dbi: float = 0.0,
        tx_loss_db: float = 0.0,
        rx_loss_db: float = 0.0,
    ):
        self.tx_power_dbm = tx_power_dbm
        self.tx_gain_dbi = tx_gain_dbi
        self.rx_gain_dbi = rx_gain_dbi
        self.tx_loss_db = tx_loss_db
        self.rx_loss_db = rx_loss_db

    def compute(
        self,
        path_loss_db: float,
        fading_db: float = 0.0,
    ) -> Tuple[float, float]:
        """
        计算接收功率

        Args:
            path_loss_db: 路径损耗 (dB)
            fading_db: 衰落损耗 (dB), 默认 0

        Returns:
            rx_power_dbm: 接收功率 (dBm)
            rx_power_dbw: 接收功率 (dBW)
        """
        # 总增益
        total_gain = self.tx_gain_dbi + self.rx_gain_dbi

        # 总损耗
        total_loss = self.tx_loss_db + self.rx_loss_db + path_loss_db + fading_db

        # 接收功率
        rx_power_dbm = self.tx_power_dbm + total_gain - total_loss
        rx_power_dbw = rx_power_dbm - 30.0

        return rx_power_dbm, rx_power_dbw

    def compute_from_distance(
        self,
        distance_m: float,
        path_loss_model,
        fading_db: float = 0.0,
    ) -> Tuple[float, float]:
        """
        从距离计算接收功率

        Args:
            distance_m: 距离 (m)
            path_loss_model: 路径损耗模型 (callable)
            fading_db: 衰落损耗 (dB)

        Returns:
            rx_power_dbm: 接收功率 (dBm)
            rx_power_dbw: 接收功率 (dBW)
        """
        path_loss_db = path_loss_model(distance_m)
        return self.compute(path_loss_db, fading_db)

    def compute_snr(
        self,
        noise_power_dbm: float,
        path_loss_db: float,
        fading_db: float = 0.0,
    ) -> float:
        """
        计算 SNR

        Args:
            noise_power_dbm: 噪声功率 (dBm)
            path_loss_db: 路径损耗 (dB)
            fading_db: 衰落损耗 (dB)

        Returns:
            snr_db: SNR (dB)
        """
        rx_power_dbm, _ = self.compute(path_loss_db, fading_db)
        snr_db = rx_power_dbm - noise_power_dbm
        return snr_db
