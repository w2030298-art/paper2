"""
Unit Conversions for Communications

dB, dBm, dBW, Watt 之间转换
"""

import numpy as np
from typing import Union


def db_to_linear(x_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    dB 转线性

    Args:
        x_db: dB 值

    Returns:
        x_linear: 线性值
    """
    return 10 ** (x_db / 10)


def linear_to_db(x_linear: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    线性转 dB

    Args:
        x_linear: 线性值

    Returns:
        x_db: dB 值
    """
    return 10 * np.log10(x_linear)


def dbm_to_watts(x_dbm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    dBm 转 Watt

    0 dBm = 1 mW

    Args:
        x_dbm: dBm 值

    Returns:
        x_watts: Watt 值
    """
    return 10 ** ((x_dbm - 30) / 10)


def watts_to_dbm(x_watts: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Watt 转 dBm

    Args:
        x_watts: Watt 值

    Returns:
        x_dbm: dBm 值
    """
    return 10 * np.log10(x_watts) + 30


def dbw_to_watts(x_dbw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    dBW 转 Watt

    0 dBW = 1 W

    Args:
        x_dbw: dBW 值

    Returns:
        x_watts: Watt 值
    """
    return 10 ** (x_dbw / 10)


def watts_to_dbw(x_watts: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Watt 转 dBW

    Args:
        x_watts: Watt 值

    Returns:
        x_dbw: dBW 值
    """
    return 10 * np.log10(x_watts)


def dbm_to_dbw(x_dbm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """dBm 转 dBW"""
    return x_dbm - 30


def dbw_to_dbm(x_dbw: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """dBW 转 dBm"""
    return x_dbw + 30


def db_to_dbm(x_db: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """dB(相对于1mW)转dBm"""
    return x_db


def dbm_to_db(x_dbm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """dBm转dB(相对于1mW)"""
    return x_dbm
