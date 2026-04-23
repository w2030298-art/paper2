"""
QAM Modulation

支持:
- QPSK, 16-QAM, 64-QAM, 256-QAM
- 格雷码映射
- 误码率计算
"""

import numpy as np
from typing import Tuple, Optional


class QAMModem:
    """
    QAM 调制解调器

    Args:
        order: 调制阶数 (4, 16, 64, 256)
    """

    def __init__(self, order: int = 16):
        if order not in [4, 16, 64, 256]:
            raise ValueError(f"Unsupported QAM order: {order}. Use 4, 16, 64, or 256.")
        self.order = order
        self.constellation = self._generate_constellation()
        self.bit_per_symbol = int(np.log2(order))

    def _generate_constellation(self) -> np.ndarray:
        """生成 QAM 星座图"""
        if self.order == 4:
            # QPSK
            return np.array([
                -1 - 1j, -1 + 1j,
                 1 - 1j,  1 + 1j
            ]) / np.sqrt(2)

        elif self.order == 16:
            # 16-QAM
            points = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    points.append(i + 1j * q)
            constellation = np.array(points)
            # 归一化平均功率
            return constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))

        elif self.order == 64:
            # 64-QAM
            points = []
            for i in range(-7, 8, 2):
                for q in range(-7, 8, 2):
                    points.append(i + 1j * q)
            constellation = np.array(points)
            return constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))

        else:  # 256
            # 256-QAM
            points = []
            for i in range(-15, 16, 2):
                for q in range(-15, 16, 2):
                    points.append(i + 1j * q)
            constellation = np.array(points)
            return constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """
        比特调制

        Args:
            bits: 输入比特, shape=(N,) 或 (N, self.bit_per_symbol)

        Returns:
            symbols: QAM 符号, shape=(N_symbols,)
        """
        if bits.ndim == 1:
            # 逐比特映射
            n_symbols = len(bits) // self.bit_per_symbol
            bits = bits[:n_symbols * self.bit_per_symbol].reshape(n_symbols, self.bit_per_symbol)

        # 格雷码转整数
        symbols = np.zeros(len(bits), dtype=complex)
        for i, byte in enumerate(bits):
            idx = self._gray_to_int(byte)
            symbols[i] = self.constellation[idx]

        return symbols

    def demodulate(
        self,
        symbols: np.ndarray,
        hard: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        软/硬解调

        Args:
            symbols: 接收符号
            hard: 硬判决 (True) 或软判决 (False)

        Returns:
            bits: 解调比特
            llr: 对数似然比 (软判决时)
        """
        if hard:
            bits = self._hard_demodulate(symbols)
            return bits, None
        else:
            llr = self._soft_demodulate(symbols)
            return None, llr

    def _hard_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """硬判决解调"""
        n_symbols = len(symbols)
        bits = np.zeros(n_symbols * self.bit_per_symbol, dtype=int)

        for i, sym in enumerate(symbols):
            # 找最近星座点
            distances = np.abs(self.constellation - sym)
            idx = np.argmin(distances)
            bits[i * self.bit_per_symbol:(i + 1) * self.bit_per_symbol] = self._int_to_gray(idx)

        return bits

    def _soft_demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        软解调 (LLR)

        对每个比特计算对数似然比
        """
        n_symbols = len(symbols)
        llr = np.zeros(n_symbols * self.bit_per_symbol, dtype=float)

        for i, sym in enumerate(symbols):
            for b in range(self.bit_per_symbol):
                # 分成两组: 比特=0 和 比特=1
                ll_sum_0 = 0.0
                ll_sum_1 = 0.0

                for idx, constellation_pt in enumerate(self.constellation):
                    bits = self._int_to_gray(idx)
                    # 复信道高斯似然
                    ll = np.exp(-np.abs(sym - constellation_pt) ** 2)

                    if bits[b] == 0:
                        ll_sum_0 += ll
                    else:
                        ll_sum_1 += ll

                # LLR
                if ll_sum_0 > 1e-10 and ll_sum_1 > 1e-10:
                    llr[i * self.bit_per_symbol + b] = np.log(ll_sum_0 / ll_sum_1)
                else:
                    llr[i * self.bit_per_symbol + b] = 0.0

        return llr

    def _gray_to_int(self, gray: np.ndarray) -> int:
        """格雷码转整数"""
        bits = gray.flatten().astype(int)
        binary = np.zeros_like(bits)
        binary[0] = bits[0]
        for i in range(1, len(bits)):
            binary[i] = bits[i] ^ binary[i - 1]
        return int(np.sum(binary * (2 ** np.arange(len(binary) - 1, -1, -1))))

    def _int_to_gray(self, idx: int) -> np.ndarray:
        """整数转格雷码"""
        binary = np.array([int(b) for b in format(idx, f'0{self.bit_per_symbol}b')], dtype=int)
        gray = np.zeros_like(binary)
        gray[0] = binary[0]
        for i in range(1, len(binary)):
            gray[i] = binary[i] ^ binary[i - 1]
        return gray

    def ber_approx(self, snr_db: float) -> float:
        """
        近似误码率 (AWGN)

        使用 Q 函数近似

        Args:
            snr_db: SNR (dB)

        Returns:
            ber: 近似误码率
        """
        from scipy.special import Q

        snr_linear = 10 ** (snr_db / 10)

        # 每比特能量
        Eb = 1.0 / self.bit_per_symbol  # 归一化

        # N0 = Eb / SNR
        N0 = Eb / snr_linear

        # 符号能量
        Es = Eb * self.bit_per_symbol

        # 噪声方差
        sigma2 = N0 / 2

        if self.order == 4:
            # QPSK
            return Q(np.sqrt(2 * Es / sigma2))
        elif self.order == 16:
            # 16-QAM (近似)
            return 3 / 2 * Q(np.sqrt(0.1 * Es / sigma2))
        elif self.order == 64:
            return 7 / 12 * Q(np.sqrt(0.1 / 7 * Es / sigma2))
        else:
            return 3 / 8 * Q(np.sqrt(0.1 / 7 * Es / sigma2))


def get_modem(modulation: str) -> QAMModem:
    """
    获取调制解调器

    Args:
        modulation: "qpsk", "16qam", "64qam", "256qam"

    Returns:
        modem: QAMModem 实例
    """
    mapping = {
        "qpsk": 4,
        "4qam": 4,
        "16qam": 16,
        "64qam": 64,
        "256qam": 256,
    }
    order = mapping.get(modulation.lower())
    if order is None:
        raise ValueError(f"Unknown modulation: {modulation}")
    return QAMModem(order)
