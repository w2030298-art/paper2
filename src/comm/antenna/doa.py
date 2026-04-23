"""
Direction of Arrival (DoA) Estimation

支持:
- MUSIC: Multiple Signal Classification
- ESPRIT: Estimation of Signal Parameters via Rotational Invariance Techniques
"""

import numpy as np
from typing import Tuple, Optional


class MUSIC:
    """
    MUSIC 算法 - 多信号分类

    利用信号子空间与噪声子空间的正交性进行 DoA 估计

    步骤:
    1. 计算协方差矩阵 R = E{xx^H}
    2. 特征值分解
    3. 噪声子空间特征矢量
    4. 谱峰搜索: P_MUSIC(θ) = 1 / (a(θ)^H @ E_n @ E_n^H @ a(θ))

    适用于: 多个不相关信号的 DoA 估计
    """

    def __init__(self, n_ant: int, d_lambda: float = 0.5, n_sig: int = 1):
        """
        Args:
            n_ant: 天线数
            d_lambda: 阵元间距 / 波长
            n_sig: 信号源数量 (用于确定噪声子空间维度)
        """
        self.n_ant = n_ant
        self.d_lambda = d_lambda
        self.n_sig = n_sig

    def _steering_vector(self, angle_rad: float) -> np.ndarray:
        """
        计算导向矢量

        Args:
            angle_rad: 波达方向 (rad)

        Returns:
            a: shape=(n_ant,)
        """
        k = 2 * np.pi / self.d_lambda
        n = np.arange(self.n_ant)
        a = np.exp(1j * k * n * np.sin(angle_rad))
        return a

    def _steering_matrix(self, angles_rad: np.ndarray) -> np.ndarray:
        """
        计算导向矢量矩阵

        Args:
            angles_rad: 波达方向数组, shape=(M,)

        Returns:
            A: shape=(n_ant, M)
        """
        k = 2 * np.pi / self.d_lambda
        n = np.arange(self.n_ant)
        A = np.exp(1j * k * np.outer(n, np.sin(angles_rad)))
        return A

    def estimate(
        self,
        X: np.ndarray,
        search_grid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        DoA 估计

        Args:
            X: 接收数据矩阵, shape=(n_ant, N) - N 为快拍数
            search_grid: 搜索角度网格 (rad), 默认 -90° ~ 90°

        Returns:
            angles: 估计的 DoA (rad), shape=(n_sig,)
            powers: 对应的伪谱峰值, shape=(n_sig,)
        """
        if search_grid is None:
            search_grid = np.linspace(-np.pi / 2, np.pi / 2, 361)

        # 协方差矩阵
        R = X @ X.conj().T / X.shape[1]  # (n_ant, n_ant)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(R)

        # 排序特征值 (降序)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 噪声子空间 (最后 n_ant - n_sig 个特征向量)
        E_n = eigenvectors[:, self.n_sig:]  # (n_ant, n_ant - n_sig)

        # MUSIC 谱
        A_grid = self._steering_matrix(search_grid)  # (n_ant, M)
        P_music = np.zeros(len(search_grid))

        for i, a in enumerate(A_grid.T):
            a = a[:, None]  # (n_ant, 1)
            denom = np.abs((a.conj().T @ E_n @ E_n.conj().T @ a)[0, 0])
            if denom > 1e-10:
                P_music[i] = 1.0 / denom
            else:
                P_music[i] = 0.0

        # 找峰值
        peak_idx = np.argsort(P_music)[-self.n_sig:][::-1]
        angles = search_grid[peak_idx]
        powers = P_music[peak_idx]

        return angles, powers

    def estimate_from_cov(
        self,
        R: np.ndarray,
        search_grid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从协方差矩阵估计 DoA

        Args:
            R: 协方差矩阵, shape=(n_ant, n_ant)
            search_grid: 搜索角度网格

        Returns:
            angles: 估计的 DoA (rad)
            powers: 对应的伪谱峰值
        """
        if search_grid is None:
            search_grid = np.linspace(-np.pi / 2, np.pi / 2, 361)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(R)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 噪声子空间
        E_n = eigenvectors[:, self.n_sig:]

        # MUSIC 谱
        A_grid = self._steering_matrix(search_grid)
        P_music = np.zeros(len(search_grid))

        for i, a in enumerate(A_grid.T):
            a = a[:, None]
            denom = np.abs((a.conj().T @ E_n @ E_n.conj().T @ a)[0, 0])
            if denom > 1e-10:
                P_music[i] = 1.0 / denom
            else:
                P_music[i] = 0.0

        peak_idx = np.argsort(P_music)[-self.n_sig:][::-1]
        angles = search_grid[peak_idx]
        powers = P_music[peak_idx]

        return angles, powers


class ESPRIT:
    """
    ESPRIT 算法 - 旋转不变子空间方法

    利用阵列几何的旋转不变性进行 DoA 估计

    要求阵列具有相同的子阵列 (如 ULA)

    步骤:
    1. 特征值分解得到信号子空间 E_s
    2. 利用旋转不变性: E_s = Φ @ E_s
    3. Φ 的特征值 exp(j*2π*d*sin(θ)/λ) 包含 DoA 信息

    适用于: 快速 DoA 估计, 无需谱搜索
    """

    def __init__(self, n_ant: int, d_lambda: float = 0.5, n_sig: int = 1):
        """
        Args:
            n_ant: 天线数
            d_lambda: 阵元间距 / 波长
            n_sig: 信号源数量
        """
        self.n_ant = n_ant
        self.d_lambda = d_lambda
        self.n_sig = n_sig

    def estimate(self, X: np.ndarray) -> np.ndarray:
        """
        DoA 估计

        Args:
            X: 接收数据矩阵, shape=(n_ant, N)

        Returns:
            angles: 估计的 DoA (rad), shape=(n_sig,)
        """
        n_snap = X.shape[1]

        # 协方差矩阵
        R = X @ X.conj().T / n_snap

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(R)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvectors = eigenvectors[:, idx]

        # 信号子空间
        E_s = eigenvectors[:, :self.n_sig]  # (n_ant, n_sig)

        # 分离两个子阵列 (假设 ULA, 移位 Δ = 1)
        E_s1 = E_s[:-1, :]  # 上子阵列 (n_ant-1, n_sig)
        E_s2 = E_s[1:, :]   # 下子阵列 (n_ant-1, n_sig)

        # 求解旋转矩阵 Φ: E_s2 = E_s1 @ Φ
        # 最小二乘: Φ = (E_s1^H E_s1)^{-1} E_s1^H E_s2
        E_s1_H = E_s1.conj().T
        # 使用伪逆
        Phi = np.linalg.lstsq(E_s1, E_s2, rcond=None)[0]

        # 特征值分解 Φ
        eigenvalues_phi, _ = np.linalg.eig(Phi)

        # DoA: angle = arcsin(angle * λ / (2πd))
        angles = np.angle(eigenvalues_phi)  # k*d*sin(θ)
        k = 2 * np.pi / self.d_lambda
        angles = np.arcsin(angles / k)

        return angles

    def estimate_from_subspace(
        self,
        E_s: np.ndarray,
    ) -> np.ndarray:
        """
        从信号子空间估计 DoA

        Args:
            E_s: 信号子空间, shape=(n_ant, n_sig)

        Returns:
            angles: 估计的 DoA (rad)
        """
        # 子阵列
        E_s1 = E_s[:-1, :]
        E_s2 = E_s[1:, :]

        # 旋转矩阵
        Phi = np.linalg.lstsq(E_s1, E_s2, rcond=None)[0]

        # 特征值
        eigenvalues_phi, _ = np.linalg.eig(Phi)

        # DoA
        k = 2 * np.pi / self.d_lambda
        angles = np.arcsin(np.angle(eigenvalues_phi) / k)

        return angles
