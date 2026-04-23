"""Helper functions."""

import random
import numpy as np
import torch
from typing import List, Sequence


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_done(terminated, truncated) -> bool:
    """Safely merge terminated/truncated into a done flag.

    Handles numpy scalars, python bools, and numpy arrays gracefully.
    Works with gymnasium env.step() outputs which may be np.bool_.
    """
    t = (
        bool(np.asarray(terminated).item())
        if np.ndim(terminated) > 0
        else bool(terminated)
    )
    tr = (
        bool(np.asarray(truncated).item())
        if np.ndim(truncated) > 0
        else bool(truncated)
    )
    return t or tr


def compute_discounted_returns(
    rewards: Sequence[float], gamma: float = 0.99
) -> np.ndarray:
    """Compute discounted returns efficiently."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    R = 0.0
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R
        returns[i] = R
    return returns


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute explained variance."""
    var_y = np.var(y_true)
    return 1 - np.var(y_true - y_pred) / var_y if var_y != 0 else 0.0


def normalize_observation(
    obs: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = 10.0
) -> np.ndarray:
    """Normalize observation with running statistics."""
    return np.clip((obs - mean) / (std + 1e-8), -clip, clip).astype(np.float32)


class RunningMeanStd:
    """Running mean and standard deviation calculator."""

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update statistics with new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Normalize data using running statistics."""
        return normalize_observation(x, self.mean, np.sqrt(self.var), clip)
