"""Action space utilities for unified action scaling and conversion."""

import numpy as np


class ActionScaler:
    """Unified action space scaling utilities."""

    @staticmethod
    def scale_continuous(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Scale from [-1, 1] to [low, high]."""
        return low + (action + 1.0) * 0.5 * (high - low)

    @staticmethod
    def unscale_continuous(
        scaled_action: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        """Scale from [low, high] to [-1, 1]."""
        return 2.0 * (scaled_action - low) / (high - low) - 1.0

    @staticmethod
    def discrete_from_continuous(action: np.ndarray) -> int:
        """Convert continuous output to discrete action via argmax."""
        return int(np.argmax(action))

    @staticmethod
    def sanitize_action(action: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf values with safe defaults and clip to [-1, 1]."""
        if not isinstance(action, np.ndarray):
            action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        if not np.isfinite(action).all():
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
        return action
