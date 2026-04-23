"""DEPRECATED: Use src.utils.buffer instead.

This module is kept for backward compatibility.
RolloutBuffer and ReplayBuffer are re-exported from src.utils.buffer.
"""

import warnings

warnings.warn(
    "rl_algorithms.utils.buffers is deprecated. Use src.utils.buffer instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.utils.buffer import ReplayBuffer, RolloutBuffer, BufferProtocol

__all__ = ["ReplayBuffer", "RolloutBuffer", "BufferProtocol"]
