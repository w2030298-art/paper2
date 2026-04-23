"""Queueing theory models: M/M/1, M/M/K, etc."""

from .mm1 import MM1Queue, MMCQueue

__all__ = [
    "MM1Queue",
    "MMCQueue",
]
