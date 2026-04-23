"""Custom exceptions for GRPO_MEC."""


class GRPOMECError(Exception):
    """Base exception for GRPO_MEC."""

    pass


class EnvironmentError(GRPOMECError):
    """Environment-related errors."""

    pass


class AgentError(GRPOMECError):
    """Agent-related errors."""

    pass


class ConfigError(GRPOMECError):
    """Configuration errors."""

    pass


class BufferError(GRPOMECError):
    """Buffer-related errors."""

    pass
