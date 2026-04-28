"""Custom exceptions for experiment orchestration."""


class ExperimentError(Exception):
    """Base class for experiment-related errors."""


class ExperimentLockError(ExperimentError):
    """Raised when experiment lock acquisition fails."""


class ExperimentNotFoundError(ExperimentError):
    """Raised when experiment files cannot be found."""


class ExperimentStateError(ExperimentError):
    """Raised when experiment state is invalid."""
