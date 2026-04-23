"""Logging utilities for training."""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class Logger:
    """
    Logger for training metrics with TensorBoard support.

    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
    ):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.log_dir)

        self.metrics: Dict[str, List[float]] = {}
        self.step = 0
        self.hparams_path = os.path.join(self.log_dir, "hparams.json")

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        with open(self.hparams_path, "w") as f:
            json.dump(params, f, indent=2)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if step is None:
            step = self.step

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            if self.use_tensorboard:
                self.writer.add_scalar(key, value, step)

        self.step += 1

    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """Log histogram of values."""
        if step is None:
            step = self.step
        if self.use_tensorboard:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text."""
        if step is None:
            step = self.step
        if self.use_tensorboard:
            self.writer.add_text(tag, text, step)

    def get_metrics(self, key: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(key, [])

    def get_last(self, key: str) -> Optional[float]:
        """Get the last value for a metric."""
        values = self.metrics.get(key, [])
        return values[-1] if values else None

    def get_mean(self, key: str, last_n: Optional[int] = None) -> float:
        """Get mean of metric values."""
        values = self.metrics.get(key, [])
        if last_n is not None:
            values = values[-last_n:]
        return np.mean(values) if values else 0.0

    def save_metrics(self, filename: str = "metrics.json"):
        """Save all metrics to a JSON file."""
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def close(self):
        """Close the logger."""
        if self.use_tensorboard:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
