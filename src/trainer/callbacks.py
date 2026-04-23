"""Trainer callback system for extensible training hooks."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base_trainer import BaseTrainer


class TrainerCallback:
    """Base callback class for trainer events."""

    def on_train_begin(self, trainer: "BaseTrainer"):
        pass

    def on_step_end(self, trainer: "BaseTrainer", step: int, info: dict):
        pass

    def on_episode_end(self, trainer: "BaseTrainer", episode: int, info: dict):
        pass

    def on_eval_end(self, trainer: "BaseTrainer", eval_info: dict):
        pass

    def on_train_end(self, trainer: "BaseTrainer"):
        pass


class LoggingCallback(TrainerCallback):
    """Callback that logs training metrics to TensorBoard if available."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir

    def on_step_end(self, trainer, step, info):
        if trainer._tb_writer is not None:
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    trainer._tb_writer.add_scalar(f"train/{k}", v, step)

    def on_eval_end(self, trainer, eval_info):
        if trainer._tb_writer is not None:
            for k, v in eval_info.items():
                if isinstance(v, (int, float)):
                    trainer._tb_writer.add_scalar(f"eval/{k}", v, trainer.total_steps)


class CheckpointCallback(TrainerCallback):
    """Callback that saves checkpoints at a given frequency."""

    def __init__(self, save_freq: int, save_dir: Optional[str] = None):
        self.save_freq = save_freq
        self.save_dir = save_dir

    def on_step_end(self, trainer, step, info):
        if step % self.save_freq == 0 and step > 0:
            path = f"{self.save_dir or trainer.save_dir}/step_{step}.pt"
            trainer.save(path)


class EarlyStoppingCallback(TrainerCallback):
    """Callback that stops training when eval reward stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_reward = float("-inf")
        self.counter = 0

    def on_eval_end(self, trainer, eval_info):
        reward = eval_info.get("eval/reward_mean", float("-inf"))
        if reward > self.best_reward + self.min_delta:
            self.best_reward = reward
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer._stop_training = True
