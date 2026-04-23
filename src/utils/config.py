"""Configuration management with OmegaConf."""

from dataclasses import dataclass, field
from typing import Optional, List

from omegaconf import OmegaConf, MISSING


@dataclass
class EnvConfig:
    name: str = "MEC-v1-game-theory-continuous-ma"
    num_edge_servers: int = 3
    num_tasks: int = 5
    max_steps: int = 100


@dataclass
class AlgorithmConfig:
    name: str = MISSING  # must specify
    type: str = "on_policy"
    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    eps_clip: float = 0.2
    group_size: int = 64
    num_epochs: int = 10
    tau: float = 0.005
    alpha: float = 0.2
    auto_entropy: bool = True
    buffer_capacity: int = 1_000_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50000


@dataclass
class TrainingConfig:
    total_timesteps: int = 100_000
    rollout_steps: int = 2048
    warmup_steps: int = 1000
    batch_size: int = 256
    eval_episodes: int = 10
    eval_freq: int = 10000
    seed: int = 42
    device: str = "auto"  # auto/cpu/cuda


@dataclass
class LoggingConfig:
    log_interval: int = 100
    eval_interval: int = 10000
    save_interval: int = 50000
    checkpoint_dir: str = "./checkpoints"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(yaml_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """
    Load configuration with priority: CLI overrides > YAML > dataclass defaults.
    """
    base_cfg = OmegaConf.structured(Config)

    if yaml_path:
        yaml_cfg = OmegaConf.load(yaml_path)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)

    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    cfg = OmegaConf.to_object(base_cfg)
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Config):
    """Validate configuration values."""
    assert cfg.algorithm.lr > 0, f"lr must be positive, got {cfg.algorithm.lr}"
    assert 0 <= cfg.algorithm.gamma <= 1, f"gamma must be in [0, 1], got {cfg.algorithm.gamma}"
    assert cfg.training.total_timesteps > 0, "total_timesteps must be positive"
    assert cfg.training.batch_size > 0, "batch_size must be positive"
