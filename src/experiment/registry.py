"""Algorithm registry and command helpers."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from .environment_profiles import (
    DEFAULT_ENVIRONMENT_PROFILE,
    profile_to_train_extra_args,
    resolve_environment_profile,
)
from .models import AlgorithmSpec
from .presets import FULL_17_ALGORITHMS


class AlgorithmRegistry:
    """Canonical algorithm registry used by experiment orchestration."""

    SUPPORTED_ALGORITHMS = list(FULL_17_ALGORITHMS)

    def __init__(self) -> None:
        self._normalized_map = {name.upper(): name for name in self.SUPPORTED_ALGORITHMS}

    def canonicalize(self, name: str) -> str:
        normalized = str(name).strip().upper()
        canonical = self._normalized_map.get(normalized)
        if canonical is None:
            raise ValueError(f"Unsupported algorithm: {name}")
        return canonical

    def validate(self, names: Sequence[str]) -> list[str]:
        return [self.canonicalize(name) for name in names]

    def config_path_for(self, algorithm_name: str) -> str:
        canonical = self.canonicalize(algorithm_name)
        return (Path("configs") / "algorithms" / f"{canonical.lower()}.yaml").as_posix()

    def build_specs(
        self,
        algorithms: Sequence[str],
        *,
        timesteps: int,
        seed: int,
        device: str,
        eval_episodes: int,
        env: str = "auto",
        environment_profile: str = DEFAULT_ENVIRONMENT_PROFILE,
    ) -> list[AlgorithmSpec]:
        profile = resolve_environment_profile(environment_profile)
        specs: list[AlgorithmSpec] = []
        for algorithm in algorithms:
            canonical = self.canonicalize(algorithm)
            specs.append(
                AlgorithmSpec(
                    name=canonical,
                    config_path=self.config_path_for(canonical),
                    timesteps=timesteps,
                    seed=seed,
                    device=device,
                    env=env,
                    eval_episodes=eval_episodes,
                    extra_args=profile_to_train_extra_args(profile),
                )
            )
        return specs


class TrainCommandBuilder:
    """Build train.py subprocess commands for one algorithm."""

    def __init__(self, project_root: Path | str = ".") -> None:
        self.project_root = Path(project_root)

    def build(
        self,
        *,
        run_id: str,
        spec: AlgorithmSpec,
        experiment_dir: Path,
        python_executable: str | None = None,
    ) -> list[str]:
        python_bin = python_executable or sys.executable
        canonical_algorithm = spec.name

        train_script = self.project_root / "scripts" / "train.py"
        save_dir = experiment_dir / "artifacts" / canonical_algorithm / "checkpoints"
        result_json = experiment_dir / "artifacts" / canonical_algorithm / "result.json"

        command = [
            python_bin,
            str(train_script),
            "--config",
            spec.config_path,
            "--algorithm",
            spec.name,
            "--env",
            spec.env,
            "--timesteps",
            str(spec.timesteps),
            "--seed",
            str(spec.seed),
            "--device",
            spec.device,
            "--save-dir",
            str(save_dir),
            "--eval-episodes",
            str(spec.eval_episodes),
            "--result-json",
            str(result_json),
        ]
        if spec.extra_args:
            command.extend(spec.extra_args)
        return command
