"""Tests for train.py resolved runtime metadata output."""

from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _Torch:
    cuda = _Cuda()


class _Space:
    shape = (3,)
    dtype = "float32"
    low = [-1.0, -1.0, -1.0]
    high = [1.0, 1.0, 1.0]


class _Env:
    observation_space = _Space()
    action_space = _Space()

    def reset(self, seed=None):
        return [0.0, 0.0, 0.0], {}


class _Agent:
    state_dim = 3
    action_dim = 3
    action_type = "continuous"
    discrete = False
    num_agents = 1


class _Trainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self) -> None:
        return None

    def evaluate(self) -> dict[str, float]:
        return {"eval/reward_mean": 1.0, "eval/e2e_latency_p95": 2.0}


def _load_train_module():
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "scripts" / "train.py"
    spec = util.spec_from_file_location("scripts_train_resolved_runtime_module", train_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_result_json_contains_resolved_runtime_config(tmp_path: Path, monkeypatch) -> None:
    train_module = _load_train_module()
    config = tmp_path / "ppo.yaml"
    config.write_text(
        """
algorithm:
  name: PPO
  num_epochs: 2
training:
  total_timesteps: 64
  rollout_steps: 16
  seed: 123
evaluation:
  num_episodes: 3
logging:
  checkpoint_dir: checkpoints/test
game_theory:
  enabled: true
""",
        encoding="utf-8",
    )
    result_json = tmp_path / "result.json"

    def _fake_load_config(path: str) -> dict:
        import yaml

        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    env = _Env()
    agent = _Agent()
    monkeypatch.setattr(
        train_module,
        "_load_training_dependencies",
        lambda: {
            "torch": _Torch(),
            "OnPolicyTrainer": _Trainer,
            "OffPolicyTrainer": _Trainer,
            "set_seed": lambda seed: None,
            "ALGO_ENV_MAP": {"PPO": "MEC-v1-game-theory-continuous-ma"},
            "ON_POLICY": {"PPO"},
            "MULTI_AGENT_ALGOS": set(),
            "create_agent": lambda *args, **kwargs: agent,
            "make_env": lambda *args, **kwargs: env,
            "load_config": _fake_load_config,
            "resolve_game_theory_config": lambda *args, **kwargs: {"enabled": True},
            "_resolve_env_overrides": lambda **kwargs: {
                key: value for key, value in kwargs.items() if value is not None
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config),
            "--algorithm",
            "PPO",
            "--environment-profile",
            "mainline-a",
            "--timesteps",
            "32",
            "--eval-episodes",
            "1",
            "--device",
            "cpu",
            "--result-json",
            str(result_json),
        ],
    )

    train_module.main()

    payload = json.loads(result_json.read_text(encoding="utf-8"))
    resolved = payload["resolved_runtime_config"]
    assert resolved["algorithm"] == "PPO"
    assert resolved["environment_profile"] == "mainline-a"
    assert resolved["train_timesteps"] == 32
    assert resolved["eval_episodes"] == 1
    assert resolved["cli_overrides"]["device"] == "cpu"
    assert resolved["trainer_kwargs"]["total_timesteps"] == 32
    assert "env" not in resolved["trainer_kwargs"]
    assert "agent" not in resolved["trainer_kwargs"]
    assert resolved["game_theory_config"]["enabled"] is True
