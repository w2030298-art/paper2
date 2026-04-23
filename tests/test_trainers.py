"""
Tests for trainers (OnPolicyTrainer, OffPolicyTrainer)
"""

import sys
import tempfile
from pathlib import Path
from uuid import uuid4
sys.path.insert(0, "src")

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock

from src.trainer.on_policy_trainer import OnPolicyTrainer
from src.trainer.off_policy_trainer import OffPolicyTrainer
from src.trainer.base_trainer import BaseTrainer
from src.environments.mec_v3 import GameTheoryDiscreteMAEnv, GameTheoryContinuousMAEnv


class DummyOnPolicyAgent:
    """Dummy on-policy agent for testing."""

    def __init__(self, state_dim, action_dim, hidden_dim=64, device="cpu", **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.update_count = 0
        self.total_steps = 0

        # Simple linear policy
        self.fc = torch.nn.Linear(state_dim, action_dim)

    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.fc(state_t)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        return action.cpu().numpy()[0], {"log_prob": 0.0, "value": 0.0}

    def update(self, batch_data):
        self.update_count += 1
        return {"loss": 0.1, "policy_loss": -0.01, "value_loss": 0.05}

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        pass

    def state_dict(self):
        return {}


class DummyOffPolicyAgent:
    """Dummy off-policy agent for testing."""

    def __init__(self, state_dim, action_dim, hidden_dim=64, device="cpu", **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.update_count = 0

    def select_action(self, state, deterministic=False):
        return np.random.randint(0, self.action_dim), {"log_prob": 0.0}

    def update(self, batch_data):
        self.update_count += 1
        return {"loss": 0.1, "q_loss": 0.05}

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        pass

    def state_dict(self):
        return {}


class TestOnPolicyTrainer:
    """Test OnPolicyTrainer."""

    def test_creation_discrete_env(self):
        """Test creating trainer with discrete env."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOnPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OnPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=1000,
            rollout_steps=64,
            log_interval=99999,  # suppress logs
            eval_interval=99999,
            save_interval=99999,
            device="cpu",
            seed=42,
            num_epochs=2,
        )
        assert trainer.total_timesteps == 1000
        assert trainer.rollout_steps == 64

    def test_creation_continuous_env(self):
        """Test creating trainer with continuous env."""
        env = GameTheoryContinuousMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOnPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
        )
        trainer = OnPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=1000,
            rollout_steps=64,
            log_interval=99999,
            eval_interval=99999,
            save_interval=99999,
            device="cpu",
            seed=42,
            num_epochs=2,
        )
        assert trainer.total_timesteps == 1000

    def test_short_training_run(self):
        """Test a very short training run (no real updates)."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOnPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OnPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=128,
            rollout_steps=64,
            log_interval=99999,
            eval_interval=99999,
            save_interval=99999,
            device="cpu",
            seed=42,
            num_epochs=1,
        )
        logs = trainer.train()
        assert trainer.total_steps >= 128
        assert trainer.update_count >= 1

    def test_collect_rollout_shape(self):
        """Test rollout data shapes."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOnPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OnPolicyTrainer(
            env=env, agent=agent, total_timesteps=128,
            rollout_steps=32, log_interval=99999, eval_interval=99999,
            save_interval=99999, device="cpu", seed=42, num_epochs=1,
        )
        rollout = trainer._collect_rollout()
        assert "states" in rollout
        assert "actions" in rollout
        assert "rewards" in rollout
        assert "dones" in rollout
        assert rollout["states"].shape[0] == 32


class TestOffPolicyTrainer:
    """Test OffPolicyTrainer."""

    def test_creation_discrete_env(self):
        """Test creating off-policy trainer."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOffPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OffPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=1000,
            rollout_steps=64,
            log_interval=99999,
            eval_interval=99999,
            save_interval=99999,
            device="cpu",
            seed=42,
            warmup_steps=50,
            update_interval=1,
        )
        assert trainer.total_timesteps == 1000
        assert trainer.warmup_steps == 50

    def test_short_training_run(self):
        """Test short off-policy training run."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOffPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OffPolicyTrainer(
            env=env,
            agent=agent,
            total_timesteps=200,
            rollout_steps=100,
            log_interval=99999,
            eval_interval=99999,
            save_interval=99999,
            device="cpu",
            seed=42,
            warmup_steps=10,
            update_interval=1,
        )
        logs = trainer.train()
        assert trainer.total_steps >= 200


class TestBaseTrainer:
    """Test BaseTrainer methods."""

    def test_seed_setting(self):
        """Test random seed setting via concrete trainer."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOnPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OnPolicyTrainer(
            env=env, agent=agent, total_timesteps=100,
            rollout_steps=50, log_interval=99999, eval_interval=99999,
            save_interval=99999, device="cpu", seed=42, num_epochs=1,
        )
        trainer._set_seed(123)
        # Just verify it doesn't crash
        assert True

    def test_trainer_save_load(self):
        """Test save/load methods."""
        env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=50)
        agent = DummyOnPolicyAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )
        trainer = OnPolicyTrainer(
            env=env, agent=agent, total_timesteps=100,
            rollout_steps=50, log_interval=99999, eval_interval=99999,
            save_interval=99999, device="cpu", seed=42, num_epochs=1,
        )
        save_path = Path(tempfile.gettempdir()) / f"paper2_test_ckpt_{uuid4().hex}.pt"
        trainer.save(str(save_path))
        # load should not crash either
        trainer.load(str(save_path))
        if save_path.exists():
            save_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
