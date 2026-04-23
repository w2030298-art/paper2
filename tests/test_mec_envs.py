"""
GameTheory MEC environment unit tests.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGameTheoryMECEnv:
    """GameTheory MEC 环境测试"""

    def test_reset_with_shapley(self):
        from src.environments.mec_v3 import GameTheoryMECEnv

        env = GameTheoryMECEnv(num_agents=3, num_edge_servers=3, max_steps=100)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, list)
        assert len(obs) == 3
        assert "shapley_allocation" in info
        assert "global_obs" in info

    def test_step_dict_actions(self):
        from src.environments.mec_v3 import GameTheoryMECEnv

        env = GameTheoryMECEnv(num_agents=3, num_edge_servers=3, max_steps=100)
        obs, info = env.reset(seed=42)
        actions = [{"target": 1, "ratio": np.array([0.0, 0.0, 0.0])} for _ in range(3)]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        assert isinstance(next_obs, list)
        assert len(rewards) == 3
        assert "individual_latencies" in info
        assert "individual_energies" in info


class TestGameTheoryDiscreteMAEnv:
    """GameTheory 离散适配环境测试"""

    def test_creation(self):
        from src.environments.mec_v3 import GameTheoryDiscreteMAEnv

        env = GameTheoryDiscreteMAEnv(num_agents=3, num_edge_servers=3)
        assert env.num_agents == 3
        assert env.action_space.n == 4 * (5 ** 3)  # (K+1) * bins^3

    def test_reset_and_step(self):
        from src.environments.mec_v3 import GameTheoryDiscreteMAEnv

        env = GameTheoryDiscreteMAEnv(num_agents=3, num_edge_servers=3)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, list)
        assert len(obs) == 3
        actions = [env.action_space.sample() for _ in range(3)]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        assert isinstance(next_obs, list)
        assert len(rewards) == 3

    def test_action_decode_encode(self):
        from src.environments.mec_v3.game_theory_adapters import (
            _decode_discrete_action,
            _encode_discrete_action,
        )

        # 测试 decode -> encode 一致性
        for action in [0, 124, 125, 249, 250, 499]:
            decoded = _decode_discrete_action(action, num_edge_servers=3)
            encoded = _encode_discrete_action(decoded["target"], decoded["ratio"], num_edge_servers=3)
            assert encoded == action, f"action {action} encode/decode mismatch: {encoded}"


class TestGameTheoryContinuousMAEnv:
    """GameTheory 连续适配环境测试"""

    def test_creation(self):
        from src.environments.mec_v3 import GameTheoryContinuousMAEnv

        env = GameTheoryContinuousMAEnv(num_agents=3, num_edge_servers=3)
        assert env.num_agents == 3
        assert env.action_space.shape == (4,)

    def test_reset_and_step(self):
        from src.environments.mec_v3 import GameTheoryContinuousMAEnv

        env = GameTheoryContinuousMAEnv(num_agents=3, num_edge_servers=3)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, list)
        assert len(obs) == 3
        actions = [env.action_space.sample() for _ in range(3)]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        assert isinstance(next_obs, list)
        assert len(rewards) == 3

    def test_action_decode(self):
        from src.environments.mec_v3.game_theory_adapters import _decode_continuous_action

        action = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
        decoded = _decode_continuous_action(action, num_edge_servers=3)
        assert decoded["target"] == 2
        assert np.allclose(decoded["ratio"], [-0.5, 0.0, 1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
