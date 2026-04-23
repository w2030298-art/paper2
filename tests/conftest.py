import pytest
import numpy as np


@pytest.fixture
def discrete_env():
    from src.environments import MECEnvDiscrete

    return MECEnvDiscrete(num_edge_servers=3, max_steps=50)


@pytest.fixture
def continuous_env():
    from src.environments import MECEnvContinuous

    return MECEnvContinuous(num_edge_servers=3, max_steps=50)


@pytest.fixture
def multi_agent_env():
    from src.environments import MECEnvMultiAgent

    return MECEnvMultiAgent(num_agents=2, num_edge_servers=3, max_steps=50)


@pytest.fixture
def dummy_batch():
    def _make(state_dim=10, action_dim=3, batch_size=32):
        return {
            "states": np.random.randn(batch_size, state_dim).astype(np.float32),
            "actions": np.random.randn(batch_size, action_dim).astype(np.float32),
            "rewards": np.random.randn(batch_size).astype(np.float32),
            "next_states": np.random.randn(batch_size, state_dim).astype(np.float32),
            "dones": np.zeros(batch_size).astype(np.float32),
            "log_probs": np.random.randn(batch_size).astype(np.float32),
        }

    return _make


@pytest.fixture
def set_deterministic():
    """Ensure test reproducibility."""
    import torch

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
