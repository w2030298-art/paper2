"""Buffer implementations for RL training."""

import torch
import numpy as np
from typing import Dict, List, Optional, Protocol, Tuple, Any


class BufferProtocol(Protocol):
    """Protocol for basic buffer operations."""

    def __len__(self) -> int: ...
    def push(self, *args, **kwargs) -> None: ...


class RolloutBuffer:
    """
    Unified On-Policy rollout buffer.

    Stores trajectory data for on-policy algorithms like PPO, GRPO.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.log_probs = np.zeros(self.capacity, dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.group_ids = np.zeros(self.capacity, dtype=np.int32)
        self.advantages = np.zeros(self.capacity, dtype=np.float32)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action,
        reward: float,
        done: bool,
        log_prob: float,
        value: float = 0.0,
        group_id: int = 0,
    ):
        """Add a single transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.group_ids[self.ptr] = group_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_batch(self, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        device = device or self.device
        indices = np.arange(self.size)
        return {
            "states": torch.tensor(self.states[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
            "log_probs": torch.tensor(self.log_probs[indices], device=device),
            "values": torch.tensor(self.values[indices], device=device),
            "group_ids": torch.tensor(self.group_ids[indices], device=device),
            "advantages": torch.tensor(self.advantages[indices], device=device),
            "returns": torch.tensor(self.returns[indices], device=device),
        }

    def get(self) -> Dict[str, torch.Tensor]:
        """Alias for get_batch() for backward compatibility."""
        return self.get_batch()

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95, last_value: float = 0.0):
        """Compute GAE advantages and returns in-place."""
        rewards = self.rewards[: self.size]
        values = self.values[: self.size]
        dones = self.dones[: self.size]
        advantages = np.zeros(self.size, dtype=np.float32)
        returns = np.zeros(self.size, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        self.advantages[: self.size] = advantages
        self.returns[: self.size] = returns

    def __len__(self) -> int:
        return self.size


class ReplayBuffer:
    """
    Unified experience replay buffer.

    Supports optional prioritized experience replay (PER).
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        device: str = "cpu",
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self.reset()

    def reset(self):
        """Reset the buffer."""
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        if self.action_dim > 1:
            self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(self.capacity, dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        if self.prioritized:
            self.priorities = np.zeros(self.capacity, dtype=np.float32)
        else:
            self.priorities = None
        self.ptr = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = 1.0,
    ):
        """Add a single transition to the buffer."""
        self.states[self.ptr] = state
        if self.action_dim > 1:
            action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
            self.actions[self.ptr] = action_arr[: self.action_dim]
        else:
            self.actions[self.ptr] = float(np.asarray(action).reshape(-1)[0])
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        if self.prioritized and self.priorities is not None:
            self.priorities[self.ptr] = priority**self.alpha
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        device = device or self.device
        if self.prioritized and self.priorities is not None:
            probs = self.priorities[: self.size] / self.priorities[: self.size].sum()
            indices = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()
        else:
            indices = np.random.choice(self.size, batch_size)
            weights = np.ones(batch_size)

        return {
            "states": torch.tensor(self.states[indices], device=device),
            "actions": torch.tensor(self.actions[indices], device=device),
            "rewards": torch.tensor(self.rewards[indices], device=device),
            "next_states": torch.tensor(self.next_states[indices], device=device),
            "dones": torch.tensor(self.dones[indices], device=device),
            "weights": torch.tensor(weights, device=device),
            "indices": indices,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for prioritized replay."""
        if self.prioritized and self.priorities is not None:
            self.priorities[indices] = priorities**self.alpha

    def __len__(self) -> int:
        return self.size
