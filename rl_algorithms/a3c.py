"""
A3C: Advantage Actor-Critic (异步版本的多线程简化实现)
多线程异步并发训练的前身

核心特点:
1. 多线程并行 — 多个worker独立与环境交互，异步更新全局网络
2. 无需经验回放 — online更新，无相关性
3. 共享全局网络 — worker拉取参数，计算梯度后push回全局
4. 计算梯度而非直接更新 — 简化版用本地副本模拟

Reference: Mnih et al. "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorNetwork, ActorDiscreteNetwork, CriticNetwork


class A3CAgent(BaseAgent):
    """
    A3C 智能体 — 多线程同步简化版本

    适用于：大规模网络优化、分布式训练场景

    简化策略：使用本地网络副本模拟多线程worker交互，
    每步计算本地梯度后更新全局网络，再同步给本地。
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        num_steps=20,
        discrete=False,
        async_mode=False,
        device="cuda",
        use_game_theory=True,
        use_shapley_credit=False,
        ctde_with_hints=False,
        warm_start_steps=0,
        warm_start_lr_scale=0.5,
    ):
        """
        Args:
            state_dim: int - 状态空间维度
            action_dim: int - 动作空间维度
            hidden_dim: int - 网络隐藏层维度 (默认256)
            lr: float - 学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            num_steps: int - 每个worker的rollout步数 (默认20)
            discrete: bool - 是否离散动作空间 (默认False)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        if async_mode:
            raise NotImplementedError("Async A3C not yet implemented. Use async_mode=False.")
        self.async_mode = async_mode

        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 维度
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        # 超参数
        self.gamma = gamma
        self.num_steps = num_steps

        # Actor (策略网络)
        if self.discrete:
            self.actor = ActorDiscreteNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Critic (价值网络)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

        # 优化器 (共享)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # 缓冲区
        self.rollout_states = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_log_probs = []
        self.rollout_values = []

        # 训练计数
        self.update_count = 0
        self.is_on_policy = True
        self.action_type = "discrete" if self.discrete else "continuous"
        self.compatible_env_types = [self.action_type]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """
        选择动作 — 同时记录rollout数据

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 动作，shape: [action_dim]
            info: dict - 包含 'log_prob': float, 'value': float
        """
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim > 1 and state_arr.shape[0] == 1:
            state_arr = state_arr[0]

        state_tensor = torch.FloatTensor(state_arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.actor.sample(state_tensor, deterministic)
            value = self.critic(state_tensor)

        # 仅在训练采样阶段记录 rollout，避免评估污染训练缓存
        if not deterministic:
            self.rollout_states.append(state_arr.copy())
            self.rollout_actions.append(action.cpu().numpy()[0])
            self.rollout_log_probs.append(log_prob.cpu().item())
            self.rollout_values.append(value.cpu().item())

        return action.cpu().numpy()[0], {
            "log_prob": log_prob.cpu().item(),
            "value": value.cpu().item(),
        }

    def record_transition(self, reward, done):
        """记录环境反馈的reward和done"""
        self.rollout_rewards.append(reward)
        self.rollout_dones.append(float(done))

    def _compute_advantages(self, next_state=None, done=False):
        """
        计算A3C的优势估计和回报 (n-step)
        """
        values = torch.FloatTensor(self.rollout_values).to(self.device)
        rewards = torch.FloatTensor(self.rollout_rewards).to(self.device)
        dones = torch.FloatTensor(self.rollout_dones).to(self.device)

        advantages = torch.zeros(len(rewards), device=self.device)
        returns = torch.zeros(len(rewards), device=self.device)

        # 计算最后一步的bootstrap
        if next_state is not None:
            with torch.no_grad():
                bootstrap = (
                    self.critic(torch.FloatTensor(next_state).unsqueeze(0).to(self.device))
                    .squeeze(-1)
                    .cpu()
                    .item()
                    if not done
                    else 0.0
                )
        else:
            bootstrap = 0.0

        gae = 0
        running_return = bootstrap

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1.0 - dones[t])
            advantages[t] = running_return - values[t]
            returns[t] = running_return

        return advantages, returns, rewards

    def update(self, batch_data):
        """
        使用收集到的rollout数据更新A3C策略
        注意：A3C update前需要调用 select_action + record_transition 来收集数据

        Args:
            batch_data: dict - 包含:
                - states: np.ndarray [batch, state_dim] - (可选)
                - actions: np.ndarray [batch, action_dim] - (可选)
                - rewards: np.ndarray [batch] - (可选)
                - next_states: np.ndarray [batch, state_dim] - (可选)
                - dones: np.ndarray [batch] - (可选)
                - log_probs: np.ndarray [batch] - (可选)

        Returns:
            info: dict - 训练信息
                - loss: float - 总损失
                - policy_loss: float - 策略损失
                - value_loss: float - 价值损失
                - entropy: float - 策略熵
                - approx_kl: float - 近似KL散度
        """
        if len(self.rollout_states) == 0 and "states" not in batch_data:
            # 没有数据可更新
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
            }

        # 优先使用内部rollout数据
        if len(self.rollout_states) > 0:
            n_samples = min(
                len(self.rollout_states),
                len(self.rollout_actions),
                len(self.rollout_log_probs),
                len(self.rollout_values),
                len(self.rollout_rewards),
                len(self.rollout_dones),
            )
            if n_samples == 0:
                return {
                    "loss": 0.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "approx_kl": 0.0,
                }

            self.rollout_states = self.rollout_states[:n_samples]
            self.rollout_actions = self.rollout_actions[:n_samples]
            self.rollout_log_probs = self.rollout_log_probs[:n_samples]
            self.rollout_values = self.rollout_values[:n_samples]
            self.rollout_rewards = self.rollout_rewards[:n_samples]
            self.rollout_dones = self.rollout_dones[:n_samples]

            states = np.asarray(self.rollout_states, dtype=np.float32)
            actions = np.asarray(
                self.rollout_actions,
                dtype=np.int64 if self.discrete else np.float32,
            )
            log_probs = np.asarray(self.rollout_log_probs, dtype=np.float32)

            # 计算优势
            next_state = batch_data.get("next_states", None)
            if next_state is not None and next_state.ndim > 1:
                next_state = next_state[-1]

            advantages, returns, rewards = self._compute_advantages(
                next_state, done=(next_state is None and len(self.rollout_states) > 0)
            )

            # 清空rollout
            self.rollout_states = []
            self.rollout_actions = []
            self.rollout_rewards = []
            self.rollout_dones = []
            self.rollout_log_probs = []
            self.rollout_values = []

        else:
            # 使用batch_data
            states = batch_data["states"]
            actions = batch_data["actions"]
            log_probs = batch_data.get("log_probs", np.zeros(len(states)))
            rewards = torch.FloatTensor(batch_data["rewards"]).to(self.device)
            returns = torch.FloatTensor(batch_data.get("values", batch_data["rewards"])).to(
                self.device
            )
            advantages = returns

        # 转换为张量
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = (
            torch.FloatTensor(actions).to(self.device)
            if actions.ndim > 1
            else torch.LongTensor(actions).to(self.device)
        )
        log_probs_t = torch.FloatTensor(log_probs).to(self.device)

        # 计算策略损失
        if self.discrete:
            dist = self.actor.get_distribution(states_t)
            new_log_probs = dist.log_prob(actions_t.squeeze(-1).long())
            entropy = dist.entropy().mean()
        else:
            dist = self.actor.get_distribution(states_t)
            new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

        # 优势标准化
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss = -(log_probs_t * advantages).mean()

        # Value loss
        values = self.critic(states_t).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        # 总损失
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
        )
        self.optimizer.step()

        self.update_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": (log_probs_t - new_log_probs.detach()).mean().item()
            if not self.discrete
            else 0.0,
        }

    def state_dict(self):
        """获取状态字典"""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.update_count = state_dict.get("update_count", 0)
