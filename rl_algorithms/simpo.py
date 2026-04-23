"""
SimPO: Simple Preference Optimization
DPO 的改进版，无需奖励模型

核心特点:
1. 无需奖励模型，直接优化策略
2. 使用 log probability 的差值替代显式奖励
3. 通过 reference policy 的 KL 散度约束策略偏离
4. 比 DPO 更简单、更稳定

Reference: "SimPO: Simple Preference Optimization" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorDiscreteNetwork, ActorNetwork
from src.utils.buffer import RolloutBuffer


class SimPOAgent(BaseAgent):
    """
    SimPO 智能体 - 简单偏好优化

    适用于: 网络切片配置、QoS优化、偏好驱动的控制任务

    训练数据格式: 偏好对 (state, action_chosen, action_rejected)
    推理时仅需 actor 网络
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        beta=0.1,
        ref_coeff=0.2,
        num_epochs=10,
        discrete=False,
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
            beta: float - 温度参数，控制偏好优化的强度 (默认0.1)
            ref_coeff: float - reference policy KL 惩罚系数 (默认0.2)
            num_epochs: int - 每次更新迭代次数 (默认10)
            discrete: bool - 是否离散动作空间 (默认False)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 维度
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        # 超参数
        self.gamma = gamma
        self.beta = beta
        self.ref_coeff = ref_coeff
        self.num_epochs = num_epochs

        # Actor 网络 (策略)
        if self.discrete:
            self.policy = ActorDiscreteNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        else:
            self.policy = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Reference policy (冻结，用于 KL 约束)
        if self.discrete:
            self.ref_policy = ActorDiscreteNetwork(state_dim, action_dim, hidden_dim).to(
                self.device
            )
        else:
            self.ref_policy = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self._freeze_ref()

        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

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

    def _freeze_ref(self):
        """冻结 reference policy"""
        for param in self.ref_policy.parameters():
            param.requires_grad = False

    def _sync_ref(self):
        """将当前策略权重复制到 reference policy"""
        self.ref_policy.load_state_dict(self.policy.state_dict())
        self._freeze_ref()

    def select_action(self, state, deterministic=False):
        """
        选择动作

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 动作，shape: [action_dim]
            info: dict - 包含 'log_prob': float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.sample(state_tensor, deterministic)

        return action.cpu().numpy()[0], {"log_prob": log_prob.cpu().item()}

    def update(self, batch_data):
        """
        使用 SimPO 更新策略

        支持两种数据格式:
        1. 偏好对格式: 包含 actions_chosen, actions_rejected
        2. 标准格式: 包含 states, actions, rewards — 自动构造偏好对

        Args:
            batch_data: dict - 批量数据

        Returns:
            info: dict - 训练信息
        """
        # 兼容标准 batch_data: 自动构造偏好对
        if "actions_chosen" not in batch_data:
            return self._update_from_standard(batch_data)

        # 数据转移到 GPU
        states = torch.FloatTensor(batch_data["states"]).to(self.device)
        actions_chosen = torch.FloatTensor(batch_data["actions_chosen"]).to(self.device)
        actions_rejected = torch.FloatTensor(batch_data["actions_rejected"]).to(self.device)

        batch_size = states.shape[0]

        total_loss = 0
        total_kl = 0
        total_margin = 0

        for epoch in range(self.num_epochs):
            # 计算 chosen 和 rejected 动作的 log probability
            if self.discrete:
                # 离散空间: 使用 Categorical
                chosen_dist = self.policy.get_distribution(states)
                log_prob_chosen = chosen_dist.log_prob(actions_chosen.squeeze(-1).long())

                rejected_dist = self.policy.get_distribution(states)
                log_prob_rejected = rejected_dist.log_prob(actions_rejected.squeeze(-1).long())

                # Reference policy
                with torch.no_grad():
                    ref_chosen = self.ref_policy.get_distribution(states)
                    ref_rejected = self.ref_policy.get_distribution(states)
                    log_ref_chosen = ref_chosen.log_prob(actions_chosen.squeeze(-1).long())
                    log_ref_rejected = ref_rejected.log_prob(actions_rejected.squeeze(-1).long())
            else:
                # 连续空间: 评估给定动作的 log prob
                chosen_dist = self.policy.get_distribution(states)
                log_prob_chosen = chosen_dist.log_prob(actions_chosen).sum(dim=-1)

                rejected_dist = self.policy.get_distribution(states)
                log_prob_rejected = rejected_dist.log_prob(actions_rejected).sum(dim=-1)

                with torch.no_grad():
                    ref_chosen = self.ref_policy.get_distribution(states)
                    ref_rejected = self.ref_policy.get_distribution(states)
                    log_ref_chosen = ref_chosen.log_prob(actions_chosen).sum(dim=-1)
                    log_ref_rejected = ref_rejected.log_prob(actions_rejected).sum(dim=-1)

            # SimPO 核心: log(pi_chosen/pi_rejected) 的 margin
            pi_logratio = log_prob_chosen - log_prob_rejected
            ref_logratio = log_ref_chosen - log_ref_rejected

            # SimPO loss = -log(sigma(beta * (log_ratio_chosen - log_ratio_rejected)))
            # 等效于最大化 chosen 相对 rejected 的 log prob 差
            simpo_input = self.beta * (pi_logratio - ref_logratio)
            simpo_loss = -F.logsigmoid(simpo_input).mean()

            # 熵正则 (鼓励探索)
            _, log_prob_cur = self.policy.sample(states)
            entropy = log_prob_cur.mean()

            # 总损失
            loss = simpo_loss - 0.01 * entropy

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # 统计
            margin = (log_prob_chosen - log_prob_rejected).mean().item()
            kl_val = (ref_logratio.abs().mean()).item()

            total_loss += loss.item()
            total_kl += kl_val
            total_margin += margin

        # 同步 reference policy (每隔一定更新步数同步一次)
        if self.update_count % 5 == 0:
            self._sync_ref()

        self.update_count += 1

        num_updates = self.num_epochs
        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_loss / num_updates,  # SimPO 的损失即为策略损失
            "entropy": -np.log(max(total_margin / num_updates, 1e-8)) if total_margin > 0 else 0,
            "approx_kl": total_kl / num_updates,
            "logratios_margin": total_margin / num_updates,
            "kl_penalty": total_kl / num_updates,
        }

    def _update_from_standard(self, batch_data, margin=0.1):
        """
        从标准 batch_data 格式构造偏好对并更新。

        策略: 取 top-k 和 bottom-k reward，并检查 margin
        """
        states = np.array(batch_data["states"])
        actions = np.array(batch_data["actions"])
        rewards = np.array(batch_data["rewards"]).flatten()

        sorted_idx = np.argsort(rewards)
        n = len(rewards)
        k = max(1, n // 4)
        chosen_idx = sorted_idx[-k:]
        rejected_idx = sorted_idx[:k]

        if rewards[chosen_idx].mean() - rewards[rejected_idx].mean() < margin:
            return self._update_ppo_style(batch_data)

        paired_data = {
            "states": states[chosen_idx],
            "actions_chosen": actions[chosen_idx],
            "actions_rejected": actions[rejected_idx],
        }
        return self.update(paired_data)

    def _update_ppo_style(self, batch_data):
        """
        PPO 式策略梯度更新 (当无法构造偏好对时的回退方案)
        """
        states = torch.FloatTensor(np.array(batch_data["states"])).to(self.device)
        actions = np.array(batch_data["actions"])
        rewards = torch.FloatTensor(np.array(batch_data["rewards"])).flatten().to(self.device)

        # 标准化奖励作为优势
        advantages = rewards
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for epoch in range(self.num_epochs):
            if self.discrete:
                dist = self.policy.get_distribution(states)
                actions_t = torch.LongTensor(actions.flatten()).to(self.device)
                log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()
            else:
                dist = self.policy.get_distribution(states)
                actions_t = torch.FloatTensor(actions).to(self.device)
                log_probs = dist.log_prob(actions_t).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

            policy_loss = -(log_probs * advantages).mean()
            loss = policy_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        self.update_count += 1
        if self.update_count % 5 == 0:
            self._sync_ref()

        return {
            "loss": total_loss / self.num_epochs,
            "policy_loss": total_loss / self.num_epochs,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "logratios_margin": 0.0,
            "kl_penalty": 0.0,
        }

    def state_dict(self):
        """获取状态字典"""
        return {
            "policy": self.policy.state_dict(),
            "ref_policy": self.ref_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.policy.load_state_dict(state_dict["policy"])
        self.ref_policy.load_state_dict(state_dict["ref_policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.update_count = state_dict.get("update_count", 0)
