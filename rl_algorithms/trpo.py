"""
TRPO: Trust Region Policy Optimization
理论完备的信任区域策略优化

核心特点:
1. KL散度约束 — 保证策略更新幅度在合理范围内
2. 共轭梯度法求解 — 避免直接求Hessian逆矩阵
3. 线搜索验证 — 确保实际改进满足预期
4. 理论保证 — 单调改进有理论下界

Reference: Schulman et al. "Trust Region Policy Optimization" (2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .utils.networks import ActorNetwork, CriticNetwork


class TRPOAgent(BaseAgent):
    """
    TRPO 智能体 — 信任区域策略优化

    适用于：理论研究基线、需要理论保证的策略优化场景
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        max_kl=0.01,
        cg_iters=10,
        backtrack_ratio=0.8,
        max_backtrack=10,
        damping=0.1,
        value_coeff=0.5,
        ent_coeff=0.01,
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
            lr: float - 价值网络学习率 (默认3e-4)
            gamma: float - 折扣因子 (默认0.99)
            max_kl: float - 最大KL散度约束 (默认0.01)
            cg_iters: int - 共轭梯度法迭代次数 (默认10)
            backtrack_ratio: float - 回溯衰减比例 (默认0.8)
            max_backtrack: int - 最大回溯次数 (默认10)
            damping: float - FIM阻尼系数 (默认0.1)
            value_coeff: float - 价值损失系数 (默认0.5)
            ent_coeff: float - 熵正则系数 (默认0.01)
            device: str - 'cuda' 或 'cpu' (默认'cuda')
        """
        # 设备设置
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 维度
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 超参数
        self.gamma = gamma
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.backtrack_ratio = backtrack_ratio
        self.max_backtrack = max_backtrack
        self.damping = damping
        self.value_coeff = value_coeff
        self.ent_coeff = ent_coeff

        # Actor (策略网络)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Critic (价值网络)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

        # 优化器 (仅用于Critic, TRPO不用于Actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # 训练计数
        self.update_count = 0
        self.is_on_policy = True
        self.action_type = "continuous"
        self.compatible_env_types = ["continuous"]
        self.use_game_theory = bool(use_game_theory)
        self.use_shapley_credit = bool(use_shapley_credit)
        self.ctde_with_hints = bool(ctde_with_hints)
        self.warm_start_steps = int(warm_start_steps)
        self.warm_start_lr_scale = float(warm_start_lr_scale)

    def select_action(self, state, deterministic=False):
        """
        选择动作

        Args:
            state: np.ndarray - 当前状态，shape: [state_dim]
            deterministic: bool - 是否确定性选择 (默认False)

        Returns:
            action: np.ndarray - 动作，shape: [action_dim]
            info: dict - 包含 'log_prob': float, 'value': float
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.sample(state_tensor, deterministic)
            value = self.critic(state_tensor)

        return action.cpu().numpy()[0], {
            "log_prob": log_prob.cpu().item(),
            "value": value.cpu().item(),
        }

    def _compute_advantages(self, values, rewards, dones):
        """计算GAE优势估计"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # lambda=0.95
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _get_flat_grad(self, loss, params, retain_graph=True):
        """获取所有参数的梯度展平向量"""
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=True)
        return torch.cat([g.view(-1) for g in grads])

    def _fvp(self, vector, states, params):
        """
        Fisher-Vector Product (FVP)
        近似 Fisher 信息矩阵与向量的乘积
        """
        # 计算KL散度: D_KL(pi_old || pi_new)
        with torch.no_grad():
            mean_old, log_std_old = self.actor(states)
            std_old = torch.exp(log_std_old)

        # 重新计算以保留计算图
        mean_new, log_std_new = self.actor(states)
        std_new = torch.exp(log_std_new)

        # 高斯分布的KL散度解析式
        kl = (
            (
                log_std_new
                - log_std_old
                + (std_old.pow(2) + (mean_old - mean_new).pow(2)) / (2 * std_new.pow(2))
                - 0.5
            )
            .sum(dim=1)
            .mean()
        )

        # Fisher gradient
        grads = torch.autograd.grad(kl, params, create_graph=True)
        flat_grad_kl = torch.cat([g.view(-1) for g in grads])

        # Fisher-Vector Product
        fvp = torch.autograd.grad((flat_grad_kl * vector).sum(), params, retain_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in fvp]) + self.damping * vector

    def _conjugate_gradient(self, fvp, b, max_iters=10, tol=1e-10):
        """Conjugate gradient solver for Ax = b with numerical safety checks."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rs_old = torch.dot(r, r)

        for _ in range(max_iters):
            Av = fvp(p)
            pAv = torch.dot(p, Av)

            if pAv.abs() < 1e-12:
                break

            alpha = rs_old / pAv

            if torch.isnan(alpha) or torch.isinf(alpha):
                break

            x = x + alpha * p
            r = r - alpha * Av
            rs_new = torch.dot(r, r)

            if rs_new < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x

    def _line_search(self, old_actor_params, new_params, states, actions, adv, expected_improve):
        """
        线搜索：找到满足KL约束的最大步长
        """
        x = torch.nn.utils.parameters_to_vector(old_actor_params)
        full_step = new_params

        with torch.no_grad():
            mean_old, log_std_old = self.actor(states)
            std_old = torch.exp(log_std_old)
            dist_old = torch.distributions.Normal(mean_old, std_old)
            old_log_probs = dist_old.log_prob(actions).sum(dim=-1)

        for i in range(self.max_backtrack):
            # 设置新参数
            new_x = x - (self.backtrack_ratio**i) * full_step
            torch.nn.utils.vector_to_parameters(new_x, self.actor.parameters())

            # 计算KL散度
            with torch.no_grad():
                mean_new, log_std_new = self.actor(states)
                std_new = torch.exp(log_std_new)
                if (not torch.isfinite(mean_new).all()) or (not torch.isfinite(std_new).all()):
                    continue

                # KL(pi_old || pi_new)
                kl = (
                    (
                        log_std_new
                        - log_std_old
                        + (std_old.pow(2) + (mean_old - mean_new).pow(2)) / (2 * std_new.pow(2))
                        - 0.5
                    )
                    .sum(dim=1)
                    .mean()
                )

                # 如果KL太大，缩小步长
                if kl.item() > self.max_kl:
                    continue

                # 计算新的策略损失改善
                dist_new = torch.distributions.Normal(mean_new, std_new)
                new_log_probs = dist_new.log_prob(actions).sum(dim=-1)
                ratio = torch.exp(new_log_probs - old_log_probs)
                loss_improve = (ratio * adv).mean().item()

                if loss_improve > 0:
                    return True

        # 恢复旧参数
        torch.nn.utils.vector_to_parameters(x, self.actor.parameters())
        return False

    def update(self, batch_data):
        """
        使用TRPO更新策略

        Args:
            batch_data: dict - 批量经验数据，包含:
                - states: np.ndarray [batch, state_dim]
                - actions: np.ndarray [batch, action_dim]
                - rewards: np.ndarray [batch]
                - next_states: np.ndarray [batch, state_dim]
                - dones: np.ndarray [batch]
                - log_probs: np.ndarray [batch]
                - values: np.ndarray [batch] (可选)

        Returns:
            info: dict - 训练信息
                - loss: float - 策略损失
                - policy_loss: float - 策略损失
                - value_loss: float - 价值损失
                - entropy: float - 策略熵
                - approx_kl: float - KL散度
        """
        states = torch.FloatTensor(batch_data["states"]).to(self.device)
        actions = torch.FloatTensor(batch_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(batch_data["rewards"]).to(self.device)
        dones = torch.FloatTensor(batch_data["dones"]).to(self.device)
        old_log_probs = torch.FloatTensor(batch_data["log_probs"]).to(self.device)

        if batch_data.get("values") is not None:
            values = torch.FloatTensor(batch_data["values"]).to(self.device).squeeze(-1)
        else:
            values = self.critic(states).squeeze(-1)

        # 计算优势
        advantages, returns = self._compute_advantages(values, rewards, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 保存旧策略参数 ---
        actor_params = list(self.actor.parameters())
        old_params = torch.nn.utils.parameters_to_vector(actor_params).clone()

        # --- 计算策略梯度 ---
        dist = self.actor.get_distribution(states)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean()

        # 获取梯度
        flat_grad = self._get_flat_grad(policy_loss, actor_params, retain_graph=True)

        # --- 共轭梯度法求解 ---
        def fvp(v):
            return self._fvp(v, states, actor_params)

        step_dir = self._conjugate_gradient(fvp, flat_grad, max_iters=self.cg_iters)

        # 计算步长
        shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
        if shs.item() > 1e-10:
            step = step_dir * torch.sqrt(self.max_kl / shs)
        else:
            step = torch.zeros_like(step_dir)
        step_norm = torch.norm(step)
        search_step = step if step_norm.item() > 1e-12 else torch.zeros_like(step)

        # --- 线搜索 ---
        success = self._line_search(
            [p.clone() for p in actor_params],  # 保存的旧参数
            search_step,
            states,
            actions,
            advantages,
            expected_improve=None,
        )

        # --- 更新Critic ---
        critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # 计算KL散度
        with torch.no_grad():
            mean_new, log_std_new = self.actor(states)
            std_new = torch.exp(log_std_new)
            mean_old = torch.zeros_like(mean_new)
            std_old = torch.ones_like(std_new)
            dist_new = torch.distributions.Normal(mean_new, std_new)
            entropy = dist_new.entropy().sum(dim=-1).mean().item()

        # 计算近似KL
        approx_kl = (old_log_probs - new_log_probs.detach()).mean().item()

        self.update_count += 1

        return {
            "loss": policy_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": critic_loss.item(),
            "entropy": entropy,
            "approx_kl": approx_kl,
        }

    def state_dict(self):
        """获取状态字典"""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_count": self.update_count,
        }

    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.update_count = state_dict.get("update_count", 0)
