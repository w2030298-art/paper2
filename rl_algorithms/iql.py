"""
IQL: Independent Q-Learning with Parameter Sharing
参数共享的独立 Q-Learning

每个智能体独立决策，但共享同一套 Q 网络参数。
Trainer 的 shared 模式会自动按 agent 展开样本。

本质上就是 DQN/DDQN，但标记为 multi-agent shared 模式。
"""

from .ddqn import DDQNAgent


class IQLAgent(DDQNAgent):
    """
    IQL 智能体 — 参数共享的独立 Q-Learning

    适用于: 多智能体离散动作合作任务
    """

    multi_agent_mode = "shared"
    compatible_env_types = ["multi_agent"]

    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99,
                 tau=0.005, batch_size=256, buffer_size=1_000_000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10000,
                 update_interval=1, device="cuda", n_agents=None,
                 use_game_theory=True, use_shapley_credit=True, ctde_with_hints=True,
                 warm_start_steps=1000, warm_start_lr_scale=0.5, **kwargs):
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
            lr=lr, gamma=gamma, tau=tau, batch_size=batch_size,
            buffer_size=buffer_size, epsilon_start=epsilon_start,
            epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
            update_interval=update_interval, device=device,
            use_game_theory=use_game_theory, use_shapley_credit=use_shapley_credit,
            ctde_with_hints=ctde_with_hints, warm_start_steps=warm_start_steps,
            warm_start_lr_scale=warm_start_lr_scale, **kwargs
        )
        self.n_agents = n_agents
