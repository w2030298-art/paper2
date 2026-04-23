"""
IPPO: Independent PPO with Parameter Sharing
多智能体参数共享 PPO

每个智能体独立决策，但共享同一套网络参数。
Trainer 的 shared 模式会自动按 agent 展开样本。

本质上就是 PPO，但标记为 multi-agent shared 模式。
"""

from .ppo import PPOAgent


class IPPOAgent(PPOAgent):
    """
    IPPO 智能体 — 参数共享的多智能体 PPO

    适用于: 多智能体合作任务、多用户 MEC 卸载
    """

    multi_agent_mode = "shared"
    compatible_env_types = ["multi_agent"]

    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99,
                 eps_clip=0.2, gae_lambda=0.95, num_epochs=10, clip_grad=0.5,
                 value_coeff=0.5, ent_coeff=0.01, device="cuda", num_agents=None, discrete=False,
                 use_game_theory=True, use_shapley_credit=True, ctde_with_hints=True,
                 warm_start_steps=1000, warm_start_lr_scale=0.5, **kwargs):
        super().__init__(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
            lr=lr, gamma=gamma, eps_clip=eps_clip, gae_lambda=gae_lambda,
            num_epochs=num_epochs, clip_grad=clip_grad, value_coeff=value_coeff,
            ent_coeff=ent_coeff, device=device, discrete=discrete,
            use_game_theory=use_game_theory, use_shapley_credit=use_shapley_credit,
            ctde_with_hints=ctde_with_hints, warm_start_steps=warm_start_steps,
            warm_start_lr_scale=warm_start_lr_scale, **kwargs
        )
        self.num_agents = num_agents
