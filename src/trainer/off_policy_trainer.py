"""
Off-Policy Trainer — 用于 SAC, DDPG, TD3, DDQN, QMIX

特点: 经验回放，数据可重复利用；agent内部管理replay buffer
"""

import numpy as np
import torch
from typing import Dict, Any
from gymnasium import spaces


from .base_trainer import BaseTrainer
from src.utils.helpers import is_done


class OffPolicyTrainer(BaseTrainer):
    """
    Off-Policy训练器

    适用于: SAC, DDPG, TD3, DDQN, QMIX

    注意: 这些算法在内部管理replay buffer，
    update()调用时数据先存入buffer，再从中采样更新
    """

    def __init__(
        self,
        env,
        agent,
        warmup_steps: int = 1000,
        update_interval: int = 1,
        **kwargs,
    ):
        super().__init__(env, agent, **kwargs)
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.step_count = 0

        # 检测动作空间类型
        self._is_discrete = isinstance(env.action_space, spaces.Discrete)
        self._is_dict = isinstance(env.action_space, spaces.Dict)
        self._is_multi_agent = hasattr(env, "num_agents") and env.num_agents > 1
        self._ma_mode = getattr(agent, "multi_agent_mode", "shared")

        self.obs_dim = env.observation_space.shape[0]

    def _collect_rollout(self) -> Dict[str, Any]:
        """
        收集 rollout 数据用于 off-policy 算法
        每个 step 都调用 update
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        global_states = []
        eq_actions = []
        shapley_values = []
        reward_terms = []
        game_hint_vectors = []
        episodes_done = 0

        obs, info = self.env.reset()
        done = False

        def _reward_terms_to_array(terms):
            if isinstance(terms, list):
                arr = []
                for t in terms:
                    if isinstance(t, dict):
                        arr.append(
                            [
                                float(t.get("r_imm", 0.0)),
                                float(t.get("r_coop", 0.0)),
                                float(t.get("r_eq", 0.0)),
                            ]
                        )
                    else:
                        arr.append([0.0, 0.0, 0.0])
                return np.asarray(arr, dtype=np.float32)
            if isinstance(terms, dict):
                return np.asarray(
                    [
                        [
                            float(terms.get("r_imm", 0.0)),
                            float(terms.get("r_coop", 0.0)),
                            float(terms.get("r_eq", 0.0)),
                        ]
                    ],
                    dtype=np.float32,
                )
            return None

        def _game_hint_vector(step_info):
            hints = step_info.get("game_hints", {})
            if not isinstance(hints, dict):
                return None
            prices = np.asarray(hints.get("equilibrium_prices", []), dtype=np.float32).reshape(-1)
            demands = np.asarray(hints.get("predicted_demands", []), dtype=np.float32).reshape(-1)
            if prices.size == 0 and demands.size == 0:
                return None
            return np.concatenate([prices, demands]).astype(np.float32)

        # 对于 off-policy，不需要一次性收集 rollout_steps
        # 只需要收集一个 episode 或者部分数据
        target_steps = min(self.rollout_steps, 2048)

        for _ in range(target_steps):
            # 动作选择
            if self._is_multi_agent:
                # 多智能体
                agent_actions = []
                for i, agent_obs in enumerate(obs):
                    with torch.no_grad():
                        action, _ = self.agent.select_action(agent_obs)
                    # 统一动作格式 (与 OnPolicyTrainer / evaluate 保持一致)
                    if isinstance(action, torch.Tensor):
                        action_val = action.cpu().numpy()
                        if action_val.ndim > 1:
                            action_val = action_val[0]
                    elif isinstance(action, np.ndarray):
                        if action.ndim > 1:
                            action = action[0]
                        action_val = action
                    else:
                        action_val = action
                    # 离散动作为 int，连续保留 ndarray
                    if self._is_discrete:
                        if isinstance(action_val, np.ndarray) and action_val.ndim >= 1:
                            if action_val.size == 1:
                                action_val = int(action_val.item())
                            else:
                                action_val = int(np.argmax(action_val, axis=-1).flatten()[0])
                        elif hasattr(action_val, "item"):
                            action_val = int(action_val.item())
                        else:
                            action_val = int(action_val)
                    else:
                        if isinstance(action_val, np.ndarray):
                            action_val = action_val.flatten()
                        else:
                            action_val = np.atleast_1d(np.asarray(action_val, dtype=np.float32)).flatten()
                    agent_actions.append(action_val)

                next_obs, step_rewards, terminated, truncated, info = self.env.step(agent_actions)
                done = is_done(terminated, truncated)

                n_agents = getattr(self.env, "num_agents", 1)
                reward_arr = np.asarray(step_rewards, dtype=np.float32).reshape(-1)
                if reward_arr.size == 0:
                    reward_arr = np.zeros(n_agents, dtype=np.float32)
                elif reward_arr.size == 1:
                    reward_arr = np.full(n_agents, float(reward_arr[0]), dtype=np.float32)
                elif reward_arr.size < n_agents:
                    reward_arr = np.pad(reward_arr, (0, n_agents - reward_arr.size), mode="edge")
                reward_arr = reward_arr[:n_agents]

                if self._ma_mode == "joint":
                    # joint 模式: 保留 [n_agents, ...] 维度
                    states.append(np.asarray(obs, dtype=np.float32))
                    actions.append(
                        np.asarray(
                            agent_actions,
                            dtype=np.int64 if self._is_discrete else np.float32,
                        )
                    )
                    rewards.append(reward_arr)
                    next_states.append(np.asarray(next_obs, dtype=np.float32))
                    dones.append(np.full(n_agents, done, dtype=np.float32))
                    # global_states
                    g_obs = info.get("global_obs", None)
                    if g_obs is None:
                        g_obs = np.concatenate([np.asarray(o, dtype=np.float32) for o in obs])
                    global_states.append(np.asarray(g_obs, dtype=np.float32))
                    if info.get("eq_actions") is not None:
                        eq_actions.append(np.asarray(info.get("eq_actions"), dtype=np.float32))
                    if info.get("shapley_allocation") is not None:
                        shapley_values.append(np.asarray(info.get("shapley_allocation"), dtype=np.float32))
                    r_terms = _reward_terms_to_array(info.get("reward_terms"))
                    if r_terms is not None:
                        reward_terms.append(r_terms)
                    hint_vec = _game_hint_vector(info)
                    if hint_vec is not None:
                        game_hint_vectors.append(hint_vec)
                else:
                    # shared 模式: 按 agent 展开为单智能体样本
                    for i in range(n_agents):
                        states.append(obs[i])
                        actions.append(agent_actions[i])
                        rewards.append(float(reward_arr[i]))
                        next_states.append(next_obs[i])
                        dones.append(done)
                        if info.get("eq_actions") is not None:
                            eq_a = np.asarray(info.get("eq_actions"), dtype=np.float32)
                            if eq_a.ndim == 2 and i < eq_a.shape[0]:
                                eq_actions.append(eq_a[i])
                        if info.get("shapley_allocation") is not None:
                            shp = np.asarray(info.get("shapley_allocation"), dtype=np.float32).reshape(-1)
                            if shp.size > i:
                                shapley_values.append(np.asarray([shp[i]], dtype=np.float32))
                        r_terms = _reward_terms_to_array(info.get("reward_terms"))
                        if r_terms is not None and r_terms.ndim == 2 and i < r_terms.shape[0]:
                            reward_terms.append(r_terms[i])
                        hint_vec = _game_hint_vector(info)
                        if hint_vec is not None:
                            game_hint_vectors.append(hint_vec)

                obs = next_obs
                if done:
                    episodes_done += 1
                    obs, info = self.env.reset()

            elif self._is_dict:
                # Dict 混合动作空间: agent 返回 {"target": ..., "ratio": ...}
                action_dict, _ = self.agent.select_action(obs, deterministic=False)

                next_obs, reward, terminated, truncated, step_info = self.env.step(action_dict)
                done = is_done(terminated, truncated)

                states.append(obs)
                actions.append(action_dict)
                rewards.append(reward)
                next_states.append(next_obs)
                dones.append(done)
                if step_info.get("eq_actions") is not None:
                    eq_a = np.asarray(step_info.get("eq_actions"), dtype=np.float32)
                    if eq_a.ndim >= 1:
                        eq_actions.append(eq_a[0] if eq_a.ndim > 1 else eq_a)
                if step_info.get("shapley_allocation") is not None:
                    shp = np.asarray(step_info.get("shapley_allocation"), dtype=np.float32).reshape(-1)
                    if shp.size > 0:
                        shapley_values.append(np.asarray([shp[0]], dtype=np.float32))
                r_terms = _reward_terms_to_array(step_info.get("reward_terms"))
                if r_terms is not None:
                    reward_terms.append(r_terms[0] if r_terms.ndim == 2 else r_terms)
                hint_vec = _game_hint_vector(step_info)
                if hint_vec is not None:
                    game_hint_vectors.append(hint_vec)

                obs = next_obs
                if done:
                    episodes_done += 1
                    obs, _ = self.env.reset()

            else:
                # 单智能体
                agent_obs = obs[0] if isinstance(obs, list) and len(obs) == 1 else obs
                # 传入原始 numpy obs，由 agent 内部处理 unsqueeze
                with torch.no_grad():
                    action, _ = self.agent.select_action(agent_obs, deterministic=False)

                    if isinstance(action, torch.Tensor):
                        action_np = action.cpu().numpy()
                        # 去掉 batch 维度: (1, n) -> (n,), (1,) -> scalar
                        if action_np.ndim > 1:
                            action_np = action_np[0]
                    elif isinstance(action, np.ndarray) and action.ndim > 1:
                        action_np = action[0]
                    else:
                        action_np = action

                    # Action NaN check
                    if isinstance(action_np, np.ndarray) and not np.isfinite(action_np).all():
                        action_np = np.nan_to_num(action_np, nan=0.0, posinf=1.0, neginf=-1.0)
                        action_np = np.clip(action_np, -1.0, 1.0)

                # 离散动作
                if self._is_discrete:
                    if isinstance(action_np, np.ndarray) and action_np.ndim >= 1:
                        action_to_env = int(np.argmax(action_np, axis=-1).flatten()[0])
                    elif hasattr(action_np, "item"):
                        action_to_env = int(action_np.item())
                    else:
                        action_to_env = int(action_np)
                else:
                    # 连续动作: 确保是 1D numpy array
                    if isinstance(action_np, np.ndarray):
                        action_to_env = action_np.flatten()
                    else:
                        action_to_env = np.atleast_1d(np.asarray(action_np)).flatten()

                next_obs, reward, terminated, truncated, step_info = self.env.step(action_to_env)
                done = is_done(terminated, truncated)
                next_obs_agent = next_obs[0] if isinstance(next_obs, list) and len(next_obs) == 1 else next_obs
                if isinstance(reward, (list, np.ndarray)):
                    reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
                    reward = float(reward_arr[0]) if reward_arr.size == 1 else float(reward_arr.mean())

                states.append(agent_obs)
                actions.append(action_to_env if self._is_discrete else action_np)
                rewards.append(reward)
                next_states.append(next_obs_agent)
                dones.append(done)
                if step_info.get("eq_actions") is not None:
                    eq_a = np.asarray(step_info.get("eq_actions"), dtype=np.float32)
                    if eq_a.ndim >= 1:
                        eq_actions.append(eq_a[0] if eq_a.ndim > 1 else eq_a)
                if step_info.get("shapley_allocation") is not None:
                    shp = np.asarray(step_info.get("shapley_allocation"), dtype=np.float32).reshape(-1)
                    if shp.size > 0:
                        shapley_values.append(np.asarray([shp[0]], dtype=np.float32))
                r_terms = _reward_terms_to_array(step_info.get("reward_terms"))
                if r_terms is not None:
                    reward_terms.append(r_terms[0] if r_terms.ndim == 2 else r_terms)
                hint_vec = _game_hint_vector(step_info)
                if hint_vec is not None:
                    game_hint_vectors.append(hint_vec)

                obs = next_obs

                if done:
                    episodes_done += 1
                    obs, _ = self.env.reset()

            self.step_count += 1

            # 构建单步 batch_data 并调用 update
            # off-policy 算法内部会处理存储和采样
            if self.step_count >= self.warmup_steps and self.step_count % self.update_interval == 0:
                if self._is_multi_agent and self._ma_mode == "joint":
                    batch_data = {
                        "states": np.asarray([states[-1]], dtype=np.float32),
                        "actions": np.asarray(
                            [actions[-1]],
                            dtype=np.int64 if self._is_discrete else np.float32,
                        ),
                        "rewards": np.asarray([rewards[-1]], dtype=np.float32),
                        "next_states": np.asarray([next_states[-1]], dtype=np.float32),
                        "dones": np.asarray([dones[-1]], dtype=np.float32),
                    }
                    if global_states:
                        batch_data["global_states"] = np.asarray([global_states[-1]], dtype=np.float32)
                    if eq_actions:
                        batch_data["eq_actions"] = np.asarray([eq_actions[-1]], dtype=np.float32)
                    if shapley_values:
                        batch_data["shapley_values"] = np.asarray([shapley_values[-1]], dtype=np.float32)
                    if reward_terms:
                        batch_data["reward_terms"] = np.asarray([reward_terms[-1]], dtype=np.float32)
                    if game_hint_vectors:
                        batch_data["game_hints"] = np.asarray([game_hint_vectors[-1]], dtype=np.float32)
                elif self._is_multi_agent and self._ma_mode == "shared":
                    # shared: 最后 n_agents 步展开为单智能体样本
                    n_agents = getattr(self.env, "num_agents", 1)
                    batch_data = {
                        "states": np.asarray(states[-n_agents:], dtype=np.float32),
                        "actions": np.asarray(
                            actions[-n_agents:],
                            dtype=np.int64 if self._is_discrete else np.float32,
                        ),
                        "rewards": np.asarray(rewards[-n_agents:], dtype=np.float32),
                        "next_states": np.asarray(next_states[-n_agents:], dtype=np.float32),
                        "dones": np.asarray(dones[-n_agents:], dtype=np.float32),
                    }
                    if len(eq_actions) >= n_agents:
                        batch_data["eq_actions"] = np.asarray(eq_actions[-n_agents:], dtype=np.float32)
                    if len(shapley_values) >= n_agents:
                        batch_data["shapley_values"] = np.asarray(shapley_values[-n_agents:], dtype=np.float32)
                    if len(reward_terms) >= n_agents:
                        batch_data["reward_terms"] = np.asarray(reward_terms[-n_agents:], dtype=np.float32)
                    if len(game_hint_vectors) >= n_agents:
                        batch_data["game_hints"] = np.asarray(game_hint_vectors[-n_agents:], dtype=np.float32)
                else:
                    batch_data = {
                        "states": np.asarray([states[-1]], dtype=np.float32),
                        "actions": np.asarray([actions[-1]]),
                        "rewards": np.asarray([rewards[-1]], dtype=np.float32),
                        "next_states": np.asarray([next_states[-1]], dtype=np.float32),
                        "dones": np.asarray([dones[-1]], dtype=np.float32),
                    }

                    if self._is_discrete:
                        batch_data["actions"] = np.asarray([actions[-1]], dtype=np.int64)
                    if eq_actions:
                        batch_data["eq_actions"] = np.asarray([eq_actions[-1]], dtype=np.float32)
                    if shapley_values:
                        batch_data["shapley_values"] = np.asarray([shapley_values[-1]], dtype=np.float32)
                    if reward_terms:
                        batch_data["reward_terms"] = np.asarray([reward_terms[-1]], dtype=np.float32)
                    if game_hint_vectors:
                        batch_data["game_hints"] = np.asarray([game_hint_vectors[-1]], dtype=np.float32)

                self.agent.update(batch_data)

        def to_array(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    return np.asarray(x)
                except ValueError:
                    return x
            return x

        return {
            "states": to_array(states),
            "actions": to_array(actions),
            "rewards": to_array(rewards),
            "next_states": to_array(next_states),
            "dones": to_array(dones),
            "global_states": to_array(global_states) if global_states else None,
            "eq_actions": to_array(eq_actions) if eq_actions else None,
            "shapley_values": to_array(shapley_values) if shapley_values else None,
            "reward_terms": to_array(reward_terms) if reward_terms else None,
            "game_hints": to_array(game_hint_vectors) if game_hint_vectors else None,
            "episodes_done": episodes_done,
        }

    def _update_step(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Off-policy: 数据已经在 _collect_rollout 中通过 agent.update() 更新了
        这里返回 agent 的最新统计信息
        """
        # Off-policy 算法在 _collect_rollout 中已经更新了
        # 返回一个占位信息
        return {
            "update_count": self.agent.update_count if hasattr(self.agent, "update_count") else 0,
        }
