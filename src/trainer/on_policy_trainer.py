"""
On-Policy Trainer — 用于 GRPO, PPO, A3C, TRPO, SimPO, MAPPO

特点: 每次更新使用新收集的rollout数据，不重复利用
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from gymnasium import spaces


from .base_trainer import BaseTrainer
from src.utils.helpers import is_done


class OnPolicyTrainer(BaseTrainer):
    """
    On-Policy训练器

    适用于: GRPO, PPO, A3C, TRPO, SimPO, MAPPO
    """

    def __init__(
        self,
        env,
        agent,
        num_epochs: int = 10,
        **kwargs,
    ):
        super().__init__(env, agent, **kwargs)
        self.num_epochs = num_epochs

        # 检测动作空间类型
        self._is_discrete = isinstance(env.action_space, spaces.Discrete)
        self._is_dict = isinstance(env.action_space, spaces.Dict)
        self._is_multi_agent = hasattr(env, "num_agents") and env.num_agents > 1
        self._ma_mode = getattr(agent, "multi_agent_mode", "shared")

        # 观测维度
        if self._is_multi_agent:
            self.obs_dim = env.observation_space.shape[0]
        else:
            self.obs_dim = env.observation_space.shape[0]

    def _collect_rollout(self) -> Dict[str, Any]:
        """
        收集一轮rollout数据

        Returns:
            rollout_data: {
                'states': list of observations,
                'actions': list of actions,
                'rewards': list of rewards,
                'next_states': list of next observations,
                'dones': list of done flags,
                'log_probs': list of log probabilities,
                'values': list of value estimates (optional),
                'episodes_done': number of episodes completed
            }
        """
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        values = []
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
                        arr.append([float(t.get("r_imm", 0.0)), float(t.get("r_coop", 0.0)), float(t.get("r_eq", 0.0))])
                    else:
                        arr.append([0.0, 0.0, 0.0])
                return np.asarray(arr, dtype=np.float32)
            if isinstance(terms, dict):
                return np.asarray([[float(terms.get("r_imm", 0.0)), float(terms.get("r_coop", 0.0)), float(terms.get("r_eq", 0.0))]], dtype=np.float32)
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

        for _ in range(self.rollout_steps):
            # 动作选择
            if self._is_multi_agent:
                # 多智能体环境
                agent_actions = []
                agent_log_probs = []
                for i, agent_obs in enumerate(obs):
                    with torch.no_grad():
                        action, info = self.agent.select_action(agent_obs)
                        # 获取log_prob
                        log_prob = info.get("log_prob", 0.0)
                        agent_log_probs.append(log_prob)
                        # 统一动作格式 (与 evaluate 保持一致)
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

                # 执行联合动作
                next_obs, step_rewards, terminated, truncated, info = self.env.step(agent_actions)

                n_agents = getattr(self.env, "num_agents", 1)
                step_done = is_done(terminated, truncated)

                if self._ma_mode == "joint":
                    # joint 模式: 保留 [n_agents, ...] 维度
                    states.append(np.asarray(obs, dtype=np.float32))
                    actions.append(
                        np.asarray(
                            agent_actions,
                            dtype=np.int64 if self._is_discrete else np.float32,
                        )
                    )
                    rewards.append(np.asarray(step_rewards, dtype=np.float32))
                    next_states.append(np.asarray(next_obs, dtype=np.float32))
                    dones.append(np.full(n_agents, step_done, dtype=np.float32))
                    log_probs.append(np.asarray(agent_log_probs, dtype=np.float32))
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
                        rewards.append(step_rewards[i])
                        next_states.append(next_obs[i])
                        dones.append(step_done)
                        log_probs.append(agent_log_probs[i])
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
                if step_done:
                    episodes_done += 1
                    obs, info = self.env.reset()

            elif self._is_dict:
                # Dict 混合动作空间: agent 返回 {"target": ..., "ratio": ...}
                action_dict, info = self.agent.select_action(obs)

                # 获取 log_prob
                log_prob = info.get("log_prob", 0.0)
                value = info.get("value", 0.0) if "value" in info else 0.0

                # 直接传递 Dict 给环境
                next_obs, reward, terminated, truncated, step_info = self.env.step(action_dict)
                done = is_done(terminated, truncated)

                # A3C 等算法需要 record_transition
                if hasattr(self.agent, "record_transition"):
                    self.agent.record_transition(reward, done)

                # 存储 Dict action（保持原始格式）
                states.append(obs)
                actions.append(action_dict)
                rewards.append(reward)
                next_states.append(next_obs)
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value)
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
                # 单智能体环境
                agent_obs = obs[0] if isinstance(obs, list) and len(obs) == 1 else obs
                # 传入原始 numpy obs，由 agent 内部处理 unsqueeze
                with torch.no_grad():
                    action, info = self.agent.select_action(agent_obs)

                    # 获取log_prob
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

                    log_prob = info.get("log_prob", 0.0)

                    # 获取value（如果agent有）
                    value = info.get("value", 0.0) if "value" in info else 0.0

                # 离散动作: 转换为标量索引
                if self._is_discrete:
                    # action_np 可能是 logits shape (n_actions,)
                    if isinstance(action_np, np.ndarray) and action_np.ndim >= 1:
                        action_scalar = int(np.argmax(action_np, axis=-1).flatten()[0])
                    elif hasattr(action_np, "item"):
                        action_scalar = int(action_np.item())
                    else:
                        action_scalar = int(action_np)
                else:
                    # 连续动作: 确保是 1D numpy array
                    if isinstance(action_np, np.ndarray):
                        action_scalar = action_np.flatten()
                    else:
                        action_scalar = np.atleast_1d(np.asarray(action_np)).flatten()

                # 执行动作
                next_obs, reward, terminated, truncated, step_info = self.env.step(action_scalar)
                done = is_done(terminated, truncated)
                next_obs_agent = next_obs[0] if isinstance(next_obs, list) and len(next_obs) == 1 else next_obs
                if isinstance(reward, (list, np.ndarray)):
                    reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
                    reward = float(reward_arr[0]) if reward_arr.size == 1 else float(reward_arr.mean())

                # A3C 等算法需要 record_transition 来记录 reward/done
                if hasattr(self.agent, "record_transition"):
                    self.agent.record_transition(reward, done)

                # 存储: 离散存储标量，连续存储原始动作
                states.append(agent_obs)
                actions.append(action_scalar if self._is_discrete else action_np)
                rewards.append(reward)
                next_states.append(next_obs_agent)
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value)
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

        # 格式化数据为 numpy array
        def to_array(x):
            if not isinstance(x, list) or len(x) == 0:
                return x

            # 安全地将每个元素转为标量
            def safe_scalar(v):
                if hasattr(v, "flatten"):
                    v = v.flatten()
                    return float(v[0]) if len(v) == 1 else v
                return float(v)

            try:
                return np.array(x)
            except ValueError:
                return np.array([safe_scalar(v) for v in x])

        rollout_data = {
            "states": to_array(states),
            "actions": to_array(actions),
            "rewards": to_array(rewards),
            "next_states": to_array(next_states),
            "dones": to_array(dones),
            "log_probs": np.array(log_probs) if log_probs else None,
            "values": np.array(values) if values else None,
            "episodes_done": episodes_done,
        }

        if global_states:
            rollout_data["global_states"] = to_array(global_states)
        if eq_actions:
            rollout_data["eq_actions"] = to_array(eq_actions)
        if shapley_values:
            rollout_data["shapley_values"] = to_array(shapley_values)
        if reward_terms:
            rollout_data["reward_terms"] = to_array(reward_terms)
        if game_hint_vectors:
            rollout_data["game_hints"] = to_array(game_hint_vectors)

        return rollout_data

    def _update_step(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一次更新（多epoch）

        对于 on-policy 算法，数据只使用一次就丢弃
        """
        # 构建 batch_data 格式
        batch_data = {
            "states": rollout_data["states"],
            "actions": rollout_data["actions"],
            "rewards": rollout_data["rewards"],
            "next_states": rollout_data["next_states"],
            "dones": rollout_data["dones"],
        }

        if rollout_data.get("log_probs") is not None:
            batch_data["log_probs"] = rollout_data["log_probs"]

        if rollout_data.get("values") is not None:
            batch_data["values"] = rollout_data["values"]

        if rollout_data.get("global_states") is not None:
            batch_data["global_states"] = rollout_data["global_states"]
        if rollout_data.get("eq_actions") is not None:
            batch_data["eq_actions"] = rollout_data["eq_actions"]
        if rollout_data.get("shapley_values") is not None:
            batch_data["shapley_values"] = rollout_data["shapley_values"]
        if rollout_data.get("reward_terms") is not None:
            batch_data["reward_terms"] = rollout_data["reward_terms"]
        if rollout_data.get("game_hints") is not None:
            batch_data["game_hints"] = rollout_data["game_hints"]

        # 调用 agent.update()
        update_info = self.agent.update(batch_data)

        # 确保返回的是字典
        if not isinstance(update_info, dict):
            update_info = {"loss": float(update_info)}

        return update_info
