"""
Base Trainer — 统一训练接口

所有训练器继承此类，确保接口一致性：
- select_action()
- update()
- evaluate()
- save()/load()
"""

import os
import sys
import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from gymnasium import spaces

from src.utils.helpers import is_done

# TensorBoard (可选依赖)
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class BaseTrainer(ABC):
    """
    训练器基类

    子类必须实现:
    - _collect_rollout(): 收集一轮数据
    - _update_step(): 执行一次更新
    """

    def __init__(
        self,
        env,
        agent,  # rl_algorithms 中的 agent
        total_timesteps: int = 500000,
        rollout_steps: int = 2048,
        eval_interval: int = 10000,
        eval_episodes: int = 10,
        min_eval_points: int = 0,
        eval_at_start: bool = False,
        save_interval: int = 50000,
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        self.env = env
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.rollout_steps = rollout_steps
        self.eval_interval = int(eval_interval)
        self.eval_episodes = int(eval_episodes)
        self.min_eval_points = max(0, int(min_eval_points or 0))
        self.eval_at_start = bool(eval_at_start)
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.log_interval = log_interval

        # 设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 随机种子
        if seed is not None:
            self._set_seed(seed)

        # 统计
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0

        # 日志
        self.train_logs: Dict[str, List[float]] = {}
        self.eval_logs: Dict[str, List[float]] = {}
        self.eval_steps: List[int] = []

        # TensorBoard
        self._tb_writer = None
        self._use_tensorboard = HAS_TENSORBOARD

        os.makedirs(save_dir, exist_ok=True)

        self.game_aware_enabled = bool(getattr(agent, "game_aware_enabled", False))
        self.game_aware_logs: Dict[str, List[float]] = {
            "dual_variable_mean": [],
            "constraint_residual_mean": [],
        }
        self.primal_dual_updater = None
        if self.game_aware_enabled:
            from src.rl_algorithms.game_aware.primal_dual import PrimalDualUpdater

            self.primal_dual_updater = PrimalDualUpdater()

    def _set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @abstractmethod
    def _collect_rollout(self) -> Dict[str, Any]:
        """收集一轮rollout数据"""
        pass

    @abstractmethod
    def _update_step(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """执行一次更新"""
        pass

    def _build_game_aware_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Attach game-aware metadata without changing the legacy batch."""
        if not self.game_aware_enabled:
            return batch
        enriched = dict(batch)
        enriched["game_aware"] = {
            "enabled": True,
            "reward_components": batch.get("reward_components", {}),
            "constraint_residuals": batch.get("constraint_residuals", {}),
        }
        return enriched

    def _apply_primal_dual_update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Update primal-dual state and record metrics when game-aware mode is enabled."""
        if not self.game_aware_enabled:
            return metrics
        from src.rl_algorithms.game_aware.primal_dual import ConstraintResiduals

        residual_payload = metrics.get("constraint_residuals")
        if isinstance(residual_payload, dict):
            residuals = ConstraintResiduals(
                latency_deadline=float(residual_payload.get("latency_deadline", 0.0)),
                energy_budget=float(residual_payload.get("energy_budget", 0.0)),
                queue_stability=float(residual_payload.get("queue_stability", 0.0)),
                migration_rate=float(residual_payload.get("migration_rate", 0.0)),
                budget_feasibility=float(residual_payload.get("budget_feasibility", 0.0)),
            )
        else:
            residuals = ConstraintResiduals(
                latency_deadline=float(metrics.get("latency_deadline_residual", metrics.get("violation_latency_mean", 0.0))),
                energy_budget=float(metrics.get("energy_budget_residual", metrics.get("violation_energy_mean", 0.0))),
                queue_stability=float(metrics.get("queue_stability_residual", metrics.get("constraint_residual", 0.0))),
                migration_rate=float(metrics.get("migration_rate_residual", 0.0)),
                budget_feasibility=float(metrics.get("budget_feasibility_residual", 0.0)),
            )
        if self.primal_dual_updater is not None:
            dual_state = self.primal_dual_updater.update_dual_variables(residuals)
        else:
            dual_state = None
        residual_values = [
            float(value)
            for value in residuals.as_dict().values()
        ]
        residual_mean = float(np.mean(residual_values)) if residual_values else 0.0
        dual_values = list(dual_state.dual_variables.values()) if dual_state is not None else []
        dual_mean = float(np.mean(dual_values)) if dual_values else 0.0
        self.game_aware_logs["constraint_residual_mean"].append(residual_mean)
        self.game_aware_logs["dual_variable_mean"].append(dual_mean)
        updated = dict(metrics)
        updated["game_aware/constraint_residual_mean"] = residual_mean
        updated["game_aware/dual_variable_mean"] = dual_mean
        for key, value in (dual_state.dual_variables if dual_state is not None else {}).items():
            updated[f"game_aware/dual/{key}"] = float(value)
        return updated

    def _log_reward_breakdown(self, reward_breakdown: Dict[str, float]) -> None:
        """Log reward component means for explainable reward reporting."""
        for key, value in reward_breakdown.items():
            if isinstance(value, (int, float)):
                log_key = f"reward_breakdown/{key}"
                self.train_logs.setdefault(log_key, []).append(float(value))

    def _effective_eval_interval(self) -> int:
        """Return the timestep interval used for checkpoint evaluation sampling."""
        interval = max(1, int(self.eval_interval))
        if self.min_eval_points > 0 and self.total_timesteps > 0:
            interval = min(interval, max(1, int(np.ceil(self.total_timesteps / self.min_eval_points))))
        return interval

    def _record_eval_logs(self, eval_info: Dict[str, float], step: Optional[int] = None) -> None:
        """Append one evaluation snapshot with an explicit training step."""
        eval_step = int(self.total_steps if step is None else step)
        self.eval_steps.append(eval_step)
        for k, v in eval_info.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                if k not in self.eval_logs:
                    self.eval_logs[k] = []
                scalar = float(v)
                self.eval_logs[k].append(scalar)
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(f"eval/{k}", scalar, eval_step)

    def train(self) -> Dict[str, List[float]]:
        """
        主训练循环

        Returns:
            train_logs: 训练日志
        """
        pbar = tqdm(total=self.total_timesteps, desc=f"Training {type(self.agent).__name__}")

        # 初始化 TensorBoard
        if self._use_tensorboard:
            tb_dir = os.path.join(self.save_dir, "tensorboard")
            self._tb_writer = SummaryWriter(log_dir=tb_dir)

        effective_eval_interval = self._effective_eval_interval()
        # Floor keeps the requested checkpoint cadence as a minimum density when
        # rollout_steps is coarser than eval_interval.
        eval_every_updates = max(1, int(effective_eval_interval // max(1, self.rollout_steps)))
        save_every_updates = max(1, int(np.ceil(self.save_interval / max(1, self.rollout_steps))))

        if self.eval_at_start:
            self._record_eval_logs(self.evaluate(), step=0)

        try:
            while self.total_steps < self.total_timesteps:
                # 收集rollout
                rollout_data = self._collect_rollout()

                # 更新
                update_info = self._update_step(rollout_data)
                update_info = self._apply_primal_dual_update(update_info)
                if isinstance(rollout_data.get("reward_breakdown"), dict):
                    self._log_reward_breakdown(rollout_data["reward_breakdown"])
                self.update_count += 1

                # 记录
                self.total_steps += self.rollout_steps
                self.episode_count += rollout_data.get("episodes_done", 0)
                pbar.update(self.rollout_steps)

                # 日志
                if self.update_count % self.log_interval == 0:
                    for k, v in update_info.items():
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            if k not in self.train_logs:
                                self.train_logs[k] = []
                            scalar = float(v)
                            self.train_logs[k].append(scalar)
                            # TensorBoard
                            if self._tb_writer is not None:
                                self._tb_writer.add_scalar(f"train/{k}", scalar, self.total_steps)

                    pbar.set_postfix(
                        {
                            k: f"{float(v):.3f}"
                            for k, v in update_info.items()
                            if isinstance(v, (int, float, np.integer, np.floating))
                        }
                    )
                # 评估
                if self.update_count % eval_every_updates == 0:
                    self._record_eval_logs(self.evaluate(), step=self.total_steps)
                # 保存
                if self.update_count % save_every_updates == 0:
                    self.save(os.path.join(self.save_dir, f"ckpt_{self.total_steps}.pt"))

        except KeyboardInterrupt:
            print(f"\nTraining interrupted at step {self.total_steps}. Saving checkpoint...")
            self.save(os.path.join(self.save_dir, f"ckpt_interrupt_{self.total_steps}.pt"))

        pbar.close()
        if self.total_steps >= self.total_timesteps:
            self.save(os.path.join(self.save_dir, "final.pt"))
        self._save_logs()

        # 关闭 TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.close()

        return self.train_logs

    def evaluate(self) -> Dict[str, float]:
        """
        评估当前策略。

        除原有 reward/latency/energy/convergence 指标外，评估阶段会从
        GameTheory/Mainline-A info 字段中抽取约束、队列、通信、公平性、价格、
        reward breakdown 等扩展指标。扩展只读取 info，不修改环境。
        """
        eval_rewards = []
        eval_latency_totals = []
        eval_energy_totals = []
        eval_latency_per_step = []
        eval_energy_per_step = []
        eval_latency_per_task = []
        eval_energy_per_task = []
        eval_episode_steps = []
        eval_episode_tasks = []
        eval_agent_reward_fairness = []
        e2e_latency_samples = []
        e2e_energy_samples = []
        social_welfare_per_step = []
        deadline_misses = 0
        total_eval_tasks = 0
        total_eval_steps = 0
        extended_samples: Dict[str, List[float]] = {}
        sample_count_before = getattr(self.agent, "sample_count", None)

        def _append_metric(key: str, value: Any) -> None:
            if value is None:
                return
            try:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return
            if arr.size == 0:
                return
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return
            extended_samples.setdefault(key, []).extend([float(x) for x in finite])

        def _jain_index(values: Any) -> float:
            try:
                arr = np.asarray(values, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return 0.0
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0
            min_v = float(np.min(arr))
            if min_v < 0.0:
                arr = arr - min_v
            denom = float(arr.size * np.sum(arr ** 2))
            if denom <= 1e-12:
                return 1.0
            return float((np.sum(arr) ** 2) / denom)

        def _numeric_values_from_mapping(payload: Any) -> np.ndarray:
            values = []
            if isinstance(payload, dict):
                iterable = payload.values()
            elif isinstance(payload, (list, tuple, np.ndarray)):
                iterable = payload
            else:
                iterable = [payload]
            for value in iterable:
                try:
                    arr = np.asarray(value, dtype=np.float64).reshape(-1)
                    values.extend(arr[np.isfinite(arr)].tolist())
                except (TypeError, ValueError):
                    continue
            return np.asarray(values, dtype=np.float64)

        def _collect_extended_info(info: Dict[str, Any], reward_vector: np.ndarray) -> None:
            reward_vector = np.asarray(reward_vector, dtype=np.float64).reshape(-1)
            reward_vector = reward_vector[np.isfinite(reward_vector)]
            if reward_vector.size:
                social_welfare_per_step.append(float(np.sum(reward_vector)))

            comm = info.get("communication_metrics") or {}
            if isinstance(comm, dict):
                for src, dst in {
                    "queue_wait_mean": "queue_wait_mean",
                    "queue_wait_max": "queue_wait_max",
                    "deadline_miss_rate": "step_deadline_miss_rate",
                    "offload_ratio_mean": "offload_ratio_mean",
                    "non_nearest_server_rate": "non_nearest_server_rate",
                    "e2e_latency_max": "e2e_latency_max",
                }.items():
                    _append_metric(dst, comm.get(src))

            constraints = info.get("constraint_metrics") or {}
            if isinstance(constraints, dict):
                for key in (
                    "penalty_mean",
                    "barrier_mean",
                    "violation_energy_mean",
                    "violation_power_mean",
                    "violation_latency_mean",
                ):
                    _append_metric(f"constraint/{key}", constraints.get(key))
                violations = [
                    constraints.get("violation_energy_mean", 0.0),
                    constraints.get("violation_power_mean", 0.0),
                    constraints.get("violation_latency_mean", 0.0),
                ]
                try:
                    _append_metric("constraint/any_violation", float(np.any(np.asarray(violations, dtype=np.float64) > 0.0)))
                except (TypeError, ValueError):
                    pass

            fairness = info.get("fairness_metrics") or {}
            if isinstance(fairness, dict):
                _append_metric("fairness/efx_satisfied", float(bool(fairness.get("efx_satisfied", False))))
                _append_metric("fairness/efx_violation", fairness.get("efx_violation"))
                _append_metric("fairness/efx_transfer_sum_abs", fairness.get("efx_transfer_sum_abs"))
                cpnet_scores = fairness.get("cpnet_scores")
                if cpnet_scores is not None:
                    _append_metric("fairness/cpnet_score_mean", _numeric_values_from_mapping(cpnet_scores))

            reward_terms = info.get("reward_terms") or []
            if isinstance(reward_terms, dict):
                reward_terms = [reward_terms]
            if isinstance(reward_terms, (list, tuple)):
                for term in reward_terms:
                    if not isinstance(term, dict):
                        continue
                    for key, value in term.items():
                        _append_metric(f"reward_component/{key}", value)

            mainline_components = info.get("mainline_a_reward_components") or {}
            if isinstance(mainline_components, dict):
                for key, value in mainline_components.items():
                    _append_metric(f"mainline_a_reward/{key}", value)

            price_metadata = info.get("dynamic_price_metadata") or {}
            if isinstance(price_metadata, dict):
                prices = _numeric_values_from_mapping(price_metadata.get("prices"))
                responses = _numeric_values_from_mapping(price_metadata.get("responses"))
                if prices.size:
                    _append_metric("pricing/price", prices)
                    _append_metric("pricing/provider_revenue_proxy", float(np.sum(prices)))
                if responses.size:
                    _append_metric("pricing/response", responses)

            queue_metrics = info.get("queue_metrics") or {}
            if isinstance(queue_metrics, dict):
                for key, value in queue_metrics.items():
                    _append_metric(f"queue/{key}", value)
            _append_metric("queue/length", info.get("queue_lengths"))

            channel_qualities = info.get("channel_qualities")
            if channel_qualities is not None:
                channel_vals = _numeric_values_from_mapping(channel_qualities)
                if channel_vals.size:
                    _append_metric("channel/quality_mean", float(np.mean(channel_vals)))
                    _append_metric("channel/quality_std", float(np.std(channel_vals)))

            components = info.get("latency_components") or []
            if isinstance(components, (list, tuple)) and components:
                targets = []
                for component in components:
                    if isinstance(component, dict) and component.get("target") is not None:
                        try:
                            targets.append(int(component.get("target")))
                        except (TypeError, ValueError):
                            continue
                if targets:
                    target_arr = np.asarray(targets, dtype=np.float64)
                    _append_metric("action/local_target_rate", float(np.mean(target_arr == 0)))
                    _append_metric("action/edge_target_rate", float(np.mean(target_arr > 0)))
                    _append_metric("action/target_mean", float(np.mean(target_arr)))

            offload = info.get("individual_offload")
            if offload is not None:
                _append_metric("action/individual_offload_mean", _numeric_values_from_mapping(offload))

        def _extract_step_metrics(
            info: Dict[str, Any], prev_task_completed: int
        ) -> tuple[float, float, int, int, np.ndarray, np.ndarray, int]:
            """Extract latency/energy for one env step with robust task counting."""
            step_latency = 0.0
            step_energy = 0.0
            task_count = 0
            latency_samples = np.asarray([], dtype=np.float64)
            energy_samples = np.asarray([], dtype=np.float64)
            deadline_miss_count = 0

            if "latency_components" in info:
                components = info.get("latency_components") or []
                latency_samples = np.asarray(
                    [float(c.get("e2e_latency", 0.0)) for c in components],
                    dtype=np.float64,
                )
                deadline_miss_count = int(
                    sum(
                        float(c.get("e2e_latency", 0.0))
                        > float(c.get("deadline", np.inf))
                        for c in components
                    )
                )
                step_latency = float(np.sum(latency_samples))
                task_count = max(task_count, int(latency_samples.size))
            elif "individual_latencies" in info:
                latencies = np.asarray(info["individual_latencies"], dtype=np.float64).reshape(-1)
                latency_samples = latencies
                step_latency = float(np.sum(latency_samples))
                task_count = max(task_count, int(latency_samples.size))
            elif "latency" in info:
                step_latency = float(info["latency"])
                latency_samples = np.asarray([step_latency], dtype=np.float64)
                task_count = max(task_count, 1)

            if "individual_energies" in info:
                energies = np.asarray(info["individual_energies"], dtype=np.float64).reshape(-1)
                energy_samples = energies
                step_energy = float(np.sum(energy_samples))
                task_count = max(task_count, int(energy_samples.size))
            elif "energy" in info:
                step_energy = float(info["energy"])
                energy_samples = np.asarray([step_energy], dtype=np.float64)
                task_count = max(task_count, 1)

            current_task_completed = info.get("task_completed", prev_task_completed)
            if isinstance(current_task_completed, (int, float, np.integer, np.floating)):
                current_task_completed = int(current_task_completed)
                task_delta = max(0, current_task_completed - prev_task_completed)
                if task_delta > 0:
                    task_count = task_delta
                prev_task_completed = current_task_completed

            if latency_samples.size == 0 and task_count > 0:
                latency_samples = np.full(task_count, step_latency / max(task_count, 1), dtype=np.float64)
            if energy_samples.size == 0 and task_count > 0:
                energy_samples = np.full(task_count, step_energy / max(task_count, 1), dtype=np.float64)

            return (
                step_latency,
                step_energy,
                task_count,
                prev_task_completed,
                latency_samples,
                energy_samples,
                deadline_miss_count,
            )

        for _ in range(self.eval_episodes):
            obs, _ = self.env.reset(seed=None)
            done = False
            episode_reward = 0.0
            episode_latency = 0.0
            episode_energy = 0.0
            episode_steps = 0
            episode_tasks = 0
            prev_task_completed = 0
            episode_agent_rewards: Optional[np.ndarray] = None
            is_multi = hasattr(self.env, "num_agents")

            while not done:
                if is_multi and isinstance(obs, list):
                    # 多智能体: 逐个选择动作
                    actions = []
                    for agent_obs in obs:
                        if hasattr(self.agent, "select_action"):
                            a, _ = self.agent.select_action(agent_obs, deterministic=True)
                        else:
                            a = self.env.action_space.sample()
                        # 统一格式
                        if isinstance(a, torch.Tensor):
                            a = a.cpu().numpy()
                        if isinstance(a, np.ndarray):
                            if a.ndim > 1:
                                a = a[0]
                            if isinstance(self.env.action_space, spaces.Discrete):
                                # size==1 直接取元素; size>1 视为 logits/probs 用 argmax
                                if a.size == 1:
                                    a = int(a.item())
                                else:
                                    a = int(np.argmax(a))
                        else:
                            if isinstance(self.env.action_space, spaces.Discrete):
                                a = int(a)
                        actions.append(a)
                    # Action NaN check
                    for idx, a in enumerate(actions):
                        if isinstance(a, np.ndarray) and not np.isfinite(a).all():
                            actions[idx] = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
                    obs, reward, terminated, truncated, info = self.env.step(actions)
                    reward_vector = np.asarray(reward if isinstance(reward, (list, np.ndarray)) else [reward], dtype=np.float64).reshape(-1)
                    episode_reward += float(np.sum(reward_vector))
                else:
                    # 单智能体
                    if hasattr(self.agent, "select_action"):
                        action, _ = self.agent.select_action(obs, deterministic=True)
                    else:
                        action = self.env.action_space.sample()
                    # 统一处理 action 格式
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    if isinstance(action, np.ndarray) and action.ndim > 1:
                        action = action[0]  # 去掉 batch 维度
                    # 离散动作: 转换为标量int
                    if isinstance(self.env.action_space, spaces.Discrete):
                        if isinstance(action, np.ndarray) and action.ndim >= 1:
                            action = int(np.argmax(action, axis=-1).flatten()[0])
                        elif hasattr(action, "item"):
                            action = int(action.item())
                        elif not isinstance(action, (int, np.number)):
                            action = int(action)
                    else:
                        # 连续动作: 确保是 1D numpy array
                        if isinstance(action, np.ndarray):
                            action = action.flatten()
                        else:
                            action = np.atleast_1d(np.asarray(action, dtype=np.float32)).flatten()
                    if isinstance(action, np.ndarray) and not np.isfinite(action).all():
                        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
                        action = np.clip(action, -1.0, 1.0)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    reward_vector = np.asarray(reward if isinstance(reward, (list, np.ndarray)) else [reward], dtype=np.float64).reshape(-1)
                    episode_reward += float(np.sum(reward_vector))

                reward_vector = reward_vector[np.isfinite(reward_vector)]
                if reward_vector.size:
                    if episode_agent_rewards is None or episode_agent_rewards.size != reward_vector.size:
                        episode_agent_rewards = np.zeros_like(reward_vector, dtype=np.float64)
                    episode_agent_rewards += reward_vector
                _collect_extended_info(info, reward_vector)

                # 统一 done 检测
                done = is_done(terminated, truncated)

                (
                    step_latency,
                    step_energy,
                    step_tasks,
                    prev_task_completed,
                    step_latency_samples,
                    step_energy_samples,
                    step_deadline_misses,
                ) = _extract_step_metrics(
                    info, prev_task_completed
                )
                episode_latency += step_latency
                episode_energy += step_energy
                episode_steps += 1
                episode_tasks += step_tasks
                total_eval_steps += 1
                total_eval_tasks += step_tasks
                deadline_misses += step_deadline_misses
                if step_latency_samples.size:
                    e2e_latency_samples.extend(step_latency_samples.tolist())
                if step_energy_samples.size:
                    e2e_energy_samples.extend(step_energy_samples.tolist())

            eval_rewards.append(episode_reward)
            eval_latency_totals.append(episode_latency)
            eval_energy_totals.append(episode_energy)
            eval_episode_steps.append(episode_steps)
            eval_episode_tasks.append(episode_tasks)
            if episode_agent_rewards is not None:
                eval_agent_reward_fairness.append(_jain_index(episode_agent_rewards))

            safe_steps = max(1, episode_steps)
            safe_tasks = max(1, episode_tasks)
            eval_latency_per_step.append(episode_latency / safe_steps)
            eval_energy_per_step.append(episode_energy / safe_steps)
            eval_latency_per_task.append(episode_latency / safe_tasks)
            eval_energy_per_task.append(episode_energy / safe_tasks)

        latency_total_mean = float(np.mean(eval_latency_totals))
        energy_total_mean = float(np.mean(eval_energy_totals))
        e2e_arr = np.asarray(e2e_latency_samples, dtype=np.float64)
        e2e_energy_arr = np.asarray(e2e_energy_samples, dtype=np.float64)
        if e2e_arr.size:
            e2e_latency_mean = float(np.mean(e2e_arr))
            e2e_latency_p95 = float(np.percentile(e2e_arr, 95))
        else:
            e2e_latency_mean = float(np.mean(eval_latency_per_task)) if eval_latency_per_task else 0.0
            e2e_latency_p95 = e2e_latency_mean
        e2e_energy_mean = (
            float(np.mean(e2e_energy_arr))
            if e2e_energy_arr.size
            else float(np.mean(eval_energy_per_task)) if eval_energy_per_task else 0.0
        )
        deadline_miss_rate = float(deadline_misses / max(total_eval_tasks, 1))
        throughput_tasks_per_step = float(total_eval_tasks / max(total_eval_steps, 1))
        comm_score = float(
            100.0
            * throughput_tasks_per_step
            * max(0.0, 1.0 - deadline_miss_rate)
            / (1.0 + e2e_latency_p95 + 0.3 * e2e_energy_mean)
        )

        result = {
            "eval/reward_mean": float(np.mean(eval_rewards)),
            "eval/reward_std": float(np.std(eval_rewards)),
            # Communication-experience aliases: latency_mean is now per-task E2E latency.
            "eval/latency_mean": e2e_latency_mean,
            "eval/energy_mean": energy_total_mean,
            "eval/e2e_latency_mean": e2e_latency_mean,
            "eval/e2e_latency_p95": e2e_latency_p95,
            "eval/deadline_miss_rate": deadline_miss_rate,
            "eval/throughput_tasks_per_step": throughput_tasks_per_step,
            "eval/comm_score": comm_score,
            # Explicit latency/energy definitions.
            "eval/latency_total_mean": latency_total_mean,
            "eval/energy_total_mean": energy_total_mean,
            "eval/latency_per_step_mean": float(np.mean(eval_latency_per_step)),
            "eval/energy_per_step_mean": float(np.mean(eval_energy_per_step)),
            "eval/latency_per_task_mean": float(np.mean(eval_latency_per_task)),
            "eval/energy_per_task_mean": float(np.mean(eval_energy_per_task)),
            "eval/episode_steps_mean": float(np.mean(eval_episode_steps)),
            "eval/episode_tasks_mean": float(np.mean(eval_episode_tasks)),
            "eval/social_welfare_mean": float(np.mean(eval_rewards)),
            "eval/social_welfare_per_step_mean": float(np.mean(social_welfare_per_step)) if social_welfare_per_step else 0.0,
            "eval/agent_reward_jain_mean": float(np.mean(eval_agent_reward_fairness)) if eval_agent_reward_fairness else 1.0,
        }

        for key, values in extended_samples.items():
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            metric_key = key.replace(" ", "_")
            result[f"eval/{metric_key}_mean"] = float(np.mean(arr))
            if metric_key in {
                "queue_wait_mean",
                "queue_wait_max",
                "constraint/penalty_mean",
                "constraint/barrier_mean",
                "pricing/price",
                "pricing/provider_revenue_proxy",
                "action/target_mean",
            }:
                result[f"eval/{metric_key}_std"] = float(np.std(arr))
            if metric_key in {"queue_wait_mean", "queue_wait_max", "pricing/price"}:
                result[f"eval/{metric_key}_p95"] = float(np.percentile(arr, 95))

        if sample_count_before is not None:
            setattr(self.agent, "sample_count", sample_count_before)
        return result

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if hasattr(self.agent, "save"):
            self.agent.save(path)
        else:
            torch.save(self.agent.state_dict(), path)

    def load(self, path: str):
        """加载模型"""
        if hasattr(self.agent, "load"):
            self.agent.load(path)
        else:
            self.agent.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=False)
            )

    def _save_logs(self):
        """保存日志"""
        log_path = os.path.join(self.save_dir, "train_logs.json")
        all_logs = {**self.train_logs}
        all_logs["eval_steps"] = self.eval_steps
        for k, v in self.eval_logs.items():
            all_logs[f"eval_{k}"] = v
        with open(log_path, "w") as f:
            json.dump(all_logs, f, indent=2)
