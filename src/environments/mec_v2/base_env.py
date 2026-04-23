"""
MEC V2 基类 — 共享信道模型、能量模型、奖励函数骨架

论文场景细节待填充，此处提供可配置的基础设施：
- 信道模型: 路径损耗 + 阴影衰落 + 瑞利衰落
- 能量模型: 发射功率 × 时间
- 奖励函数: 延迟 + 能耗 + 截止期限惩罚
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


def _make_pathloss_model(
    channel_model_type: str,
    carrier_freq: float,
    h_bs: float = 25.0,
    h_ut: float = 1.5,
    scenario: str = "umi",
):
    """
    工厂函数: 创建路径损耗模型

    Args:
        channel_model_type: "simple" (简化) | "3gpp_los" | "3gpp_nlos"
        carrier_freq: 载波频率 (Hz)
        h_bs: 基站高度 (m)
        h_ut: 终端高度 (m)
        scenario: 3GPP 场景 "umi" | "uma" | "rma"

    Returns:
        路径损耗模型对象 (有 __call__(distance) -> float_dB 接口)
    """
    if channel_model_type in ("3gpp_los", "3gpp_nlos"):
        try:
            from src.comm.channel.pathloss import PathLoss3GPP_LOS, PathLoss3GPP_NLOS

            if channel_model_type == "3gpp_los":
                return PathLoss3GPP_LOS(
                    carrier_freq=carrier_freq, h_bs=h_bs, h_ut=h_ut, scenario=scenario
                )
            else:
                return PathLoss3GPP_NLOS(
                    carrier_freq=carrier_freq, h_bs=h_bs, h_ut=h_ut, scenario=scenario
                )
        except ImportError:
            import warnings

            warnings.warn("src.comm not available, falling back to simple pathloss model")
            return None
    return None


class MECChannelModel:
    """
    信道模型 — 可配置路径损耗和衰落
    """

    def __init__(
        self,
        carrier_freq: float = 2.4e9,  # Hz, 载波频率
        tx_power_local: float = 0.1,  # W, 本地发射功率
        tx_power_edge: float = 0.2,  # W, 边缘基站发射功率
        bandwidth: float = 20e6,  # Hz, 信道带宽
        noise_power: float = -100,  # dBm, 噪声功率
        path_loss_exponent: float = 3.0,  # 路径损耗指数 (simple 模型)
        shadow_std: float = 0.0,  # 阴影衰落标准差(dB)
        fading_type: str = "none",  # "none" | "rayleigh"
        fading_param: float = 1.0,  # 瑞利衰落参数
        channel_model_type: str = "simple",  # "simple" | "3gpp_los" | "3gpp_nlos"
        h_bs: float = 25.0,  # 基站高度 (m), 3GPP 模型
        h_ut: float = 1.5,  # 终端高度 (m), 3GPP 模型
        scenario: str = "umi",  # 3GPP 场景
        rng: Optional[np.random.Generator] = None,
    ):
        self.carrier_freq = carrier_freq
        self.tx_power_local = tx_power_local
        self.tx_power_edge = tx_power_edge
        self.bandwidth = bandwidth
        self.noise_power_dbm = noise_power
        self.noise_power_w = 10 ** ((noise_power - 30) / 10)  # dBm → W
        self.path_loss_exponent = path_loss_exponent
        self.shadow_std = shadow_std
        self.fading_type = fading_type
        self.fading_param = fading_param
        self._rng = rng if rng is not None else np.random.default_rng()

        # 预计算常数 (simple 模型)
        self.speed_of_light = 3e8
        self.reference_distance = 1.0  # m
        self.reference_loss = (
            20 * np.log10(self.carrier_freq * self.reference_distance / self.speed_of_light) + 8
        )

        # 3GPP 路径损耗模型 (可选)
        self._3gpp_model = _make_pathloss_model(
            channel_model_type, carrier_freq, h_bs=h_bs, h_ut=h_ut, scenario=scenario
        )
        self.channel_model_type = channel_model_type

    def compute_path_loss(self, distance: float) -> float:
        """计算路径损耗(dB) — 优先使用 3GPP 模型，回退到简化模型"""
        if self._3gpp_model is not None:
            return self._3gpp_model(distance)
        # 简化模型回退
        if distance < self.reference_distance:
            distance = self.reference_distance
        return self.reference_loss + 10 * self.path_loss_exponent * np.log10(
            distance / self.reference_distance
        )

    def compute_snr(self, distance: float, tx_power_w: float) -> float:
        """计算SNR(线性)"""
        path_loss_db = self.compute_path_loss(distance)
        shadow = self._rng.normal(0, self.shadow_std) if self.shadow_std > 0 else 0.0
        # 防止 log10 收到非正值产生 NaN
        tx_power_mw = max(tx_power_w * 1000, 1e-12)
        rx_power_dbm = 10 * np.log10(tx_power_mw) - path_loss_db + shadow
        rx_power_w = 10 ** ((rx_power_dbm - 30) / 10)

        snr_linear = rx_power_w / self.noise_power_w

        if self.fading_type == "rayleigh":
            channel_gain = self._rng.exponential(self.fading_param)
            snr_linear *= channel_gain

        return snr_linear

    def transmission_time(self, data_size_bits: float, distance: float, tx_power_w: float) -> float:
        """计算传输时间(秒)"""
        snr = self.compute_snr(distance, tx_power_w)
        capacity = self.bandwidth * np.log2(1 + snr)  # bps
        if capacity < 1e-9:
            return float("inf")
        return data_size_bits / capacity


class MECEnergyModel:
    """
    能量消耗模型
    """

    def __init__(
        self,
        local_cpu_cycles_per_joule: float = 1e9,  # CPU每焦耳cycles
        circuit_power: float = 0.01,  # W, 电路功率
    ):
        self.local_cpu_cycles_per_joule = local_cpu_cycles_per_joule
        self.circuit_power = circuit_power

    def local_energy(self, cpu_cycles: float) -> float:
        """本地计算能耗(J)"""
        return cpu_cycles / self.local_cpu_cycles_per_joule

    def transmission_energy(self, tx_power_w: float, time_s: float) -> float:
        """传输能耗(J)"""
        return tx_power_w * time_s


class MECRewardShaper:
    """
    奖励函数 — 可配置的延迟/能耗/截止期限惩罚
    """

    def __init__(
        self,
        latency_weight: float = 1.0,
        energy_weight: float = 0.1,
        deadline_weight: float = 2.0,
        reward_scale: float = 0.01,
    ):
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.deadline_weight = deadline_weight
        self.reward_scale = reward_scale

    def compute(
        self,
        latency: float,
        energy: float,
        deadline: float,
    ) -> float:
        """
        计算奖励

        Args:
            latency: 端到端延迟(秒)
            energy: 总能耗(焦耳)
            deadline: 任务截止期限(秒)

        Returns:
            reward: float (负值表示惩罚)
        """
        # 截止期限惩罚
        if latency > deadline:
            penalty = (latency - deadline) * self.deadline_weight
        else:
            penalty = 0.0

        total_cost = self.latency_weight * latency + self.energy_weight * energy + penalty

        # 缩放奖励到合理范围
        return -total_cost * self.reward_scale


class BaseMECEnv(gym.Env):
    """
    MEC环境基类

    子类需要实现:
    - _get_obs(): 返回观测
    - _process_action(): 处理动作
    - reset(): 重置环境
    - step(): 执行动作
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_edge_servers: int = 3,
        num_tasks: int = 5,
        max_steps: int = 100,
        task_arrival_prob: float = 0.5,
        # 信道参数
        carrier_freq: float = 2.4e9,
        tx_power_local: float = 0.1,
        tx_power_edge: float = 0.2,
        bandwidth: float = 20e6,
        noise_power: float = -100,
        path_loss_exponent: float = 3.0,
        fading_type: str = "none",
        channel_model_type: str = "simple",  # "simple" | "3gpp_los" | "3gpp_nlos"
        h_bs: float = 25.0,
        h_ut: float = 1.5,
        scenario: str = "umi",
        # 能量参数
        local_cpu_cycles_per_joule: float = 1e9,
        # 奖励参数
        latency_weight: float = 1.0,
        energy_weight: float = 0.1,
        deadline_weight: float = 2.0,
        # 距离参数 (服务器位置)
        server_distances: Optional[list] = None,  # 距离列表，默认[10, 20, 30]m
        # 任务参数
        task_data_size_range: Tuple[float, float] = (1e4, 1e6),  # bits
        task_cpu_cycles_range: Tuple[float, float] = (1e6, 1e9),  # cycles
        task_deadline_range: Tuple[float, float] = (0.5, 5.0),  # 秒
        # 观测参数
        normalize_obs: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_edge_servers = num_edge_servers
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.task_arrival_prob = task_arrival_prob
        self.normalize_obs = normalize_obs
        self.render_mode = render_mode

        # 服务器距离
        if server_distances is None:
            self.server_distances = np.array([10.0 + i * 10.0 for i in range(num_edge_servers)])
        else:
            self.server_distances = np.array(server_distances)

        # 任务特性
        self.task_data_range = task_data_size_range
        self.task_cpu_range = task_cpu_cycles_range
        self.task_deadline_range = task_deadline_range

        # 初始化模型
        self.channel = MECChannelModel(
            carrier_freq=carrier_freq,
            tx_power_local=tx_power_local,
            tx_power_edge=tx_power_edge,
            bandwidth=bandwidth,
            noise_power=noise_power,
            path_loss_exponent=path_loss_exponent,
            fading_type=fading_type,
            channel_model_type=channel_model_type,
            h_bs=h_bs,
            h_ut=h_ut,
            scenario=scenario,
        )

        self.energy_model = MECEnergyModel(
            local_cpu_cycles_per_joule=local_cpu_cycles_per_joule,
        )

        self.reward_shaper = MECRewardShaper(
            latency_weight=latency_weight,
            energy_weight=energy_weight,
            deadline_weight=deadline_weight,
        )

        # 观测归一化器
        self.obs_rms = None
        if normalize_obs:
            from src.utils.helpers import RunningMeanStd

            self.obs_rms = RunningMeanStd(shape=(self._get_obs_dim(),))

        # 状态变量
        self.current_step = 0
        self.current_task = None
        self.queue_lengths = None
        self.total_energy = 0.0
        self.total_latency = 0.0
        self.task_completed = 0

        # 动作空间和观测空间（子类覆盖）
        self.action_space = None
        self.observation_space = None

    def _get_obs_dim(self) -> int:
        """返回观测维度（子类实现）"""
        raise NotImplementedError

    def _generate_task(self) -> Dict[str, Any]:
        """生成新任务"""
        rng = self.np_random
        task_type = rng.integers(0, self.num_tasks)
        data_size = rng.uniform(*self.task_data_range)
        cpu_cycles = rng.uniform(*self.task_cpu_range)
        deadline = rng.uniform(*self.task_deadline_range)
        return {
            "type": task_type,
            "data_size": data_size,
            "cpu_cycles": cpu_cycles,
            "deadline": deadline,
            "arrival_step": self.current_step,
        }

    def _compute_latency(self, task: Dict, action: int) -> float:
        """计算任务延迟（子类可覆盖）"""
        raise NotImplementedError

    def _compute_energy(self, task: Dict, action: int, latency: float) -> float:
        """计算任务能耗"""
        if action == 0:
            # 本地处理
            return self.energy_model.local_energy(task["cpu_cycles"])
        else:
            # 边缘处理
            server_idx = action - 1
            tx_power = self.channel.tx_power_edge
            tx_time = self.channel.transmission_time(
                task["data_size"], self.server_distances[server_idx], tx_power
            )
            return self.energy_model.transmission_energy(tx_power, tx_time)

    def _get_obs(self) -> np.ndarray:
        """返回观测（子类实现）"""
        raise NotImplementedError

    def _process_action(self, action: int) -> Tuple[float, float]:
        """
        处理动作，计算延迟和能耗

        Returns:
            latency, energy
        """
        task = self.current_task
        if task is None:
            return 0.0, 0.0

        if action == 0:
            # 本地计算
            processing_time = task["cpu_cycles"] / self._get_local_cpu_freq()
            tx_time = 0.0
        else:
            # 边缘卸载
            server_idx = action - 1
            distance = self.server_distances[server_idx]
            tx_power = self.channel.tx_power_edge
            tx_time = self.channel.transmission_time(task["data_size"], distance, tx_power)
            # 边缘计算时间（简化）
            processing_time = task["cpu_cycles"] / self._get_edge_cpu_freq()

        latency = tx_time + processing_time
        energy = self._compute_energy(task, action, latency)

        return latency, energy

    def _get_local_cpu_freq(self) -> float:
        """本地CPU频率 (cycles/s)"""
        return 1e9  # 1 GHz

    def _get_edge_cpu_freq(self) -> float:
        """边缘服务器CPU频率 (cycles/s)"""
        return 5e9  # 5 GHz

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        self.channel._rng = self.np_random

        self.current_step = 0
        self.current_task = self._generate_task()
        self.queue_lengths = np.zeros(self.num_edge_servers)
        self.total_energy = 0.0
        self.total_latency = 0.0
        self.task_completed = 0

        obs = self._get_obs()

        if self.normalize_obs and self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(1, -1))
            obs = self.obs_rms.normalize(obs.reshape(1, -1)).squeeze()

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        self.current_step += 1

        latency, energy = self._process_action(action)

        self.total_latency += latency
        self.total_energy += energy
        self.task_completed += 1

        reward = self.reward_shaper.compute(latency, energy, self.current_task["deadline"])

        obs = self._get_obs()

        if self.normalize_obs and self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(1, -1))
            obs = self.obs_rms.normalize(obs.reshape(1, -1)).squeeze()

        terminated = False
        truncated = self.current_step >= self.max_steps

        # 新任务到达
        if self.np_random.random() < self.task_arrival_prob:
            self.current_task = self._generate_task()
        else:
            self.current_task = None

        info = {
            "latency": latency,
            "energy": energy,
            "deadline": self.current_task["deadline"] if self.current_task else 0.0,
            "queue_lengths": self.queue_lengths.copy(),
            "task_completed": self.task_completed,
            "avg_latency": self.total_latency / max(1, self.task_completed),
            "avg_energy": self.total_energy / max(1, self.task_completed),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染环境状态"""
        if self.render_mode == "ansi":
            return str(self._get_obs())
        elif self.render_mode == "human":
            print(f"Step {self.current_step}/{self.max_steps}")
            print(f"Queue lengths: {self.queue_lengths}")
            if self.current_task:
                print(
                    f"Task: data={self.current_task['data_size']:.1e}b, "
                    f"cpu={self.current_task['cpu_cycles']:.1e}, "
                    f"deadline={self.current_task['deadline']:.2f}s"
                )
            print("-" * 40)

    def close(self):
        """清理资源"""
        pass
