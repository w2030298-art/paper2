# Graph Report - paper2  (2026-04-28)

## Corpus Check
- 76 files · ~84,721 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1218 nodes · 2620 edges · 34 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 1007 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]

## God Nodes (most connected - your core abstractions)
1. `BaseAgent` - 107 edges
2. `ActorNetwork` - 59 edges
3. `BaseTrainer` - 51 edges
4. `BaseMECEnv` - 45 edges
5. `OnPolicyTrainer` - 45 edges
6. `GameTheoryMECEnv` - 44 edges
7. `ActorDiscreteNetwork` - 43 edges
8. `ReplayBuffer` - 43 edges
9. `OffPolicyTrainer` - 41 edges
10. `MECRewardShaper` - 36 edges

## Surprising Connections (you probably didn't know these)
- `DDPGAgent` --uses--> `ReplayBuffer`  [INFERRED]
  rl_algorithms/ddpg.py → src/utils/buffer.py
- `DDPG 智能体 — 确定性策略梯度 (连续动作)      适用于：连续资源分配、MIMO预编码、机器人控制` --uses--> `ReplayBuffer`  [INFERRED]
  rl_algorithms/ddpg.py → src/utils/buffer.py
- `DDQNAgent` --uses--> `ReplayBuffer`  [INFERRED]
  rl_algorithms/ddqn.py → src/utils/buffer.py
- `DDQN: Double Deep Q-Network 解决DQN过估计问题的经典价值型算法  核心特点: 1. 目标解耦 — 选择动作和目标Q值使用不同网络` --uses--> `ReplayBuffer`  [INFERRED]
  rl_algorithms/ddqn.py → src/utils/buffer.py
- `DDQN 智能体 — Double Deep Q-Network (离散动作)      适用于：离散信道选择、频谱分配、离散动作决策任务` --uses--> `ReplayBuffer`  [INFERRED]
  rl_algorithms/ddqn.py → src/utils/buffer.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (62): BaseMECEnv, PathLoss3GPP_LOS, PathLoss3GPP_NLOS, Path Loss Models  支持: - 自由空间路径损耗 (Freespace) - 3GPP TR 38.901 LOS / NLOS 路径损耗 -, 3GPP TR 38.901 NLOS 路径损耗模型      PL_NLOS = max(PL_LOS, PL_NLOS_specific)      Arg, 计算 NLOS 路径损耗 (dB)          Args:             distance_2d: 2D 水平距离 (m)          R, 3GPP TR 38.901 LOS 路径损耗模型      适用于: UMi, UMa, RMa 等城市场景      PL = 28.0 + 22*log1, 计算 LOS 路径损耗 (dB)          Args:             distance_2d: 2D 水平距离 (m)          Re (+54 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (69): ABC, A3C: Advantage Actor-Critic (异步版本的多线程简化实现) 多线程异步并发训练的前身  核心特点: 1. 多线程并行 — 多个work, 选择动作 — 同时记录rollout数据          Args:             state: np.ndarray - 当前状态，shape:, 使用收集到的rollout数据更新A3C策略         注意：A3C update前需要调用 select_action + record_transit, A3C 智能体 — 多线程同步简化版本      适用于：大规模网络优化、分布式训练场景      简化策略：使用本地网络副本模拟多线程worker交互，, Args:             state_dim: int - 状态空间维度             action_dim: int - 动作空间维度, ActionResult, BaseAgent (+61 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (75): BaseAgent, DDQNAgent, PPOAgent, A3CAgent, 计算A3C的优势估计和回报 (n-step), COMAAgent, DDPGAgent, DDPG 智能体 — 确定性策略梯度 (连续动作)      适用于：连续资源分配、MIMO预编码、机器人控制 (+67 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (58): BaseTrainer, GameTheoryDiscreteMAEnv, GameTheoryMECEnv 离散动作适配器 (多智能体)      每个智能体的动作空间: Discrete((K+1) * bins^3)     观测, Return the registered algorithm name, accepting case-insensitive input., Mirror stream writes to terminal and file., 运行单算法评测。      环境自动选择: ALGO_ENV_MAP 保证了每个算法在正确的环境类型上评测，     确保离散/连续/多智能体算法各在其适对应的, Best-effort numeric conversion for aggregation., Format metric value for table output, with NA fallback. (+50 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (63): Exception, benchmark_heuristic(), benchmark_single(), _build_heuristic_action(), _canonical_algorithm_name(), create_agent(), _encode_discrete_action(), _extract_step_metrics() (+55 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (49): ArrayFactor, Antenna Array Models  支持: - ULA: 均匀线阵 (Uniform Linear Array) - URA: 均匀矩形阵 (Unifo, 阵列因子计算器      阵列因子 AF(θ) = sum_{n=0}^{N-1} w_n * exp(j * 2π * d * n * sin(θ)), 计算阵列因子          Args:             angles_rad: 方向角度 (rad), shape=(M,), 估算 3dB 波束宽度 (rad)          近似公式: θ_3dB ≈ 0.886 * λ / (N * d * cos(θ_0)), 均匀线阵 (Uniform Linear Array)      天线沿直线等间距排列      Args:         n_ant: 天线数, 导向矢量计算器      导向矢量 a(θ) 描述了信号从角度 θ 入射时，各天线单元的相位差      Args:         n_ant: 天线数, 计算导向矢量          Args:             angle_rad: 方向 (rad), shape=(M,)          Retur (+41 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (31): Protocol, DDPGActor, DDPGQNetwork, OUNoise, DDPG: Deep Deterministic Policy Gradient Actor-Critic 处理连续动作空间  核心特点: 1. 确定性, Args:             state_dim: int - 状态空间维度             action_dim: int - 动作空间维度, DDPG 更新 — off-policy，先存buffer再采样更新, Ornstein-Uhlenbeck 噪声 — 时序相关的探索噪声 (+23 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (36): JakesFading, NakagamiFading, Fading Channel Models  支持: - Rayleigh: 经典瑞利衰落 (无直视径) - Rician: 莱斯衰落 (有直视径) - Nak, 生成 Nakagami-m 衰落信道系数          Args:             size: 输出形状          Returns:, 返回信道增益 |h|^2 (Gamma 分布), 对数正态阴影衰落      X_dB ~ N(mu, sigma^2)      用于: 建筑物遮挡导致的快衰落     常与路径损耗叠加: PL_total, 生成阴影衰落 (dB)          Args:             size: 输出形状          Returns:, 瑞利衰落信道      h ~ CN(0, 1)     |h|^2 ~ Exp(1) (指数分布)      用于: 城市宏基站无直视径 (NLOS) 场景 (+28 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (22): Propagation models: link budget, SNR, mobility., LinkBudget, Link Budget Calculator  计算通信链路的总增益与损耗, 链路预算计算器      计算发射功率经路径损耗、增益损耗后的接收功率      公式: P_rx = P_tx + G_tx + G_rx - L_tx -, 计算接收功率          Args:             path_loss_db: 路径损耗 (dB)             fading_db:, 从距离计算接收功率          Args:             distance_m: 距离 (m)             path_loss_mo, 计算 SNR          Args:             noise_power_dbm: 噪声功率 (dBm)             path_l, GaussMarkovMobility (+14 more)

### Community 9 - "Community 9"
Cohesion: 0.07
Nodes (30): _fmt_or_na(), load_results(), main(), _pick_energy_metric(), _pick_first_numeric(), _pick_latency_metric(), plot_latency_energy(), plot_reward_comparison() (+22 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (15): Queueing theory models: M/M/1, M/M/K, etc., MM1Queue, MMcKQueue, MMCQueue, M/M/1 and M/M/K Queue Models  排队论模型用于分析服务器队列性能, M/M/c 排队系统 (多服务器)      - 到达: 泊松过程, 率 λ     - 服务: 指数分布, 率 μ     - 服务员: c 个      A, M/M/1 排队系统      - 到达: 泊松过程, 率 λ     - 服务: 指数分布, 率 μ     - 服务员: 1 个      Args:, 所有服务器都忙的概率 (Erlang C 公式的一部分)          P_wait = (C_c(α) * α^c) / (c! * (1 - ρ) + (+7 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (15): _decode_continuous_action(), _decode_discrete_action(), _encode_discrete_action(), GameTheoryContinuousMAEnv, GameTheory MEC 环境适配层  将 GameTheoryMECEnv 的 Dict 混合动作空间适配为标准 gym 空间， 供单/多智能体 RL 算, Args:             actions: List[int] or int, 每个智能体的离散动作索引, GameTheoryMECEnv 连续动作适配器 (多智能体)      每个智能体的动作空间: Box(-1, 1, (4,))     观测空间与底层环境一, Args:             actions: List[np.ndarray] or np.ndarray, 每个智能体的 4D 连续动作向量 (+7 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (11): Signal processing: noise models., AWGNChannel, NoiseFigure, Noise Models  支持: - 热噪声 (Thermal Noise) - AWGN 信道 - 噪声系数 (Noise Figure), 添加噪声          Args:             x: 输入信号             signal_power_dbm: 信号功率 (dBm), 噪声系数 (Noise Figure)      F = SNR_in / SNR_out (线性)      Args:         noise_figu, 热噪声模型      N = k * T * B (W)      Args:         temperature_k: 噪声温度 (K), 默认 290K, 输出噪声功率          Args:             bandwidth_hz: 带宽 (Hz)             input_snr_db (+3 more)

### Community 13 - "Community 13"
Cohesion: 0.13
Nodes (15): 将阴影衰落叠加到路径损耗上          Args:             path_loss_db: 路径损耗 (dB)          Return, BenchmarkAnalyzer, generate_excel_analysis(), generate_markdown_report(), main(), Get success/failure summary, Get top N algorithms by metric, Get statistics for algorithm category (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.13
Nodes (10): Modulation schemes: QAM, PSK, PAM., get_modem(), QAMModem, QAM Modulation  支持: - QPSK, 16-QAM, 64-QAM, 256-QAM - 格雷码映射 - 误码率计算, 软解调 (LLR)          对每个比特计算对数似然比, QAM 调制解调器      Args:         order: 调制阶数 (4, 16, 64, 256), 近似误码率 (AWGN)          使用 Q 函数近似          Args:             snr_db: SNR (dB), 获取调制解调器      Args:         modulation: "qpsk", "16qam", "64qam", "256qam"      R (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (13): db_to_linear(), dbm_to_watts(), dbw_to_watts(), linear_to_db(), Unit Conversions for Communications  dB, dBm, dBW, Watt 之间转换, dB 转线性      Args:         x_db: dB 值      Returns:         x_linear: 线性值, 线性转 dB      Args:         x_linear: 线性值      Returns:         x_db: dB 值, dBm 转 Watt      0 dBm = 1 mW      Args:         x_dbm: dBm 值      Returns: (+5 more)

### Community 16 - "Community 16"
Cohesion: 0.21
Nodes (4): MADDPGActor, MADDPGCritic, MADDPG: Multi-Agent Deep Deterministic Policy Gradient 多智能体DDPG — 集中训练分散执行的连续动作算, _soft_update()

### Community 17 - "Community 17"
Cohesion: 0.23
Nodes (11): AgentError, BufferError, ConfigError, EnvironmentError, GRPOMECError, Custom exceptions for GRPO_MEC., Environment-related errors., Agent-related errors. (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.2
Nodes (10): AlgorithmConfig, Config, EnvConfig, load_config(), LoggingConfig, Configuration management with OmegaConf., Load configuration with priority: CLI overrides > YAML > dataclass defaults., Validate configuration values. (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.25
Nodes (3): ActionScaler, Action space utilities for unified action scaling and conversion., Unified action space scaling utilities.

### Community 20 - "Community 20"
Cohesion: 0.29
Nodes (2): Ensure test reproducibility., set_deterministic()

### Community 21 - "Community 21"
Cohesion: 0.67
Nodes (1): Custom environments for MEC (Multi-Edge Computing).

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): GRPO-MEC: Group Relative Policy Optimization for Multi-Edge Computing.

### Community 23 - "Community 23"
Cohesion: 2.0
Nodes (1): Communication Foundation Library (comm)  通用通信基础组件，支持 MEC、MIMO、OFDM 等场景的模块化复用。  模

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): MEC V2 — Base environment only (base_env.py is kept as parent class for GameTheo

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): 初始化智能体          Args:             state_dim: int - 状态空间维度             action_dim

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): 选择动作          Args:             state: np.ndarray - 当前状态，shape: [state_dim] 或 [b

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): 更新策略          Args:             batch_data: dict - 批量经验数据，包含:                 -

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): 返回 t 时刻的位置          Returns:             x, y: 位置 (m)

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): 返回 t 时刻的速度          Returns:             vx, vy: 速度 (m/s)

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Scale from [-1, 1] to [low, high].

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Scale from [low, high] to [-1, 1].

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Convert continuous output to discrete action via argmax.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Replace NaN/Inf values with safe defaults and clip to [-1, 1].

## Knowledge Gaps
- **198 isolated node(s):** `计算A3C的优势估计和回报 (n-step)`, `Base Agent Interface for RL Algorithms All algorithms must inherit from this cla`, `Standardized return type for select_action.`, `Standardized return type for update.`, `初始化智能体          Args:             state_dim: int - 状态空间维度             action_dim` (+193 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 20`** (7 nodes): `continuous_env()`, `discrete_env()`, `dummy_batch()`, `multi_agent_env()`, `conftest.py`, `Ensure test reproducibility.`, `set_deterministic()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (3 nodes): `Custom environments for MEC (Multi-Edge Computing).`, `_register_if_missing()`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (2 nodes): `__init__.py`, `GRPO-MEC: Group Relative Policy Optimization for Multi-Edge Computing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (2 nodes): `Communication Foundation Library (comm)  通用通信基础组件，支持 MEC、MIMO、OFDM 等场景的模块化复用。  模`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `MEC V2 — Base environment only (base_env.py is kept as parent class for GameTheo`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `初始化智能体          Args:             state_dim: int - 状态空间维度             action_dim`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `选择动作          Args:             state: np.ndarray - 当前状态，shape: [state_dim] 或 [b`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `更新策略          Args:             batch_data: dict - 批量经验数据，包含:                 -`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `返回 t 时刻的位置          Returns:             x, y: 位置 (m)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `返回 t 时刻的速度          Returns:             vx, vy: 速度 (m/s)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Scale from [-1, 1] to [low, high].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Scale from [low, high] to [-1, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Convert continuous output to discrete action via argmax.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Replace NaN/Inf values with safe defaults and clip to [-1, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseAgent` connect `Community 1` to `Community 16`, `Community 2`, `Community 6`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `Channel models: path loss, fading, MIMO.` connect `Community 7` to `Community 0`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `BaseTrainer` connect `Community 3` to `Community 1`, `Community 4`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Are the 101 inferred relationships involving `BaseAgent` (e.g. with `A3CAgent` and `A3C: Advantage Actor-Critic (异步版本的多线程简化实现) 多线程异步并发训练的前身  核心特点: 1. 多线程并行 — 多个work`) actually correct?**
  _`BaseAgent` has 101 INFERRED edges - model-reasoned connections that need verification._
- **Are the 54 inferred relationships involving `ActorNetwork` (e.g. with `A3CAgent` and `A3C: Advantage Actor-Critic (异步版本的多线程简化实现) 多线程异步并发训练的前身  核心特点: 1. 多线程并行 — 多个work`) actually correct?**
  _`ActorNetwork` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `BaseTrainer` (e.g. with `TrainerCallback` and `LoggingCallback`) actually correct?**
  _`BaseTrainer` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `BaseMECEnv` (e.g. with `PathLoss3GPP_LOS` and `PathLoss3GPP_NLOS`) actually correct?**
  _`BaseMECEnv` has 30 INFERRED edges - model-reasoned connections that need verification._