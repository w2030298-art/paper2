# RL-MEC Benchmark

**17-Algorithm MADRL Benchmark on GameTheory MEC Environment**

17 种强化学习算法在统一的 GameTheory MEC 边缘计算环境上的公平对比评测框架。支持单智能体/多智能体、离散/连续动作空间的完整训练-评估-Benchmark 流水线，全部算法统一在 GameTheory 适配环境上运行。

---

## 目录

- [核心特性](#核心特性)
- [支持的算法](#支持的算法)
- [支持的环境](#支持的环境)
- [项目结构](#项目结构)
- [Agent 接口规范](#agent-接口规范)
- [环境详细说明](#环境详细说明)
- [训练框架](#训练框架)
- [Benchmark 系统](#benchmark-系统)
- [通信模块](#通信模块)
- [神经网络架构](#神经网络架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [测试](#测试)
- [依赖](#依赖)
- [License](#license)

---

## 核心特性

- **17 种 RL 算法**：涵盖 On-Policy、Off-Policy、Multi-Agent 三大类别（含 6 个新增多智能体算法）
- **统一 GameTheory 环境**：所有算法在同一 MEC 博弈论环境上评测，确保公平对比
- **动作空间适配**：原生混合动作空间（Dict）通过离散/连续适配器自动转换为标准 Gym 空间
- **公平对比设计**：统一 Agent 接口、统一训练框架、自动算法-环境匹配
- **完整 Benchmark 流水线**：多 seed 支持、自动评估、JSON 结果存储、可视化图表
- **可扩展架构**：BaseAgent 抽象、BaseMECEnv 基类、Trainer 继承体系

---

## 支持的算法

| 类别 | 算法 | 动作空间 | 网络架构 | 核心机制 | 参考文献 |
|------|------|----------|----------|----------|----------|
| On-Policy | **GRPO** | 连续 | ActorNetwork | 组相对奖励、无 Critic、PPO 裁剪 | DeepSeek-R1 (2024) |
| On-Policy | **PPO** | 连续/离散 | Actor + Critic | GAE 优势估计、裁剪目标、多 epoch 更新 | Schulman et al. (2017) |
| On-Policy | **A3C** | 离散/连续 | Actor + Critic | 异步多线程（简化版本地副本）、n-step 优势 | Mnih et al. (2016) |
| On-Policy | **TRPO** | 连续 | Actor + Critic | KL 约束、共轭梯度法、线搜索 | Schulman et al. (2015) |
| On-Policy | **SimPO** | 离散 | Policy + RefPolicy | 无需奖励模型、偏好对优化、KL 约束 | SimPO (2024) |
| Off-Policy | **SAC** | 连续 | 双 Q + Actor | 最大熵、自动 alpha 调节、重参数化 | Haarnoja et al. (2018) |
| Off-Policy | **DDPG** | 连续 | Actor + Critic + Target | 确定性策略、OU 噪声探索、经验回放 | Lillicrap et al. (2016) |
| Off-Policy | **TD3** | 连续 | 双 Q Critic + Actor | Twin Q、延迟策略更新、目标平滑正则 | Fujimoto et al. (2018) |
| Off-Policy | **DDQN** | 离散 | DuelingQNetwork | 目标解耦、Dueling 架构、epsilon-greedy | Hasselt et al. (2016) |
| Multi-Agent | **MAPPO** | 离散 | Actor(分散) + Critic(集中) | 集中训练分散执行、参数共享、PPO 裁剪 | Yu et al. (2022) |
| Multi-Agent | **QMIX** | 离散 | Q-net + MixingNetwork | 值分解、单调性约束、Hypernetwork 混合 | Rashid et al. (2018) |
| Multi-Agent | **COMA** | 离散 | Actor + Counterfactual Critic | 反事实基线、集中 Critic、信用分配 | Foerster et al. (2018) |
| Multi-Agent | **IPPO** | 离散 | Actor + Critic (独立) | 独立 PPO、参数共享/独立可选 | de Witt et al. (2020) |
| Multi-Agent | **VDN** | 离散 | Q-net | 值分解网络、简单求和混合 | Sunehag et al. (2018) |
| Multi-Agent | **MADDPG** | 连续 | Actor + Critic (集中) | 集中 Critic、分散 Actor、经验回放 | Lowe et al. (2017) |
| Multi-Agent | **IQL** | 离散 | Q-net | 独立 Q-Learning、值分解简化 | Tan (1993) |
| Multi-Agent | **MATD3** | 连续 | 双 Q Critic + Actor | TD3 多智能体扩展、目标平滑 | Ackermann et al. (2019) |

### 新增 6 个多智能体算法

| 算法 | 类型 | 特点 |
|------|------|------|
| **COMA** | On-Policy (离散) | 反事实基线解决信用分配问题 |
| **IPPO** | On-Policy (离散) | 独立 PPO，每个智能体独立策略更新 |
| **VDN** | Off-Policy (离散) | 简单值分解，Q_total = sum(Q_i) |
| **MADDPG** | Off-Policy (连续) | 集中 Critic 观察全局状态，分散 Actor 执行 |
| **IQL** | Off-Policy (离散) | 独立 Q-Learning，无需全局状态 |
| **MATD3** | Off-Policy (连续) | MADDPG 的 Twin Q 改进版 |

### 算法超参数默认值

| 算法 | `hidden_dim` | `lr` | `gamma` | `batch_size` | 特殊参数 |
|------|-------------|------|---------|-------------|----------|
| GRPO | 256 | 3e-4 | 0.99 | — | `eps_clip=0.2, group_size=64, num_epochs=10` |
| PPO | 256 | 3e-4 | 0.99 | — | `eps_clip=0.2, gae_lambda=0.95, num_epochs=10` |
| A3C | 256 | 3e-4 | 0.99 | — | `num_steps=20, discrete=True/False` |
| TRPO | 256 | 3e-4 | 0.99 | — | `max_kl=0.01, cg_iters=10, damping=0.1` |
| SimPO | 256 | 3e-4 | 0.99 | — | `beta=0.1, ref_coeff=0.2, discrete=True` |
| SAC | 256 | 3e-4 | 0.99 | 256 | `tau=0.005, alpha=0.2, auto_entropy=True, buffer=1M` |
| DDPG | 256 | 3e-4 | 0.99 | 256 | `tau=0.005, OU噪声(theta=0.15,sigma=0.2), buffer=1M` |
| TD3 | 256 | 3e-4 | 0.99 | 256 | `tau=0.005, noise_std=0.2, noise_clip=0.5, policy_delay=2, buffer=1M` |
| DDQN | 256 | 3e-4 | 0.99 | 256 | `tau=0.005, epsilon: 1.0→0.05 over 10000 steps, buffer=1M` |
| MAPPO | 256 | 3e-4 | 0.99 | — | `eps_clip=0.2, gae_lambda=0.95, num_agents=3, num_epochs=10, discrete=True` |
| QMIX | 64 | 3e-4 | 0.99 | 256 | `tau=0.005, n_agents=3, epsilon: 1.0→0.05 over 10000, buffer=1M` |
| COMA | 256 | 3e-4 | 0.99 | — | `num_agents=3, num_epochs=10, discrete=True` |
| IPPO | 256 | 3e-4 | 0.99 | — | `eps_clip=0.2, num_agents=3, num_epochs=10, discrete=True` |
| VDN | 64 | 3e-4 | 0.99 | 256 | `tau=0.005, n_agents=3, epsilon: 1.0→0.05 over 10000, buffer=1M` |
| MADDPG | 256 | 3e-4 | 0.99 | 256 | `tau=0.005, n_agents=3, buffer=1M` |
| IQL | 64 | 3e-4 | 0.99 | 256 | `tau=0.005, n_agents=3, epsilon: 1.0→0.05 over 10000, buffer=1M` |
| MATD3 | 256 | 3e-4 | 0.99 | 256 | `tau=0.005, n_agents=3, noise_std=0.2, policy_delay=2, buffer=1M` |

---

## 支持的环境

### 核心环境：GameTheoryMECEnv

所有 17 个算法统一在 GameTheory 环境上运行。环境原生支持混合动作空间（Dict），通过离散/连续适配器转换为标准 Gym 空间供不同算法使用。

#### GameTheoryMECEnv — 博弈论 MEC (原生)

| 属性 | 值 |
|------|-----|
| 继承 | `BaseMECEnv` |
| 动作空间 | `Dict({"target": Discrete(K+1), "ratio": Box(-1,1,(3,))})` |
| 观测空间 | `[queue_K, snr_K, load_K, 5个任务特征]` |
| Gymnasium ID | `MEC-v1-game-theory` |
| 核心机制 | Stackelberg 博弈、Shapley 值分配、用户移动性 |

**博弈论模型**：
- **StackelbergGame**：领导者（基站）根据需求调整价格；跟随者（用户）选择最低成本基站
- **ShapleyValueCalculator**：合作联盟的 Shapley 值计算，用于分配个体奖励
- 每 10 步计算博弈均衡，奖励中包含博弈均衡信息

#### GameTheoryDiscreteMAEnv — 离散适配器

| 属性 | 值 |
|------|-----|
| 动作空间 | `Discrete((K+1) × 5³)` — 离散量化 5 档 `[-1.0, -0.5, 0.0, 0.5, 1.0]` |
| 观测空间 | 同 GameTheoryMECEnv |
| Gymnasium ID | `MEC-v1-game-theory-discrete-ma` |
| 适用算法 | 离散动作算法（A3C, SimPO, DDQN, MAPPO, QMIX, COMA, IPPO, VDN, IQL） |

#### GameTheoryContinuousMAEnv — 连续适配器

| 属性 | 值 |
|------|-----|
| 动作空间 | `Box(-1, 1, (4,))` — `[target_selector, ratio_0, ratio_1, ratio_2]` |
| 观测空间 | 同 GameTheoryMECEnv |
| Gymnasium ID | `MEC-v1-game-theory-continuous-ma` |
| 适用算法 | 连续动作算法（GRPO, PPO, TRPO, SAC, DDPG, TD3, MADDPG, MATD3） |

### 算法-环境统一映射

```
离散动作算法 ──→ GameTheoryDiscreteMAEnv
连续动作算法 ──→ GameTheoryContinuousMAEnv
```

单智能体算法（如 A3C, SAC）使用 `num_agents=1` 的适配器环境；多智能体算法（如 MAPPO, QMIX）使用 `num_agents=3` 的环境。

---

## 项目结构

```
paper2/
├── rl_algorithms/                # 17 个 RL 算法实现
│   ├── __init__.py
│   ├── base_agent.py             #   BaseAgent 抽象基类
│   ├── grpo.py                   #   GRPO
│   ├── ppo.py                    #   PPO（支持离散/连续双模式）
│   ├── sac.py                    #   SAC
│   ├── ddqn.py                   #   DDQN
│   ├── ddpg.py                   #   DDPG
│   ├── td3.py                    #   TD3
│   ├── a3c.py                    #   A3C
│   ├── trpo.py                   #   TRPO
│   ├── simpo.py                  #   SimPO
│   ├── mappo.py                  #   MAPPO
│   ├── qmix.py                   #   QMIX
│   ├── coma.py                   #   COMA (新增)
│   ├── ippo.py                   #   IPPO (新增)
│   ├── vdn.py                    #   VDN (新增)
│   ├── maddpg.py                 #   MADDPG (新增)
│   ├── iql.py                    #   IQL (新增)
│   ├── matd3.py                  #   MATD3 (新增)
│   └── utils/
│       ├── networks.py           #   共享网络
│       └── buffers.py            #   ReplayBuffer, RolloutBuffer
│
├── src/
│   ├── environments/
│   │   ├── __init__.py           #   GameTheory 环境导出 + Gym 注册
│   │   ├── mec_v2/
│   │   │   ├── __init__.py       #     BaseMECEnv 基类保留
│   │   │   └── base_env.py       #     信道/能耗/奖励模型
│   │   └── mec_v3/
│   │       ├── __init__.py       #     GameTheory 注册
│   │       ├── game_theory_env.py#     GameTheoryMECEnv
│   │       └── game_theory_adapters.py # 离散/连续适配器
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── base_trainer.py       #   BaseTrainer（评估/保存/日志/种子）
│   │   ├── on_policy_trainer.py  #   OnPolicyTrainer（shared/joint 双模式）
│   │   └── off_policy_trainer.py #   OffPolicyTrainer（shared/joint 双模式）
│   ├── comm/                     #   通信信道模型
│   │   ├── antenna/              #     天线: 阵列、波束赋形、DoA
│   │   ├── channel/              #     信道: MIMO, 衰落, 路径损耗
│   │   ├── modulation/           #     调制: QAM
│   │   ├── propagation/          #     传播: SNR, 链路预算
│   │   ├── queueing/             #     排队: M/M/1
│   │   ├── signal/               #     信号: 噪声
│   │   └── utils/                #     工具: 单位转换
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py            #   set_seed, RunningMeanStd
│       ├── buffer.py             #   ReplayBuffer(PER), RolloutBuffer
│       └── exceptions.py         #   自定义异常
│
├── scripts/
│   ├── benchmark.py             #   17 算法对比评测 CLI 入口
│   ├── train.py                  #   单算法训练脚本
│   ├── experiment_manager.py     #   实验编排入口
│   ├── backup_experiment.py      #   实验结果备份入口
│   ├── analyze_convergence_failures.py # 收敛失败分析
│   └── plot_results.py           #   结果可视化与质量报告
│
├── configs/
│   ├── default.yaml             #   默认全局配置
│   └── algorithms/               #   17 个算法 YAML 配置
│       ├── grpo.yaml, ppo.yaml, sac.yaml, ddqn.yaml, ddpg.yaml,
│       ├── td3.yaml, a3c.yaml, trpo.yaml, simpo.yaml,
│       ├── mappo.yaml, qmix.yaml, coma.yaml, ippo.yaml,
│       ├── vdn.yaml, maddpg.yaml, iql.yaml, matd3.yaml
│
├── tests/
│   ├── test_mec_envs.py          #   GameTheory 环境单元测试
│   ├── test_trainers.py          #   训练器测试（GameTheory 环境）
│   └── test_algorithms_on_envs.py#   17 算法集成测试
│
├── docs/
│   ├── mec_v3_game_theory.md     #   GameTheory 环境详细文档
│   ├── RL-MEC 系统模型框架方案.md #   系统框架方案
│   └── reporting/                #   汇报模板与流程图
│       ├── status_snapshot.md
│       ├── mentor_report_template.md
│       ├── README.md
│       └── diagrams/
│           ├── benchmark_flow.md
│           ├── system_architecture.md
│           ├── training_flow.md
│           └── environment_models.md
│
├── ref_papers/                   #   参考文献 PDF
├── requirements.txt              #   依赖列表
├── pyproject.toml               #   项目配置（依赖、工具链、pytest）
└── README.md                     #   本文件
```

---

## Agent 接口规范

所有 17 个算法必须实现 `BaseAgent` 抽象基类：

```python
class BaseAgent(ABC):
    @abstractmethod
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4,
                 gamma=0.99, device='cuda', **kwargs): ...

    @abstractmethod
    def select_action(self, state, deterministic=False) -> tuple:
        """返回 (action, info)，info 必须包含 'log_prob'"""
        ...

    @abstractmethod
    def update(self, batch_data: dict) -> dict:
        """返回训练信息，必须包含 'loss'"""
        ...

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict): ...

    # 可选：多智能体模式标识
    multi_agent_mode: str = "shared"  # "shared" 或 "joint"
```

### 关键约定

| 约定 | 说明 |
|------|------|
| `state` 输入 | Trainer 传入原始 numpy obs，**Agent 内部负责 unsqueeze** |
| `action` 输出 | 返回 numpy array 或 int，Trainer 负责格式转换 |
| `info` 字典 | 必须包含 `log_prob` 键（on-policy 算法必须） |
| `batch_data` 输入 | Trainer 统一构造，包含 `states, actions, rewards, next_states, dones, log_probs` |
| 多智能体 | `multi_agent_mode="shared"` 按 agent 展开为单智能体样本；`"joint"` 保留 `[n_agents, ...]` 维度 |

---

## 环境详细说明

### GameTheoryMECEnv

基于 Stackelberg 博弈 + Shapley 值协作分配的移动边缘计算环境。

**动作空间（原生 Dict）**：
```python
Dict({
    "target": Discrete(K+1),  # 0=本地处理, 1..K=基站k
    "ratio": Box(-1, 1, (3,))  # [offload_ratio, cpu_freq_ratio, tx_power_ratio]
})
```

**观测空间**：每个智能体观测维度 `3K + 5 + M*K + (K+1) + M + 3K`（含价格历史/Shapley历史/动作频率/ΔSNR/ΔQ/ρ）

| 特征组 | 描述 |
|--------|------|
| `queue/snr/load` | 各基站队列长度、信道SNR、CPU负载 |
| `task_features(5)` | 电量/截止时间/数据量/CPU需求/移动强度 |
| `price_history` | 最近 `M` 步均衡定价 |
| `action_frequency` | 他方动作目标频率统计 |
| `shapley_history` | 最近 `M` 步 Shapley |
| `delta_snr/delta_q/rho` | 信道变化、拥塞导数、M/M/1 利用率 |

**核心模块**：
- `OptimalPricingMechanism` + `BilevelGameSolver`：凸优化定价 + Stackelberg/IBR 双层求解
- `MonteCarloShapley`：Shapley 蒙特卡洛近似（含 antithetic）
- `QueueingDelayModel`/`DVFSEnergyModel`/`Channel3GPP`：排队、非理想DVFS、3GPP+干扰
- `ConstraintProjection` + `HierarchicalReward`：可微约束与分层奖励
- `EFXFairAllocation` + `CPNetPreferenceModel`：公平修复与偏好建模

### 适配器层

**GameTheoryDiscreteMAEnv**：将 Dict 动作编码为单个离散索引 `(K+1) × 5³`，支持 5 档离散量化。

**GameTheoryContinuousMAEnv**：将 Dict 动作编码为 4D 连续向量 `[target_selector, ratio_0, ratio_1, ratio_2]`。

### 信道模型 (MECChannelModel)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `carrier_freq` | 2.4 GHz | 载波频率 |
| `bandwidth` | 20 MHz | 信道带宽 |
| `noise_power` | -100 dBm | 噪声功率 |

**路径损耗**：`PL(d) = 128.1 + 37.6 × log10(d/1000)` (d in meters)

**SNR 计算**：`SNR = tx_power_w × |h|² / (noise_power_w × bandwidth)`

**传输速率**：`rate = bandwidth × log2(1 + SNR)`

### 能耗模型 (MECEnergyModel)

- **本地能耗**：`E_local = cpu_cycles / cycles_per_joule`
- **传输能耗**：`E_tx = tx_power_w × transmission_time`

### 奖励塑形

```
reward = -(latency_weight × latency + energy_weight × energy
            + deadline_weight × max(0, latency - deadline)) × reward_scale
```

| 参数 | 默认值 |
|------|--------|
| `latency_weight` | 1.0 |
| `energy_weight` | 0.1 |
| `deadline_weight` | 2.0 |
| `reward_scale` | 0.01 |

---

## 训练框架

### On-Policy Trainer

**适用算法**：GRPO, PPO, A3C, TRPO, SimPO, MAPPO, COMA, IPPO

**多智能体模式**：
- `shared` 模式：按 agent 展开为单智能体样本（适用于 IPPO, A3C 等独立更新算法）
- `joint` 模式：保留 `[n_agents, ...]` 维度（适用于 MAPPO, QMIX 等联合更新算法）

### Off-Policy Trainer

**适用算法**：SAC, DDPG, TD3, DDQN, QMIX, VDN, MADDPG, IQL, MATD3

**数据流**：每步收集 transition → 存入 agent 内部 buffer → 采样更新

### 动作格式转换

| 环境类型 | 动作格式 | 转换方式 |
|----------|----------|----------|
| 离散 | `int` 或 `np.ndarray` | `np.argmax(action)` → `int` |
| 连续 | `np.ndarray (4,)` | `action.flatten()` |
| 多智能体 | `List[int]` 或 `List[np.ndarray]` | 直接传入 |

---

## Benchmark 系统

### 算法-环境自动匹配

```
离散动作算法 (A3C, SimPO, DDQN, QMIX, COMA, IPPO, VDN, IQL, MAPPO)
    └── GameTheoryDiscreteMAEnv(num_agents=1 或 3)

连续动作算法 (GRPO, PPO, TRPO, SAC, DDPG, TD3, MADDPG, MATD3)
    └── GameTheoryContinuousMAEnv(num_agents=1 或 3)
```

| 算法 | 环境 | Trainer 类型 |
|------|------|-------------|
| GRPO, PPO, TRPO | `MEC-v1-game-theory-continuous-ma` (num_agents=1) | On-Policy |
| SAC, DDPG, TD3 | `MEC-v1-game-theory-continuous-ma` (num_agents=1) | Off-Policy |
| A3C, SimPO, DDQN | `MEC-v1-game-theory-discrete-ma` (num_agents=1) | On/Off-Policy |
| MAPPO, QMIX, COMA, IPPO, VDN, IQL | `MEC-v1-game-theory-discrete-ma` (num_agents=3) | On/Off-Policy |
| MADDPG, MATD3 | `MEC-v1-game-theory-continuous-ma` (num_agents=3) | Off-Policy |

### 配置文件

每个算法有独立 YAML 配置文件 (`configs/algorithms/*.yaml`)，包含：

```yaml
algorithm:
  name: grpo
  type: on_policy
  hidden_dim: 256
  lr: 3.0e-4
  gamma: 0.99
  eps_clip: 0.2
  group_size: 64
  num_epochs: 10
training:
  total_timesteps: 100000
  rollout_steps: 2048
  seed: 42
game_theory:
  enabled: true
  use_shapley_credit: true
  ctde_with_hints: true
  warm_start_steps: 1000
  shapley_samples: 128
  reward_weights: [0.5, 0.3, 0.2]
  efx_enabled: true
  cpnet_enabled: true
  efx_transfer_rate: 0.5
```

### Benchmark 命令

```bash
# 全部 17 个算法
python scripts/benchmark.py --all --timesteps 100000

# 指定算法
python scripts/benchmark.py --algorithms GRPO PPO SAC --timesteps 50000

# 多 seed 取平均
python scripts/benchmark.py --algorithms GRPO --seeds 42 123 456

# 指定设备
python scripts/benchmark.py --all --device cpu

# 可扩展实验矩阵 (small / medium / large)
python scripts/benchmark.py --all --scale medium --timesteps 50000

# 启发式基线一起跑 (Greedy/Random/Local-only/Full-offload)
python scripts/benchmark.py --all --include-heuristics --episodes 10

# 关闭 EFX/CP-net 消融
python scripts/benchmark.py --algorithms GRPO MAPPO --efx-enabled false --cpnet-enabled false
```

### GameTheory Integration Matrix

| 算法 | 模式 | Shapley信用 | CTDE提示 | Warm-Start |
|------|------|-------------|-----------|------------|
| GRPO/MAPPO/QMIX/COMA/IPPO/VDN/MADDPG/IQL/MATD3 | 深度融合 | 开启 | 开启 | 开启 |
| PPO/SAC/DDQN/DDPG/TD3/A3C/TRPO/SimPO | 兼容模式 | 可选 | 可选 | 可选 |

### Pipeline with Game-Theory Signals

```mermaid
flowchart LR
    A["env.step/reset"] --> B["info: game_hints/shapley/reward_terms/queue/constraint/fairness"]
    B --> C["OnPolicy/OffPolicy Trainer"]
    C --> D["batch_data: global_states/eq_actions/shapley_values/reward_terms"]
    D --> E["算法 update() (按能力消费，未使用字段忽略)"]
```

### 结果可视化

```bash
python scripts/plot_results.py --input results/benchmark.json --output figures/ --format pdf
```

生成 4 种图表：奖励对比、训练时间、时延-能耗散点图、汇总表。

**颜色编码**（17 算法）：
GRPO=红, PPO=蓝, SAC=绿, DDQN=橙, DDPG=紫, TD3=青, A3C=棕, TRPO=深灰, SimPO=粉,
MAPPO=浅蓝, QMIX=棕褐, COMA=橙红, IPPO=蓝灰, VDN=浅绿, MADDPG=橙黄, IQL=天蓝, MATD3=紫红

---

## 通信模块 (comm)

`src/comm/` 提供物理层通信模型，供 GameTheoryMECEnv 使用：

| 子模块 | 主要内容 |
|--------|----------|
| `antenna/` | 天线阵列、波束赋形、到达角估计 |
| `channel/` | MIMO 信道、衰落模型、路径损耗 |
| `modulation/` | QAM 调制 |
| `propagation/` | SNR 计算、链路预算 |
| `queueing/` | M/M/1 排队模型 |
| `signal/` | 噪声模型 |
| `utils/` | 单位转换 |

---

## 神经网络架构

所有网络定义在 `rl_algorithms/utils/networks.py`：

| 网络 | 架构 | 用途 |
|------|------|------|
| `ActorNetwork` | 2 层 MLP → `(mean, log_std)` | 连续策略 |
| `ActorDiscreteNetwork` | 2 层 MLP → `logits` | 离散策略 |
| `CriticNetwork` | 2 层 MLP → `V(s)` | 价值估计 |
| `QNetwork` | 2 层 MLP → `Q(s,a)` | Q 值 |
| `DuelingQNetwork` | 分离 V + A 流 | Q 值 (DDQN) |
| `MixingNetwork` | Hypernetwork 生成权重 | QMIX 值分解 |

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/RL-MEC-Benchmark.git
cd RL-MEC-Benchmark

# 安装依赖
pip install -r requirements.txt

# 或从 pyproject.toml 安装（推荐）
pip install -e .
```

**Python ≥ 3.10**, **PyTorch ≥ 2.0**, **CUDA** (可选，推荐)

---

## 快速开始

### 单算法训练（离散）

```python
from src.environments.mec_v3 import GameTheoryDiscreteMAEnv
from rl_algorithms.a3c import A3CAgent
from src.trainer.on_policy_trainer import OnPolicyTrainer

env = GameTheoryDiscreteMAEnv(num_agents=1, num_edge_servers=3, max_steps=100)
agent = A3CAgent(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.n,
                 hidden_dim=256, discrete=True, device="cpu")

trainer = OnPolicyTrainer(env=env, agent=agent,
                          total_timesteps=100000,
                          rollout_steps=2048)
trainer.train()
```

### 单算法训练（连续）

```python
from src.environments.mec_v3 import GameTheoryContinuousMAEnv
from rl_algorithms.sac import SACAgent
from src.trainer.off_policy_trainer import OffPolicyTrainer

env = GameTheoryContinuousMAEnv(num_agents=1, num_edge_servers=3, max_steps=100)
agent = SACAgent(state_dim=env.observation_space.shape[0],
                 action_dim=env.action_space.shape[0],
                 hidden_dim=256, device="cuda")

trainer = OffPolicyTrainer(env=env, agent=agent,
                           total_timesteps=200000,
                           warmup_steps=1000)
trainer.train()
```

### 多智能体训练

```python
from src.environments.mec_v3 import GameTheoryDiscreteMAEnv
from rl_algorithms.mappo import MAPPOAgent
from src.trainer.on_policy_trainer import OnPolicyTrainer

env = GameTheoryDiscreteMAEnv(num_agents=3, num_edge_servers=3, max_steps=100)
obs_list, info = env.reset()
agent = MAPPOAgent(state_dim=env.observation_space.shape[0],
                   action_dim=env.action_space.n,
                   hidden_dim=256, num_agents=3, discrete=True, device="cpu")

trainer = OnPolicyTrainer(env=env, agent=agent,
                          total_timesteps=100000, rollout_steps=2048)
trainer.train()
```

### Benchmark 多算法对比

```bash
# 全部 17 个算法
python scripts/benchmark.py --all --timesteps 100000

# 指定算法
python scripts/benchmark.py --algorithms GRPO PPO MAPPO QMIX --timesteps 50000

# 多 seed 取平均
python scripts/benchmark.py --all --seeds 42 123 456
```

### 结果可视化

```bash
python scripts/plot_results.py --input results/benchmark.json --output figures/ --format pdf
```

---

## 测试

```bash
# 全部测试
pytest tests/ -v

# 仅环境测试
pytest tests/test_mec_envs.py -v

# 仅训练器测试
pytest tests/test_trainers.py -v

# 仅算法-环境集成测试
pytest tests/test_algorithms_on_envs.py -v
```

### 测试覆盖

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_mec_envs.py` | GameTheoryMECEnv(2) + DiscreteMAEnv(3) + ContinuousMAEnv(2) |
| `test_trainers.py` | OnPolicyTrainer(离散+连续+训练) + OffPolicyTrainer(训练) + BaseTrainer(种子/保存) |
| `test_algorithms_on_envs.py` | 17 算法 × GameTheory 适配器 + 17 个 YAML 配置有效性 |

---

## 依赖

```
numpy>=1.24
scipy>=1.10
torch>=2.0
gymnasium>=0.29
matplotlib>=3.7
seaborn>=0.12
tensorboard>=2.13
tqdm>=4.65
pyyaml>=6.0
pytest>=7.0
```

---

## License

MIT
