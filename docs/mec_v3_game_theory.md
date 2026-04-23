# GameTheoryMECEnv — 方案一：多基站协作博弈 + MADRL

## 概述

基于 Stackelberg 博弈 + Shapley 值协作分配的移动边缘计算环境。

**核心特性：**
- Stackelberg 博弈：领导者（基站）定价，跟随者（用户）响应
- Shapley 值：联盟收益公平分配
- GRPO 初始化提示：博弈均衡作为 RL 策略初始化

**适用算法：** GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3

**Gymnasium ID：** `MEC-v1-game-theory`

---

## 动作空间

```python
Dict({
    "target": Discrete(K+1),  # 0=本地处理, 1..K=基站k
    "ratio": Box(-1,1,(3,))   # [offload_ratio, cpu_freq_ratio, tx_power_ratio]
})
```

| 动作键 | 类型 | 描述 |
|--------|------|------|
| `target` | Discrete(K+1) | 卸载目标（0=本地，1-K=基站k） |
| `ratio[0]` | Box | 卸载比例 [0, 1] |
| `ratio[1]` | Box | CPU频率比例 [0, 1] |
| `ratio[2]` | Box | 传输功率比例 [0, 1] |

---

## 观测空间

每个智能体观测维度：`(3K + 5,)`

| 索引范围 | 描述 |
|----------|------|
| `[0, K)` | 各基站队列长度（归一化） |
| `[K, 2K)` | 各基站信道 SNR（归一化） |
| `[2K, 3K)` | 各基站 CPU 负载 |
| `[3K, 3K+5)` | 任务特征：能量/截止时间/数据量/CPU需求/移动速度 |

---

## 核心模块

### 1. StackelbergGame

领导者-跟随者博弈求解器：

```python
class StackelbergGame:
    def leader_update_price(self, demands)    # 领导者更新定价
    def follower_best_response(self, costs)   # 跟随者最优响应
    def compute_user_costs(...)               # 计算用户总成本
```

### 2. ShapleyValueCalculator

联盟博弈收益分配：

```python
class ShapleyValueCalculator:
    def compute_shapley(coalition_values)  # 计算 Shapley 值
```

---

## 使用示例

```python
from src.environments.mec_v3 import GameTheoryMECEnv

env = GameTheoryMECEnv(
    num_agents=3,        # 用户数量
    num_edge_servers=3,   # 基站数量
    enable_game_init=True,  # 启用博弈初始化提示
    enable_shapley=True     # 启用 Shapley 分配
)

obs, info = env.reset()
actions = [
    {"target": 1, "ratio": [0.5, 0.3, 0.7]},
    {"target": 2, "ratio": [0.3, 0.5, 0.6]},
    {"target": 0, "ratio": [0.0, 0.8, 0.0]},
]
next_obs, rewards, terminated, truncated, info = env.step(actions)
```

---

## 奖励函数

基础奖励 + Shapley 权重：

```python
reward = base_reward * shapley_allocation[agent_id]
```

其中 `base_reward` 来自 MECRewardShaper（延迟+能耗+截止期限惩罚）。

---

## 状态信息（info）

```python
{
    "global_obs": np.ndarray,           # 全局状态
    "agent_tasks": [Dict],                # 各智能体任务
    "game_hints": {                       # 博弈均衡提示
        "equilibrium_prices": np.ndarray,
        "predicted_demands": np.ndarray,
    },
    "shapley_allocation": np.ndarray,    # Shapley 收益分配
    "queue_lengths": np.ndarray,         # 基站队列长度
    "individual_latencies": [float],     # 各智能体延迟
    "individual_energies": [float],      # 各智能体能耗
}
```

---

## 文件位置

`src/environments/mec_v3/game_theory_env.py`
