# RL\-MEC 系统模型框架方案

# RL\-MEC 系统模型框架方案

> 基于强化学习的通信MEC网络性能优化 — 系统模型设计
生成时间：2026\-04\-10
资料基础：43篇论文/资料（searcher 31篇 \+ catcher 12篇）

---

## 一、背景资料覆盖

<table><tbody>
<tr>
<td>

来源

</td>
<td>

论文数

</td>
<td>

重点方向

</td>
</tr>
<tr>
<td>

searcher

</td>
<td>

31篇

</td>
<td>

MEC架构、RL算法应用、State/Action/Reward设计、仿真平台、IEEE顶刊、联合优化、MADRL、6G/IoT/数字孪生

</td>
</tr>
<tr>
<td>

catcher

</td>
<td>

12篇

</td>
<td>

场景对比（车联网/工业IoT/医疗）、博弈论\+RL混合、网络切片、边缘智能、绿色MEC、开源项目、ETSI/3GPP标准化、benchmark

</td>
</tr>
</tbody></table>

### 关键论文推荐

<table><tbody>
<tr>
<td>

论文

</td>
<td>

年份

</td>
<td>

期刊

</td>
<td>

核心价值

</td>
</tr>
<tr>
<td>

Parameterized DRL With Hybrid Action Space

</td>
<td>

2024

</td>
<td>

IEEE

</td>
<td>

混合动作空间MEC（离散卸载\+连续资源分配）

</td>
</tr>
<tr>
<td>

MA\-MEC: Multi\-Agent MEC

</td>
<td>

2024

</td>
<td>

Springer

</td>
<td>

多边缘服务器独立智能体 \+ MADRL协同

</td>
</tr>
<tr>
<td>

Dec\-POMDP for Collaborative MEC

</td>
<td>

2024

</td>
<td>

GitHub

</td>
<td>

去中心化POMDP \+ 多层队列模型

</td>
</tr>
<tr>
<td>

Multi\-Agent RL Driven Resource Game

</td>
<td>

2025

</td>
<td>

Nature SR

</td>
<td>

博弈论\+MARL资源竞争建模

</td>
</tr>
<tr>
<td>

Secure Active RIS\-Assisted MEC

</td>
<td>

2024

</td>
<td>

IEEE

</td>
<td>

Active RIS提升物理层安全 \+ MEC卸载

</td>
</tr>
<tr>
<td>

Digital Twin Enabled 6G MEC

</td>
<td>

2025

</td>
<td>

IEEE

</td>
<td>

DT虚拟副本主动预测调度

</td>
</tr>
<tr>
<td>

Joint Service Caching \&amp; Resource Allocation

</td>
<td>

2024

</td>
<td>

IEEE TWC

</td>
<td>

协作MEC \+ 服务缓存联合优化

</td>
</tr>
<tr>
<td>

Stackelberg Game\-based MARL for MEC

</td>
<td>

2024

</td>
<td>

IEEE TNSE

</td>
<td>

Stackelberg博弈 \+ 多智能体RL定价博弈

</td>
</tr>
</tbody></table>

---

## 二、方案一：多基站协作博弈 \+ MADRL 联合优化框架 ⭐推荐

### 系统架构

- **顶层：MADRL 全局优化**

    - GRPO/MAPPO 驱动计算卸载 \+ 资源分配决策

    - CTDE架构：中心化训练 → 去中心化执行

- **底层：基站间博弈交互**

    - Stackelberg博弈：领导者基站定价策略

    - 联盟博弈：Shapley值收益分配

    - EFX/CP\-nets：公平性与偏好建模

    - 博弈均衡 → 辅助RL算法选型与策略初始化

- **State/Action/Reward 设计**

    - State: \[用户队列长度, 信道SNR, BS负载, 设备剩余能量, 任务截止时间, 移动速度\]

    - Action: 混合空间（离散:卸载目标 \+ 连续:资源比例）

    - Reward: \-α·时延 \-β·能耗 \+γ·SLA满足度

### 系统模型数学描述

**网络模型**：

- K个基站（BS），每个BS配备MEC服务器，计算能力 F\_k

- U个用户设备（UE），每个UE有任务到达队列

- 时隙 t 内，用户 u 生成任务 J\_u\(t\) = \(D\_u, C\_u, T\_u^max\)

    - D\_u: 数据量（bits），C\_u: 所需CPU周期数，T\_u^max: 时延阈值

**卸载决策**：

- x\_\{u,k\} ∈ \{0,1\}：用户u是否卸载到基站k

- α\_u ∈ \[0,1\]：部分卸载比例（0=全本地，1=全卸载）

**时延模型**：

- 本地计算：T\_local = C\_u / f\_u^local

- 边缘计算：T\_edge = D\_u / R\_\{u,k\} \+ C\_u / f\_\{u,k\} \+ D\_u^result / R\_\{u,k\}^dl

    - R\_\{u,k\} = B·log2\(1 \+ p\_u·h\_\{u,k\} / σ²\)：上行传输速率

**能耗模型**：

- 本地：E\_local = κ·\(f\_u^local\)²·C\_u

- 边缘：E\_edge = p\_u·D\_u / R\_\{u,k\}

**优化目标**：

```
min_{x,α,f,p}  Σ_u [ α_u·T_local + (1-α_u)·T_edge + λ_u·(α_u·E_local + (1-α_u)·E_edge) ]
s.t.
  C1: Σ_k x_{u,k} ≤ 1          （每任务最多卸载到一个基站）
  C2: Σ_u x_{u,k}·f_{u,k} ≤ F_k  （基站k的CPU容量约束）
  C3: T_total(u) ≤ T_u^max       （时延SLA约束）
  C4: E_total(u) ≤ E_u^max       （能量预算约束）
  C5: p_u ∈ [0, P_max]           （发射功率约束）
```

**博弈论底层**：

- 基站间资源竞争 → Stackelberg博弈建模

- 领导者（宏基站）定价，跟随者（微基站）响应

- 均衡解作为RL策略的初始化点

**RL算法设计**：

- 算法：GRPO（Group Relative Policy Optimization）

- 优势：Group Relative思想适合多智能体场景，无需单独Value网络

- 训练：CTDE中心化训练，执行时去中心化

- 策略网络：双头Actor（离散卸载头 \+ 连续资源头）

### 仿真方案

<table><tbody>
<tr>
<td>

组件

</td>
<td>

选型

</td>
<td>

说明

</td>
</tr>
<tr>
<td>

RL框架

</td>
<td>

Stable\-Baselines3 / Tianshou

</td>
<td>

PPO/GRPO快速原型

</td>
</tr>
<tr>
<td>

网络仿真

</td>
<td>

ns3\-gym 或 自建Gym环境

</td>
<td>

信道模型\+任务生成

</td>
</tr>
<tr>
<td>

信道模型

</td>
<td>

3GPP TR 38\.901

</td>
<td>

5G NR标准信道

</td>
</tr>
<tr>
<td>

任务模型

</td>
<td>

Poisson到达 \+ 随机数据量

</td>
<td>

标准MEC任务模型

</td>
</tr>
<tr>
<td>

对比基线

</td>
<td>

DQN / DDPG / 随机卸载 / 全本地 / 全卸载

</td>
<td>

学术对比

</td>
</tr>
</tbody></table>

### 适合期刊

- IEEE Transactions on Vehicular Technology \(TVT\)

- IEEE Transactions on Wireless Communications \(TWC\)

- IEEE Internet of Things Journal \(IoTJ\)

### 优劣势分析

<table><tbody>
<tr>
<td>

维度

</td>
<td>

评估

</td>
</tr>
<tr>
<td>

与研究框架匹配度

</td>
<td>

⭐⭐⭐⭐⭐ 底层博弈\+顶层GRPO完全一致

</td>
</tr>
<tr>
<td>

创新性

</td>
<td>

⭐⭐⭐⭐ 博弈\+RL混合是前沿交叉方向

</td>
</tr>
<tr>
<td>

实现难度

</td>
<td>

🟡 中等，博弈均衡\+RL训练需调参

</td>
</tr>
<tr>
<td>

仿真可行性

</td>
<td>

✅ RLlib/ns3\-gym成熟可用

</td>
</tr>
<tr>
<td>

发表竞争力

</td>
<td>

✅ 博弈\+MARL在MEC方向仍有空间

</td>
</tr>
</tbody></table>

---

## 三、方案二：RIS辅助MEC \+ 参数化深度RL 混合动作空间框架

### 系统架构

- **RIS辅助通信层**

    - 主动RIS / 可旋转RIS 提升信道质量

    - RIS相位优化 → RL联合决策变量

    - 解决Passive RIS \&\#34;双衰落\&\#34;效应

- **参数化DRL：混合动作空间**

    - 离散头：卸载目标选择 \{BS0, BS1, \.\.\., BSK\}

    - 连续头：CPU分配比例 \[0, F\_max\]

    - 连续头：RIS相位优化 \[0, 2π\]^M

    - 连续头：传输功率 \[P\_min, P\_max\]

- **联合优化目标**

    - min 时延 \+ 能耗 \+ 缓存切换成本

    - s\.t\. QoS约束 / 功率约束 / RIS硬件约束

### 系统模型数学描述

**RIS信道模型**：

- h\_\{u,k\} = h\_\{u,k\}^direct \+ h\_\{u,R\}^H · Θ · h\_\{R,k\}

- Θ = diag\(β₁e^\{jθ₁\}, \.\.\., β\_Me^\{jθ\_M\}\)：RIS反射矩阵

- β\_m ∈ \{0,1\}：开关控制（Active RIS: β\_m 可连续调节）

- θ\_m ∈ \[0, 2π\)：相位偏移

**优化目标**：

```
min_{x,α,f,p,θ}  Σ_u [ α·T_local + (1-α)·T_edge_RIS ]
s.t.
  C1-C5: 同方案一
  C6: Σ_m |β_m|² ≤ P_RIS          （Active RIS功率约束）
  C7: θ_m ∈ [0, 2π), ∀m            （相位约束）
  C8: ||c_k||₀ ≤ S_max             （缓存容量约束）
```

**RL设计**：

- 算法：参数化PPO（Multi\-Pass PPO）

- State扩展：\[用户队列, 信道SNR, BS负载, RIS状态, 缓存状态\]

- Action扩展：4头输出（卸载\+资源\+RIS相位\+功率）

- Reward：\-α·T \- β·E \- γ·C\_cache（含缓存切换成本）

### 适合期刊

- IEEE Transactions on Wireless Communications \(TWC\)

- IEEE Journal on Selected Areas in Communications \(JSAC\)

### 优劣势分析

<table><tbody>
<tr>
<td>

维度

</td>
<td>

评估

</td>
</tr>
<tr>
<td>

与研究框架匹配度

</td>
<td>

⭐⭐⭐ 仅RL层，无博弈论

</td>
</tr>
<tr>
<td>

创新性

</td>
<td>

⭐⭐⭐⭐⭐ RIS\+MEC是最热前沿

</td>
</tr>
<tr>
<td>

实现难度

</td>
<td>

🔴 较高，RIS信道建模复杂

</td>
</tr>
<tr>
<td>

仿真可行性

</td>
<td>

⚠️ 需自建RIS模型

</td>
</tr>
<tr>
<td>

发表竞争力

</td>
<td>

✅ RIS方向竞争激烈但高IF

</td>
</tr>
</tbody></table>

---

## 四、方案三：数字孪生MEC \+ 联邦MARL 主动预测调度框架

### 系统架构

- **数字孪生层：虚拟MEC网络镜像**

    - 实时同步物理网络状态

    - 预测性资源分配（被动响应→主动预测）

    - 任务到达率预测 \+ 信道状态预测

- **联邦MARL：隐私保护多智能体学习**

    - 各基站本地训练 → 加密梯度聚合

    - 自适应聚合频率（降低通信开销）

    - 差分隐私保护机制

- **系统模型**

    - 多用户 \+ 多BS \+ 边缘\-云协同

    - 能量收集\(EH\)集成 → 可持续MEC

    - DAG任务依赖图建模

### 系统模型数学描述

**数字孪生模型**：

- DT\_k\(t\)：基站k在时隙t的数字孪生镜像

- DT\_k\(t\) = \{队列状态, 信道状态, 资源状态, 预测状态\}

- 同步延迟：Δt\_sync（物理→虚拟同步间隔）

**能量收集模型**：

- E\_harvest\(u,t\) \~ f\_solar\(wind, area, efficiency\)

- 电池状态：B\_u\(t\+1\) = min\{B\_max, B\_u\(t\) \- E\_consumed \+ E\_harvest\}

**DAG任务模型**：

- 任务图 G = \(V, E\)，V=子任务集，E=依赖关系

- 拓扑排序约束：子任务v\_j必须在v\_i完成后执行（若\(v\_i,v\_j\)∈E）

**联邦学习**：

- 本地更新：θ\_k^\{local\} ← θ\_k \- η·∇L\_k\(D\_k\)

- 全局聚合：θ^\{global\} ← Σ\_k \(n\_k/n\)·θ\_k^\{local\}

- 自适应聚合频率：通信轮次基于梯度散度自适应调整

### 适合期刊

- IEEE Internet of Things Journal \(IoTJ\)

- IEEE Transactions on Mobile Computing \(TMC\)

### 优劣势分析

<table><tbody>
<tr>
<td>

维度

</td>
<td>

评估

</td>
</tr>
<tr>
<td>

与研究框架匹配度

</td>
<td>

⭐⭐⭐ 联邦RL层，无博弈论

</td>
</tr>
<tr>
<td>

创新性

</td>
<td>

⭐⭐⭐⭐⭐ DT\+MEC是6G核心概念

</td>
</tr>
<tr>
<td>

实现难度

</td>
<td>

🔴 高，DT平台不成熟

</td>
</tr>
<tr>
<td>

仿真可行性

</td>
<td>

❌ 需自建DT\+FL仿真平台

</td>
</tr>
<tr>
<td>

发表竞争力

</td>
<td>

✅ 概念新颖但验证难度大

</td>
</tr>
</tbody></table>

---

## 五、三方案综合对比

<table><tbody>
<tr>
<td>

维度

</td>
<td>

方案一（博弈\+MADRL）

</td>
<td>

方案二（RIS\+参数化DRL）

</td>
<td>

方案三（DT\+联邦MARL）

</td>
</tr>
<tr>
<td>

与研究框架匹配度

</td>
<td>

⭐⭐⭐⭐⭐

</td>
<td>

⭐⭐⭐

</td>
<td>

⭐⭐⭐

</td>
</tr>
<tr>
<td>

创新性

</td>
<td>

⭐⭐⭐⭐

</td>
<td>

⭐⭐⭐⭐⭐

</td>
<td>

⭐⭐⭐⭐⭐

</td>
</tr>
<tr>
<td>

实现难度

</td>
<td>

🟡 中等

</td>
<td>

🔴 较高

</td>
<td>

🔴 高

</td>
</tr>
<tr>
<td>

仿真可行性

</td>
<td>

✅ 成熟

</td>
<td>

⚠️ 需自建

</td>
<td>

❌ 不成熟

</td>
</tr>
<tr>
<td>

适合期刊

</td>
<td>

TVT/TWC/IoTJ

</td>
<td>

TWC/JSAC

</td>
<td>

IoTJ/TMC

</td>
</tr>
<tr>
<td>

发表周期

</td>
<td>

6\-9月

</td>
<td>

8\-12月

</td>
<td>

10\-15月

</td>
</tr>
<tr>
<td>

与博弈论框架兼容

</td>
<td>

✅ 完全兼容

</td>
<td>

❌ 需新增博弈层

</td>
<td>

❌ 需新增博弈层

</td>
</tr>
</tbody></table>

---

## 六、推荐结论

**首选方案一**：多基站协作博弈 \+ MADRL联合优化框架

理由：

1. 与底层博弈论（EFX/CP\-nets）\+ 顶层GRPO研究框架完全一致

2. 实现难度可控，仿真平台成熟（RLlib/ns3\-gym）

3. 博弈\+RL混合是前沿交叉方向，创新点充足

4. 发表周期相对较短

**可考虑融合**：将方案二的RIS辅助通信作为方案一的增强模块（RIS辅助信道 → 纳入State空间），在方案一基础上拓展。

---

*资料来源：searcher 31篇 \+ catcher 12篇 = 43篇论文/资料*
*生成时间：2026\-04\-10*

