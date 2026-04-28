# 基于博弈论引导的多智能体深度强化学习移动边缘计算联合优化：系统模型与理论框架

> **完整理论模型初稿**
> 
> 框架：多基站协作博弈 + MADRL (GRPO) 联合优化
> 
> 版本：v1.0 — 2026-04-24

---

## 符号总表 (Notation Summary)

为保证全文符号一致性，以下列出本文所用核心数学符号。所有向量默认为列向量，集合用花体字母表示。

### 网络拓扑与索引

| 符号 | 定义 | 取值范围 |
|------|------|----------|
| $\mathcal{K}$ | 基站（BS）集合 | $\mathcal{K} = \{1, 2, \ldots, K\}$ |
| $\mathcal{U}$ | 用户设备（UE）集合 | $\mathcal{U} = \{1, 2, \ldots, U\}$ |
| $K$ | 基站总数 | 典型值 3–10 |
| $U$ | 用户设备总数 | 典型值 6–20 |
| $t$ | 离散时隙索引 | $t \in \{0, 1, \ldots, T\}$ |
| $\Delta t$ | 时隙持续时间 | 单位：秒 |

### 任务参数

| 符号 | 定义 | 单位 |
|------|------|------|
| $\mathcal{J}_i(t)$ | 用户 $i$ 在时隙 $t$ 的计算任务 | 三元组 $(D_i, C_i, T_i^{\max})$ |
| $D_i$ | 任务输入数据量 | bits |
| $C_i$ | 任务所需 CPU 周期数 | cycles |
| $T_i^{\max}$ | 任务最大容忍时延（截止时间） | 秒 |
| $D_i^{\text{res}}$ | 任务计算结果数据量 | bits |

### 决策变量

| 符号 | 定义 | 取值域 |
|------|------|--------|
| $x_{i,k}$ | 用户 $i$ 是否卸载至基站 $k$ 的二元指示 | $\{0, 1\}$ |
| $\alpha_i$ | 用户 $i$ 的卸载比例 | $[0, 1]$ |
| $f_i^{\text{loc}}$ | 用户 $i$ 本地 CPU 频率 | $[f_{\min}, f_{\max}]$ Hz |
| $f_{i,k}^{\text{edge}}$ | 基站 $k$ 分配给用户 $i$ 的 CPU 频率 | Hz |
| $P_i$ | 用户 $i$ 的上行发射功率 | $[P_{\min}, P_{\max}]$ W |
| $p_k$ | 基站 $k$ 的服务定价 | $[p_{\min}, p_{\max}]$ 单位价格 |

### 信道与通信参数

| 符号 | 定义 | 单位 |
|------|------|------|
| $B$ | 系统信道带宽 | Hz |
| $f_c$ | 载波频率 | GHz |
| $h_{i,k}$ | 用户 $i$ 到基站 $k$ 的信道系数 | 复数 |
| $d_{i,k}$ | 用户 $i$ 到基站 $k$ 的二维距离 | m |
| $\text{PL}(d)$ | 路径损耗函数 | dB |
| $\sigma^2$ | 加性高斯白噪声功率 | W |
| $\text{SINR}_{i,k}$ | 用户 $i$ 在基站 $k$ 的信干噪比 | 线性 |
| $R_{i,k}$ | 用户 $i$ 到基站 $k$ 的上行传输速率 | bps |
| $h_{\text{BS}}$ | 基站天线高度 | m |
| $h_{\text{UT}}$ | 用户终端天线高度 | m |

### 排队与计算参数

| 符号 | 定义 | 单位 |
|------|------|------|
| $\lambda_k$ | 基站 $k$ 的任务到达率 | tasks/s |
| $\mu_k$ | 基站 $k$ 的服务率 | tasks/s |
| $\rho_k$ | 基站 $k$ 的利用率 $\rho_k = \lambda_k / \mu_k$ | 无量纲 |
| $F_k$ | 基站 $k$ MEC 服务器总 CPU 能力 | cycles/s |
| $\kappa_{\text{dyn}}$ | 动态功耗系数（CMOS 有效开关电容） | $\approx 10^{-27}$ |
| $\hat{\alpha}$ | DVFS 工艺相关指数 | $[2.5, 3.0]$, 典型 2.7 |
| $P_{\text{leak}}$ | 泄漏功耗 | W |

### 博弈论参数

| 符号 | 定义 |
|------|------|
| $\phi_i$ | 用户 $i$ 的 Shapley 值 |
| $v(S)$ | 联盟 $S \subseteq \mathcal{U}$ 的联盟值函数 |
| $\eta_k$ | 基站 $k$ 的价格敏感度参数 |
| $D_k(p_k)$ | 基站 $k$ 在价格 $p_k$ 下的需求函数 |
| $c_k$ | 基站 $k$ 的边际成本系数 |
| $d_k^0$ | 基站 $k$ 的基础需求量 |

### RL 算法参数

| 符号 | 定义 |
|------|------|
| $\pi_{\theta_i}$ | 智能体 $i$ 的参数化策略 |
| $o_i(t)$ | 智能体 $i$ 在时隙 $t$ 的局部观测 |
| $s(t)$ | 全局状态 |
| $a_i(t)$ | 智能体 $i$ 的动作 |
| $r_i(t)$ | 智能体 $i$ 的即时奖励 |
| $\gamma$ | 折扣因子 |
| $\epsilon$ | PPO/GRPO 剪切比率 |

---

## I. 系统模型 (System Model)

### I-A. 网络架构 (Network Architecture)

考虑一个由 $K$ 个基站（Base Station, BS）和 $U$ 个移动用户设备（User Equipment, UE）组成的异构多接入边缘计算（MEC）网络。每个基站 $k \in \mathcal{K}$ 配备一台 MEC 服务器，具有计算能力 $F_k$（cycles/s）。基站按照预设拓扑部署于二维平面上，其位置坐标记为 $\mathbf{q}_k = (x_k^{\text{BS}}, y_k^{\text{BS}}, h_{\text{BS}})$，其中 $h_{\text{BS}}$ 为天线高度（典型值 25 m）。

用户设备 $i \in \mathcal{U}$ 在二维平面内移动，其在时隙 $t$ 的位置记为 $\mathbf{u}_i(t) = (x_i(t), y_i(t), h_{\text{UT}})$，其中 $h_{\text{UT}}$ 为终端天线高度（典型值 1.5 m）。每个用户设备具有有限的本地计算能力 $f_i^{\text{loc}} \in [f_{\min}, f_{\max}]$ 和能量预算 $E_i^{\max}$。

系统运行在时分架构下，整个运行周期被划分为 $T$ 个等长时隙，每个时隙持续 $\Delta t$ 秒。在每个时隙 $t$ 中，每个用户 $i$ 独立生成一个计算任务 $\mathcal{J}_i(t) = (D_i(t), C_i(t), T_i^{\max}(t))$。

**任务到达模型：** 任务到达过程建模为泊松过程，到达率为 $\lambda_{\text{arr}}$（tasks/slot）。每个任务的参数独立采样：

$$D_i(t) \sim \text{Uniform}(D_{\min}, D_{\max})$$

$$C_i(t) \sim \text{Uniform}(C_{\min}, C_{\max})$$

$$T_i^{\max}(t) \sim \text{Uniform}(T_{\min}^{\max}, T_{\max}^{\max})$$

其中典型参数设置为 $D_{\min} = 100$ KB，$D_{\max} = 500$ KB，$C_{\min} = 10^8$ cycles，$C_{\max} = 10^9$ cycles，$T_{\min}^{\max} = 0.5$ s，$T_{\max}^{\max} = 2.0$ s。

每个用户需要做出以下联合决策：

1. **卸载目标选择**：选择本地处理（$x_{i,k} = 0, \forall k$）或卸载至某个基站 $k$（$x_{i,k} = 1$），且每个任务至多卸载至一个基站：$\sum_{k \in \mathcal{K}} x_{i,k} \leq 1$。

2. **部分卸载比例**：确定卸载比例 $\alpha_i \in [0, 1]$，其中 $\alpha_i = 0$ 表示全部本地计算，$\alpha_i = 1$ 表示全部卸载。

3. **计算资源分配**：选择本地 CPU 频率 $f_i^{\text{loc}}$ 和向基站请求的边缘 CPU 频率 $f_{i,k}^{\text{edge}}$。

4. **传输功率控制**：选择上行传输功率 $P_i \in [P_{\min}, P_{\max}]$。

### I-B. 通信模型 (Communication Model)

#### I-B-1. 3GPP TR 38.901 信道模型

采用 3GPP TR 38.901 标准定义的 UMi（Urban Micro-cell）Street Canyon 信道模型，支持 LOS（视距）和 NLOS（非视距）两种传播状态。

**LOS 概率模型：**

用户 $i$ 到基站 $k$ 之间存在 LOS 路径的概率为：

$$P_{\text{LOS}}(d_{i,k}) = \min\left(1, \frac{18}{d_{i,k}}\right) \cdot \left(1 - e^{-d_{i,k}/36}\right) + e^{-d_{i,k}/36}$$

其中 $d_{i,k} = \|\mathbf{u}_i(t) - \mathbf{q}_k\|_2$ 为用户 $i$ 与基站 $k$ 之间的二维距离（m），且 $d_{i,k} \geq 1$ m。

**路径损耗模型：**

LOS 路径损耗（dB）：

$$\text{PL}_{\text{LOS}}(d_{i,k}) = 32.4 + 21 \cdot \log_{10}(d_{i,k}^{\text{3D}}) + 20 \cdot \log_{10}(f_c)$$

NLOS 路径损耗（dB）：

$$\text{PL}_{\text{NLOS}}(d_{i,k}) = 35.3 \cdot \log_{10}(d_{i,k}^{\text{3D}}) + 22.4 + 21.3 \cdot \log_{10}(f_c) - 0.3 \cdot (h_{\text{UT}} - 1.5)$$

其中 $d_{i,k}^{\text{3D}} = \sqrt{d_{i,k}^2 + (h_{\text{BS}} - h_{\text{UT}})^2}$ 为三维距离，$f_c$ 为载波频率（GHz），典型值 3.5 GHz。

**综合路径损耗（概率加权）：**

$$\text{PL}(d_{i,k}) = P_{\text{LOS}}(d_{i,k}) \cdot \text{PL}_{\text{LOS}}(d_{i,k}) + \left(1 - P_{\text{LOS}}(d_{i,k})\right) \cdot \text{PL}_{\text{NLOS}}(d_{i,k})$$

**阴影衰落：** 在路径损耗基础上叠加对数正态阴影衰落 $\xi \sim \mathcal{N}(0, \sigma_{\text{SF}}^2)$（dB 域），其中 LOS 场景 $\sigma_{\text{SF}} = 4$ dB，NLOS 场景 $\sigma_{\text{SF}} = 7.82$ dB。

**小尺度衰落：** 采用瑞利衰落模型描述多径效应。信道系数 $h_{i,k} \sim \mathcal{CN}(0, 1)$，信道增益 $|h_{i,k}|^2$ 服从均值为 1 的指数分布。在存在 LOS 分量的场景下，可扩展为莱斯衰落模型，K 因子典型值 $K_{\text{Rician}} = 3$。

#### I-B-2. 多用户干扰与 SINR 模型

考虑同信道用户间的共道干扰，用户 $i$ 在基站 $k$ 处的接收 SINR 为：

$$\text{SINR}_{i,k} = \frac{P_i \cdot |h_{i,k}|^2 \cdot 10^{-\text{PL}(d_{i,k})/10}}{\sum_{j \neq i, j \in \mathcal{U}_k} P_j \cdot |h_{j,k}|^2 \cdot 10^{-\text{PL}(d_{j,k})/10} + \sigma^2}$$

其中 $\mathcal{U}_k$ 为当前卸载至基站 $k$ 的用户子集，$\sigma^2$ 为热噪声功率：

$$\sigma^2 = k_B \cdot T_0 \cdot B = 10^{(\sigma^2_{\text{dBm}} - 30)/10} \quad \text{(W)}$$

其中 $\sigma^2_{\text{dBm}} = -174 + 10 \cdot \log_{10}(B)$（dBm），$k_B = 1.38 \times 10^{-23}$ J/K 为玻尔兹曼常数，$T_0 = 290$ K 为参考噪声温度，$B$ 为信道带宽（Hz），典型值 20 MHz。

**上行传输速率（Shannon 容量）：**

$$R_{i,k} = B \cdot \log_2(1 + \text{SINR}_{i,k}) \quad \text{(bps)}$$

**下行传输速率：** 结果回传速率 $R_{i,k}^{\text{dl}}$ 的建模方式类似，但由于基站发射功率通常远大于用户设备，下行 SINR 一般较高，可简化处理或采用与上行相同的 Shannon 公式。

#### I-B-3. RIS 辅助信道模型（可选扩展）

若系统部署 $M$ 个元素的可重构智能超表面（RIS），信道模型扩展为：

$$\tilde{h}_{i,k} = h_{i,k}^{\text{direct}} + \mathbf{h}_{i,R}^H \cdot \boldsymbol{\Theta} \cdot \mathbf{h}_{R,k}$$

其中 $\boldsymbol{\Theta} = \text{diag}(\beta_1 e^{j\theta_1}, \ldots, \beta_M e^{j\theta_M})$ 为 RIS 反射矩阵，$\beta_m \in \{0, 1\}$（被动 RIS）或 $\beta_m \in [0, \beta_{\max}]$（主动 RIS），$\theta_m \in [0, 2\pi)$ 为第 $m$ 个 RIS 元素的相位偏移。主动 RIS 受功率约束 $\sum_m |\beta_m|^2 \leq P_{\text{RIS}}$。

### I-C. 计算模型 (Computation Model)

#### I-C-1. 本地计算模型

用户 $i$ 选择在本地执行任务的比例为 $(1 - \alpha_i)$。本地计算时延为：

$$T_i^{\text{loc}} = \frac{(1 - \alpha_i) \cdot C_i}{f_i^{\text{loc}}}$$

#### I-C-2. 非理想 DVFS 能耗模型

传统的理想 DVFS（动态电压频率调节）模型假设计算能耗正比于 $\kappa f^2 C$，忽略了泄漏功耗和非理想缩放效应。本文采用更精确的非理想 DVFS 模型：

$$E_i^{\text{loc}} = \left(\kappa_{\text{dyn}} \cdot (f_i^{\text{loc}})^{\hat{\alpha} - 1} + \frac{P_{\text{leak}}}{f_i^{\text{loc}}}\right) \cdot (1 - \alpha_i) \cdot C_i$$

其中 $\hat{\alpha} \in [2.5, 3.0]$ 为 CMOS 工艺相关参数（本文取 $\hat{\alpha} = 2.7$），$\kappa_{\text{dyn}} \approx 10^{-27}$ 为动态功耗系数（有效开关电容），$P_{\text{leak}}$ 为泄漏功耗（典型值 0.01 W）。

**最优 CPU 频率推导：** 给定时延-能耗权衡参数 $\lambda$，求解能耗最小化问题：

$$\min_{f} \quad E(f) + \lambda \cdot T(f) = \kappa_{\text{dyn}} f^{\hat{\alpha}-1} C + P_{\text{leak}} \frac{C}{f} + \lambda \frac{C}{f}$$

对 $f$ 求导令其为零：

$$(\hat{\alpha} - 1) \cdot \kappa_{\text{dyn}} \cdot f^{\hat{\alpha}-2} \cdot C = (P_{\text{leak}} + \lambda) \cdot \frac{C}{f^2}$$

解得最优频率：

$$f_i^* = \left(\frac{P_{\text{leak}} + \lambda}{(\hat{\alpha} - 1) \cdot \kappa_{\text{dyn}}}\right)^{1/\hat{\alpha}}$$

实际频率受限于 $f_i^{\text{loc}} = \text{clip}(f_i^*, f_{\min}, f_{\max})$，其中 $f_{\min} = 0.5$ GHz，$f_{\max} = 3.0$ GHz。

#### I-C-3. 边缘计算模型

当用户 $i$ 将比例为 $\alpha_i$ 的任务卸载至基站 $k$ 时，边缘计算总时延包含以下分量：

**上行传输时延：**

$$T_i^{\text{ul}} = \frac{\alpha_i \cdot D_i}{R_{i,k}}$$

**边缘计算时延：**

$$T_i^{\text{exec}} = \frac{\alpha_i \cdot C_i}{f_{i,k}^{\text{edge}}}$$

**排队等待时延（M/M/1 模型）：** 见 I-C-4 节。

**下行回传时延：**

$$T_i^{\text{dl}} = \frac{\alpha_i \cdot D_i^{\text{res}}}{R_{i,k}^{\text{dl}}}$$

**边缘传输能耗：**

$$E_i^{\text{edge}} = P_i \cdot T_i^{\text{ul}} = P_i \cdot \frac{\alpha_i \cdot D_i}{R_{i,k}}$$

#### I-C-4. M/M/1 排队时延模型

将每个 MEC 服务器 $k$ 建模为 M/M/1 排队系统，以刻画高负载下的排队效应。

**到达率：** 基站 $k$ 的聚合任务到达率为：

$$\lambda_k = \sum_{i \in \mathcal{U}_k} \frac{\alpha_i \cdot D_i}{\Delta t}$$

其中 $\mathcal{U}_k = \{i \in \mathcal{U} : x_{i,k} = 1\}$ 为选择卸载至基站 $k$ 的用户集合。

**服务率：** 基站 $k$ 的服务率基于其 CPU 能力和平均任务 CPU 需求：

$$\mu_k = \frac{F_k}{\bar{C}}$$

其中 $\bar{C} = \mathbb{E}[C_i]$ 为平均任务 CPU 需求。

**利用率与稳定性条件：**

$$\rho_k = \frac{\lambda_k}{\mu_k}, \quad \rho_k < 1 \quad \text{（稳定性条件）}$$

在仿真中引入稳定性裕度 $\rho_k^{\max} = 0.95$，即 $\rho_k = \text{clip}(\lambda_k / \mu_k, 0, \rho_k^{\max})$。

**平均排队时延：**

$$T_k^{\text{queue}} = \frac{\rho_k}{\mu_k \cdot (1 - \rho_k)}$$

**平均系统时延（排队 + 服务）：**

$$T_k^{\text{system}} = \frac{1}{\mu_k \cdot (1 - \rho_k)}$$

**队列长度更新：** 基站 $k$ 的队列长度按如下规则在时隙间更新：

$$Q_k(t+1) = \max\left(Q_k(t) + \lambda_k(t) - \mu_k, \; 0\right)$$

#### I-C-5. 总时延与总能耗模型

用户 $i$ 的总时延采用部分卸载的加权模型。由于本地处理与边缘处理可以并行执行，总时延取二者的最大值：

$$T_i^{\text{total}} = \max\left(T_i^{\text{loc}}, \; T_i^{\text{ul}} + T_k^{\text{queue}} + T_i^{\text{exec}} + T_i^{\text{dl}}\right)$$

当采用顺序执行模式（保守估计）时：

$$T_i^{\text{total}} = (1 - \alpha_i) \cdot T_i^{\text{loc}} + \alpha_i \cdot \left(T_i^{\text{ul}} + T_k^{\text{queue}} + T_i^{\text{exec}} + T_i^{\text{dl}}\right)$$

用户 $i$ 的总能耗：

$$E_i^{\text{total}} = E_i^{\text{loc}} + E_i^{\text{edge}}$$

展开为：

$$E_i^{\text{total}} = \left(\kappa_{\text{dyn}} (f_i^{\text{loc}})^{\hat{\alpha}-1} + \frac{P_{\text{leak}}}{f_i^{\text{loc}}}\right)(1-\alpha_i)C_i + P_i \cdot \frac{\alpha_i D_i}{R_{i,k}}$$

### I-D. 移动性模型 (Mobility Model)

用户移动性采用高斯-马尔可夫模型，兼顾时间相关性和随机性。

**位置更新：** 在每个时隙 $t$，用户 $i$ 的位置按如下规则更新：

$$\mathbf{u}_i(t+1) = \mathbf{u}_i(t) + \mathbf{v}_i(t) \cdot \Delta t + \boldsymbol{\epsilon}_i(t)$$

其中速度向量 $\mathbf{v}_i(t) = (v_x, v_y)$，$\boldsymbol{\epsilon}_i(t) \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{mob}}^2 \mathbf{I}_2)$ 为位置扰动项。

**速度模型（高斯-马尔可夫）：**

$$v_i(t+1) = \beta_{\text{GM}} \cdot v_i(t) + (1 - \beta_{\text{GM}}) \cdot \bar{v} + \sigma_v \sqrt{1 - \beta_{\text{GM}}^2} \cdot w_i(t)$$

其中 $\beta_{\text{GM}} \in [0, 1]$ 为记忆参数（控制时间相关性），$\bar{v}$ 为均值速度，$\sigma_v$ 为速度标准差，$w_i(t) \sim \mathcal{N}(0, 1)$ 为高斯白噪声。

**边界反射：** 当用户移动至部署区域边界时，采用弹性反射策略将其限制在有效区域内：

$$\mathbf{u}_i(t) = \text{clip}(\mathbf{u}_i(t), \mathbf{u}_{\min}, \mathbf{u}_{\max})$$

**初始化：** 用户初始位置在部署区域内随机分布：

$$\theta_i^{(0)} \sim \text{Uniform}(0, 2\pi), \quad r_i^{(0)} \sim \text{Uniform}(r_{\min}, r_{\max})$$

$$\mathbf{u}_i(0) = (r_i^{(0)} \cos \theta_i^{(0)}, \; r_i^{(0)} \sin \theta_i^{(0)}, \; h_{\text{UT}})$$

其中 $r_{\min} = 8$ m，$r_{\max} = 30$ m。最大用户速度 $v_{\max} = 5$ m/s（对应步行至慢速骑行场景）。

---

## II. 问题建模 (Problem Formulation)

### II-A. 联合优化目标

本文的核心目标是在满足 QoS 约束条件下，联合优化所有用户的卸载决策、计算资源分配和传输功率控制，以最小化系统加权时延-能耗代价。

**原始优化问题 (P1)：**

$$\textbf{(P1):} \quad \min_{\mathbf{x}, \boldsymbol{\alpha}, \mathbf{f}, \mathbf{P}, \mathbf{p}} \quad \frac{1}{U} \sum_{i=1}^{U} \left[ w_T \cdot \frac{T_i^{\text{total}}}{T_i^{\max}} + w_E \cdot \frac{E_i^{\text{total}}}{E_i^{\max}} + w_D \cdot \left(\max\left(0, \frac{T_i^{\text{total}} - T_i^{\max}}{T_i^{\max}}\right)\right)^2 \right]$$

**约束条件：**

$$\text{C1（卸载唯一性）:} \quad \sum_{k=1}^{K} x_{i,k} \leq 1, \quad \forall i \in \mathcal{U}$$

$$\text{C2（边缘 CPU 容量）:} \quad \sum_{i \in \mathcal{U}_k} f_{i,k}^{\text{edge}} \leq F_k, \quad \forall k \in \mathcal{K}$$

$$\text{C3（时延 QoS）:} \quad T_i^{\text{total}} \leq T_i^{\max}, \quad \forall i \in \mathcal{U}$$

$$\text{C4（能量预算）:} \quad E_i^{\text{total}} \leq E_i^{\max}, \quad \forall i \in \mathcal{U}$$

$$\text{C5（发射功率范围）:} \quad P_{\min} \leq P_i \leq P_{\max}, \quad \forall i \in \mathcal{U}$$

$$\text{C6（系统功率上限）:} \quad \sum_{i=1}^{U} P_i \leq P_{\text{total}}^{\max}$$

$$\text{C7（队列稳定性）:} \quad \rho_k < 1, \quad \forall k \in \mathcal{K}$$

$$\text{C8（卸载比例）:} \quad 0 \leq \alpha_i \leq 1, \quad \forall i \in \mathcal{U}$$

$$\text{C9（CPU 频率范围）:} \quad f_{\min} \leq f_i^{\text{loc}} \leq f_{\max}, \quad \forall i \in \mathcal{U}$$

其中 $w_T$、$w_E$、$w_D$ 分别为时延、能耗和截止期限违约的权重系数（本文取 $w_T = 0.55$，$w_E = 0.10$，$w_D = 0.20$，另有排队权重 $w_Q = 0.15$）。

### II-B. NP-Hard 性分析

**定理（NP-Hard 性）：** 问题 (P1) 是 NP-Hard 的。

**证明思路：**

1. 即使在单基站（$K=1$）、无排队（$\rho_k = 0$）的简化场景下，问题 (P1) 退化为一个混合整数非线性规划（MINLP）问题：离散卸载决策 $x_{i,k} \in \{0, 1\}$ 与连续资源分配 $(\alpha_i, f_i, P_i)$ 的耦合使得问题在一般情况下不可多项式时间求解。

2. 将经典的多处理器任务调度问题（已知 NP-Hard）规约至 (P1) 的特例：令 $\alpha_i \in \{0, 1\}$（纯卸载/纯本地），问题退化为将 $U$ 个任务分配至 $K+1$ 个处理器（$K$ 个 MEC 服务器 + 1 个本地设备），以最小化加权完成时间——此即多处理器调度问题的变体。

3. 多用户间的干扰耦合（SINR 中的 $\sum_{j \neq i} P_j |h_{j,k}|^2$ 项）使得即使在连续松弛后，目标函数也是关于决策变量的非凸函数。

因此，直接求解 (P1) 在计算上不可行，这激发了本文采用博弈论建模与深度强化学习相结合的方法。

### II-C. 约束处理机制

为在 RL 框架中处理约束 C1–C9，本文引入**约束投影-惩罚-障碍混合机制**：

**参数化动作空间：** 策略网络输出归一化至 $[-1, 1]$ 的连续动作向量，通过仿射映射到物理域：

$$f_i^{\text{loc}} = f_{\min} + \frac{a_f + 1}{2} \cdot (f_{\max} - f_{\min})$$

$$P_i = P_{\min} + \frac{a_P + 1}{2} \cdot (P_{\max} - P_{\min})$$

$$\alpha_i = \frac{a_\alpha + 1}{2}$$

其中 $a_f, a_P, a_\alpha \in [-1, 1]$ 为策略网络输出。卸载目标 $x_{i,k}$ 通过离散动作头输出。

**约束投影层：** 对于能量预算约束 C4，若估计能耗 $\hat{E}_i > E_i^{\max}$，则按比例缩放频率和功率：

$$\text{scale} = \sqrt{E_i^{\max} / \hat{E}_i}, \quad f_i \leftarrow f_i \cdot \text{scale}, \quad P_i \leftarrow P_i \cdot \text{scale}$$

对于系统功率约束 C6，若 $\sum_i P_i > P_{\text{total}}^{\max}$，则按比例缩放每个用户的功率。

**二次惩罚项：**

$$\text{Penalty} = \mu_{\text{pen}} \cdot \left(v_E^2 + v_T^2 + v_P^2\right)$$

其中 $v_E = \max(0, \hat{E}_i - E_i^{\max})$，$v_T = \max(0, \hat{T}_i - T_i^{\max})$，$v_P = \max(0, \sum_i P_i - P_{\text{total}}^{\max})$，$\mu_{\text{pen}} = 0.05$ 为惩罚系数。

**对数障碍项：**

$$\text{Barrier} = -\mu_{\text{bar}} \cdot \left(\log(E_i^{\max} - \hat{E}_i) + \log(T_i^{\max} - \hat{T}_i) + \log(P_{\text{total}}^{\max} - \sum_i P_i)\right)$$

其中 $\mu_{\text{bar}} = 10^{-4}$ 为障碍系数。障碍项在可行域内部平滑地趋近约束边界时增加代价，起到隐式约束的作用。

---

## III. 博弈论框架 (Game-Theoretic Framework)

本文构建**双层博弈**结构：上层为 MEC 运营商（基站）之间的 Stackelberg 定价博弈，下层为用户之间的 Nash 资源竞争博弈，并通过 Shapley 值机制实现联盟收益的公平分配。

### III-A. Stackelberg 博弈模型

#### III-A-1. 领导者-跟随者结构

将基站（MEC 运营商）建模为**领导者**，将用户设备建模为**跟随者**，形成 Stackelberg 博弈的双层结构：

- **上层（领导者问题）：** 各基站 $k$ 选择服务定价 $p_k$ 以最大化利润：

$$\max_{p_k} \quad \Pi_k(p_k) = p_k \cdot D_k(p_k) - C_k\left(D_k(p_k)\right)$$

$$\text{s.t.} \quad p_{\min} \leq p_k \leq p_{\max}, \quad \forall k \in \mathcal{K}$$

- **下层（跟随者 Nash 博弈）：** 给定定价向量 $\mathbf{p} = (p_1, \ldots, p_K)$，用户 $i$ 选择卸载策略 $a_i$ 以最小化个体成本：

$$\min_{a_i} \quad J_i(a_i, \mathbf{a}_{-i}; \mathbf{p}) = w_T \cdot T_i(a_i, \mathbf{a}_{-i}) + w_E \cdot E_i(a_i) + p_{k_i} \cdot D_i(a_i)$$

其中 $k_i$ 为用户 $i$ 选择的目标基站。

#### III-A-2. 需求函数与成本函数

**对数线性需求模型：** 基站 $k$ 在价格 $p_k$ 下的需求量为：

$$D_k(p_k) = d_k^0 \cdot \exp(-\eta_k \cdot p_k)$$

其中 $d_k^0 > 0$ 为基础需求量，$\eta_k > 0$ 为价格敏感度参数。该模型满足经济学中需求函数的标准性质：$D_k'(p_k) < 0$（需求关于价格严格递减），且 $D_k(p_k) > 0, \forall p_k < \infty$。

**二次成本函数：** 基站 $k$ 的运营成本为：

$$C_k(x) = c_k \cdot x^2$$

其中 $c_k > 0$ 为边际成本系数，二次形式体现了资源利用的边际成本递增效应。

#### III-A-3. 最优定价的凸性分析

**命题 1（利润函数的凹性）：** 基站 $k$ 的利润函数 $\Pi_k(p_k)$ 关于 $p_k$ 为严格凹函数。

**证明：**

$$\Pi_k(p_k) = p_k \cdot d_k^0 \cdot e^{-\eta_k p_k} - c_k \cdot \left(d_k^0 \cdot e^{-\eta_k p_k}\right)^2$$

一阶导数：

$$\frac{d\Pi_k}{dp_k} = d_k^0 e^{-\eta_k p_k} - \eta_k p_k d_k^0 e^{-\eta_k p_k} + 2 c_k \eta_k (d_k^0)^2 e^{-2\eta_k p_k}$$

$$= d_k^0 e^{-\eta_k p_k} (1 - \eta_k p_k) + 2 c_k \eta_k (d_k^0)^2 e^{-2\eta_k p_k}$$

二阶导数：

$$\frac{d^2\Pi_k}{dp_k^2} = 2 \eta_k \underbrace{(-d_k^0 e^{-\eta_k p_k})}_{<0} + \eta_k^2 p_k d_k^0 e^{-\eta_k p_k} - 2 c_k (d_k^0)^2 \left(\eta_k^2 e^{-2\eta_k p_k} + 4 \eta_k^2 e^{-2\eta_k p_k}\right)$$

可以验证，对于 $\eta_k > 0$，$c_k > 0$，$d_k^0 > 0$，二阶条件 $d^2\Pi_k/dp_k^2 < 0$ 在 $p_k \geq 0$ 时恒成立。因此 $\Pi_k(p_k)$ 为严格凹函数，从而最优定价问题为凸优化问题。 $\square$

**最优定价求解（KKT 条件 + 牛顿迭代）：**

令 $d\Pi_k/dp_k = 0$，由于该一阶条件的隐式非线性特征（含 Lambert W 函数结构），采用牛顿法迭代求解：

$$p_k^{(n+1)} = p_k^{(n)} - \frac{d\Pi_k/dp_k|_{p_k^{(n)}}}{d^2\Pi_k/dp_k^2|_{p_k^{(n)}}}$$

初始化 $p_k^{(0)} = (p_{\min} + p_{\max}) / 2$，迭代至 $|d\Pi_k/dp_k| < \epsilon_{\text{tol}}$（$\epsilon_{\text{tol}} = 10^{-6}$），并将结果投影至 $[p_{\min}, p_{\max}]$。

### III-B. 用户间 Nash 均衡

#### III-B-1. 用户非合作博弈建模

在给定领导者定价 $\mathbf{p}$ 的条件下，$U$ 个用户形成一个 $U$ 人非合作博弈 $\mathcal{G} = \langle \mathcal{U}, \{A_i\}_{i \in \mathcal{U}}, \{J_i\}_{i \in \mathcal{U}} \rangle$：

- **参与者集合：** $\mathcal{U} = \{1, \ldots, U\}$
- **策略空间：** $A_i = \{0, 1, \ldots, K\} \times [0, 1] \times [f_{\min}, f_{\max}] \times [P_{\min}, P_{\max}]$
- **代价函数：** $J_i(a_i, \mathbf{a}_{-i}; \mathbf{p})$ 如 III-A-1 节定义

**Nash 均衡条件：** 策略组合 $\mathbf{a}^* = (a_1^*, \ldots, a_U^*)$ 构成 Nash 均衡当且仅当：

$$J_i(a_i^*, \mathbf{a}_{-i}^*; \mathbf{p}) \leq J_i(a_i, \mathbf{a}_{-i}^*; \mathbf{p}), \quad \forall a_i \in A_i, \; \forall i \in \mathcal{U}$$

即任何用户在其他用户策略不变的情况下，单方面偏离均衡策略都不会降低其代价。

#### III-B-2. 迭代最佳响应求解算法

采用**迭代最佳响应（Iterative Best Response, IBR）** 算法求解用户间 Nash 均衡：

**Algorithm 1: Iterative Best Response for Nash Equilibrium**

```
输入：定价向量 p, 初始策略 a^(0), 最大迭代次数 N_IBR, 容差 ε_NE
输出：近似 Nash 均衡 a*

1. for n = 1 to N_IBR do
2.     for i = 1 to U do
3.         a_i^(n) ← arg min_{a_i ∈ A_i} J_i(a_i, a_{-i}^(n-1); p)   // 最佳响应
4.     end for
5.     if ||a^(n) - a^(n-1)||_∞ < ε_NE then
6.         return a^(n)  // 收敛
7.     end if
8. end for
9. return a^(N_IBR)
```

其中步骤 3 的单用户最佳响应问题在给定 $\mathbf{a}_{-i}$ 的条件下退化为单变量优化，可通过解析或梯度下降方法高效求解。

**收敛性保证：** 当代价函数 $J_i$ 具有对角严格凹（diagonally strictly concave）性质时，IBR 保证收敛至唯一 Nash 均衡（Rosen, 1965）。

### III-C. Stackelberg 均衡存在性与唯一性

**定理 1（Stackelberg 均衡存在性与唯一性）：** 在以下条件下，所提双层 Stackelberg 博弈存在唯一均衡 $(\mathbf{p}^*, \mathbf{a}^*)$：

**(i)** 需求函数 $D_k(p_k) = d_k^0 \exp(-\eta_k p_k)$ 严格递减且对数凹；

**(ii)** 成本函数 $C_k(x) = c_k x^2$ 为凸函数；

**(iii)** 用户代价函数 $J_i$ 关于 $a_i$ 凸且关于 $\mathbf{a}_{-i}$ 连续。

**证明思路：**

1. **跟随者最佳响应 $\text{BR}_i(\mathbf{p})$ 的存在性：** 由条件 (iii)，对于固定的 $\mathbf{p}$ 和 $\mathbf{a}_{-i}$，用户 $i$ 的优化问题为凸优化问题，在紧凸策略空间上具有唯一最优解。由 KKT 条件保证最佳响应的存在性。

2. **$\text{BR}_i(\mathbf{p})$ 关于 $\mathbf{p}$ 的连续性：** 由 Berge 最大值定理（Maximum Theorem），在连续参数化的约束集上，极值函数（此处为最佳响应映射）关于参数连续。

3. **领导者问题在 BR 映射下为凸：** 由命题 1，$\Pi_k(p_k)$ 为严格凹函数。将跟随者最佳响应 $\text{BR}(\mathbf{p})$ 代入领导者目标后，利润函数保持严格凹性（因需求函数的对数凹性保证了复合函数的凹性）。

4. **唯一性：** 由严格凹性，领导者的最优定价 $\mathbf{p}^*$ 唯一，进而由跟随者 Nash 均衡的唯一性（源于对角严格凹条件），Stackelberg 均衡唯一。 $\square$

### III-D. Shapley 值协作博弈

#### III-D-1. 联盟博弈建模

在多基站协作场景下，引入 Shapley 值实现联盟收益的公平分配。定义联盟博弈 $(\mathcal{U}, v)$：

- **参与者集合：** $\mathcal{U} = \{1, \ldots, U\}$
- **联盟值函数：** $v : 2^{\mathcal{U}} \to \mathbb{R}$，$v(S)$ 表示联盟 $S$ 中的智能体协同行动时获得的总收益

**联盟值函数定义：**

$$v(S) = \sum_{i \in S} r_i^{\text{coop}}(S) - \sum_{i \in S} r_i^{\text{indep}}$$

其中 $r_i^{\text{coop}}(S)$ 为用户 $i$ 在联盟 $S$ 中协作时获得的通信性能收益，$r_i^{\text{indep}}$ 为用户 $i$ 独立行动时的基线收益。协作收益来源于基站间的负载均衡和干扰协调。

#### III-D-2. Shapley 值定义与性质

用户 $i$ 的 **Shapley 值**定义为其对所有可能联盟的边际贡献的加权平均：

$$\phi_i(v) = \sum_{S \subseteq \mathcal{U} \setminus \{i\}} \frac{|S|! \cdot (U - |S| - 1)!}{U!} \cdot \left[v(S \cup \{i\}) - v(S)\right]$$

Shapley 值满足以下公理化性质（保证分配的合理性）：

1. **效率性（Efficiency）：** $\sum_{i=1}^{U} \phi_i(v) = v(\mathcal{U})$，即所有用户的 Shapley 值之和等于大联盟的总价值。

2. **对称性（Symmetry）：** 若用户 $i$ 和 $j$ 对所有联盟的边际贡献相同，则 $\phi_i = \phi_j$。

3. **虚拟参与者（Null Player）：** 若用户 $i$ 对所有联盟的边际贡献为零，则 $\phi_i = 0$。

4. **可加性（Additivity）：** $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$。

#### III-D-3. 蒙特卡洛近似 Shapley 值

由于精确计算 Shapley 值的复杂度为 $O(2^U)$（需遍历所有联盟子集），当用户数较多时不可行。本文采用**蒙特卡洛排列采样法**结合**对偶变量（Antithetic）方差缩减**技术进行近似：

**Algorithm 2: Monte Carlo Shapley with Antithetic Sampling**

```
输入：用户数 U, 联盟值函数 v(·), 采样数 M, 是否使用对偶 use_antithetic
输出：近似 Shapley 值 φ̂ = (φ̂_1, ..., φ̂_U)

1. 初始化 φ̂ ← 0, n_total ← 0
2. for m = 1 to M do
3.     随机生成排列 π = permutation(1, ..., U)
4.     permutations ← {π}
5.     if use_antithetic then
6.         permutations ← permutations ∪ {reverse(π)}   // 对偶排列
7.     end if
8.     for each σ ∈ permutations do
9.         S ← ∅, v_prev ← 0
10.        for j = 1 to U do
11.            i ← σ(j)
12.            S ← S ∪ {i}
13.            v_curr ← v(S)
14.            φ̂_i ← φ̂_i + (v_curr - v_prev)
15.            v_prev ← v_curr
16.        end for
17.        n_total ← n_total + 1
18.    end for
19. end for
20. φ̂ ← φ̂ / n_total
21. return φ̂
```

**复杂度分析：** 精确 Shapley 值的计算复杂度为 $O(U! \cdot T_v)$（$T_v$ 为联盟值函数一次评估时间），蒙特卡洛近似将其降至 $O(M \cdot U \cdot T_v)$，其中 $M$ 为采样数。对偶变量技术使方差界降低约一半：$\text{Var}(\hat{\phi}_i) \leq V_{\max}^2 / M$。本文取 $M = 128$。

### III-E. EFX 公平分配与 CP-nets 偏好建模

#### III-E-1. EFX (Envy-Free up to any item) 公平性

在资源分配后，引入 EFX 公平性验证层，确保分配满足"无嫉妒至一物"条件：

**EFX 条件：** 对任意用户 $i, j$，令 $A_i$ 为用户 $i$ 的分配包（所获得的基站资源份额），则：

$$v_i(A_i) \geq v_i(A_j \setminus \{g\}), \quad \forall g \in A_j \text{（价值最小的物品）}$$

即用户 $i$ 对自己的分配的评价不低于其对用户 $j$ 的分配移除一个最不值钱物品后的评价。

**EFX 修复机制：** 当 Shapley 值分配违反 EFX 条件时，通过转移支付机制进行修复：

$$\text{deficit}_{i,j} = v_i(A_j \setminus \{g\}) - v_i(A_i)$$

$$\text{transfer}_i \mathrel{+}= \text{deficit}_{i,j} \cdot \tau / 2, \quad \text{transfer}_j \mathrel{-}= \text{deficit}_{i,j} \cdot \tau / 2$$

其中 $\tau = 0.5$ 为转移速率，最大迭代 32 次。转移支付被裁剪到 $[-0.05, 0.05]$ 以保证训练稳定性。

#### III-E-2. CP-nets (Conditional Preference Networks) 偏好建模

引入 CP-nets 为每个用户建模其对不同基站的条件偏好结构。CP-net 是一个有向无环图（DAG），其中每个节点表示一个偏好属性（如时延优先 vs. 能耗优先），边表示条件依赖关系。

**偏好打分函数：** 对于用户 $i$ 选择基站 $k$ 的偏好得分：

$$\text{pref}_{i,k} = w_{\text{dist}} \cdot \frac{1}{d_{i,k} + 1} + w_{\text{load}} \cdot (1 - \rho_k) + w_{\text{sinr}} \cdot \text{norm}(\text{SINR}_{i,k})$$

CP-net 偏好得分被整合至 EFX 的估值函数中，使得 EFX 修复考虑用户的真实偏好。

---

## IV. MADRL 算法设计 (Multi-Agent Deep Reinforcement Learning)

### IV-A. Dec-POMDP 建模

将多用户 MEC 问题建模为**去中心化部分可观测马尔可夫决策过程（Dec-POMDP）**：

$$\mathcal{M} = \langle \mathcal{U}, \mathcal{S}, \{\mathcal{O}_i\}, \{\mathcal{A}_i\}, P, \{R_i\}, \gamma \rangle$$

各元素定义如下：

- **智能体集合** $\mathcal{U} = \{1, \ldots, U\}$
- **全局状态空间** $\mathcal{S}$：包含所有基站队列长度、信道状态、用户位置和任务信息
- **局部观测空间** $\mathcal{O}_i$：智能体 $i$ 仅能观测自身相关的部分状态
- **动作空间** $\mathcal{A}_i$：混合动作空间（离散 + 连续）
- **状态转移函数** $P(s' | s, \mathbf{a})$：环境动力学
- **奖励函数** $R_i(s, \mathbf{a})$：分层奖励
- **折扣因子** $\gamma = 0.99$

### IV-B. 观测空间设计

每个智能体 $i$ 的局部观测向量 $o_i(t)$ 的维度为 $(3K + 5)$，结构如下：

$$o_i(t) = \left[\underbrace{Q_1/Q_{\max}, \ldots, Q_K/Q_{\max}}_{K \text{个基站队列长度（归一化）}}, \; \underbrace{\widetilde{\text{SINR}}_{i,1}, \ldots, \widetilde{\text{SINR}}_{i,K}}_{K \text{个信道 SINR（归一化）}}, \; \underbrace{L_1, \ldots, L_K}_{K \text{个基站 CPU 负载}}, \; \underbrace{e_i, t_i^{\text{ddl}}, d_i, c_i, v_i}_{5 \text{个任务/移动特征}}\right]$$

其中：
- $Q_k / Q_{\max}$：基站 $k$ 的归一化队列长度
- $\widetilde{\text{SINR}}_{i,k}$：归一化 SINR（经 sigmoid 或 min-max 变换）
- $L_k$：基站 $k$ 的 CPU 利用率
- $e_i$：用户 $i$ 的剩余能量（归一化）
- $t_i^{\text{ddl}}$：任务截止时间（归一化）
- $d_i$：任务数据量（归一化）
- $c_i$：任务 CPU 需求（归一化）
- $v_i$：用户移动速度（归一化）

**全局状态：** 中心化 Critic 的输入为全局状态 $s(t)$，包含所有智能体的局部观测拼接，以及博弈均衡提示：

$$s(t) = [o_1(t), \ldots, o_U(t), \mathbf{p}^*(t), \mathbf{a}^*_{\text{eq}}(t)]$$

### IV-C. 混合动作空间设计

每个智能体的动作空间为**混合空间**（Dict 空间），包含一个离散头和一个连续头：

$$a_i = \left\{\text{target}: x_i \in \{0, 1, \ldots, K\}, \quad \text{ratio}: (r_\alpha, r_f, r_P) \in [-1, 1]^3\right\}$$

**离散头（Discrete Head）：** 输出卸载目标选择，0 表示本地处理，$1 \sim K$ 表示卸载至对应基站。

**连续头（Continuous Head）：** 输出三个归一化比例值，通过参数化动作空间映射至物理量：

$$\alpha_i = \frac{r_\alpha + 1}{2} \in [0, 1]$$

$$f_i^{\text{loc}} = f_{\min} + \frac{r_f + 1}{2} \cdot (f_{\max} - f_{\min}) \in [f_{\min}, f_{\max}]$$

$$P_i = P_{\min} + \frac{r_P + 1}{2} \cdot (P_{\max} - P_{\min}) \in [P_{\min}, P_{\max}]$$

### IV-D. 分层奖励函数设计

奖励函数由三个层次组成，分别对应通信性能、协作行为和博弈一致性：

#### IV-D-1. 即时奖励 $r_{\text{imm}}$

$$r_{\text{imm}} = -\left(w_T \cdot \frac{T_i^{\text{total}}}{T_i^{\max}} + w_Q \cdot \frac{T_k^{\text{queue}}}{T_i^{\max}} + w_D \cdot \left(\max\left(0, \frac{T_i^{\text{total}} - T_i^{\max}}{T_i^{\max}}\right)\right)^2 + w_E \cdot \frac{E_i^{\text{total}}}{E_i^{\max}} + \delta_{\text{nn}}\right)$$

其中 $\delta_{\text{nn}}$ 为非最近基站惩罚项：当用户选择非最近基站时施加额外惩罚（典型值 0.02），以引导合理的基站选择。

#### IV-D-2. 协作奖励 $r_{\text{coop}}$

基于 Shapley 值的协作奖励衡量用户对团队协作的贡献：

$$r_{\text{coop}} = \text{clip}\left(\phi_i \cdot \left[\bar{J}^{\text{baseline}} - \bar{J}^{\text{observed}}\right], -1, 1\right)$$

其中 $\bar{J}^{\text{baseline}} = \frac{1}{U} \sum_{i} J_i^{\text{local-only}}$ 为所有用户采用纯本地策略时的平均代价，$\bar{J}^{\text{observed}} = \frac{1}{U} \sum_{i} J_i^{\text{current}}$ 为当前策略下的平均代价。正的协作收益意味着当前的联合策略优于独立策略。

#### IV-D-3. 均衡一致性奖励 $r_{\text{eq}}$

引导策略靠近 Stackelberg 博弈均衡解：

$$r_{\text{eq}} = -\frac{\|\mathbf{a}_i - \mathbf{a}_i^*(s)\|^2}{\dim(\mathcal{A})}$$

其中 $\mathbf{a}_i^*(s)$ 为 Stackelberg 均衡下用户 $i$ 的最优动作（由双层博弈求解器计算），$\dim(\mathcal{A})$ 为动作空间维度。该项确保 RL 策略不会偏离博弈论指导的合理策略空间。

#### IV-D-4. 总奖励与自适应权重

$$R_i = \hat{\alpha} \cdot r_{\text{imm}} + \hat{\beta} \cdot r_{\text{coop}} + \hat{\gamma} \cdot r_{\text{eq}} - \text{Penalty} - \text{Barrier} + r_{\text{fair}}$$

其中 $r_{\text{fair}} = \text{clip}(\text{EFX\_transfer}_i, -0.05, 0.05)$ 为 EFX 公平转移支付。

**自适应权重调节：** 为缓解不同奖励分量间的梯度冲突，引入基于历史相关性的自适应权重机制：

$$\hat{\beta}^{(t)} = \hat{\beta}^{(0)} \cdot \max\left(0.5, \; 1 + \text{corr}(r_{\text{imm}}^{(t-W:t)}, r_{\text{coop}}^{(t-W:t)})\right)$$

$$\hat{\gamma}^{(t)} = \hat{\gamma}^{(0)} \cdot \max\left(0.5, \; 1 + \text{corr}(r_{\text{imm}}^{(t-W:t)}, r_{\text{eq}}^{(t-W:t)})\right)$$

然后重新归一化：$(\hat{\alpha}, \hat{\beta}, \hat{\gamma}) \leftarrow (\hat{\alpha}, \hat{\beta}, \hat{\gamma}) / (\hat{\alpha} + \hat{\beta} + \hat{\gamma})$。

其中 $W = 50$ 为历史窗口长度，$\text{corr}(\cdot, \cdot)$ 为皮尔逊相关系数。当 $r_{\text{imm}}$ 与 $r_{\text{coop}}$ 的负相关性增强时（表明两者冲突），自动降低 $\hat{\beta}$；反之亦然。

初始权重设置为 $(\hat{\alpha}^{(0)}, \hat{\beta}^{(0)}, \hat{\gamma}^{(0)}) = (0.8, 0.1, 0.1)$。

### IV-E. CTDE 架构与 GRPO 算法

#### IV-E-1. 中心化训练-去中心化执行（CTDE）

本文采用 CTDE（Centralized Training with Decentralized Execution）范式：

**中心化 Critic（训练阶段）：** 观测全局状态 $s(t)$（包含所有智能体的局部观测和博弈均衡提示），输出值函数估计：

$$V_\psi(s) : \mathcal{S} \to \mathbb{R}$$

Critic 网络结构包含均衡嵌入模块：

$$s_{\text{critic}} = [s(t), \; \text{EqEncoder}(\mathbf{a}_1^*, \ldots, \mathbf{a}_U^*)]$$

其中 EqEncoder 为一个线性层 $\mathbb{R}^{U \cdot \dim(\mathcal{A})} \to \mathbb{R}^{64}$，将均衡动作嵌入为紧凑表示。

**分散化 Actor（执行阶段）：** 每个智能体 $i$ 仅基于其局部观测 $o_i(t)$ 输出动作：

$$\pi_{\theta_i}(a_i | o_i) : \mathcal{O}_i \to \Delta(\mathcal{A}_i)$$

**网络架构：**
- Actor：$\text{Linear}(|\mathcal{O}_i|, 256) \to \text{ReLU} \to \text{Linear}(256, 128) \to \text{ReLU} \to$ 双头输出
  - 离散头：$\text{Linear}(128, K+1) \to \text{Softmax}$
  - 连续头：$\text{Linear}(128, 3) \to \text{Tanh}$

- Critic：$\text{Linear}(|\mathcal{S}| + 64, 512) \to \text{ReLU} \to \text{Linear}(512, 256) \to \text{ReLU} \to \text{Linear}(256, 1)$

#### IV-E-2. GRPO (Group Relative Policy Optimization) 算法

GRPO 是 PPO 的多智能体扩展，核心思想为**组相对优势估计**，无需独立的 Value 网络：

**组相对优势计算：** 对于每个状态 $s$，采样一组大小为 $G$ 的动作 $\{a_i^{(1)}, \ldots, a_i^{(G)}\}$，计算各动作对应的回报 $\{R^{(1)}, \ldots, R^{(G)}\}$，然后标准化：

$$\hat{A}_i^{(g)} = \frac{R^{(g)} - \text{mean}(\{R^{(j)}\}_{j=1}^G)}{\text{std}(\{R^{(j)}\}_{j=1}^G) + \epsilon}$$

其中 $G = 4$ 为组大小。

**GRPO 策略更新目标：**

$$\mathcal{L}^{\text{GRPO}}(\theta_i) = -\mathbb{E}\left[\min\left(\frac{\pi_{\theta_i}(a_i | o_i)}{\pi_{\theta_i^{\text{old}}}(a_i | o_i)} \hat{A}_i, \; \text{clip}\left(\frac{\pi_{\theta_i}(a_i | o_i)}{\pi_{\theta_i^{\text{old}}}(a_i | o_i)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right)\right]$$

其中 $\epsilon = 0.2$ 为剪切比率。

**熵正则化：**

$$\mathcal{L}^{\text{total}}(\theta_i) = \mathcal{L}^{\text{GRPO}}(\theta_i) - c_{\text{ent}} \cdot H[\pi_{\theta_i}(\cdot | o_i)]$$

其中 $c_{\text{ent}} = 0.01$ 为熵系数，$H[\cdot]$ 为策略分布的熵。

#### IV-E-3. Shapley 值信用分配

在 CTDE 框架中，团队奖励通过 Shapley 值进行信用分配，替代传统的 QMIX/VDN 值分解：

$$r_i^{\text{individual}} = \phi_i \cdot R^{\text{team}} / \sum_j \phi_j$$

当 $\sum_j \phi_j \approx 0$ 时，退化为均匀分配 $r_i^{\text{individual}} = R^{\text{team}} / U$。

### IV-F. 博弈论引导的 Warm-Start 机制

为加速 RL 收敛，利用博弈均衡解进行策略网络的预训练（Warm-Start）：

**Algorithm 3: Game-Theoretic Warm-Start for MADRL**

```
输入：环境 env, Stackelberg 求解器 stack_solver, 预训练轮数 N_warm
输出：预训练策略网络 {π_θ_i}

1. for episode = 1 to N_warm do
2.     s ← env.reset()
3.     p*, {a*_i} ← stack_solver.solve(s)   // 求解 Stackelberg 均衡
4.     for each agent i do
5.         // 监督学习：模仿均衡动作
6.         loss_i ← ||π_θ_i(o_i) - a*_i||²
7.         θ_i ← θ_i - η_warm · ∇loss_i
8.     end for
9.     // 逐步增加探索噪声
10.    noise_scale ← max(0, 1 - episode/N_warm) · σ_init
11. end for
12. // 切换至标准 GRPO 训练
13. for episode = N_warm+1 to N_total do
14.    标准 GRPO 训练循环
15. end for
```

超参数：$N_{\text{warm}} = 1000$，$\eta_{\text{warm}} = \eta \cdot 0.5$（Warm-Start 学习率为标准学习率的一半）。

---

## V. 收敛性与复杂度分析

### V-A. 收敛性分析

**假设 1（奖励有界）：** $|R_i(s, \mathbf{a})| \leq R_{\max} < \infty$，$\forall s, \mathbf{a}$。

**假设 2（Lipschitz 连续性）：** 奖励函数 $R_i$ 和状态转移概率 $P$ 关于 $(s, \mathbf{a})$ 均为 Lipschitz 连续函数，Lipschitz 常数为 $L_R$ 和 $L_P$。

**假设 3（Robbins-Monro 条件）：** 学习率序列 $\{\eta^{(n)}\}$ 满足：

$$\sum_{n=1}^{\infty} \eta^{(n)} = \infty, \quad \sum_{n=1}^{\infty} (\eta^{(n)})^2 < \infty$$

**定理 2（收敛至 $\epsilon$-近似均衡）：** 在假设 1–3 下，所提 GRPO-Shapley 算法的策略序列 $\{\pi_\theta^{(t)}\}$ 满足：

$$\lim_{T \to \infty} \mathbb{E}\left[\delta_{\text{SE}}^{(T)}\right] = 0$$

其中 $\delta_{\text{SE}}^{(T)} = \|\mathbf{a}_\theta^{(T)} - \mathbf{a}^*\|$ 为策略动作与 Stackelberg 均衡动作之间的偏差。

更精确地，存在常数 $C > 0$，使得在有限时间 $T$ 内：

$$\mathbb{E}\left[\delta_{\text{SE}}^{(T)}\right] \leq \epsilon + O\left(\frac{1}{\sqrt{T}}\right)$$

**证明思路：**

1. 由 PPO/GRPO 的剪切目标，策略更新的步长被控制在信赖域内，保证了策略序列的稳定性。

2. Shapley 值信用分配保持了多智能体梯度估计的无偏性（效率性公理保证 $\sum_i \phi_i = v(\mathcal{U})$）。

3. 均衡一致性奖励 $r_{\text{eq}}$ 在策略空间中引入了一个势函数 $\Phi(\pi) = -\sum_i \|\pi_{\theta_i}(\cdot) - \pi_i^*(\cdot)\|^2$，使得 GRPO 的策略梯度方向与势函数梯度方向渐近一致。

4. 结合随机逼近理论（Stochastic Approximation），在 Robbins-Monro 条件下，策略序列几乎必然收敛至势函数的稳定点，即均衡附近。

### V-B. 复杂度分析

#### V-B-1. 时间复杂度（每步）

| 算法模块 | 复杂度 |
|---------|--------|
| Actor 前向传播 | $O(U \cdot |\mathcal{O}| \cdot H)$ |
| Critic 前向传播 | $O(|\mathcal{S}| \cdot H)$ |
| Stackelberg 求解（牛顿迭代） | $O(K \cdot N_{\text{Newton}})$ |
| Nash IBR（内循环） | $O(U \cdot N_{\text{IBR}} \cdot K)$ |
| Monte Carlo Shapley | $O(M \cdot U \cdot T_v)$ |
| EFX 验证与修复 | $O(U^2 \cdot K \cdot N_{\text{EFX}})$ |

其中 $H = 256$ 为隐藏层维度，$N_{\text{Newton}} \leq 100$ 为牛顿迭代上限，$N_{\text{IBR}} \leq 200$ 为 IBR 迭代上限，$M = 128$ 为 Shapley 采样数，$T_v$ 为联盟值一次评估时间，$N_{\text{EFX}} \leq 32$ 为 EFX 修复迭代上限。

#### V-B-2. 总体复杂度对比

| 算法 | 每步时间复杂度 | 空间复杂度 | Shapley 开销 |
|------|--------------|------------|-------------|
| DQN | $O(|\mathcal{S}| \cdot |\mathcal{A}|)$ | $O(B \cdot |\mathcal{S}|)$ | — |
| DDPG | $O(|\mathcal{S}| \cdot |\mathcal{A}|)$ | $O(B \cdot (|\mathcal{S}| + |\mathcal{A}|))$ | — |
| MAPPO | $O(U \cdot |\mathcal{S}| \cdot |\mathcal{A}|)$ | $O(U \cdot |\theta|)$ | — |
| QMIX | $O(U^2 \cdot |\mathcal{S}| \cdot |\mathcal{A}|)$ | $O(U \cdot |\theta| + U^2)$ | — |
| **本文 (GRPO+Game)** | $O(U \cdot |\mathcal{S}| \cdot |\mathcal{A}| + M \cdot U \cdot T_v)$ | $O(U \cdot |\theta| + M \cdot U)$ | $O(M \cdot U \cdot T_v)$ |

其中 $B$ 为 batch size，$U$ 为智能体数，$M$ 为蒙特卡洛采样数。

---

## VI. 仿真设置与参数配置

### VI-A. 仿真环境参数

| 参数类别 | 参数名 | 符号 | 默认值 |
|---------|--------|------|--------|
| **网络拓扑** | 基站数 | $K$ | 3 |
| | 用户数 | $U$ | 3–20 |
| | 基站间距 | — | 15 m |
| | 基站高度 | $h_{\text{BS}}$ | 25 m |
| | 终端高度 | $h_{\text{UT}}$ | 1.5 m |
| **信道参数** | 载波频率 | $f_c$ | 3.5 GHz |
| | 信道带宽 | $B$ | 20 MHz |
| | 噪声功率谱密度 | — | −174 dBm/Hz |
| | 阴影衰落标准差 (LOS) | $\sigma_{\text{SF}}$ | 4 dB |
| | 阴影衰落标准差 (NLOS) | $\sigma_{\text{SF}}$ | 7.82 dB |
| | 衰落类型 | — | Rayleigh |
| **计算参数** | 边缘 CPU 频率 | $F_k$ | 5 GHz |
| | 本地 CPU 频率范围 | $[f_{\min}, f_{\max}]$ | [0.5, 3.0] GHz |
| | 动态功耗系数 | $\kappa_{\text{dyn}}$ | $10^{-27}$ |
| | DVFS 指数 | $\hat{\alpha}$ | 2.7 |
| | 泄漏功耗 | $P_{\text{leak}}$ | 0.01 W |
| **任务参数** | 数据量范围 | $[D_{\min}, D_{\max}]$ | [100, 500] KB |
| | CPU 需求范围 | $[C_{\min}, C_{\max}]$ | [$10^8$, $10^9$] cycles |
| | 截止时间范围 | $[T_{\min}^{\max}, T_{\max}^{\max}]$ | [0.5, 2.0] s |
| **功率参数** | 用户发射功率范围 | $[P_{\min}, P_{\max}]$ | [0.01, 0.5] W |
| | 系统功率上限 | $P_{\text{total}}^{\max}$ | 1.5 W |
| **排队参数** | 稳定性裕度 | $\rho_k^{\max}$ | 0.95 |
| **移动性** | 用户初始半径 | $[r_{\min}, r_{\max}]$ | [8, 30] m |
| | 最大速度 | $v_{\max}$ | 5 m/s |
| **博弈论** | 价格范围 | $[p_{\min}, p_{\max}]$ | [0.1, 2.0] |
| | 价格敏感度 | $\eta_k$ | 0.35 |
| | 边际成本系数 | $c_k$ | 0.12 |
| | 基础需求量 | $d_k^0$ | 1.0 |
| | Shapley 采样数 | $M$ | 128 |
| | EFX 转移速率 | $\tau$ | 0.5 |
| **预算约束** | 能量预算 | $E_i^{\max}$ | 10.0 J |
| | 时延预算 | $T_i^{\max}$ | 2.0 s |

### VI-B. GRPO 训练超参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 折扣因子 | $\gamma$ | 0.99 |
| GAE λ | $\lambda_{\text{GAE}}$ | 0.95 |
| PPO 剪切比率 | $\epsilon$ | 0.2 |
| 熵系数 | $c_{\text{ent}}$ | 0.01 |
| 值函数系数 | $c_{\text{val}}$ | 0.5 |
| GRPO 组大小 | $G$ | 4 |
| 学习率 | $\eta$ | $3 \times 10^{-4}$ |
| 批大小 | $B$ | 64 |
| Rollout 步数 | — | 2048 |
| 更新 Epoch 数 | — | 10 |
| 最大梯度范数 | — | 0.5 |
| 总训练步数 | — | 500,000 |
| 每 Episode 步数 | — | 100 |
| 隐藏层维度 | $H$ | 256 |
| 激活函数 | — | ReLU |
| Warm-Start 步数 | $N_{\text{warm}}$ | 1,000 |
| Warm-Start 学习率比例 | — | 0.5 |
| 奖励权重 | $(\hat{\alpha}, \hat{\beta}, \hat{\gamma})$ | (0.8, 0.1, 0.1) |
| 探索初始 $\epsilon$ | — | 1.0 |
| 探索终止 $\epsilon$ | — | 0.05 |
| 探索衰减步数 | — | 50,000 |

### VI-C. 参数化实验矩阵

| 配置 | $K$ (BS) | $U$ (Users) | 观测维度 | 动作维度 | 预计训练时间 |
|------|----------|-------------|----------|----------|-------------|
| Small | 3 | 6 | ~46 | 4 | ~2h |
| Medium | 5 | 10 | ~71 | 4 | ~8h |
| Large | 10 | 20 | ~126 | 4 | ~24h |

### VI-D. 基线算法对比

| 类别 | 算法 | 特点 |
|------|------|------|
| 启发式 | Greedy, Random, Local-only, Full-offload | 无学习能力 |
| 单智能体 DRL | DQN, DDPG, TD3, SAC | 无多智能体协调 |
| 多智能体 DRL | QMIX, MAPPO (w/o game), MADDPG | 无博弈论层 |
| **本文** | **GRPO + Stackelberg + Shapley** | 完整框架 |

### VI-E. Ablation Study 清单

| ID | 消融项 | 描述 | 预期影响 |
|----|--------|------|----------|
| A1 | w/o Shapley | 用均匀分配替代 Shapley 值 | 公平性下降，收敛变慢 |
| A2 | w/o 博弈初始化 | 随机初始化策略（取消 Warm-Start） | 前期探索效率低 |
| A3 | w/o 移动性 | 用户静止 | 性能上界，验证移动性挑战 |
| A4 | w/o 协作奖励 | 仅用即时奖励 $r_{\text{imm}}$ | 自私策略，系统效率低 |
| A5 | w/o 排队模型 | 去除 M/M/1 排队时延 | 高负载性能估计失真 |
| A6 | w/o 均衡奖励 | 去除 $r_{\text{eq}}$ | 策略偏离均衡 |
| A7 | w/o DVFS | 使用理想 $\kappa f^2 C$ 模型 | 能耗估计偏乐观 |

---

## VII. 理论贡献总结

本文的主要理论贡献包括：

1. **双层博弈-RL 融合框架：** 首次将 Stackelberg 定价博弈（上层）、用户间 Nash 均衡（下层）与多智能体 GRPO 算法有机融合，博弈均衡解既作为 RL 策略的初始化点（Warm-Start），又作为训练过程中的引导信号（均衡一致性奖励 $r_{\text{eq}}$）。

2. **Shapley 值信用分配机制：** 将 Shapley 值引入多智能体 RL 的信用分配问题，替代传统的 QMIX/VDN 值分解方法，并通过蒙特卡洛对偶采样降低计算复杂度。Shapley 值的公理化性质为分配的公平性和效率性提供了理论保证。

3. **EFX 公平约束与 CP-nets 偏好：** 在 Shapley 分配的基础上引入 EFX 公平性验证和 CP-nets 条件偏好建模，确保资源分配不仅高效且公平，并尊重用户的异质偏好结构。

4. **完整的系统模型堆栈：** 系统模型涵盖 3GPP TR 38.901 信道（LOS/NLOS 概率路径损耗 + 多用户干扰 SINR）、非理想 DVFS 能耗（含泄漏功耗和工艺相关指数）、M/M/1 排队时延、高斯-马尔可夫移动性模型，以及混合动作空间的约束投影-惩罚-障碍处理机制。

5. **收敛性理论保证：** 证明了在标准假设下，所提 GRPO-Shapley 算法收敛至 $\epsilon$-近似 Stackelberg 均衡，收敛速率为 $O(1/\sqrt{T})$。

---

## 附录 A. 仿真平台技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| RL 框架 | Stable-Baselines3 / Tianshou | PPO/GRPO 快速原型 |
| 环境接口 | Gymnasium (gymnasium.Env) | 标准化 RL 环境 API |
| 信道模型 | 3GPP TR 38.901 (自实现) | UMi/UMa/RMa 场景 |
| 排队模型 | M/M/1 (自实现) | 排队论基础模块 |
| 博弈论求解 | 牛顿法 + IBR (自实现) | Stackelberg + Nash |
| 衰落模型 | Rayleigh/Rician/Nakagami (自实现) | 信道衰落库 |
| 移动性模型 | Gaussian-Markov (自实现) | 用户移动轨迹 |
| 对比基线 | DQN / DDPG / TD3 / SAC / MAPPO / QMIX / MADDPG | 全面对比 |

## 附录 B. 支持的 RL 算法列表

本仿真平台支持以下算法与环境的对接：

| 算法类型 | 算法名称 | 适用场景 |
|---------|---------|---------|
| 单智能体 On-Policy | PPO, TRPO, A3C | 单用户基线 |
| 单智能体 Off-Policy | DQN (DDQN), DDPG, TD3, SAC | 单用户基线 |
| 多智能体值分解 | QMIX, VDN, IQL | 协作多智能体 |
| 多智能体策略梯度 | MAPPO, IPPO, MADDPG, MATD3, COMA | 协作/竞争多智能体 |
| 本文算法 | GRPO (+ Stackelberg + Shapley) | 博弈-RL 融合 |
| 偏好对齐 | SimPO | 人类偏好对齐扩展 |

---

*文档结束 — 初稿版本 v1.0*