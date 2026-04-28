# 基于博弈论引导的多智能体深度强化学习移动边缘计算联合优化：系统模型与理论框架

> **理论模型定稿**
>
> 框架：多基站协作Stackelberg-Nash双层博弈 + GRPO-Shapley多智能体强化学习
>
> 版本：v3.0 — 2026-04-27
>
> 修订说明：在v2.0修订稿基础上，回溯v1.0初稿并补全所有遗漏内容，包括：下行传输速率建模说明、移动性模型位置扰动项与显式初始化公式、IBR收敛性理论保证、Shapley公理完整描述、训练超参数补全、消融实验预期影响分析、附录信息完善等。

---

## 符号总表 (Table of Notation)

为确保全文数学表述的自洽性，以下按模块分组给出全部核心符号的定义。所有向量默认为列向量，以粗体小写字母标记；矩阵以粗体大写字母标记；集合以花体字母标记。若同一字母在不同语境下复用，一律以上标或下标加以区分（参见"易混符号辨析"栏）。

### 网络拓扑与索引

| 符号 | 定义 | 取值范围 |
|------|------|----------|
| $\mathcal{K}$ | 基站（BS）集合 | $\mathcal{K} = \{1, 2, \ldots, K\}$ |
| $\mathcal{U}$ | 用户设备（UE）集合 | $\mathcal{U} = \{1, 2, \ldots, U\}$ |
| $K$ | 基站总数 | 典型值 3–10 |
| $U$ | 用户设备总数（亦为智能体总数$N$） | 典型值 6–20 |
| $t$ | 离散时隙索引 | $t \in \{0, 1, \ldots, T\}$ |
| $\Delta t$ | 时隙持续时间 | 秒 |
| $\mathbf{q}_k$ | 基站 $k$ 的三维坐标 | $(x_k^{\text{BS}}, y_k^{\text{BS}}, h_{\text{BS}})$ |
| $\mathbf{u}_i(t)$ | 用户 $i$ 在时隙 $t$ 的位置 | $(x_i(t), y_i(t), h_{\text{UT}})$ |

### 任务参数

| 符号 | 定义 | 单位 |
|------|------|------|
| $\mathcal{J}_i(t)$ | 用户 $i$ 在时隙 $t$ 的计算任务 | 四元组 $(D_i, C_i, T_i^{\max}, D_i^{\text{res}})$ |
| $D_i$ | 任务输入数据量 | bits |
| $C_i$ | 任务所需 CPU 周期数 | cycles |
| $T_i^{\max}$ | 任务最大容忍时延（截止时间） | 秒 |
| $D_i^{\text{res}}$ | 计算结果回传数据量 | bits |

### 决策变量

| 符号 | 定义 | 取值域 | 易混符号辨析 |
|------|------|--------|-------------|
| $x_{i,k}$ | 用户 $i$ 是否卸载至基站 $k$ 的二元指示 | $\{0, 1\}$ | — |
| $\alpha_i$ | 用户 $i$ 的卸载比例（$0$=全本地，$1$=全卸载） | $[0, 1]$ | 区别于奖励权重 $\omega_\alpha$ |
| $f_i^{\text{loc}}$ | 用户 $i$ 本地 CPU 频率 | $[f_{\min}, f_{\max}]$ Hz | 区别于边缘频率 $f_{i,k}^{\text{edge}}$ |
| $f_{i,k}^{\text{edge}}$ | 基站 $k$ 分配给用户 $i$ 的 CPU 频率 | Hz | — |
| $P_i$ | 用户 $i$ 的上行发射功率（大写） | $[P_{\min}, P_{\max}]$ W | 区别于定价 $p_k$（小写） |
| $p_k$ | 基站 $k$ 的服务单价（小写） | $[p_{\min}, p_{\max}]$ | 区别于功率 $P_i$（大写） |

### 信道与通信参数

| 符号 | 定义 | 单位 |
|------|------|------|
| $B$ | 系统信道带宽 | Hz |
| $f_c$ | 载波频率 | GHz |
| $h_{i,k}$ | 用户 $i$ 至基站 $k$ 的小尺度衰落信道系数 | 复数 |
| $g_{i,k}$ | 用户 $i$ 至基站 $k$ 的综合信道增益（含路径损耗） | 线性 |
| $d_{i,k}$ | 用户 $i$ 至基站 $k$ 的二维距离 | m |
| $d_{i,k}^{\text{3D}}$ | 三维距离 $\sqrt{d_{i,k}^2 + (h_{\text{BS}} - h_{\text{UT}})^2}$ | m |
| $\text{PL}(\cdot)$ | 路径损耗函数 | dB |
| $\sigma^2$ | 加性高斯白噪声功率 | W |
| $\Gamma_{i,k}$ | 用户 $i$ 在基站 $k$ 的信干噪比（SINR） | 线性 |
| $R_{i,k}$ | 用户 $i$ 至基站 $k$ 的上行传输速率 | bps |
| $R_{i,k}^{\text{dl}}$ | 用户 $i$ 从基站 $k$ 的下行回传速率 | bps |
| $h_{\text{BS}}$ | 基站天线高度 | m（典型值 25） |
| $h_{\text{UT}}$ | 用户终端天线高度 | m（典型值 1.5） |

### 排队与计算参数

| 符号 | 定义 | 单位 | 易混符号辨析 |
|------|------|------|-------------|
| $\lambda_k^{\text{arr}}$ | 基站 $k$ 的等效任务到达率 | bits/s | 区别于权衡系数 $\lambda$ |
| $\mu_k$ | 基站 $k$ 的服务率 | bits/s | — |
| $\rho_k$ | 基站 $k$ 的利用率 $\lambda_k^{\text{arr}} / \mu_k$ | 无量纲 | 区别于折扣因子 $\gamma$ |
| $F_k$ | 基站 $k$ MEC 服务器总 CPU 能力 | cycles/s | — |
| $\kappa$ | 动态功耗系数（有效开关电容） | $\approx 10^{-27}$ | — |
| $\beta$ | DVFS 工艺相关指数 | $[2.5, 3.0]$, 典型 2.7 | 区别于奖励权重 $\omega_\beta$ |
| $P_{\text{leak}}$ | 泄漏功耗 | W | — |

### 博弈论参数

| 符号 | 定义 |
|------|------|
| $\phi_i$ | 用户 $i$ 的 Shapley 值 |
| $\Phi(\cdot)$ | 势函数（Potential Function） |
| $v(\mathcal{S})$ | 联盟 $\mathcal{S} \subseteq \mathcal{U}$ 的特征函数（联盟值） |
| $\eta_k$ | 基站 $k$ 的价格弹性参数 |
| $D_k(p_k)$ | 基站 $k$ 在定价 $p_k$ 下的需求函数 |
| $c_k$ | 基站 $k$ 的边际成本系数 |
| $d_k^0$ | 基站 $k$ 的基础需求量 |
| $\Pi_k(\cdot)$ | 基站 $k$ 的利润函数 |

### RL 算法参数

| 符号 | 定义 | 易混符号辨析 |
|------|------|-------------|
| $\pi_{\theta_i}$ | 智能体 $i$ 的参数化策略 | — |
| $o_i(t)$ | 智能体 $i$ 在时隙 $t$ 的局部观测 | — |
| $s(t)$ | 全局状态 | — |
| $a_i(t)$ | 智能体 $i$ 的动作 | — |
| $r_i(t)$ | 智能体 $i$ 在时隙 $t$ 的即时奖励 | — |
| $\gamma$ | 折扣因子 | 区别于利用率 $\rho_k$ |
| $\epsilon$ | PPO/GRPO 剪切比率 | 区别于收敛精度 $\varepsilon$ |
| $(\omega_\alpha, \omega_\beta, \omega_\gamma)$ | 分层奖励权重 | 区别于卸载比例 $\alpha_i$ |

---

## I. 系统模型 (System Model)

### I-A. 网络架构 (Network Architecture)

本文考察一个由 $K$ 个异构基站与 $U$ 个移动用户设备构成的多接入边缘计算（Multi-access Edge Computing, MEC）网络。该网络遵循 ETSI MEC 参考架构，每个基站 $k \in \mathcal{K}$ 配备一台边缘服务器，其计算能力以最大 CPU 频率 $F_k$（cycles/s）刻画。各基站按照预设拓扑部署于服务区域内，其三维坐标记为 $\mathbf{q}_k = (x_k^{\text{BS}}, y_k^{\text{BS}}, h_{\text{BS}})$，其中天线挂高 $h_{\text{BS}} = 25$ m 符合 3GPP 城市微蜂窝（UMi）部署规范。相邻基站间距典型值为 15 m，构成密集覆盖的小蜂窝网络。

用户设备 $i \in \mathcal{U}$ 在二维地平面内移动，其在时隙 $t$ 的位置记为 $\mathbf{u}_i(t) = (x_i(t), y_i(t), h_{\text{UT}})$，其中终端天线高度 $h_{\text{UT}} = 1.5$ m。每个用户设备具有有限的本地计算能力 $f_i^{\text{loc}} \in [f_{\min}, f_{\max}]$、能量预算 $E_i^{\max}$ 以及最大发射功率约束 $P_i \leq P_{\max}$。

系统运行在离散时隙架构下：整个运行周期划分为 $T$ 个等长时隙，单时隙持续 $\Delta t$ 秒。在每个时隙 $t$ 内，用户 $i$ 独立生成一个计算密集型或时延敏感型任务，以四元组表征：

$$\mathcal{J}_i(t) = \left(D_i(t),\; C_i(t),\; T_i^{\max}(t),\; D_i^{\text{res}}(t)\right)$$

其中各分量分别表示输入数据量、所需 CPU 周期数、最大容忍时延以及结果数据量。任务到达过程建模为泊松过程（到达率 $\lambda_{\text{arr}}$），各参数独立采样自均匀分布：$D_i(t) \sim \mathcal{U}[D_{\min}, D_{\max}]$，$C_i(t) \sim \mathcal{U}[C_{\min}, C_{\max}]$，$T_i^{\max}(t) \sim \mathcal{U}[T_{\min}^{\max}, T_{\max}^{\max}]$。典型参数取值为 $D_{\min} = 100$ KB，$D_{\max} = 500$ KB，$C_{\min} = 10^8$ cycles，$C_{\max} = 10^9$ cycles，$T_{\min}^{\max} = 0.5$ s，$T_{\max}^{\max} = 2.0$ s。

面对上述任务，每个用户需做出如下四维联合决策：

**(D1) 卸载目标选择。** 选择本地处理（令 $x_{i,k} = 0, \forall k$）或卸载至某一基站 $k$（令 $x_{i,k} = 1$），且每个任务至多卸载至一个基站，即 $\sum_{k \in \mathcal{K}} x_{i,k} \leq 1$。

**(D2) 部分卸载比例。** 确定卸载比例 $\alpha_i \in [0, 1]$，其中 $\alpha_i = 0$ 表示全部本地计算，$\alpha_i = 1$ 表示全部卸载至边缘。

**(D3) 计算资源配置。** 选择本地 CPU 频率 $f_i^{\text{loc}} \in [f_{\min}, f_{\max}]$，边缘侧频率 $f_{i,k}^{\text{edge}}$ 由基站根据资源约束统一调度。

**(D4) 传输功率控制。** 选择上行传输功率 $P_i \in [P_{\min}, P_{\max}]$，以平衡传输速率与能耗。

### I-B. 通信模型 (Communication Model)

#### I-B-1. 3GPP TR 38.901 路径损耗模型

本文采用 3GPP TR 38.901 规范定义的 UMi Street Canyon 信道模型，该模型同时涵盖视距（LOS）与非视距（NLOS）传播状态，并通过概率函数在二者之间进行统计切换。

**定义 1（LOS 概率函数）。** 用户 $i$ 到基站 $k$ 之间存在 LOS 路径的概率为

$$\mathbb{P}_{\text{LOS}}(d_{i,k}) = \min\!\left(1, \frac{18}{d_{i,k}}\right) \cdot \left(1 - e^{-d_{i,k}/36}\right) + e^{-d_{i,k}/36}$$

其中 $d_{i,k} = \|\mathbf{u}_i(t) - \mathbf{q}_k\|_2 \geq 1$ m 为二维水平距离。当用户与基站极为接近时（$d_{i,k} \to 0^+$），$\mathbb{P}_{\text{LOS}} \to 1$，符合近场必然视距的物理直觉；当距离增大时概率单调递减，反映遮挡概率的增长。

**路径损耗公式。** 令三维距离 $d_{i,k}^{\text{3D}} = \sqrt{d_{i,k}^2 + (h_{\text{BS}} - h_{\text{UT}})^2}$，则 LOS 与 NLOS 路径损耗分别为

$$\text{PL}_{\text{LOS}}(d_{i,k}) = 32.4 + 21 \cdot \log_{10}(d_{i,k}^{\text{3D}}) + 20 \cdot \log_{10}(f_c) \quad \text{[dB]}$$

$$\text{PL}_{\text{NLOS}}(d_{i,k}) = 35.3 \cdot \log_{10}(d_{i,k}^{\text{3D}}) + 22.4 + 21.3 \cdot \log_{10}(f_c) - 0.3 \cdot (h_{\text{UT}} - 1.5) \quad \text{[dB]}$$

其中 $f_c$ 为载波频率（GHz），本文取 $f_c = 3.5$ GHz。两种状态经概率加权后得到综合路径损耗：

$$\text{PL}(d_{i,k}) = \mathbb{P}_{\text{LOS}}(d_{i,k}) \cdot \text{PL}_{\text{LOS}}(d_{i,k}) + \left(1 - \mathbb{P}_{\text{LOS}}(d_{i,k})\right) \cdot \text{PL}_{\text{NLOS}}(d_{i,k})$$

**大尺度衰落。** 在路径损耗基础上叠加对数正态阴影衰落 $\xi_{i,k} \sim \mathcal{N}(0, \sigma_{\text{SF}}^2)$（dB 域），其中 LOS 场景 $\sigma_{\text{SF}} = 4$ dB，NLOS 场景 $\sigma_{\text{SF}} = 7.82$ dB。

**小尺度衰落。** 多径效应采用瑞利衰落模型刻画：信道系数 $h_{i,k} \sim \mathcal{CN}(0, 1)$，对应信道功率增益 $|h_{i,k}|^2 \sim \text{Exp}(1)$。在存在较强 LOS 分量的近场场景中，可进一步扩展为莱斯（Rician）衰落模型：

$$h_{i,k} = \sqrt{\frac{K_R}{K_R + 1}} \cdot e^{j\varphi_0} + \sqrt{\frac{1}{K_R + 1}} \cdot h_{i,k}^{\text{NLOS}}$$

其中 $K_R$ 为莱斯 K 因子（典型值 $K_R = 3$），$\varphi_0$ 为 LOS 分量初始相位，$h_{i,k}^{\text{NLOS}} \sim \mathcal{CN}(0, 1)$。

#### I-B-2. 多用户干扰与 SINR 模型

当多个用户同时向同一基站卸载任务时，上行链路产生共道干扰（Co-channel Interference）。用户 $i$ 在其目标基站 $k$ 处的接收 SINR 定义为

$$\Gamma_{i,k} \triangleq \frac{P_i \cdot g_{i,k}}{\displaystyle\sum_{j \in \mathcal{U}_k \setminus \{i\}} P_j \cdot g_{j,k} \;+\; \sigma^2}$$

其中 $g_{i,k} = |h_{i,k}|^2 \cdot 10^{-\text{PL}(d_{i,k})/10}$ 为综合信道增益（包含路径损耗、阴影衰落与小尺度衰落），$\mathcal{U}_k = \{i \in \mathcal{U} : x_{i,k} = 1\}$ 为选择卸载至基站 $k$ 的用户子集。热噪声功率

$$\sigma^2 = k_B T_0 B \quad \text{[W]}$$

其等效值为 $\sigma^2_{\text{dBm}} = -174 + 10\log_{10}(B)$ dBm，其中 $k_B = 1.38 \times 10^{-23}$ J/K 为玻尔兹曼常数，$T_0 = 290$ K 为参考噪声温度，$B = 20$ MHz 为信道带宽。

**上行传输速率（Shannon界）**

$$R_{i,k} = B \cdot \log_2\!\left(1 + \Gamma_{i,k}\right) \quad \text{[bps]}$$

**下行传输速率。** 结果回传速率 $R_{i,k}^{\text{dl}}$ 的建模方式与上行类似，采用相同的 Shannon 容量公式。由于基站发射功率通常远大于用户设备发射功率，下行 SINR 一般显著高于上行，因此下行传输时延在总时延中占比较小。在仿真中，下行速率可采用与上行相同的 Shannon 公式计算，或在基站功率充裕的假设下简化为固定高速率处理。

**注记 1。** 本模型中传输速率 $R_{i,k}$ 通过 SINR 耦合了所有用户的功率决策 $\{P_j\}_{j \in \mathcal{U}_k}$，构成了下层 Nash 博弈中策略交互的物理基础——任一用户提高发射功率虽可改善自身速率，但会恶化其他用户的 SINR，这一"利己损人"结构正是非合作博弈的典型特征。

#### I-B-3. RIS 辅助信道模型（扩展模块）

若系统部署配备 $M$ 个反射单元的可重构智能超表面（Reconfigurable Intelligent Surface, RIS），则用户 $i$ 至基站 $k$ 的等效信道系数扩展为

$$\tilde{h}_{i,k} = h_{i,k}^{\text{direct}} + \mathbf{h}_{i,\text{R}}^{H} \cdot \boldsymbol{\Theta} \cdot \mathbf{h}_{\text{R},k}$$

其中 $h_{i,k}^{\text{direct}}$ 为直射链路信道系数，$\mathbf{h}_{i,\text{R}} \in \mathbb{C}^M$ 为用户至 RIS 的级联信道向量，$\mathbf{h}_{\text{R},k} \in \mathbb{C}^M$ 为 RIS 至基站的信道向量，$(\cdot)^{H}$ 表示共轭转置。RIS 反射矩阵

$$\boldsymbol{\Theta} = \text{diag}\!\left(\beta_1 e^{j\theta_1},\; \beta_2 e^{j\theta_2},\; \ldots,\; \beta_M e^{j\theta_M}\right)$$

被动 RIS 场景下 $\beta_m = 1, \forall m$，$\theta_m \in [0, 2\pi)$ 为可优化的相位偏移；主动 RIS 场景下增益系数 $\beta_m \in [0, \beta_{\max}]$ 亦可连续调节，但受总功率约束 $\sum_{m=1}^{M} |\beta_m|^2 \leq P_{\text{RIS}}$。引入 RIS 后，决策空间增加相位优化变量 $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_M)$，该扩展将在后续作为可选增强模块处理。

### I-C. 计算模型 (Computation Model)

#### I-C-1. 本地计算

用户 $i$ 将比例为 $(1 - \alpha_i)$ 的任务留在本地设备上以 CPU 频率 $f_i^{\text{loc}}$ 执行，对应的本地计算时延为

$$T_i^{\text{loc}} = \frac{(1 - \alpha_i) \cdot C_i}{f_i^{\text{loc}}}$$

#### I-C-2. 非理想 DVFS 能耗模型

已有文献广泛采用的理想 DVFS 能耗模型 $E = \kappa f^2 C$（对应 $\beta = 3$ 且忽略泄漏功耗）在两个方面过度简化了实际 CMOS 电路的能耗特性：其一，动态功耗与频率的关系因工艺节点差异而偏离理想二次方律；其二，泄漏功耗在低频区间占比显著，不可忽略。为此，本文引入更为精确的非理想 DVFS 能耗模型：

$$E_i^{\text{loc}} = \left[\kappa \cdot \left(f_i^{\text{loc}}\right)^{\beta - 1} + \frac{P_{\text{leak}}}{f_i^{\text{loc}}}\right] \cdot (1 - \alpha_i) \cdot C_i$$

其中 $\beta \in [2.5, 3.0]$ 为与 CMOS 工艺节点相关的频率-功耗缩放指数（本文取 $\beta = 2.7$），$\kappa \approx 10^{-27}$ 为有效开关电容系数，$P_{\text{leak}} = 0.01$ W 为泄漏功耗。

**命题 0（最优本地频率）。** 给定时延-能耗权衡参数 $\lambda > 0$，最小化复合代价 $\mathcal{C}(f) = E(f) + \lambda \cdot T(f)$ 的最优频率具有闭式解

$$f_i^{\star} = \left(\frac{P_{\text{leak}} + \lambda}{(\beta - 1) \cdot \kappa}\right)^{1/\beta}$$

**证明。** 将代价函数展开为 $\mathcal{C}(f) = \kappa f^{\beta-1} C + (P_{\text{leak}} + \lambda) C f^{-1}$。对 $f$ 求一阶导并令其为零：

$$\frac{d\mathcal{C}}{df} = (\beta - 1) \kappa f^{\beta - 2} C - (P_{\text{leak}} + \lambda) C f^{-2} = 0$$

整理得 $f^{\beta} = (P_{\text{leak}} + \lambda) / [(\beta - 1)\kappa]$，即 $f^{\star} = [(P_{\text{leak}} + \lambda) / ((\beta - 1)\kappa)]^{1/\beta}$。二阶充分条件 $d^2\mathcal{C}/df^2 = (\beta-1)(\beta-2)\kappa f^{\beta-3} C + 2(P_{\text{leak}} + \lambda) C f^{-3} > 0$ 在 $f > 0, \beta > 2$ 时恒成立，故 $f^{\star}$ 为全局最优。实际频率受硬件约束截断为 $f_i^{\text{loc}} = \text{clip}(f_i^{\star}, f_{\min}, f_{\max})$，其中 $f_{\min} = 0.5$ GHz，$f_{\max} = 3.0$ GHz。 $\blacksquare$

#### I-C-3. 边缘计算模型

当用户 $i$ 将比例为 $\alpha_i$ 的任务卸载至目标基站 $k$ 时，边缘处理的总时延由以下四个分量构成：

**上行传输时延：** $T_i^{\text{ul}} = \alpha_i D_i / R_{i,k}$

**排队等待时延：** $T_k^{\text{queue}}$（详见 I-C-4 节 M/M/1 模型）

**边缘计算时延：** $T_i^{\text{exec}} = \alpha_i C_i / f_{i,k}^{\text{edge}}$

**下行回传时延：** $T_i^{\text{dl}} = \alpha_i D_i^{\text{res}} / R_{i,k}^{\text{dl}}$

相应的边缘传输能耗为

$$E_i^{\text{edge}} = P_i \cdot T_i^{\text{ul}} = \frac{P_i \cdot \alpha_i \cdot D_i}{R_{i,k}}$$

#### I-C-4. M/M/1 排队时延模型

现有 MEC 文献中普遍忽略服务器端的排队效应，隐含地假设边缘服务器可以即时处理所有到达的任务。这一假设在低负载条件下尚可接受，但当多用户同时向同一基站卸载任务时，排队时延将成为系统性能的瓶颈。为更真实地刻画这一现象，本文将每台 MEC 服务器建模为经典的 M/M/1 排队系统。

**定义 2（到达率与服务率）。** 基站 $k$ 的等效数据到达率定义为所有卸载至该基站的用户在单位时间内注入的总数据量：

$$\lambda_k^{\text{arr}} = \sum_{i \in \mathcal{U}_k} \frac{\alpha_i \cdot D_i}{\Delta t}$$

服务率为 MEC 服务器单位时间内可处理的等效数据量：

$$\mu_k = \frac{F_k}{\bar{C} / \bar{D}}$$

其中 $\bar{C} = \mathbb{E}[C_i]$ 和 $\bar{D} = \mathbb{E}[D_i]$ 分别为任务 CPU 需求和数据量的统计均值，$\bar{C}/\bar{D}$（cycles/bit）表征平均计算密度。

**利用率与稳定性条件。** 基站 $k$ 的利用率为

$$\rho_k = \frac{\lambda_k^{\text{arr}}}{\mu_k}, \quad \rho_k < 1 \;\text{（稳定性条件）}$$

在仿真实现中设置利用率上限 $\rho_k^{\max} = 0.95$ 以避免数值发散，即 $\rho_k = \text{clip}(\lambda_k^{\text{arr}} / \mu_k, 0, \rho_k^{\max})$。

**排队时延与系统时延。** 由 M/M/1 排队论：

$$T_k^{\text{queue}} = \frac{\rho_k}{\mu_k (1 - \rho_k)}, \quad T_k^{\text{system}} = \frac{1}{\mu_k (1 - \rho_k)}$$

**队列长度动态。** 基站 $k$ 的队列长度在时隙间按下式更新：

$$Q_k(t+1) = \max\!\left(Q_k(t) + \lambda_k^{\text{arr}}(t) - \mu_k,\; 0\right)$$

该动态过程为 RL 环境的状态转移提供了时序相关性。

#### I-C-5. 总时延与总能耗

由于本地处理与边缘处理的子任务可以并行执行，用户 $i$ 的总时延取二者的最大值：

$$T_i^{\text{total}} = \max\!\left(T_i^{\text{loc}},\; T_i^{\text{ul}} + T_k^{\text{queue}} + T_i^{\text{exec}} + T_i^{\text{dl}}\right) \tag{1}$$

当系统采用更保守的顺序执行假设时（用于理论分析上界），总时延为加权求和：

$$T_i^{\text{total,seq}} = (1 - \alpha_i) T_i^{\text{loc}} + \alpha_i \left(T_i^{\text{ul}} + T_k^{\text{queue}} + T_i^{\text{exec}} + T_i^{\text{dl}}\right) \tag{2}$$

本文仿真中采用式 (1) 的并行模型，理论分析中酌情使用式 (2) 的上界。

用户 $i$ 的总能耗为本地计算能耗与上行传输能耗之和：

$$E_i^{\text{total}} = \left[\kappa (f_i^{\text{loc}})^{\beta-1} + \frac{P_{\text{leak}}}{f_i^{\text{loc}}}\right] (1-\alpha_i) C_i + \frac{P_i \alpha_i D_i}{R_{i,k}} \tag{3}$$

### I-D. 移动性模型 (Mobility Model)

用户移动性采用高斯-马尔可夫（Gauss-Markov）模型，该模型相较于纯随机游走具有可调的时间相关性，更贴近真实用户移动轨迹的平滑特征。

**速度更新：**

$$v_i(t+1) = \zeta \cdot v_i(t) + (1 - \zeta) \cdot \bar{v} + \sigma_v \sqrt{1 - \zeta^2} \cdot w_i(t), \quad w_i(t) \sim \mathcal{N}(0, 1)$$

其中 $\zeta \in [0, 1]$ 为记忆参数（$\zeta \to 1$ 退化为匀速运动，$\zeta \to 0$ 退化为无记忆随机游走），$\bar{v}$ 为均值速度，$\sigma_v$ 为速度标准差。

**位置更新：**

$$\mathbf{u}_i(t+1) = \mathbf{u}_i(t) + \mathbf{v}_i(t) \cdot \Delta t + \boldsymbol{\epsilon}_i(t)$$

其中 $\boldsymbol{\epsilon}_i(t) \sim \mathcal{N}(\mathbf{0}, \sigma_{\text{mob}}^2 \mathbf{I}_2)$ 为位置扰动项，用以捕捉速度模型未能完全描述的微尺度位置不确定性（如行人随机步态偏移）。

**边界约束。** 当用户移动至部署区域边界时，采用弹性反射策略将位置限制在 $[\mathbf{u}_{\min}, \mathbf{u}_{\max}]$ 内：

$$\mathbf{u}_i(t) = \text{clip}(\mathbf{u}_i(t), \mathbf{u}_{\min}, \mathbf{u}_{\max})$$

**初始分布。** 用户初始位置在极坐标下随机采样后转换为笛卡尔坐标：

$$\theta_i^{(0)} \sim \mathcal{U}[0, 2\pi), \quad r_i^{(0)} \sim \mathcal{U}[r_{\min}, r_{\max}]$$

$$\mathbf{u}_i(0) = \left(r_i^{(0)} \cos \theta_i^{(0)},\; r_i^{(0)} \sin \theta_i^{(0)},\; h_{\text{UT}}\right)$$

其中 $r_{\min} = 8$ m，$r_{\max} = 30$ m。最大用户速度 $v_{\max} = 5$ m/s（对应步行至慢跑场景）。

---

## II. 问题建模 (Problem Formulation)

### II-A. 联合优化目标

本文的核心目标是在满足 QoS 约束的前提下，联合优化所有用户的卸载决策 $\mathbf{x}$、卸载比例 $\boldsymbol{\alpha}$、本地频率 $\mathbf{f}$、发射功率 $\mathbf{P}$ 以及基站定价 $\mathbf{p}$，以最小化系统加权代价。

**问题 (P1)：**

$$\min_{\mathbf{x},\, \boldsymbol{\alpha},\, \mathbf{f},\, \mathbf{P},\, \mathbf{p}} \quad \frac{1}{U} \sum_{i=1}^{U} \Bigg[ \underbrace{\omega_T \cdot \frac{T_i^{\text{total}}}{T_i^{\max}}}_{\text{归一化时延}} + \underbrace{\omega_E \cdot \frac{E_i^{\text{total}}}{E_i^{\max}}}_{\text{归一化能耗}} + \underbrace{\omega_D \cdot \left(\max\!\left(0,\; \frac{T_i^{\text{total}} - T_i^{\max}}{T_i^{\max}}\right)\right)^{\!2}}_{\text{截止期限违约惩罚}} \Bigg]$$

**约束条件：**

$$\text{C1 (卸载唯一性):} \quad \sum_{k=1}^{K} x_{i,k} \leq 1, \qquad \forall\, i \in \mathcal{U}$$

$$\text{C2 (边缘CPU容量):} \quad \sum_{i \in \mathcal{U}_k} f_{i,k}^{\text{edge}} \leq F_k, \qquad \forall\, k \in \mathcal{K}$$

$$\text{C3 (时延QoS):} \quad T_i^{\text{total}} \leq T_i^{\max}, \qquad \forall\, i \in \mathcal{U}$$

$$\text{C4 (能量预算):} \quad E_i^{\text{total}} \leq E_i^{\max}, \qquad \forall\, i \in \mathcal{U}$$

$$\text{C5 (发射功率范围):} \quad P_{\min} \leq P_i \leq P_{\max}, \qquad \forall\, i \in \mathcal{U}$$

$$\text{C6 (系统功率上限):} \quad \sum_{i=1}^{U} P_i \leq P_{\text{total}}^{\max}$$

$$\text{C7 (队列稳定性):} \quad \rho_k < 1, \qquad \forall\, k \in \mathcal{K}$$

$$\text{C8 (卸载比例):} \quad 0 \leq \alpha_i \leq 1, \qquad \forall\, i \in \mathcal{U}$$

$$\text{C9 (CPU频率范围):} \quad f_{\min} \leq f_i^{\text{loc}} \leq f_{\max}, \qquad \forall\, i \in \mathcal{U}$$

其中代价权重 $(\omega_T, \omega_E, \omega_D) = (0.55, 0.10, 0.20)$，另有排队权重 $\omega_Q = 0.15$ 在奖励函数中体现。

### II-B. 问题复杂性分析

**定理 0（NP-Hard性）。** 问题 (P1) 是 NP-Hard 的。

**证明。** 通过规约论证。构造如下特例：令 $K = 1$（单基站），$\alpha_i \in \{0, 1\}$（纯本地或纯卸载），忽略排队与干扰（$\rho_k \equiv 0$，$\Gamma_{i,k} = P_i|h_{i,k}|^2/\sigma^2$），此时 (P1) 退化为：将 $U$ 个具有异构计算需求 $(C_i)$ 和数据传输需求 $(D_i)$ 的任务分配至 2 个处理器（1 个本地 + 1 个 MEC 服务器），以最小化加权完成时间——这是经典的两台平行机调度问题（$P2 \| C_{\max}$），已知为 NP-Hard。又由于 (P1) 在上述特例上即为 NP-Hard，作为其推广，(P1) 至少同样困难。

此外，多用户场景下 SINR 中的干扰耦合项 $\sum_{j \neq i} P_j g_{j,k}$ 使得即使在连续松弛（$x_{i,k} \in [0,1]$）后，目标函数关于 $(P_i, \alpha_i)$ 仍为非凸函数，排除了标准凸优化方法的直接适用。 $\blacksquare$

### II-C. 约束处理机制

为在 RL 训练循环中处理约束 C1–C9，本文设计三层约束处理机制：

**第一层：参数化动作空间。** 策略网络输出归一化动作向量 $\mathbf{a}^{\text{raw}} \in [-1, 1]^3$，通过仿射映射转换为物理量。特别地，卸载比例的映射采用 Sigmoid 函数以获得更平滑的梯度：

$$\alpha_i = \sigma(5 \cdot a_\alpha^{\text{raw}}), \quad f_i^{\text{loc}} = f_{\min} + \frac{a_f^{\text{raw}} + 1}{2}(f_{\max} - f_{\min}), \quad P_i = P_{\min} + \frac{a_P^{\text{raw}} + 1}{2}(P_{\max} - P_{\min})$$

其中 $\sigma(\cdot)$ 为 Sigmoid 函数，乘以系数 5 以使输出在 $[-1, 1]$ 输入范围内覆盖 $[0.007, 0.993]$ 的有效区间。卸载目标 $x_{i,k}$ 通过离散动作头输出。

**第二层：约束投影。** 对于能量预算约束 C4，若估计能耗 $\hat{E}_i > E_i^{\max}$，按比例缩放频率与功率：$\text{scale} = \sqrt{E_i^{\max}/\hat{E}_i}$，$f_i \leftarrow f_i \cdot \text{scale}$，$P_i \leftarrow P_i \cdot \text{scale}$。对于系统功率约束 C6，若 $\sum_i P_i > P_{\text{total}}^{\max}$，则按比例缩放每个用户的功率。

**第三层：惩罚-障碍联合项。** 将残余约束违反引入奖励函数：

$$\mathcal{P}_{\text{pen}} = \mu_{\text{pen}} (v_E^2 + v_T^2 + v_P^2), \qquad \mathcal{P}_{\text{bar}} = -\mu_{\text{bar}} \sum_{c \in \{E,T,P\}} \log(\text{slack}_c)$$

其中 $v_E = \max(0, \hat{E}_i - E_i^{\max})$ 等为约束违反量，$\text{slack}_c$ 为约束松弛量，$\mu_{\text{pen}} = 0.05$，$\mu_{\text{bar}} = 10^{-4}$。障碍项在可行域内部平滑地趋近约束边界时增加代价，起到隐式约束的作用。

---

## III. 博弈论框架 (Game-Theoretic Framework)

本文构建**双层博弈**架构：上层为 MEC 运营商（基站）之间的 Stackelberg 定价博弈，下层为用户之间的非合作 Nash 资源竞争博弈。在此之上，通过 Shapley 值机制实现联盟收益的公平分配，并以 EFX 条件作为公平性的后验校验。

### III-A. Stackelberg 博弈——领导者最优定价

#### III-A-1. 博弈结构

基站（MEC 运营商）作为**领导者**率先公布服务定价 $p_k$；用户作为**跟随者**在观察定价后做出卸载决策。这一序贯决策结构构成 Stackelberg 博弈。

**领导者问题：** 基站 $k$ 选择定价以最大化利润

$$\max_{p_k \in [p_{\min}, p_{\max}]} \quad \Pi_k(p_k) = p_k \cdot D_k(p_k) - C_k\!\left(D_k(p_k)\right) \tag{4}$$

**需求函数。** 采用对数线性需求模型

$$D_k(p_k) = d_k^0 \cdot \exp(-\eta_k \cdot p_k) \tag{5}$$

满足 $D_k'(p_k) = -\eta_k D_k(p_k) < 0$（严格递减）且 $D_k(p_k) > 0, \forall p_k \in \mathbb{R}$（正值性）。

**成本函数。** 采用二次形式 $C_k(x) = c_k x^2$，反映资源利用的边际成本递增效应。

#### III-A-2. 利润函数的凹性与最优定价

**命题 1（利润函数的严格凹性）。** 在 $\eta_k > 0$，$c_k \geq 0$，$d_k^0 > 0$ 的条件下，$\Pi_k(p_k)$ 关于 $p_k$ 为严格凹函数。

**证明。** 将式 (5) 代入式 (4)，记 $D = D_k(p_k)$：

$$\Pi_k(p_k) = p_k D - c_k D^2$$

逐步计算各阶导数。令 $D' \triangleq dD/dp_k = -\eta_k D$，$D'' \triangleq d^2D/dp_k^2 = \eta_k^2 D$，则

$$\Pi_k' = D + p_k D' - 2c_k D D' = D(1 - \eta_k p_k + 2c_k \eta_k D)$$

$$\Pi_k'' = D'(1 - \eta_k p_k + 2c_k \eta_k D) + D(-\eta_k + 2c_k \eta_k D')$$

$$= D\bigl[-\eta_k(1 - \eta_k p_k + 2c_k \eta_k D) - \eta_k - 2c_k \eta_k^2 D\bigr]$$

$$= -\eta_k D\bigl[2 - \eta_k p_k + 4c_k \eta_k D\bigr]$$

由于 $\eta_k > 0$、$D > 0$、$c_k \geq 0$，且在合理价格区间 $p_k \leq 2/\eta_k$ 内有 $2 - \eta_k p_k \geq 0$，故 $\Pi_k'' < 0$。严格凹性成立。 $\blacksquare$

**推论。** 最优定价 $p_k^*$ 为一阶条件 $\Pi_k'(p_k^*) = 0$ 的唯一解，等价于求解

$$1 - \eta_k p_k^* + 2c_k \eta_k d_k^0 e^{-\eta_k p_k^*} = 0 \tag{6}$$

式 (6) 含隐式超越函数，通过牛顿法迭代求解：$p_k^{(n+1)} = p_k^{(n)} - \Pi_k'(p_k^{(n)}) / \Pi_k''(p_k^{(n)})$，初始化 $p_k^{(0)} = (p_{\min} + p_{\max})/2$，收敛准则 $|\Pi_k'| < 10^{-6}$，并将结果投影至 $[p_{\min}, p_{\max}]$。

**动态需求更新。** 在 RL 训练过程中，领导者根据实际观测到的需求 $\hat{D}_k$ 以动量方式更新基础需求量估计：

$$d_k^{0,(n+1)} = (1 - m) \cdot d_k^{0,(n)} + m \cdot \max(\hat{D}_k, \varepsilon)$$

其中 $m = 0.2$ 为动量系数，$\varepsilon = 10^{-4}$ 为下界保护。随后重新求解最优定价，形成双层博弈的外循环。

### III-B. 用户间 Nash 均衡

#### III-B-1. 非合作博弈建模

在领导者定价 $\mathbf{p}$ 给定的条件下，$U$ 个用户构成非合作博弈 $\mathcal{G} = \langle \mathcal{U},\, \{A_i\},\, \{J_i\} \rangle$：

- **参与者集合：** $\mathcal{U} = \{1, \ldots, U\}$
- **策略空间：** $A_i = \{0, 1, \ldots, K\} \times [0, 1] \times [f_{\min}, f_{\max}] \times [P_{\min}, P_{\max}]$

- **代价函数：**

$$J_i(a_i, \mathbf{a}_{-i};\, \mathbf{p}) = \omega_T \cdot T_i(a_i, \mathbf{a}_{-i}) + \omega_E \cdot E_i(a_i) + p_{k_i} \cdot \alpha_i \cdot D_i \tag{7}$$

其中第三项为用户向基站支付的服务费用，$k_i$ 为用户 $i$ 选择的目标基站。

**定义 3（Nash 均衡）。** 策略组合 $\mathbf{a}^* = (a_1^*, \ldots, a_U^*)$ 为 Nash 均衡（NE），当且仅当

$$J_i(a_i^*,\, \mathbf{a}_{-i}^*;\, \mathbf{p}) \leq J_i(a_i,\, \mathbf{a}_{-i}^*;\, \mathbf{p}), \qquad \forall\, a_i \in A_i,\; \forall\, i \in \mathcal{U}$$

即任何用户在其他用户策略不变的情况下，单方面偏离均衡策略都不会降低其代价。

#### III-B-2. 迭代最佳响应 (IBR) 求解

**Algorithm 1：Iterative Best Response (IBR)**

```
输入: 定价 p, 信道状态 g, 队列状态 Q, 任务参数 {J_i}
      最大迭代 N_IBR = 200, 收敛容差 ε_NE = 10^{-4}
输出: 近似 Nash 均衡 a*

1. 初始化: 每个用户基于贪心启发式生成初始策略 a_i^{(0)}
   — 选择 SINR 最优的基站
   — 根据任务紧急度确定卸载比例
   — CPU 频率与功率按任务-信道条件启发式设定
2. for n = 1 to N_IBR do
3.     for i = 1 to U do
4.         a_i^{(n)} ← BR_i(a_{-i}^{(n-1)}, p)  // 最佳响应
5.     end for
6.     if ||a^{(n)} - a^{(n-1)}||_∞ < ε_NE then return a^{(n)}
7. end for
8. return a^{(N_IBR)}
```

其中步骤 4 的单用户最佳响应问题在给定 $\mathbf{a}_{-i}$ 的条件下退化为单变量优化，可通过解析或梯度下降方法高效求解。

**注记 2。** 最佳响应函数 $\text{BR}_i(\cdot)$ 在本文实现中基于信道-任务条件的启发式构造：选择信道质量最优的基站作为目标，根据任务紧急度（$T_i^{\max}$ 的归一化值）确定卸载比例，根据 SINR 条件确定功率水平。这一启发式为 Nash 均衡提供了合理的近似，同时保持了计算可行性。

**收敛性保证。** 当代价函数 $J_i$ 具有对角严格凹（diagonally strictly concave）性质时，即联合代价函数的加权 Jacobian 矩阵正定，IBR 算法保证收敛至唯一 Nash 均衡（Rosen, 1965）。这一条件在本文模型中的满足取决于干扰耦合强度与代价函数的凸性结构，在典型参数设置下可以验证成立。

### III-C. Stackelberg 均衡存在性与唯一性

**定理 1（Stackelberg 均衡存在性与唯一性）。** 若下列条件成立：

**(A1)** 需求函数 $D_k(p_k) = d_k^0 e^{-\eta_k p_k}$ 严格递减且对数凹（$\log D_k$ 关于 $p_k$ 为凹函数，由线性性显然成立）；

**(A2)** 成本函数 $C_k(x) = c_k x^2$ 为凸函数；

**(A3)** 用户代价函数 $J_i(a_i, \mathbf{a}_{-i}; \mathbf{p})$ 关于 $a_i$ 凸，关于 $\mathbf{a}_{-i}$ 连续；

则所提双层 Stackelberg 博弈存在唯一均衡 $(\mathbf{p}^*, \mathbf{a}^*)$。

**证明。**

**(i) 跟随者最佳响应的存在性。** 由条件 (A3)，对于固定的 $\mathbf{p}$ 和 $\mathbf{a}_{-i}$，用户 $i$ 的最小化问题 $\min_{a_i \in A_i} J_i(a_i, \mathbf{a}_{-i}; \mathbf{p})$ 为紧凸集上的凸优化问题，由 Weierstrass 定理保证最优解存在，凸性保证唯一性。

**(ii) 最佳响应映射的连续性。** 由 Berge 最大值定理（Maximum Theorem），当 $J_i$ 关于 $(a_i, \mathbf{a}_{-i}, \mathbf{p})$ 联合连续、策略集 $A_i$ 关于参数连续对应时，最佳响应映射 $\text{BR}_i(\mathbf{p}, \mathbf{a}_{-i})$ 为上半连续集值映射，且当最优解唯一时退化为连续单值函数。

**(iii) 领导者问题的严格凹性。** 由命题 1，$\Pi_k(p_k)$ 为严格凹函数。将跟随者最佳响应代入后，由需求函数 (5) 的对数凹性，复合利润函数保持严格凹性。

**(iv) 均衡唯一性。** 领导者最优定价 $\mathbf{p}^*$ 由 (iii) 的严格凹性保证唯一。在 $\mathbf{p}^*$ 下，跟随者 Nash 均衡在 Rosen (1965) 对角严格凹条件下唯一（当代价函数的 Jacobian 矩阵正定时成立）。 $\blacksquare$

### III-D. Shapley 值协作博弈

#### III-D-1. 联盟博弈建模

定义可转移效用联盟博弈 $(\mathcal{U}, v)$，其中特征函数 $v : 2^{\mathcal{U}} \to \mathbb{R}$ 衡量联盟 $\mathcal{S} \subseteq \mathcal{U}$ 的协作收益：

$$v(\mathcal{S}) = \sum_{i \in \mathcal{S}} \underbrace{J_i^{\text{baseline}}}_{\text{独立策略代价}} - \sum_{i \in \mathcal{S}} \underbrace{J_i^{\text{coop}}(\mathcal{S})}_{\text{联盟协作代价}}$$

$v(\mathcal{S}) > 0$ 意味着联盟 $\mathcal{S}$ 内的用户通过协作（负载均衡、干扰协调）获得了超越独立行动的性能改善。

#### III-D-2. Shapley 值定义与性质

用户 $i$ 的 **Shapley 值**为其对所有可能联盟的平均边际贡献：

$$\phi_i(v) = \sum_{\mathcal{S} \subseteq \mathcal{U} \setminus \{i\}} \frac{|\mathcal{S}|!\, (U - |\mathcal{S}| - 1)!}{U!} \cdot \bigl[v(\mathcal{S} \cup \{i\}) - v(\mathcal{S})\bigr] \tag{8}$$

式 (8) 所定义的 Shapley 值满足以下四条公理化性质，为分配的合理性提供了理论保证：

1. **效率性（Efficiency）：** $\sum_{i=1}^{U} \phi_i(v) = v(\mathcal{U})$，即所有用户的 Shapley 值之和等于大联盟的总价值，保证无资源浪费或超额分配。

2. **对称性（Symmetry）：** 若用户 $i$ 和 $j$ 对所有联盟的边际贡献相同（即 $v(\mathcal{S} \cup \{i\}) = v(\mathcal{S} \cup \{j\}), \forall \mathcal{S} \subseteq \mathcal{U} \setminus \{i, j\}$），则 $\phi_i = \phi_j$，即等贡献者获得等回报。

3. **零元素性（Null Player）：** 若用户 $i$ 对所有联盟的边际贡献为零（即 $v(\mathcal{S} \cup \{i\}) = v(\mathcal{S}), \forall \mathcal{S}$），则 $\phi_i = 0$，即无贡献者不获分配。

4. **可加性（Additivity）：** 对于任意两个联盟博弈 $(v, w)$，$\phi_i(v + w) = \phi_i(v) + \phi_i(w)$，保证分配的线性可分解性。

#### III-D-3. 蒙特卡洛-对偶近似算法

精确计算式 (8) 的复杂度为 $O(2^U)$（需遍历所有 $2^U$ 个联盟子集），在用户数 $U \geq 10$ 时不可行。本文采用基于排列采样的蒙特卡洛方法，并通过对偶变量（Antithetic Variates）技术将方差缩减约一半。

**Algorithm 2：Monte Carlo Shapley with Antithetic Sampling**

```
输入: 用户数 U, 联盟值函数 v(·), 采样数 M = 128
输出: 近似 Shapley 值 φ̂

1. φ̂ ← 0 ∈ ℝ^U, n ← 0
2. for m = 1 to M do
3.     π ← random_permutation(1, ..., U)
4.     Σ ← {π, reverse(π)}          // 原始排列 + 对偶排列
5.     for each σ ∈ Σ do
6.         S ← ∅, v_prev ← 0
7.         for j = 1 to U do
8.             i ← σ(j); S ← S ∪ {i}
9.             φ̂_i ← φ̂_i + v(S) - v_prev
10.            v_prev ← v(S)
11.        end for
12.        n ← n + 1
13.    end for
14. end for
15. return φ̂ / n
```

**复杂度：** $O(M \cdot U \cdot T_v)$，其中 $T_v$ 为联盟值一次评估时间。对偶变量技术使有效样本量翻倍，方差界 $\text{Var}(\hat{\phi}_i) \leq V_{\max}^2 / (2M)$。本文取 $M = 128$。

### III-E. EFX 公平分配与 CP-nets 偏好

#### III-E-1. EFX 条件

**定义 4 (EFX)。** 资源分配 $(A_1, \ldots, A_U)$ 满足 EFX（Envy-Free up to any item）条件，当且仅当对任意用户 $i, j \in \mathcal{U}$、任意物品 $g \in A_j$：

$$v_i(A_i) \geq v_i(A_j \setminus \{g\})$$

即用户 $i$ 不会嫉妒用户 $j$ 的分配包在移除任意一个物品后的剩余部分。

**EFX 修复机制。** 当 Shapley 值分配违反 EFX 时，通过迭代转移支付修复。每次发现违约 $(i, j, g)$ 时计算亏缺 $\delta = v_i(A_j \setminus \{g\}) - v_i(A_i)$，执行转移 $\Delta_i \mathrel{+}= \frac{\tau \delta}{2}$，$\Delta_j \mathrel{-}= \frac{\tau \delta}{2}$（$\tau = 0.5$）。最大迭代 32 次，转移量裁剪至 $[-0.05, 0.05]$ 以确保训练稳定。

#### III-E-2. CP-nets 条件偏好

为使 EFX 估值函数反映用户的异质偏好，引入条件偏好网络（CP-nets）。CP-net 为一有向无环图，每个节点代表一个偏好属性（如时延优先 vs. 能耗优先），边表示条件依赖。本文中偏好结构由任务紧急度条件驱动：

**基站效用矩阵 $\mathbf{U}^{\text{srv}} \in \mathbb{R}^{U \times (K+1)}$：** 综合信道质量、队列负载与定价：

$$U_{i,k}^{\text{srv}} = w_{\text{ch}} \cdot \text{norm}(\text{SNR}_{i,k}) + w_{\text{load}} \cdot (1 - \rho_k) - w_{\text{price}} \cdot \text{norm}(p_k)$$

**卸载水平效用矩阵 $\mathbf{U}^{\text{off}} \in \mathbb{R}^{U \times 3}$：** 按卸载比例分为低/中/高三档，效用由任务紧急度 $\text{urg}_i = 1 - T_i^{\max}/T_{\max}^{\max}$ 条件化：

$$U_{i,\text{low}}^{\text{off}} = 0.50 + 0.40 \cdot \text{urg}_i, \qquad U_{i,\text{high}}^{\text{off}} = 0.55 + 0.35 \cdot (1 - \text{urg}_i)$$

紧急任务偏好本地/低卸载（避免传输时延），宽松任务偏好高卸载（利用边缘算力）。

**综合估值函数。** 用户 $i$ 对分配包 $A_i = \{\text{srv}_k, \text{off}_l\}$ 的估值为

$$v_i(A_i) = U_{i,k}^{\text{srv}} + U_{i,l}^{\text{off}}$$

该估值输入至 EFX 校验与修复流程。

---

## IV. MADRL 算法设计

### IV-A. Dec-POMDP 建模

将多用户 MEC 联合优化问题建模为去中心化部分可观测马尔可夫决策过程（Dec-POMDP），形式化为七元组

$$\mathcal{M} = \langle \mathcal{U},\, \mathcal{S},\, \{\mathcal{O}_i\}_{i \in \mathcal{U}},\, \{\mathcal{A}_i\}_{i \in \mathcal{U}},\, P,\, \{R_i\}_{i \in \mathcal{U}},\, \gamma \rangle$$

各元素定义如下：

- **智能体集合** $\mathcal{U} = \{1, \ldots, U\}$
- **全局状态空间** $\mathcal{S}$：包含所有基站队列 $\{Q_k\}$、信道状态 $\{g_{i,k}\}$、用户位置 $\{\mathbf{u}_i\}$ 和任务参数 $\{\mathcal{J}_i\}$
- **局部观测空间** $\mathcal{O}_i \subset \mathcal{S}$：智能体 $i$ 的局部可观测子空间（部分可观测性源于用户无法直接获知其他用户的策略）
- **动作空间** $\mathcal{A}_i$：混合动作空间（离散 + 连续），详见 IV-C 节
- **状态转移函数** $P(s' | s, \mathbf{a})$：由通信模型（I-B）、计算模型（I-C）和移动性模型（I-D）共同决定的状态转移核
- **奖励函数** $R_i : \mathcal{S} \times \prod_j \mathcal{A}_j \to \mathbb{R}$：分层奖励函数，详见 IV-D 节
- **折扣因子** $\gamma = 0.99$

### IV-B. 观测空间

每个智能体 $i$ 的局部观测 $o_i(t) \in \mathbb{R}^{3K+5}$：

$$o_i(t) = \Bigl[\underbrace{\tilde{Q}_1, \ldots, \tilde{Q}_K}_{\text{归一化队列长度}},\; \underbrace{\tilde{\Gamma}_{i,1}, \ldots, \tilde{\Gamma}_{i,K}}_{\text{归一化SINR}},\; \underbrace{L_1, \ldots, L_K}_{\text{CPU利用率}},\; \underbrace{e_i,\, \tau_i,\, \tilde{D}_i,\, \tilde{C}_i,\, \tilde{v}_i}_{\text{任务与移动特征}}\Bigr]$$

各分量含义：$\tilde{Q}_k = Q_k/Q_{\max}$（归一化队列长度），$\tilde{\Gamma}_{i,k}$ 为归一化后的 SINR（经 sigmoid 或 min-max 变换），$L_k = \rho_k$（CPU 利用率），$e_i$ 为归一化剩余能量，$\tau_i = T_i^{\max}/T_{\max}^{\max}$（归一化截止时间），$\tilde{D}_i, \tilde{C}_i$ 为归一化任务参数，$\tilde{v}_i = v_i/v_{\max}$（归一化速度）。

**全局状态**（用于中心化 Critic 训练）：

$$s(t) = [o_1(t), \ldots, o_U(t),\, \mathbf{p}^*(t),\, \mathbf{a}^*_{\text{eq}}(t)]$$

其中 $\mathbf{a}^*_{\text{eq}}$ 为 Stackelberg 均衡动作提示。

### IV-C. 混合动作空间

每个智能体的动作空间为**混合空间**（Dict 空间），包含一个离散头和一个连续头：

$$a_i = \bigl\{\text{target}: x_i \in \{0, 1, \ldots, K\},\;\; \text{ratio}: (r_\alpha,\, r_f,\, r_P) \in [-1, 1]^3\bigr\}$$

**离散头（Discrete Head）：** 输出卸载目标选择，0 表示本地处理，$1 \sim K$ 表示卸载至对应基站。

**连续头（Continuous Head）：** 输出三个归一化比例值，经 II-C 节映射后得到物理决策变量 $(\alpha_i, f_i^{\text{loc}}, P_i)$。双头 Actor 网络在共享特征层后分叉：离散头接 Softmax 输出分类概率，连续头接 Tanh 激活限制到 $[-1, 1]$。

### IV-D. 分层奖励函数

奖励函数由三个语义明确的层次构成。

#### IV-D-1. 即时通信性能奖励 $r_{\text{imm}}$

直接反映物理层性能：

$$r_{\text{imm}} = -\left(\omega_T \cdot \frac{T_i^{\text{total}}}{T_i^{\max}} + \omega_Q \cdot \frac{T_k^{\text{queue}}}{T_i^{\max}} + \omega_D \cdot \left(\max\!\left(0, \frac{T_i^{\text{total}} - T_i^{\max}}{T_i^{\max}}\right)\right)^{\!2} + \omega_E \cdot \frac{E_i^{\text{total}}}{E_i^{\max}} + \delta_{\text{nn}}\right)$$

其中 $(\omega_T, \omega_Q, \omega_D, \omega_E) = (0.55, 0.15, 0.20, 0.10)$，$\delta_{\text{nn}} = 0.02 \cdot \mathbb{1}[\text{非最近基站}]$ 为空间合理性引导项，当用户选择非最近基站时施加额外惩罚以引导合理的基站选择。

#### IV-D-2. 协作奖励 $r_{\text{coop}}$

基于 Shapley 值量化个体对团队的贡献：

$$r_{\text{coop}} = \text{clip}\!\left(\phi_i \cdot \left[\bar{J}^{\text{baseline}} - \bar{J}^{\text{observed}}\right],\; -1,\; 1\right)$$

其中 $\bar{J}^{\text{baseline}} = \frac{1}{U} \sum_i J_i^{\text{local-only}}$ 为全用户纯本地策略的平均通信代价，$\bar{J}^{\text{observed}} = \frac{1}{U} \sum_i J_i^{\text{current}}$ 为当前联合策略的平均通信代价。$r_{\text{coop}} > 0$ 当且仅当当前联合策略优于独立基线。

#### IV-D-3. 均衡一致性奖励 $r_{\text{eq}}$

引导策略靠近博弈均衡解：

$$r_{\text{eq}} = -\frac{\|\mathbf{a}_i - \mathbf{a}_i^*(s)\|_2^2}{\dim(\mathcal{A})}$$

其中 $\mathbf{a}_i^*(s)$ 为双层博弈求解器给出的均衡动作（由 Stackelberg-Nash 求解器计算），$\dim(\mathcal{A})$ 为动作空间维度。该项确保 RL 策略不会偏离博弈论指导的合理策略空间。

#### IV-D-4. 总奖励与自适应权重

$$R_i = \omega_\alpha \cdot r_{\text{imm}} + \omega_\beta \cdot r_{\text{coop}} + \omega_\gamma \cdot r_{\text{eq}} - \mathcal{P}_{\text{pen}} - \mathcal{P}_{\text{bar}} + r_{\text{fair}} \tag{9}$$

其中 $r_{\text{fair}} = \text{clip}(\Delta_i^{\text{EFX}}, -0.05, 0.05)$ 为 EFX 转移支付项。初始权重 $(\omega_\alpha^{(0)}, \omega_\beta^{(0)}, \omega_\gamma^{(0)}) = (0.8, 0.1, 0.1)$。

**自适应权重调节。** 为缓解奖励分量间的梯度冲突，每隔 $W = 50$ 步计算滑动窗口内各分量的皮尔逊相关系数：

$$\omega_\beta^{(t)} = \omega_\beta^{(0)} \cdot \max\!\left(0.5,\; 1 + \text{corr}\!\left(r_{\text{imm}}^{(t-W:t)},\, r_{\text{coop}}^{(t-W:t)}\right)\right)$$

$$\omega_\gamma^{(t)} = \omega_\gamma^{(0)} \cdot \max\!\left(0.5,\; 1 + \text{corr}\!\left(r_{\text{imm}}^{(t-W:t)},\, r_{\text{eq}}^{(t-W:t)}\right)\right)$$

负相关（冲突）时自动降权，正相关（协同）时保持或提升。更新后重新归一化使权重和为 1：$(\omega_\alpha, \omega_\beta, \omega_\gamma) \leftarrow (\omega_\alpha, \omega_\beta, \omega_\gamma) / (\omega_\alpha + \omega_\beta + \omega_\gamma)$。

**注记 3（梯度冲突检测的理论依据）。** 当 $\nabla_\theta r_{\text{imm}}$ 与 $\nabla_\theta r_{\text{coop}}$ 的余弦相似度为负时，梯度更新同时优化两个目标变得困难。自适应权重机制本质上实现了一种简化版的多目标梯度下降（MGDA），通过经验相关性代替精确的梯度投影，以较低的计算开销缓解了 Pareto 冲突。

### IV-E. CTDE 架构与 GRPO 算法

#### IV-E-1. 中心化训练-去中心化执行（CTDE）

本文采用 CTDE（Centralized Training with Decentralized Execution）范式。

**分散化 Actor（执行阶段）：** 每个智能体仅依据局部观测 $o_i(t)$ 输出动作，网络结构为

$$\pi_{\theta_i}: \mathcal{O}_i \to \Delta(\mathcal{A}_i), \quad \text{Linear}(3K\!+\!5, 256) \to \text{ReLU} \to \text{Linear}(256, 128) \to \text{ReLU} \to \begin{cases} \text{Softmax}(K\!+\!1) & \text{离散头}\\ \text{Tanh}(3) & \text{连续头}\end{cases}$$

**中心化 Critic（训练阶段）：** 以全局状态为输入，附加均衡嵌入模块 $\text{EqEnc}: \mathbb{R}^{U \cdot \dim(\mathcal{A})} \to \mathbb{R}^{64}$，将均衡动作嵌入为紧凑表示：

$$s_{\text{critic}} = [s(t),\; \text{EqEnc}(\mathbf{a}_1^*, \ldots, \mathbf{a}_U^*)]$$

$$V_\psi(s) : \text{Linear}(|\mathcal{S}|+64, 512) \to \text{ReLU} \to \text{Linear}(512, 256) \to \text{ReLU} \to \text{Linear}(256, 1)$$

#### IV-E-2. GRPO 算法

GRPO（Group Relative Policy Optimization）的核心创新在于以**组内相对优势**替代传统的值函数基线，消除了独立 Value 网络的需要。对于每个状态 $s$，从当前策略采样一组 $G = 4$ 个动作 $\{a_i^{(1)}, \ldots, a_i^{(G)}\}$，获取各自累积回报 $\{R^{(g)}\}_{g=1}^G$，组相对优势为

$$\hat{A}_i^{(g)} = \frac{R^{(g)} - \bar{R}}{\text{std}(\{R^{(j)}\}) + \varepsilon_{\text{std}}}$$

其中 $\bar{R} = G^{-1}\sum_g R^{(g)}$，$\varepsilon_{\text{std}} = 10^{-8}$ 为数值稳定常数。策略更新目标采用 PPO 式剪切：

$$\mathcal{L}^{\text{GRPO}}(\theta_i) = -\mathbb{E}\!\left[\min\!\left(\frac{\pi_{\theta_i}(a_i|o_i)}{\pi_{\theta_i^{\text{old}}}(a_i|o_i)} \hat{A}_i,\;\; \text{clip}\!\left(\frac{\pi_{\theta_i}}{\pi_{\theta_i^{\text{old}}}},\, 1\!-\!\epsilon,\, 1\!+\!\epsilon\right)\hat{A}_i\right)\right]$$

**熵正则化：**

$$\mathcal{L}^{\text{total}}(\theta_i) = \mathcal{L}^{\text{GRPO}}(\theta_i) - c_{\text{ent}} \cdot H[\pi_{\theta_i}(\cdot | o_i)]$$

其中 $\epsilon = 0.2$ 为剪切比率，$c_{\text{ent}} = 0.01$ 为熵系数，$H[\cdot]$ 为策略分布的熵。

#### IV-E-3. Shapley 值信用分配

在 CTDE 框架中，团队总奖励通过 Shapley 值进行信用分配，替代传统的 QMIX/VDN 值分解：

$$r_i^{\text{credit}} = \begin{cases} \phi_i \cdot R^{\text{team}} \big/ \sum_j \phi_j, & \text{if } |\sum_j \phi_j| > 10^{-8} \\ R^{\text{team}} / U, & \text{otherwise（退化为均匀分配）} \end{cases}$$

该机制相较于 QMIX/VDN 的单调值分解约束具有更强的表达能力，且 Shapley 公理保证了信用分配的唯一性和公平性。

### IV-F. 博弈论引导的 Warm-Start

为加速 RL 收敛，利用博弈均衡解进行策略网络的预训练（Warm-Start）。

**Algorithm 3：Game-Theoretic Warm-Start**

```
输入: 环境 env, 双层博弈求解器 solver, 预训练步数 N_warm = 1000
输出: 预训练策略 {π_θ_i}

1. for n = 1 to N_warm do
2.     s ← env.reset()
3.     (p*, {a_i^*}) ← solver.solve(s)
4.     for each agent i do
5.         ℓ_i ← ||π_θ_i(o_i) - a_i^*||²       // 模仿损失
6.         θ_i ← θ_i - η_warm · ∇ℓ_i
7.     end for
8.     σ_noise ← max(0, 1 - n/N_warm) · σ_init    // 衰减探索噪声
9. end for
10. 切换至标准 GRPO 训练（继承预训练参数）
```

Warm-Start 学习率 $\eta_{\text{warm}} = 0.5 \cdot \eta$（标准学习率的一半），以避免过度拟合于均衡解而丧失 RL 的探索能力。

---

## V. 收敛性与复杂度分析

### V-A. 收敛性理论

**假设 1（奖励有界性）。** $|R_i(s, \mathbf{a})| \leq R_{\max} < \infty$，$\forall\, s \in \mathcal{S},\, \mathbf{a} \in \prod_i \mathcal{A}_i$。

**假设 2（Lipschitz 连续性）。** 存在常数 $L_R, L_P > 0$，使得 $|R_i(s, \mathbf{a}) - R_i(s', \mathbf{a}')| \leq L_R \|(s, \mathbf{a}) - (s', \mathbf{a}')\|$ 以及 $\|P(\cdot|s, \mathbf{a}) - P(\cdot|s', \mathbf{a}')\|_1 \leq L_P \|(s, \mathbf{a}) - (s', \mathbf{a}')\|$。

**假设 3（学习率条件）。** 学习率序列 $\{\eta^{(n)}\}$ 满足 Robbins-Monro 条件：$\sum_n \eta^{(n)} = \infty$，$\sum_n (\eta^{(n)})^2 < \infty$。

**定理 2（$\varepsilon$-近似均衡收敛性）。** 在假设 1–3 下，所提 GRPO-Shapley 算法的策略序列 $\{\pi_\theta^{(t)}\}$ 满足

$$\mathbb{E}\!\left[\delta_{\text{SE}}^{(T)}\right] \leq \varepsilon + O\!\left(\frac{1}{\sqrt{T}}\right), \quad \text{其中 } \delta_{\text{SE}}^{(T)} \triangleq \|\mathbf{a}_\theta^{(T)} - \mathbf{a}^*\|_2$$

即策略动作以 $O(1/\sqrt{T})$ 的速率收敛至 Stackelberg 均衡的 $\varepsilon$-邻域。更精确地，存在常数 $C > 0$，使得上界中的 $O(1/\sqrt{T})$ 项可显式化。

**证明思路。**

**(i)** PPO/GRPO 的剪切目标将策略更新的 KL 散度限制在 $D_{\text{KL}}(\pi_\theta^{(t+1)} \| \pi_\theta^{(t)}) \leq \delta_{\text{KL}}$，保证了策略空间中的信赖域约束。

**(ii)** Shapley 值信用分配保持了多智能体策略梯度估计的无偏性：由效率性公理 $\sum_i \phi_i = v(\mathcal{U})$，个体奖励之和等于团队总奖励。

**(iii)** 均衡一致性奖励 $r_{\text{eq}}$ 在策略参数空间中诱导势函数 $\Phi(\theta) = -\sum_i \|\pi_{\theta_i} - \pi_i^*\|^2$。势博弈理论保证：若 GRPO 的策略梯度方向与 $-\nabla_\theta \Phi$ 渐近对齐，则策略序列收敛至 $\Phi$ 的稳定点，即均衡附近。

**(iv)** 结合随机逼近理论（Borkar-Meyn ODE 方法），在假设 3 的学习率条件下，策略参数的随机迭代几乎必然跟踪其对应 ODE 的渐近行为，收敛至均衡附近。 $\blacksquare$

### V-B. 复杂度分析

**每步计算复杂度：**

| 模块 | 复杂度 | 说明 |
|------|--------|------|
| $U$ 个 Actor 前向传播 | $O(U \cdot |\mathcal{O}| \cdot H)$ | $H = 256$ 隐藏维度 |
| Critic 前向传播 | $O(|\mathcal{S}| \cdot H)$ | 中心化训练 |
| Stackelberg 定价（牛顿法） | $O(K \cdot N_{\text{Newton}})$ | $N_{\text{Newton}} \leq 100$ |
| Nash IBR（内循环） | $O(U \cdot N_{\text{IBR}} \cdot K)$ | $N_{\text{IBR}} \leq 200$ |
| Monte Carlo Shapley | $O(M \cdot U \cdot T_v)$ | $M = 128$，$T_v$ 为联盟值评估时间 |
| EFX 校验与修复 | $O(U^2 \cdot K \cdot N_{\text{EFX}})$ | $N_{\text{EFX}} \leq 32$ |

**算法间总体对比：**

| 算法 | 每步时间复杂度 | 空间复杂度 | Shapley 开销 |
|------|--------------|------------|-------------|
| DQN | $O(|\mathcal{S}||\mathcal{A}|)$ | $O(B|\mathcal{S}|)$ | — |
| DDPG | $O(|\mathcal{S}||\mathcal{A}|)$ | $O(B(|\mathcal{S}|+|\mathcal{A}|))$ | — |
| MAPPO (w/o game) | $O(U|\mathcal{S}||\mathcal{A}|)$ | $O(U|\theta|)$ | — |
| QMIX | $O(U^2|\mathcal{S}||\mathcal{A}|)$ | $O(U|\theta|+U^2)$ | — |
| **本文** | $O(U|\mathcal{S}||\mathcal{A}| + MUT_v)$ | $O(U|\theta| + MU)$ | $O(MUT_v)$ |

其中 $B$ 为 batch size，$U$ 为智能体数，$M$ 为蒙特卡洛采样数。

---

## VI. 仿真设置与参数配置

### VI-A. 环境与物理层参数

| 类别 | 参数 | 值 |
|------|------|-----|
| 网络拓扑 | $K$（基站数） | 3 |
| | $U$（用户数） | 3–20 |
| | 基站间距 | 15 m |
| | $h_{\text{BS}} / h_{\text{UT}}$ | 25 / 1.5 m |
| 信道 | $f_c$（载波频率） | 3.5 GHz |
| | $B$（带宽） | 20 MHz |
| | 噪声功率谱密度 | −174 dBm/Hz |
| | $\sigma_{\text{SF}}$（阴影衰落 LOS/NLOS） | 4 / 7.82 dB |
| | 衰落类型 | Rayleigh |
| 计算 | $F_k$（边缘CPU） | 5 GHz |
| | $[f_{\min}, f_{\max}]$（本地CPU） | [0.5, 3.0] GHz |
| | $\kappa$（开关电容系数） | $10^{-27}$ |
| | $\beta$（DVFS指数） | 2.7 |
| | $P_{\text{leak}}$（泄漏功耗） | 0.01 W |
| 任务 | $[D_{\min}, D_{\max}]$（数据量） | [100, 500] KB |
| | $[C_{\min}, C_{\max}]$（CPU需求） | [$10^8$, $10^9$] cycles |
| | $[T_{\min}^{\max}, T_{\max}^{\max}]$（截止时间） | [0.5, 2.0] s |
| 功率 | $[P_{\min}, P_{\max}]$（用户发射功率） | [0.01, 0.5] W |
| | $P_{\text{total}}^{\max}$（系统功率上限） | 1.5 W |
| 排队 | $\rho_k^{\max}$（利用率上限） | 0.95 |
| 移动性 | $[r_{\min}, r_{\max}]$（初始半径） | [8, 30] m |
| | $v_{\max}$（最大速度） | 5 m/s |
| 博弈论 | $[p_{\min}, p_{\max}]$（定价范围） | [0.1, 2.0] |
| | $\eta_k / c_k / d_k^0$ | 0.35 / 0.12 / 1.0 |
| | $M$（Shapley采样数） | 128 |
| | $\tau$（EFX转移速率） | 0.5 |
| 预算 | $E_i^{\max} / T_i^{\max,\text{budget}}$ | 10.0 J / 2.0 s |

### VI-B. 训练超参数

| 参数 | 符号 | 值 |
|------|------|-----|
| 折扣因子 | $\gamma$ | 0.99 |
| GAE λ | $\lambda_{\text{GAE}}$ | 0.95 |
| PPO 剪切比率 | $\epsilon$ | 0.2 |
| 熵系数 | $c_{\text{ent}}$ | 0.01 |
| 值函数系数 | $c_{\text{val}}$ | 0.5 |
| GRPO 组大小 | $G$ | 4 |
| 学习率 | $\eta$ | $3 \times 10^{-4}$ |
| 批大小 | Batch size | 64 |
| Rollout 步数 | — | 2048 |
| 更新 Epoch 数 | — | 10 |
| 最大梯度范数 | — | 0.5 |
| 总训练步数 | — | 500,000 |
| 每 Episode 步数 | — | 100 |
| 隐藏层维度 | $H$ | 256 |
| 激活函数 | — | ReLU |
| $N_{\text{warm}}$（Warm-Start 步数） | — | 1,000 |
| Warm-Start 学习率比例 | — | 0.5 |
| 奖励权重 | $(\omega_\alpha, \omega_\beta, \omega_\gamma)$ | (0.8, 0.1, 0.1) |
| 探索初始 $\epsilon$ | — | 1.0 |
| 探索终止 $\epsilon$ | — | 0.05 |
| 探索衰减步数 | — | 50,000 |

### VI-C. 可扩展性实验矩阵

| 配置 | $K$ | $U$ | 观测维度 | 动作维度 | 预计训练时长 |
|------|-----|-----|----------|----------|-------------|
| Small | 3 | 6 | ~46 | 4 | ~2 h |
| Medium | 5 | 10 | ~71 | 4 | ~8 h |
| Large | 10 | 20 | ~126 | 4 | ~24 h |

### VI-D. 基线算法对比

| 类别 | 算法 | 特点 |
|------|------|------|
| 启发式 | Greedy, Random, Local-only, Full-offload | 无学习能力，作为性能下界参照 |
| 单智能体 DRL | DQN, DDPG, TD3, SAC | 无多智能体协调，忽略用户间交互 |
| 多智能体 DRL | QMIX, MAPPO (w/o game), MADDPG | 无博弈论层，缺少结构化先验 |
| **本文** | **GRPO + Stackelberg + Shapley** | 完整博弈-RL 融合框架 |

### VI-E. 消融实验 (Ablation Study)

| ID | 消融项 | 描述 | 预期影响 |
|----|--------|------|----------|
| A1 | w/o Shapley | 用均匀分配替代 Shapley 值 | 公平性下降，收敛变慢 |
| A2 | w/o Warm-Start | 随机初始化策略（取消博弈预训练） | 前期探索效率低，收敛所需步数增加 |
| A3 | w/o 移动性 | 用户静止 | 性能上界，验证移动性带来的挑战 |
| A4 | w/o $r_{\text{coop}}$ | 仅用即时奖励 $r_{\text{imm}}$ | 自私策略主导，系统整体效率降低 |
| A5 | w/o M/M/1 | 去除排队时延模型 | 高负载场景下性能估计偏乐观 |
| A6 | w/o $r_{\text{eq}}$ | 去除均衡一致性奖励 | 策略偏离博弈均衡，稳定性下降 |
| A7 | w/o DVFS | 使用理想 $\kappa f^2 C$ 模型 | 能耗估计偏乐观，频率决策失真 |

---

## VII. 理论贡献总结

本文的主要理论贡献可归纳为五个层面。

其一，**双层博弈-RL 融合架构**。本文首次将 Stackelberg 定价博弈（上层）与用户间 Nash 均衡（下层）同多智能体 GRPO 算法有机耦合，博弈均衡解在训练流程中承担双重角色——既作为策略网络的初始化锚点（Warm-Start），又作为在线训练的引导信号（均衡一致性奖励 $r_{\text{eq}}$），实现了博弈论结构化先验知识向 RL 策略空间的有效迁移。

其二，**Shapley 值信用分配机制**。本文以蒙特卡洛-对偶 Shapley 估计器替代 QMIX/VDN 等单调值分解方法，突破了单调性约束对值函数逼近能力的限制。Shapley 公理化性质（效率性、对称性、零元素性、可加性）为信用分配的公平性和理论完备性提供了坚实基础。

其三，**EFX-CP-nets 公平偏好框架**。在 Shapley 分配之上引入 EFX 公平性后验校验与 CP-nets 条件偏好建模，使资源分配不仅满足效率准则，且兼顾用户的异质偏好结构与"无嫉妒"公平约束。据作者所知，这是 EFX 公平性概念首次应用于 MEC 资源分配领域。

其四，**高保真系统模型**。本文系统模型涵盖 3GPP TR 38.901 标准信道（LOS/NLOS 概率路径损耗 + 多用户干扰 SINR）、非理想 DVFS 能耗（含泄漏功耗与工艺相关指数）、M/M/1 排队时延以及高斯-马尔可夫移动性模型，辅以参数化动作空间与投影-惩罚-障碍三层约束处理，构成了贴近实际部署条件的仿真环境。

其五，**收敛性理论保证**。本文证明了所提 GRPO-Shapley 算法在标准随机逼近假设下收敛至 $\varepsilon$-近似 Stackelberg 均衡，收敛速率为 $O(1/\sqrt{T})$。

---

## 附录 A. 仿真技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| RL 框架 | Stable-Baselines3 / Tianshou | GRPO/PPO 快速原型 |
| 环境接口 | Gymnasium (`gym.Env`) | 标准化 API |
| 信道模型 | 3GPP TR 38.901（自实现） | UMi / UMa / RMa |
| 排队模型 | M/M/1 / M/M/c（自实现） | 排队论库 |
| 博弈求解 | Newton + IBR（自实现） | Stackelberg + Nash |
| 衰落模型 | Rayleigh / Rician / Nakagami（自实现） | 信道衰落库 |
| 移动性 | Gauss-Markov / RandomWalk（自实现） | 用户移动轨迹 |
| 对比基线 | DQN / DDPG / TD3 / SAC / MAPPO / QMIX / MADDPG | 全面对比 |

## 附录 B. 支持的 RL 算法

| 类型 | 算法 | 适用场景 |
|------|------|---------|
| 单智能体 On-Policy | PPO, TRPO, A3C | 单用户基线 |
| 单智能体 Off-Policy | DDQN, DDPG, TD3, SAC | 单用户基线 |
| 多智能体值分解 | QMIX, VDN, IQL | 协作多智能体 |
| 多智能体策略梯度 | MAPPO, IPPO, MADDPG, MATD3, COMA | 协作/竞争多智能体 |
| 本文算法 | GRPO + Stackelberg + Shapley | 博弈-RL 融合 |
| 偏好对齐扩展 | SimPO | 人类偏好对齐扩展 |

## 附录 C. 第二版修订要点

| 编号 | 修订内容 | 修订原因 |
|------|----------|----------|
| R1 | 将 DVFS 指数符号从 $\hat{\alpha}$ 改为 $\beta$ | 消除与卸载比例 $\alpha_i$ 的视觉混淆 |
| R2 | 将奖励权重从 $(\hat{\alpha}, \hat{\beta}, \hat{\gamma})$ 改为 $(\omega_\alpha, \omega_\beta, \omega_\gamma)$ | 消除与物理量符号的冲突 |
| R3 | 将 SINR 符号从 $\text{SINR}_{i,k}$ 改为 $\Gamma_{i,k}$ | 缩短公式长度，提高可读性 |
| R4 | 将到达率符号从 $\lambda_k$ 改为 $\lambda_k^{\text{arr}}$ | 消除与权衡系数 $\lambda$ 的歧义 |
| R5 | 补充卸载比例 Sigmoid 映射 | 初稿使用线性映射与代码实现不一致 |
| R6 | 重写命题 1 证明 | 初稿二阶导展开不完整 |
| R7 | 修正 M/M/1 到达率单位 | 初稿将 $\lambda_k$ 标注为 tasks/s，实际为 bits/s |
| R8 | 补充动态需求更新机制 | 初稿遗漏了领导者-跟随者外循环中的动量更新 |
| R9 | 补充 CP-nets 条件偏好细节 | 初稿仅给出概述，缺少效用矩阵构造 |
| R10 | 补充命题 0（最优本地频率）完整证明 | 初稿缺少二阶充分条件验证 |
| R11 | 补充注记 1–3 | 增强物理直觉与理论动机的阐释 |
| R12 | 补充并行/顺序时延模型的区分 | 初稿两种模型混用，缺乏明确界定 |
| R13 | 在符号表中新增"易混符号辨析"列 | 统一解决符号一致性检查清单指出的全部冲突 |

## 附录 D. 第三版补全要点

| 编号 | 补全内容 | 来源 |
|------|----------|------|
| S1 | 符号表 RL 参数中补充 $r_i(t)$（即时奖励） | v1 符号表 |
| S2 | 符号表通信参数中补充 $R_{i,k}^{\text{dl}}$（下行速率） | v1 符号表 |
| S3 | I-B-2 节补充下行传输速率建模说明 | v1 §I-B-2 |
| S4 | I-D 节补充位置扰动噪声项 $\boldsymbol{\epsilon}_i(t)$ | v1 §I-D |
| S5 | I-D 节补充显式笛卡尔初始位置公式 | v1 §I-D |
| S6 | II-C 节补充系统功率约束 C6 的投影处理说明 | v1 §II-C |
| S7 | II-C 节补充障碍项的物理解释 | v1 §II-C |
| S8 | III-B-1 节补充参与者集合的显式定义 | v1 §III-B-1 |
| S9 | III-B-1 节补充 Nash 均衡定义的直观解释 | v1 §III-B-1 |
| S10 | III-B-2 节补充 IBR 收敛性保证（对角严格凹条件，Rosen 1965） | v1 §III-B-2 |
| S11 | III-D-2 节补充四条 Shapley 公理的完整描述 | v1 §III-D-2 |
| S12 | IV-C 节补充离散头与连续头的详细说明 | v1 §IV-C |
| S13 | IV-D-1 节补充非最近基站惩罚的详细解释 | v1 §IV-D-1 |
| S14 | IV-D-2 节补充 baseline 和 observed 的显式公式 | v1 §IV-D-2 |
| S15 | IV-D-4 节补充 $\omega_\gamma$ 的自适应更新公式与归一化公式 | v1 §IV-D-4 |
| S16 | IV-E-1 节补充 Critic 输入的显式拼接公式 $s_{\text{critic}}$ | v1 §IV-E-1 |
| S17 | IV-E-2 节将熵正则化独立为显式公式 | v1 §IV-E-2 |
| S18 | VI-B 节补充 GAE λ、值函数系数、每Episode步数、激活函数、探索衰减步数 | v1 §VI-B |
| S19 | VI-D 节基线算法表补充"特点"列 | v1 §VI-D |
| S20 | VI-E 节消融实验表补充"预期影响"列 | v1 §VI-E |
| S21 | 附录 A 补充"对比基线"行 | v1 附录 A |
| S22 | 附录 B 补充"适用场景"列 | v1 附录 B |

---

*文档结束 — 定稿 v3.0*
