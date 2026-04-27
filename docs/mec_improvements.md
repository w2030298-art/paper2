# GameTheory MEC 系统框架改进方案

## 1. 博弈论模型深化

### 1.1 基于凸优化的最优定价机制

**问题分析**：当前 `StackelbergGame` 仅使用需求阈值 0.8/0.3 做乘性调整，缺乏最优性保证。领导者（MEC 运营商）定价问题应建模为凸优化。

**数学公式**：

领导者利润最大化问题：

```
max_p  Σ_k [p_k · D_k(p_k) - C_k(D_k(p_k))]
s.t.   p_min ≤ p_k ≤ p_max,  ∀k
```

其中需求函数 `D_k(p_k) = d_k^0 · exp(-η_k · p_k)`（对数线性需求模型），成本函数 `C_k(x) = c_k · x^2`。

**凸性证明条件**：利润函数 `Π_k(p_k) = p_k · d_k^0 · exp(-η_k · p_k) - c_k · [d_k^0 · exp(-η_k · p_k)]^2`

二阶条件：`d²Π_k/dp_k² < 0` 成立当且仅当 `η_k > 0` 且 `c_k · (d_k^0)^2 · η_k^2 · exp(-2η_k p_k) > 0`（始终成立），因此利润函数关于 `p_k` 为严格凹函数 ⟹ 凸优化。

最优价格闭式解（KKT 条件）：

```
p_k* = 1/η_k + 2·c_k·d_k^0·exp(-η_k·p_k*)/1
```

需用 Lambert W 函数或牛顿迭代求解。

**伪代码**：

```python
class OptimalPricingMechanism:
    def __init__(self, eta, c, d0, p_min, p_max, tol=1e-6):
        self.eta = eta      # 价格敏感度 shape (K,)
        self.c = c          # 边际成本系数 shape (K,)
        self.d0 = d0        # 基础需求量 shape (K,)
        self.p_min, self.p_max = p_min, p_max
        self.tol = tol

    def demand(self, p):
        return self.d0 * np.exp(-self.eta * p)

    def profit_gradient(self, p):
        """dΠ/dp for each BS k"""
        D = self.demand(p)
        dD_dp = -self.eta * D
        grad = D + p * dD_dp - 2 * self.c * D * dD_dp
        return grad

    def solve_optimal_price(self, max_iter=100):
        """Newton's method for KKT conditions"""
        p = (self.p_min + self.p_max) / 2 * np.ones_like(self.eta)
        for _ in range(max_iter):
            g = self.profit_gradient(p)
            H = self._profit_hessian(p)
            delta = -g / (H + 1e-8)
            p = np.clip(p + delta, self.p_min, self.p_max)
            if np.max(np.abs(g)) < self.tol:
                break
        return p

    def _profit_hessian(self, p):
        D = self.demand(p)
        eta = self.eta
        dD = -eta * D
        d2D = eta**2 * D
        return 2*dD + p*d2D - 2*self.c*(dD**2 + D*d2D)
```

### 1.2 双层博弈：Nash均衡 + Stackelberg均衡

**动机**：用户间存在资源竞争（非合作博弈），领导者-跟随者之间为 Stackelberg 博弈，形成双层结构。

**数学模型**：

- 上层（领导者）：MEC 运营商选择定价 `p = (p_1,...,p_K)` 最大化利润
- 下层（跟随者 Nash 博弈）：用户 `i` 在给定 `p` 下选择卸载策略 `a_i` 最小化个体成本：

```
min_{a_i}  J_i(a_i, a_{-i}; p) = α·T_i(a_i, a_{-i}) + β·E_i(a_i) + p_{k_i}·D_i(a_i)
```

Nash 均衡条件：`∀i, J_i(a_i*, a_{-i}*; p) ≤ J_i(a_i, a_{-i}*; p)`

**求解算法**：迭代最佳响应（Iterative Best Response, IBR）

```python
class BilevelGameSolver:
    def __init__(self, pricing_mechanism, users, max_outer=50, max_inner=200):
        self.pricing = pricing_mechanism
        self.users = users
        self.max_outer = max_outer
        self.max_inner = max_inner

    def solve(self):
        """Bilevel: outer Stackelberg, inner Nash"""
        p = self.pricing.solve_optimal_price()  # initial pricing

        for t_outer in range(self.max_outer):
            # Inner loop: Nash equilibrium via iterative best response
            actions = self._solve_nash_equilibrium(p)

            # Outer loop: update pricing based on realized demand
            demand = self._compute_demand(actions)
            p_new = self.pricing.update_with_demand(demand)

            if np.linalg.norm(p_new - p) < 1e-5:
                break
            p = p_new

        return p, actions

    def _solve_nash_equilibrium(self, prices):
        """Iterative best response for user Nash game"""
        N = len(self.users)
        actions = [u.initial_action() for u in self.users]

        for t_inner in range(self.max_inner):
            old_actions = [a.copy() for a in actions]
            for i in range(N):
                actions[i] = self.users[i].best_response(
                    actions[:i] + actions[i+1:], prices
                )
            # Check convergence
            max_diff = max(
                np.linalg.norm(actions[i] - old_actions[i]) for i in range(N)
            )
            if max_diff < 1e-4:
                break

        return actions
```

### 1.3 基于势函数的精确 Shapley 值与蒙特卡洛近似

**问题分析**：当前简化边际贡献忽略了联盟顺序效应。精确 Shapley 值计算复杂度为 `O(2^N)`，需蒙特卡洛近似。

**势函数定义**：若博弈 `G = (N, v)` 为势博弈（potential game），则存在势函数 `Φ: A → R` 使得：

```
∀i, ∀a_i, a_i': v_i(a_i, a_{-i}) - v_i(a_i', a_{-i}) = Φ(a_i, a_{-i}) - Φ(a_i', a_{-i})
```

在此框架下，Shapley 值的精确计算：

```
φ_i(v) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)!/|N|!] · [v(S∪{i}) - v(S)]
```

**蒙特卡洛近似方案**（ApproShapley）：

```python
class MonteCarloShapley:
    def __init__(self, n_agents, coalition_value_fn, n_samples=1000):
        self.N = n_agents
        self.v = coalition_value_fn  # v(S) -> float
        self.n_samples = n_samples

    def compute(self):
        """Monte Carlo approximation of Shapley values"""
        shapley_values = np.zeros(self.N)

        for _ in range(self.n_samples):
            # Random permutation
            perm = np.random.permutation(self.N)
            coalition = set()
            prev_value = 0.0

            for agent in perm:
                coalition.add(agent)
                curr_value = self.v(frozenset(coalition))
                marginal = curr_value - prev_value
                shapley_values[agent] += marginal
                prev_value = curr_value

        shapley_values /= self.n_samples
        return shapley_values

    def compute_with_antithetic(self):
        """Variance reduction via antithetic sampling"""
        shapley_vals = np.zeros(self.N)

        for _ in range(self.n_samples // 2):
            perm = np.random.permutation(self.N)
            anti_perm = perm[::-1]  # antithetic permutation

            for p in [perm, anti_perm]:
                coalition = set()
                prev_val = 0.0
                for agent in p:
                    coalition.add(agent)
                    curr_val = self.v(frozenset(coalition))
                    shapley_vals[agent] += curr_val - prev_val
                    prev_val = curr_val

        shapley_vals /= self.n_samples
        return shapley_vals
```

**复杂度分析**：精确计算 `O(N! · T_v)` → 蒙特卡洛 `O(M · N · T_v)`，其中 `M` 为采样数，`T_v` 为联盟值函数评估时间。方差界：`Var(φ̂_i) ≤ V_max^2 / M`。

### 1.4 EFX 公平分配与 CP-nets 偏好集成

**动机**：资源分配需满足"无嫉妒至一物"（EFX）公平性。

**EFX 条件**：对任意用户 `i, j`，令 `A_i` 为 `i` 的分配包，则：
```
v_i(A_i) ≥ v_i(A_j \ {g})  ∀g ∈ A_j（价值最小的物品）
```

**集成点**：在 Shapley 收益分配后，加入 EFX 校验层。若 EFX 被违反，进行转移支付补偿。

```python
class EFXFairAllocation:
    def __init__(self, n_agents, valuations):
        self.N = n_agents
        self.v = valuations  # v[i][bundle] -> value

    def check_efx(self, allocation):
        """Check if allocation satisfies EFX"""
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                A_j = allocation[j]
                for item in A_j:
                    reduced = A_j - {item}
                    if self.v[i](allocation[i]) < self.v[i](reduced):
                        return False, (i, j, item)
        return True, None

    def repair_allocation(self, allocation, shapley_values):
        """Post-hoc EFX repair via transfer payments"""
        is_efx, violation = self.check_efx(allocation)
        transfers = np.zeros(self.N)

        while not is_efx:
            i, j, item = violation
            # Transfer payment from j to i
            deficit = self.v[i](allocation[j] - {item}) - self.v[i](allocation[i])
            transfers[i] += deficit * 0.5
            transfers[j] -= deficit * 0.5
            # Re-check after transfer
            is_efx, violation = self.check_efx(allocation)

        return transfers
```

---

## 2. 系统模型严谨性

### 2.1 M/M/1 排队时延模型

**改进动机**：当前时延模型忽略服务器排队效应，低估高负载时延。

**数学模型**：

MEC 服务器 `k` 建模为 M/M/1 队列：
- 到达率：`λ_k = Σ_{i∈U_k} (1-α_i) · D_i / t_slot`
- 服务率：`μ_k = f_k^edge / C_avg`（CPU周期/秒 ÷ 平均任务CPU需求）
- 利用率：`ρ_k = λ_k / μ_k`，需保证 `ρ_k < 1`
- 平均排队时延：`T_queue_k = ρ_k / [μ_k · (1 - ρ_k)]`
- 平均系统时延：`T_system_k = 1 / [μ_k · (1 - ρ_k)]`

**修正后总时延**：

```
T_total_i = α_i · T_local_i + (1-α_i) · [T_trans_i + T_queue_k + T_exec_i^edge]
```

```python
class QueueingDelayModel:
    def __init__(self, service_rate, stability_margin=0.95):
        self.mu = service_rate
        self.rho_max = stability_margin

    def compute_delay(self, arrival_rates):
        """M/M/1 queueing delay per server"""
        rho = np.clip(arrival_rates / self.mu, 0, self.rho_max)
        T_queue = rho / (self.mu * (1 - rho))
        T_system = 1.0 / (self.mu * (1 - rho))
        return T_queue, T_system, rho

    def compute_arrival_rate(self, offload_ratios, task_sizes, t_slot):
        """Aggregate arrival rate at each BS"""
        # offload_ratios: (N_users,), task_sizes: (N_users,)
        return np.sum((1 - offload_ratios) * task_sizes) / t_slot
```

### 2.2 非理想 DVFS 能耗模型

**改进动机**：理想 DVFS 假设 `E = κ·f²·C` 忽略了泄漏功耗和非理想缩放。

**修正模型**：

```
E_compute = (κ_dyn · f^α + P_leak) · C / f
         = κ_dyn · f^(α-1) · C + P_leak · C / f
```

其中 `α ∈ [2.5, 3.0]` 为工艺相关参数（CMOS: α ≈ 2.7），`P_leak` 为泄漏功耗。

**最优频率**（能耗-时延权衡）：

```
min_f  E(f) + λ · T(f) = κ_dyn · f^(α-1) · C + P_leak · C/f + λ · C/f
```

对 `f` 求导令其为零：`(α-1)·κ_dyn·f^(α-2)·C = (P_leak + λ)·C/f²`

```python
class DVFSEnergyModel:
    def __init__(self, kappa_dyn=1e-27, alpha=2.7, P_leak=0.01,
                 f_min=0.5e9, f_max=3.0e9):
        self.kappa = kappa_dyn
        self.alpha = alpha
        self.P_leak = P_leak
        self.f_min = f_min
        self.f_max = f_max

    def compute_energy(self, freq, cpu_cycles):
        """Non-ideal DVFS energy model"""
        E_dyn = self.kappa * freq**(self.alpha - 1) * cpu_cycles
        E_leak = self.P_leak * cpu_cycles / freq
        return E_dyn + E_leak

    def optimal_frequency(self, cpu_cycles, delay_weight):
        """Solve for energy-optimal frequency"""
        # (alpha-1)*kappa*f^(alpha-2) = (P_leak + lambda)/f^2
        # f^alpha = (P_leak + lambda) / [(alpha-1)*kappa]
        f_opt = ((self.P_leak + delay_weight) /
                 ((self.alpha - 1) * self.kappa)) ** (1.0 / self.alpha)
        return np.clip(f_opt, self.f_min, self.f_max)
```

### 2.3 3GPP TR 38.901 LOS/NLOS 信道模型

**概率 LOS 模型**：

```
P_LOS(d) = min(1, 18/d) · (1 - exp(-d/36)) + exp(-d/36)
```

（注：以上为 UMi Street Canyon 模型；原文中的简化公式亦可使用）

**路径损耗**：

```
LOS:  PL_LOS(d) = 32.4 + 21·log10(d) + 20·log10(f_c)
NLOS: PL_NLOS(d) = 35.3·log10(d) + 22.4 + 21.3·log10(f_c) - 0.3·(h_UT - 1.5)
```

**综合路径损耗**：

```
PL(d) = P_LOS(d) · PL_LOS(d) + (1 - P_LOS(d)) · PL_NLOS(d)
```

```python
class Channel3GPP:
    def __init__(self, fc_ghz=3.5, h_bs=25, h_ut=1.5):
        self.fc = fc_ghz
        self.h_bs = h_bs
        self.h_ut = h_ut

    def prob_los(self, d):
        """UMi Street Canyon LOS probability"""
        return np.minimum(1, 18.0/d) * (1 - np.exp(-d/36)) + np.exp(-d/36)

    def path_loss_los(self, d):
        return 32.4 + 21*np.log10(d) + 20*np.log10(self.fc)

    def path_loss_nlos(self, d):
        return (35.3*np.log10(d) + 22.4 +
                21.3*np.log10(self.fc) - 0.3*(self.h_ut - 1.5))

    def composite_path_loss(self, d):
        p = self.prob_los(d)
        return p * self.path_loss_los(d) + (1-p) * self.path_loss_nlos(d)

    def compute_sinr(self, d_target, d_interferers, p_target, p_interferers,
                     sigma2=-174+10*np.log10(20e6)):
        """SINR with inter-user interference"""
        PL_t = self.composite_path_loss(d_target)
        S = p_target - PL_t  # dBm

        I_total = 0
        for d_j, p_j in zip(d_interferers, p_interferers):
            PL_j = self.composite_path_loss(d_j)
            I_total += 10**((p_j - PL_j) / 10)  # linear

        S_lin = 10**(S/10)
        N_lin = 10**(sigma2/10)
        sinr = S_lin / (I_total + N_lin)
        return 10 * np.log10(sinr)
```

### 2.4 多用户干扰传输速率模型

**修正 Shannon 速率**：

```
R_i = B · log2(1 + SINR_i)

SINR_i = p_i · |h_{i,k}|² / (Σ_{j≠i} p_j · |h_{j,k}|² + σ²)
```

---

## 3. 观测空间增强

### 3.1 完整特征工程表

| 特征组 | 维度 | 描述 |
|--------|------|------|
| 基站队列长度 | K | 各 BS 当前队列任务数 |
| 信道 SNR | K | 到各 BS 的瞬时 SNR |
| BS CPU 负载 | K | 各 BS CPU 利用率 ∈ [0,1] |
| 任务特征 | 5 | (数据量, CPU需求, 最大时延, 优先级, 任务类型) |
| **新增：均衡价格历史** | M·K | 最近 M 步各 BS 的均衡定价 |
| **新增：他方动作统计** | K+1 | 其他智能体选择各目标的频率 |
| **新增：Shapley历史** | M | 最近 M 步的 Shapley 值 |
| **新增：CSI差分 ΔSNR** | K | SNR 一阶差分（变化趋势） |
| **新增：队列导数 ΔQ** | K | 队列长度一阶差分（拥塞趋势） |
| **新增：排队利用率 ρ** | K | 各 BS 的 M/M/1 利用率 |
| **总计** | **3K + 5 + M·K + (K+1) + M + K + K + K** | |

设 `M=5, K=5` 时总维度：`15 + 5 + 25 + 6 + 5 + 5 + 5 + 5 = 71`

```python
class EnhancedObservation:
    def __init__(self, K, M=5):
        self.K = K
        self.M = M
        self.price_history = deque(maxlen=M)
        self.action_counts = np.zeros(K + 1)
        self.shapley_history = deque(maxlen=M)
        self.prev_snr = np.zeros(K)
        self.prev_queue = np.zeros(K)

    def get_obs(self, queue_lengths, snr, cpu_load, task_features,
                current_prices, current_shapley, rho):
        # CSI差分
        delta_snr = snr - self.prev_snr
        self.prev_snr = snr.copy()

        # 队列导数
        delta_q = queue_lengths - self.prev_queue
        self.prev_queue = queue_lengths.copy()

        # 更新历史
        self.price_history.append(current_prices)
        self.shapley_history.append(current_shapley)

        # 价格历史填充
        price_hist = np.zeros(self.M * self.K)
        for t, p in enumerate(self.price_history):
            price_hist[t*self.K:(t+1)*self.K] = p

        # Shapley历史填充
        shap_hist = np.zeros(self.M)
        for t, s in enumerate(self.shapley_history):
            shap_hist[t] = s

        # 他方动作频率（归一化）
        action_freq = self.action_counts / max(self.action_counts.sum(), 1)

        obs = np.concatenate([
            queue_lengths, snr, cpu_load,       # 3K
            task_features,                       # 5
            price_hist,                          # M*K
            action_freq,                         # K+1
            shap_hist,                           # M
            delta_snr,                           # K
            delta_q,                             # K
            rho                                  # K
        ])
        return obs

    @property
    def obs_dim(self):
        return 3*self.K + 5 + self.M*self.K + (self.K+1) + self.M + 3*self.K
```

---

## 4. 奖励函数重构

### 4.1 分层奖励设计

**数学定义**：

**即时奖励**：
```
r_imm = -w_l · T_total / T_max - w_e · E_total / E_budget - w_d · max(0, T_total - T_max)² / T_max²
```

**协作奖励**（基于 Shapley 值）：
```
r_coop = φ_i · [V(N) - Σ_j V({j})] / |N|
```
其中 `V(N)` 为全联盟收益，`V({j})` 为单智能体独立收益。

**博弈一致性奖励**：
```
r_eq = -|| a_i - a_i^*(s) ||² / dim(A)
```
其中 `a_i^*(s)` 为 Stackelberg 均衡下的最优动作。

**总奖励**：
```
R_i = α · r_imm + β · r_coop + γ · r_eq
```

**梯度分析**：
- `∇r_imm` 直接关联物理量梯度，信号清晰
- `∇r_coop` 通过 Shapley 值间接传播，可能有高方差 → 需基线减去
- `∇r_eq` 引导策略靠近均衡，但可能与 `r_imm` 冲突 → 需动态权重

**奖励冲突检测与缓解**：

```python
class HierarchicalReward:
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2,
                 conflict_threshold=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.conflict_threshold = conflict_threshold

    def compute(self, T_total, E_total, T_max, E_budget,
                shapley_val, coalition_value, indiv_values,
                action, equilibrium_action):
        # Immediate reward
        r_imm = (-0.4 * T_total / T_max
                 - 0.3 * E_total / E_budget
                 - 0.3 * max(0, T_total - T_max)**2 / T_max**2)

        # Cooperative reward
        cooperation_gain = coalition_value - sum(indiv_values)
        r_coop = shapley_val * cooperation_gain / len(indiv_values)

        # Equilibrium reward
        action_diff = np.linalg.norm(action - equilibrium_action)
        r_eq = -action_diff**2 / len(action)

        # Conflict detection: check gradient cosine similarity
        rewards = np.array([r_imm, r_coop, r_eq])
        total = self.alpha * r_imm + self.beta * r_coop + self.gamma * r_eq

        return total, {'r_imm': r_imm, 'r_coop': r_coop, 'r_eq': r_eq}

    def adaptive_weights(self, reward_history, window=50):
        """Dynamically adjust weights to mitigate conflicts"""
        if len(reward_history) < window:
            return self.alpha, self.beta, self.gamma

        recent = reward_history[-window:]
        r_imm_seq = [r['r_imm'] for r in recent]
        r_coop_seq = [r['r_coop'] for r in recent]
        r_eq_seq = [r['r_eq'] for r in recent]

        # Cosine similarity between reward gradients
        corr_ic = np.corrcoef(r_imm_seq, r_coop_seq)[0, 1]
        corr_ie = np.corrcoef(r_imm_seq, r_eq_seq)[0, 1]

        # If negative correlation (conflict), reduce conflicting weight
        beta_adj = self.beta * max(0.5, 1 + corr_ic)
        gamma_adj = self.gamma * max(0.5, 1 + corr_ie)

        # Renormalize
        total = self.alpha + beta_adj + gamma_adj
        return self.alpha/total, beta_adj/total, gamma_adj/total
```

---

## 5. RL算法与博弈论的深度融合


### 5.2 Shapley 值信用分配

**替代 QMIX/VDN 的简单值分解**：

```python
class ShapleyCredit:
    def __init__(self, n_agents, shapley_calculator):
        self.N = n_agents
        self.shapley = shapley_calculator

    def assign_credit(self, team_reward, observations, actions):
        """Shapley-based credit assignment for MAPPO"""
        def coalition_value(S):
            # Simulate coalition S acting, others taking default
            return self._evaluate_coalition(S, observations, actions)

        phi = self.shapley.compute()  # Monte Carlo Shapley values

        # Normalize to sum to team reward
        phi_sum = np.sum(phi)
        if abs(phi_sum) > 1e-8:
            phi = phi * team_reward / phi_sum
        else:
            phi = np.full(self.N, team_reward / self.N)

        return phi  # Individual rewards for each agent
```

### 5.3 CTDE 架构设计

**中央 Critic**：观测全局状态 `s_global = [s_1, ..., s_N, p^*, a^*_eq]`（含博弈均衡提示）

**分散 Actor**：仅观测局部 `o_i`，输出动作 `a_i`

```python
class CTDEWithGameTheory(nn.Module):
    def __init__(self, local_obs_dim, global_state_dim, action_dim, n_agents):
        super().__init__()
        self.n_agents = n_agents

        # Decentralized actors (local observation only)
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(local_obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ) for _ in range(n_agents)
        ])

        # Centralized critic (global state + equilibrium hints)
        critic_input = global_state_dim + action_dim  # equilibrium action
        self.critic = nn.Sequential(
            nn.Linear(critic_input, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Equilibrium embedding
        self.eq_encoder = nn.Linear(action_dim * n_agents, 64)

    def act(self, local_obs, agent_id):
        """Decentralized execution"""
        return self.actors[agent_id](local_obs)

    def evaluate(self, global_state, eq_actions):
        """Centralized training with equilibrium hints"""
        eq_embed = self.eq_encoder(eq_actions.flatten())
        critic_input = torch.cat([global_state, eq_embed])
        return self.critic(critic_input)
```

### 5.4 Warm-Start 伪代码

```
Algorithm: Game-Theoretic Warm-Start for MADRL
Input: Environment env, Stackelberg solver stack_solver, N episodes
Output: Pre-trained policy networks {π_θ_i}

1. for episode = 1 to N_warmup:
2.   s = env.reset()
3.   p*, {a*_i} = stack_solver.solve(s)   // Solve Stackelberg
4.   for each agent i:
5.     // Supervised pre-training: imitate equilibrium
6.     loss_i = ||π_θ_i(o_i) - a*_i||²
7.     θ_i ← θ_i - lr · ∇loss_i
8.   end for
9.   // Add exploration noise gradually
10.  noise_scale = max(0, 1 - episode/N_warmup) * σ_init
11. end for
12. // Transition to RL fine-tuning with reduced lr
13. for episode = N_warmup+1 to N_total:
14.   Run standard MAPPO/GRPO with warm-started θ
15. end for
```

---

## 6. 动作空间与约束处理

### 6.1 参数化动作映射

```python
class ParameterizedActionSpace:
    def __init__(self, K, f_min=0.5e9, f_max=3e9, P_min=0.01, P_max=0.5):
        self.K = K
        self.f_min, self.f_max = f_min, f_max
        self.P_min, self.P_max = P_min, P_max

    def map_action(self, raw_action):
        """Map raw NN output [-1,1] to physical parameters"""
        target = raw_action['target']  # Discrete(K+1), unchanged

        ratio = raw_action['ratio']    # Box(-1,1,(3,))
        # Sigmoid for offload ratio [0, 1]
        offload_ratio = torch.sigmoid(ratio[0] * 5)  # 5 for sharper mapping

        # Affine for CPU frequency [f_min, f_max]
        cpu_freq = self.f_min + (ratio[1] + 1) / 2 * (self.f_max - self.f_min)

        # Affine for TX power [P_min, P_max]
        tx_power = self.P_min + (ratio[2] + 1) / 2 * (self.P_max - self.P_min)

        return {
            'target': target,
            'offload_ratio': offload_ratio,
            'cpu_freq': cpu_freq,
            'tx_power': tx_power
        }
```

### 6.2 约束投影层

```python
class ConstraintProjection(nn.Module):
    """Differentiable constraint projection using optimization layer"""
    def __init__(self, K, E_max, T_max, P_total_max):
        super().__init__()
        self.K = K
        self.E_max = E_max
        self.T_max = T_max
        self.P_total = P_total_max

    def forward(self, actions, state):
        """Project actions onto feasible set C1-C5"""
        offload = actions['offload_ratio']
        freq = actions['cpu_freq']
        power = actions['tx_power']

        # C1: 0 ≤ offload ≤ 1 (already ensured by sigmoid)
        # C2: f_min ≤ f ≤ f_max (already ensured by affine)
        # C3: P_min ≤ p ≤ P_max (already ensured by affine)

        # C4: Total energy budget
        E_est = self._estimate_energy(offload, freq, power, state)
        if E_est > self.E_max:
            scale = (self.E_max / E_est) ** 0.5
            freq = freq * scale
            power = power * scale

        # C5: Total power constraint (multi-agent)
        total_power = power.sum()
        if total_power > self.P_total:
            power = power * (self.P_total / total_power)

        return {'offload_ratio': offload, 'cpu_freq': freq, 'tx_power': power}
```

### 6.3 软惩罚 vs 硬裁剪分析

**硬裁剪**问题：`a_clipped = clip(a, a_min, a_max)` → 梯度在边界处为零，导致策略梯度消失。

**软惩罚**方案：

```
L_constraint = μ · Σ_c max(0, g_c(a))²
```

其中 `g_c(a) ≤ 0` 为约束条件，`μ` 为惩罚系数（自适应增大）。

**梯度分析**：

- 硬裁剪：`∂a_clip/∂a = 0` 当 `a` 超界 → 无信息梯度
- 软惩罚：`∂L/∂a = 2μ · g_c(a) · ∇g_c(a)` → 梯度持续存在且指向可行域

**建议**：使用 Log-barrier + 软惩罚混合：

```
L_barrier = -ε · Σ_c log(-g_c(a))    (interior penalty)
L_penalty = μ · Σ_c max(0, g_c(a))²  (exterior penalty)
L_total = L_barrier + L_penalty
```

---

## 7. 可扩展性与实验设计

### 7.1 参数化实验矩阵

| 配置 | K (BS) | U (Users) | 观测维度 | 动作维度 | 预计训练时间 |
|------|--------|-----------|----------|----------|-------------|
| Small | 3 | 6 | ~46 | 4 | ~2h |
| Medium | 5 | 10 | ~71 | 4 | ~8h |
| Large | 10 | 20 | ~126 | 4 | ~24h |

### 7.2 Ablation Study 清单

| ID | 消融项 | 描述 | 预期影响 |
|----|--------|------|----------|
| A1 | w/o Shapley | 用均匀分配替代Shapley | 公平性下降，收敛变慢 |
| A2 | w/o 博弈初始化 | 随机初始化策略 | 前期探索效率低 |
| A3 | w/o 移动性 | 用户静止 | 性能上界，验证移动性挑战 |
| A4 | w/o 协作奖励 | 仅用即时奖励 | 自私策略，系统效率低 |
| A5 | w/o 排队模型 | 去除M/M/1 | 高负载性能失真 |
| A6 | w/o 均衡奖励 | 去除r_eq | 策略偏离均衡 |
| A7 | w/o DVFS | 理想κf²C模型 | 能耗估计偏乐观 |

### 7.3 基线算法对比

| 类别 | 算法 | 特点 |
|------|------|------|
| 启发式 | Greedy, Random, Local-only, Full-offload | 无学习能力 |
| 单智能体 DRL | DQN, DDPG, TD3, SAC | 无多智能体协调 |
| 多智能体 DRL | QMIX, MAPPO (no game), MADDPG | 无博弈论层 |
| **本文** | **GRPO+Stackelberg+Shapley** | 完整框架 |

### 7.4 收敛性分析框架

**策略偏差与均衡偏离**：

定义策略偏差 `Δπ = D_KL(π_θ || π_eq)`，Stackelberg 均衡偏离量 `δ_SE = ||a_θ - a^*||`

**定理（非正式）**：在满足以下条件时策略梯度方法收敛至 ε-近似均衡：
1. 奖励函数 Lipschitz 连续且有界
2. 转移概率关于动作连续
3. 学习率衰减满足 Robbins-Monro 条件

则 `E[δ_SE(T)] ≤ ε + O(1/√T)`

---

## 8. 论文写作建议

### 8.1 Mathematical Notation 一致性检查清单

| 符号 | 含义 | 注意事项 |
|------|------|----------|
| `α_i` | 用户i的本地计算比例 | 避免与奖励权重α混用 |
| `p_k` | BS k 的定价 | 与传输功率 `P_i` 区分大小写 |
| `f_i` | 用户i的CPU频率 | 与BS频率 `f_k^edge` 区分 |
| `φ_i` | 用户i的Shapley值 | 不要与势函数Φ混淆 |
| `ρ_k` | BS k的利用率 | 不要与折扣因子混淆 |
| `λ` | 能耗-时延权衡系数 | 与到达率 `λ_k` 区分（用不同下标） |

### 8.2 建议论文结构

```
I.   Introduction
II.  System Model
     A. Network Architecture
     B. Communication Model (3GPP channel + interference)
     C. Computation Model (DVFS + queueing)
     D. Mobility Model
III. Problem Formulation  ← 新增独立小节
     A. Optimization Objective (P1: min-max)
     B. Constraint Analysis (C1-C5)
     C. NP-hardness Analysis
IV.  Game-Theoretic Framework
     A. Stackelberg Game (Leader pricing, Follower response)
     B. Nash Equilibrium among Users
     C. Shapley Value-based Cooperative Game
     D. Equilibrium Existence and Uniqueness (Theorem 1)
V.   MADRL Algorithm Design
     A. CTDE Architecture
     B. GRPO with Game-Theoretic Initialization
     C. Shapley-based Credit Assignment
     D. Convergence Analysis (Theorem 2)
VI.  Simulation Results
     A. Setup & Parameters
     B. Convergence Analysis
     C. Performance Comparison
     D. Ablation Study
     E. Scalability Analysis
     F. Complexity Analysis  ← 新增
VII. Conclusion
```

### 8.3 关键定理格式

**Theorem 1** (Stackelberg 均衡存在性与唯一性)：在需求函数 `D_k(p)` 严格递减且对数凹、成本函数 `C_k(x)` 凸的条件下，所提 Stackelberg 博弈存在唯一均衡 `(p*, a*)`。

*证明思路*：
1. 跟随者最佳响应 `BR_i(p)` 的存在性 ← 用户优化问题的凸性（KKT条件）
2. `BR_i(p)` 关于 `p` 的连续性 ← Berge's 最大值定理
3. 领导者问题在 `BR` 映射下为凸 ← 1.1节的凸性证明
4. 唯一性 ← 严格凹利润函数

**Theorem 2** (收敛性)：在 Assumption 1-3 下，所提 GRPO-Shapley 算法的策略序列 `{π_θ^t}` 满足 `lim_{t→∞} E[δ_SE^t] = 0`。

### 8.4 复杂度分析表

| 算法 | 时间复杂度/步 | 空间复杂度 | Shapley 开销 |
|------|--------------|------------|-------------|
| DQN | O(|S|·|A|) | O(B·|S|) | — |
| DDPG | O(|S|·|A|) | O(B·(|S|+|A|)) | — |
| MAPPO | O(N·|S|·|A|) | O(N·|θ|) | — |
| QMIX | O(N²·|S|·|A|) | O(N·|θ| + N²) | — |
| **Ours** | O(N·|S|·|A| + M·N·T_v) | O(N·|θ| + M·N) | O(M·N·T_v) |

其中 `B`=batch size, `N`=agents, `M`=Monte Carlo samples, `T_v`=coalition evaluation time.

---

## 总结：优先实施路径

| 优先级 | 改进项 | 影响面 | 实现难度 |
|--------|--------|--------|----------|
| P0 | 排队时延模型 (2.1) | System Model 章节基础 | 低 |
| P0 | 参数化动作空间 (6.1) | 所有实验 | 低 |
| P1 | 凸优化定价 (1.1) | 博弈模块核心 | 中 |
| P1 | 分层奖励 (4.1) | 训练效果 | 中 |
| P1 | 3GPP信道 (2.3) | 模型可信度 | 低 |
| P2 | 蒙特卡洛Shapley (1.3) | 公平性精度 | 中 |
| P2 | CTDE + warm-start (5.3-5.4) | 收敛速度 | 高 |
| P3 | EFX公平 (1.4) | 理论贡献 | 高 |
| P3 | 双层博弈 (1.2) | 论文深度 | 高 |
