# docs/references/ref-mainline-a-overhaul-v4.1.md

## 修正说明

本版修正两个边界：

1. `paper2` 与仿真实验是同一个项目。所有 benchmark、oracle、ablation、OOD、plot、publication gate 均纳入 `docs/inbox/plan.md` 的模块 20，不再拆出单独仿真实验项目。
2. 论文改写与代码/实验计划不同步执行。本轮只生成写作资产、公式库存、图表库存与待确认问题，不直接改论文正文，也不输出独立论文 dispatch。

## 目标定位

本轮不再把论文定位为“又一个 Stackelberg 定价机制”，而是定位为：

> 在队列、信道、迁移状态强耦合的动态 MEC 中，提出 game-guided constrained MARL。Stackelberg 结构提供状态依赖价格先验，primal-dual 更新负责约束满足，game-aware critic 提升策略学习稳定性与可解释性。

## 为什么旧 L2/L3 降级

当前 L2/L3 formal convergence 验证的是旧系统模型、旧状态空间、旧 reward 与旧 benchmark 管线。系统模型大改后：

1. observation 增加 queue/channel/migration/heterogeneity features。
2. reward 拆成 delay、energy、queue、migration、price、constraint 等 components。
3. pricing 从静态参数变成状态依赖控制变量。
4. benchmark 目标从旧算法收敛转向新模型下的 oracle / ablation / OOD 证据链。

因此旧 L2/L3 只能作为 legacy baseline，不应作为新模型主结论或主图门禁。

## MEC 模型升级要点

### 任务模型

- 传统独立任务保留为单节点 DAG。
- 新增 `TaskDAGSpec`，支持依赖任务与语义权重。
- 不在第一阶段引入完整 network-flow scheduler，避免工程爆炸；先支持 DAG 拓扑与卸载约束。

### 队列模型

- `MM1QueueModel`：legacy 映射。
- `MMCQueueModel`：多服务器近似。
- `ParallelQueueApproxModel`：多 CPU / 多 worker。
- `FiniteCapacityQueueModel`：容量与 drop probability。
- 所有模型必须处理 `rho >= 1` 的 bounded penalty，禁止 NaN/Inf。

### 异构协作边缘

- 节点类型：BS、RSU、UAV、PEER_DEVICE、CLOUD。
- 默认 `bs_only`，协作默认关闭。
- 协作成本至少包括：数据迁移、状态迁移、结果回传。
- 实验里必须比较 no migration / nearest edge / cooperative migration。

### 能耗模型

- 采用 network-level DVFS abstraction。
- 设备本地计算、传输、边缘计算、迁移能耗拆分。
- 不引入 leakage、temperature、sleep-wakeup，避免硬件级模型偏离主线。

### 移动性模型

- 默认 Markov cell transition。
- 支持 trace adapter，但不要求第一阶段有真实轨迹数据。
- handover、migration risk、service interruption penalty 必须进入状态与 reward。

### 通信模型

- 理论：`AnalyticRateModel`，pathloss + average SINR + Shannon rate。
- 仿真：`ThreeGppLiteRateModel`，UMi/UMa、LoS/NLoS、shadowing 的简化派生。
- 消融：Rayleigh / pathloss-only，用于证明结论不依赖单一物理层颗粒度。

## 动态 Stackelberg 定价

建议价格函数：

```text
p_i(t)=clip(p0_i + alpha_q Q_i(t) - alpha_h H_i(t) + alpha_m M_i(t), p_min, p_max)
```

- `Q_i(t)`：queue pressure，价格单调非降。
- `H_i(t)`：channel quality。默认解释为高质量降低单位风险，因此价格项使用负号；若实现选择高质量抬价，论文必须同步解释为稀缺收益定价。
- `M_i(t)`：migration / handover risk，价格单调非降。
- `clip` 保证价格边界。

## 理论升级边界

可以写：

- equilibrium existence under compact action sets and continuous utilities。
- uniqueness sufficient condition under strict concavity / contraction。
- monotonic demand response under convex cost and price-linear payment。
- state-price monotonicity under nonnegative coefficients。
- primal-dual residual bound under bounded gradient/noise assumptions。
- constraint violation rate with conservative `O(1/sqrt(T))` discussion。

不能写：

- 无条件唯一均衡。
- 任意非凸神经策略的全局收敛。
- 旧 L2/L3 证明新模型收敛。

## 新实验链

- N0：smoke correctness。
- N1：small-scale oracle comparison。
- N2：ablation：no_price、no_queue、no_channel、no_migration、no_dual、no_cooperation。
- N3：OOD：用户数、边缘数、移动性、信道模型、队列模型变化。

主图必须来自 N1/N2/N3，而不是 legacy L2/L3。

## 论文改写处理方式

本轮只生成：

- `writing_ref/paper2_mainline_a_revision/model_change_inventory.md`
- `writing_ref/paper2_mainline_a_revision/equation_inventory.md`
- `writing_ref/paper2_mainline_a_revision/experiment_figure_inventory.md`
- `docs/paper_revision_pending_questions.md`
- `docs/paper_revision_manifest.md`

不直接生成论文正文 patch。论文改写需用户补充“有所区别”的具体差异后再进入单独写作迭代。
