# docs/inbox/plan.md：paper2 主线A系统模型大换血计划 v4.1

## 元信息

- 项目：`paper2`
- 计划版本：`system-model-overhaul-v4.1`
- 变更类型：`patch`
- 创建日期：2026-05-04
- 最后更新：2026-05-04
- 上一版本：`slimming-plan-v3`
- GitHub 仓库：`w2030298-art/paper2`
- 项目边界：`paper2` 与仿真对比实验是同一个项目；实验 runner、benchmark、plot、oracle、OOD、ablation 全部纳入本 plan，不再拆成独立仿真实验项目。
- 论文边界：论文改写不是本 plan 的执行主线；本 plan 只生成 `writing_ref/` 写作资产与 `docs/paper_revision_manifest.md`，不假设执行端直接改论文正文。
- 本版目标：基于主线A，把 paper2 从“静态 Stackelberg + 传统 MEC + 常规 MARL benchmark”升级为“队列/信道/迁移状态依赖动态定价 + game-guided constrained MARL + 可解释实验链”。
- 关键调整：
  - 当前模块 14 的 L2/L3 formal convergence 只验证旧模型，不再作为新模型主门禁。
  - L2/L3 结果只保留为 legacy baseline，不进入新论文主结论。
  - 新模型实验链统一在模块 20 内实现：N0 smoke、N1 oracle、N2 ablation、N3 OOD。
- 明确不做：
  - 不把当前 L2/L3 写成新模型收敛证据。
  - 不恢复已移除的 `docs_paper/`、`scripts/evaluate.py`、`scripts/generate_report.py`。
  - 不把 `experiments/`、`results/`、`figures/`、`logs/`、`checkpoints/` 重新纳入 Git tracking。
  - 不声称数学意义的全局收敛保证。
  - 不在未收到论文源文件/目标期刊/现有章节差异前直接改论文正文。

### 变更记录

| 版本 | 日期 | 变更摘要 |
|------|------|---------|
| slimming-plan-v1 | 2026-05-02 | repo hygiene、生成产物 tracking 移除、旧入口/旧工具删除 |
| slimming-plan-v2 | 2026-05-03 | 追加 targeted debugging 方案；50k 只作为预验证 |
| slimming-plan-v3 | 2026-05-03 | 正式工程收敛验证协议：L0/L1/L2/L3、raw/clean/quality report、禁止越级收敛结论 |
| system-model-overhaul-v4 | 2026-05-04 | 初版系统模型大换血计划；误拆 paper2 与仿真实验为两个项目 |
| system-model-overhaul-v4.1 | 2026-05-04 | 修正项目边界：paper2 与仿真实验合并；论文改写降级为独立写作资产，不进入代码 dispatch |

---

## Status

> 执行端读到此区块即可恢复上下文。

- 当前阶段：Mainline-A review fix completed for C-1/C-2/H-1/H-2/H-3/H-4; awaiting user/Web review.
- 当前模块：模块 14R-21 已进入实现/修复态；review scope 项为 `DONE_PENDING_REVIEW`，正式 N0/N1/N2/N3 训练仍为 `NOT_STARTED`。
- 整体进度：`21 / 21` 模块已有实现资产；实验基础设施已 dry-run，正式实验结果未产生。
- 状态：`NEEDS_REVIEW`
- 当前背景：
  - 旧模块 14 的 L2 job `l2_20260504_171744` 已停止并降级为 legacy baseline。
  - 用户确认：由于系统模型将大改，当前 L2/L3 对新模型参考意义较小。
  - 用户确认：paper2 与仿真实验是同一个项目。
  - 用户说明：论文改写与代码/实验计划有所区别，不应按同一执行链下发。
- 阻塞项：
  - Critical/High 审查项已修复并通过 dry-run/单元测试；仍需用户/Web 审核 review scope。
  - 论文正文改写需要另行基于论文源文件与用户差异要求生成，不阻塞 paper2 代码/实验计划。
  - dashboard 兼容性仍是外部复核项，不阻塞主线A系统模型实现。
- 本版总原则：
  - 先退役旧收敛门禁，再做模型大改。
  - 所有新模型默认 `enabled: false`，由 `configs/system_model_mainline_a.yaml` 显式开启。
  - 新模型另建 smoke → small-oracle → ablation → OOD 的实验链，不复用旧 L2/L3 结论。
  - 论文主结论只允许引用新模型实验链的正式结果；当前只有 dry-run 和单元测试，不得写作正式实验结论。

### Last Iteration Summary

- 模块 12-13 已完成仓库瘦身与非核心功能删除。
- 模块 14 已建立旧模型工程收敛验证协议并启动 L2，但该协议现在降级为 legacy baseline。
- 本轮进入系统模型大换血：MEC 建模、动态 Stackelberg、game-aware critic / primal-dual MARL、消融/OOD/oracle、理论与论文写作资产同步准备。

### Pending Decisions

- 用户/Web 审核 C-1/C-2/H-1/H-2/H-3/H-4 修复结果。
- 正式 N0/N1/N2/N3 训练是否开启，以及对应算力/时间窗口。
- 论文源文件实际位置、目标期刊/模板、当前章节结构、用户所说“论文改写有所区别”的具体差异。
- 3GPP 派生仿真默认场景暂定 `UMi`；如论文场景偏车联网，后续可切换为 `UMa`/RSU 派生配置。

---

# 历史模块冻结区

## 模块 1-13：历史已完成模块 `[DONE]`

- **scope: auto**
- 状态：保持完成，不重复执行，不回滚瘦身决策。
- 约束：
  - 不恢复旧入口。
  - 不恢复 `docs_paper/` 入仓。
  - 不恢复 callback 扩展机制。
  - 不把生成产物纳入 Git tracking。
- 验证：
  ```bash
  test ! -f scripts/evaluate.py
  test ! -f scripts/generate_report.py
  test ! -d docs_paper
  git status --short experiments results figures logs checkpoints | grep -E "^[AM]" && exit 1 || true
  pytest -q
  ```

## 模块 14：formal convergence verification protocol `[SUPERSEDED_FOR_NEW_MODEL]`

- **scope: review**
- 状态：不删除历史产物，但不再作为新模型主门禁。
- 处理方式：
  - 归档为 legacy baseline。
  - 禁止把旧 L2/L3 结果写入新模型主图或新论文主结论。
  - 保留旧协议文档用于说明“pre-overhaul convergence baseline”。

---

# 模块 14R：legacy convergence retirement

## 概述

- 职责：停止旧 L2/L3 对后续开发的阻塞作用，把旧收敛验证降级为 legacy baseline，并建立新模型实验门禁。
- 前置依赖：已有模块 14 L2 run 信息。
- 输出：
  - `docs/legacy_convergence_retirement.md`
  - `docs/convergence_publication_gate.md` 更新
  - `docs/report.md` 更新
  - `docs/progress.md` 更新
- 预计步骤数：5

## Step 1：检查并优雅停止旧 L2 background job

- **scope: review**
- 操作：
  - 检查 PID `26860` 是否仍存在。
  - 若仍在运行，优雅终止；保留 `experiments/formal_convergence/l2/l2_20260504_171744/manifest.json`、stdout/stderr log、已完成中间结果。
  - 若已完成，不启动 L3。
  - 不删除任何已有日志。
- 新增：`docs/legacy_convergence_retirement.md`
- 记录：
  - `run_id`
  - PID 状态
  - 停止/完成时间
  - 已产生文件路径
  - 降级原因：system model overhaul invalidates old formal gate relevance
- 验证：
  ```bash
  test -f docs/legacy_convergence_retirement.md
  grep -n "legacy baseline" docs/legacy_convergence_retirement.md
  grep -n "must not be used as main claim" docs/legacy_convergence_retirement.md
  ```

## Step 2：更新 publication gate

- **scope: auto**
- 修改：`docs/convergence_publication_gate.md`
- 要求：
  - 新增 `legacy_pre_overhaul` evidence level。
  - 明确旧 L1/L2/L3 只能进入 appendix/debug 或 regression reference。
  - 新论文主图必须来自模块 20 的新模型实验。
- 验证：
  ```bash
  grep -n "legacy_pre_overhaul" docs/convergence_publication_gate.md
  grep -n "new system model" docs/convergence_publication_gate.md
  ```

## Step 3：新增新模型验证等级定义

- **scope: auto**
- 修改：`docs/formal_convergence_protocol.md`
- 新增：
  - `N0`: model smoke check
  - `N1`: small-scale oracle comparison
  - `N2`: ablation validation
  - `N3`: OOD generalization validation
- 要求：
  - N0-N3 是新模型实验链，不继承旧 L1-L3 的结论。
- 验证：
  ```bash
  grep -n "N0" docs/formal_convergence_protocol.md
  grep -n "N3" docs/formal_convergence_protocol.md
  ```

## Step 4：更新 docs 状态

- **scope: auto**
- 修改：
  - `docs/report.md`
  - `docs/progress.md`
  - `docs/issues.md`
- 要求：
  - `docs/report.md` 状态改为 `IN_PROGRESS`，当前阶段指向模块 14R / 15。
  - issues 追加旧 L2/L3 降级说明，不标 bug。
- 验证：
  ```bash
  grep -n "legacy convergence" docs/report.md
  grep -n "system-model-overhaul-v4.1" docs/progress.md
  ```

## Step 5：防误报检查

- **scope: auto**
- 操作：
  - 检查 docs 中是否仍把旧 L2/L3 写作新模型正式结论。
  - 检查是否启动了 L3 新任务。
- 验证：
  ```bash
  grep -R "verified_converged_under_protocol" docs/ | grep -v "legacy" | grep -v "protocol" && exit 1 || true
  ps -p 26860 || true
  ```

---

# 模块 15：MEC 系统模型模块化基座

## 概述

- 职责：新增独立、可测试、可替换的 MEC 系统模型层，覆盖任务模型、队列模型、异构协作边缘、能耗模型、移动性建模、通信双轨模型。
- 前置依赖：模块 14R 完成。
- 输出：
  - `src/mec_model/`
  - `configs/system_model_mainline_a.yaml`
  - `tests/test_mec_model_*.py`
  - `docs/references/ref-mainline-a-system-model.md`
- 预计步骤数：10

## Step 1：创建 MEC 模型包骨架

- **scope: auto**
- 新增：
  - `src/mec_model/__init__.py`
  - `src/mec_model/types.py`
  - `src/mec_model/state.py`
- 在 `types.py` 中定义：
  - `TaskId = NewType("TaskId", str)`
  - `NodeId = NewType("NodeId", str)`
  - `UserId = NewType("UserId", str)`
  - `ChannelModelType = Literal["analytic", "3gpp_lite", "rayleigh", "pathloss_only"]`
  - `QueueModelType = Literal["mm1", "mmc", "parallel", "finite_capacity"]`
- 在 `state.py` 中定义：
  - `SystemState`
  - `UserState`
  - `EdgeNodeState`
  - `QueueSnapshot`
  - `MobilitySnapshot`
  - `ChannelSnapshot`
  - `MigrationSnapshot`
- 验证：
  ```bash
  python - <<'PY'
  from src.mec_model.state import SystemState
  from src.mec_model.types import ChannelModelType
  print("mec_model import ok")
  PY
  ```

## Step 2：实现任务模型

- **scope: review**
- 新增：`src/mec_model/tasks.py`
- 必须实现：
  - `TaskSegment`
  - `TaskDAGSpec`
  - `TaskArrivalProcess`
  - `generate_independent_task_batch(num_users, tasks_per_user, seed)`
  - `generate_dag_task_batch(num_users, dag_template, seed)`
  - `validate_task_dag(task_spec)`
  - `topological_task_order(task_spec)`
- 设计要求：
  - 传统标量任务是 `TaskDAGSpec` 的单节点特例。
  - DAG task 不强制替换 legacy env；通过 config 启用。
  - 每个 task segment 必含 `data_size_bits`、`cpu_cycles`、`deadline_s`、`predecessors`、`semantic_weight`。
- 验证：
  ```bash
  pytest tests/test_mec_model_tasks.py -q
  ```

## Step 3：实现多队列/并行队列模型

- **scope: review**
- 新增：`src/mec_model/queues.py`
- 必须实现：
  - `MM1QueueModel`
  - `MMCQueueModel`
  - `ParallelQueueApproxModel`
  - `FiniteCapacityQueueModel`
  - `compute_waiting_delay(queue_snapshot, model_type)`
  - `compute_queue_pressure(queue_snapshot)`
  - `compute_drop_probability(queue_snapshot)`
  - `estimate_deadline_miss_rate(queue_snapshot, deadline_s)`
- 设计要求：
  - legacy 单队列映射到 `MM1QueueModel`。
  - `rho >= 1` 返回 bounded penalty，不产生 NaN/Inf。
- 验证：
  ```bash
  pytest tests/test_mec_model_queues.py -q
  ```

## Step 4：实现异构协作边缘模型

- **scope: review**
- 新增：`src/mec_model/edge_topology.py`
- 必须实现：
  - `EdgeNodeSpec`
  - `CooperationLink`
  - `CooperativeEdgeGraph`
  - `select_candidate_edges(system_state, user_id)`
  - `compute_migration_cost(task_spec, source_node, target_node)`
  - `compute_cooperation_gain(source_node, target_node, task_spec)`
- 节点类型：
  - `BS`
  - `RSU`
  - `UAV`
  - `PEER_DEVICE`
  - `CLOUD`
- 设计要求：
  - 默认只启用 `BS`。
  - 协作只作为可选扩展，不破坏原 user-BS 两层结构。
  - 迁移成本包含数据迁移、状态迁移、结果回传。
- 验证：
  ```bash
  pytest tests/test_mec_model_edge_topology.py -q
  ```

## Step 5：实现能耗模型

- **scope: auto**
- 新增：`src/mec_model/energy.py`
- 必须实现：
  - `EnergyBreakdown`
  - `compute_local_energy(cpu_cycles, frequency_hz, kappa)`
  - `compute_tx_energy(tx_power_w, tx_time_s)`
  - `compute_edge_compute_energy(cpu_cycles, frequency_hz, kappa_edge)`
  - `compute_migration_energy(data_bits, link_rate_bps, tx_power_w)`
- 设计要求：
  - 采用 network-level DVFS abstraction。
  - 不引入 temperature/leakage/sleep-wakeup 硬件级模型。
- 验证：
  ```bash
  pytest tests/test_mec_model_energy.py -q
  ```

## Step 6：实现移动性与服务连续性模型

- **scope: review**
- 新增：`src/mec_model/mobility.py`
- 必须实现：
  - `MarkovMobilityModel`
  - `TraceMobilityModel`
  - `HandoverEvent`
  - `ServiceContinuityState`
  - `sample_next_location(state, action, rng)`
  - `detect_handover(prev_state, next_state)`
  - `compute_service_interruption_penalty(handover_event, migration_state)`
- 设计要求：
  - 默认 Markov cell transition。
  - 支持 `no_migration` / `nearest_edge` / `cooperative_migration` 三种策略。
  - 移动性强度进入 OOD 配置。
- 验证：
  ```bash
  pytest tests/test_mec_model_mobility.py -q
  ```

## Step 7：实现通信双轨模型

- **scope: review**
- 新增：`src/mec_model/channel.py`
- 必须实现：
  - `AnalyticRateModel`
  - `ThreeGppLiteRateModel`
  - `RayleighRateModel`
  - `PathlossOnlyRateModel`
  - `compute_shannon_rate_bps(bandwidth_hz, sinr_linear)`
  - `compute_sinr_linear(tx_power_w, pathloss_linear, interference_w, noise_w)`
- 理论分析：
  - `AnalyticRateModel` 使用 pathloss + average SINR + Shannon rate。
- 仿真：
  - `ThreeGppLiteRateModel` 使用 UMi/UMa、LoS/NLoS 概率、路径损耗、shadowing 的简化派生模型。
- 验证：
  ```bash
  pytest tests/test_mec_model_channel.py -q
  ```

## Step 8：新增系统模型配置

- **scope: auto**
- 新增：`configs/system_model_mainline_a.yaml`
- 内容：
  ```yaml
  system_model:
    enabled: false
    task_model: independent
    queue_model: mm1
    edge_topology: bs_only
    cooperation_enabled: false
    mobility_model: markov
    channel_model:
      theory: analytic
      simulation: 3gpp_lite
    energy_model: dvfs_network_level
  compatibility:
    preserve_legacy_default: true
    no_effect_on_legacy_convergence_module14: true
  ```
- 验证：
  ```bash
  python - <<'PY'
  import yaml
  cfg = yaml.safe_load(open("configs/system_model_mainline_a.yaml", encoding="utf-8"))
  assert cfg["system_model"]["enabled"] is False
  assert cfg["compatibility"]["preserve_legacy_default"] is True
  PY
  ```

## Step 9：新增模型层单元测试

- **scope: auto**
- 新增：
  - `tests/test_mec_model_tasks.py`
  - `tests/test_mec_model_queues.py`
  - `tests/test_mec_model_edge_topology.py`
  - `tests/test_mec_model_energy.py`
  - `tests/test_mec_model_mobility.py`
  - `tests/test_mec_model_channel.py`
- 验证：
  ```bash
  pytest tests/test_mec_model_tasks.py tests/test_mec_model_queues.py tests/test_mec_model_edge_topology.py tests/test_mec_model_energy.py tests/test_mec_model_mobility.py tests/test_mec_model_channel.py -q
  ```

## Step 10：编写系统模型实现参考

- **scope: auto**
- 新增：`docs/references/ref-mainline-a-system-model.md`
- 内容：
  - 任务模型升级边界。
  - 队列模型选择。
  - 异构协作边缘默认关闭。
  - 能耗模型采用 network-level DVFS。
  - 移动性策略。
  - 理论通信模型与仿真通信模型双轨。
- 验证：
  ```bash
  test -f docs/references/ref-mainline-a-system-model.md
  grep -n "AnalyticRateModel" docs/references/ref-mainline-a-system-model.md
  grep -n "ThreeGppLiteRateModel" docs/references/ref-mainline-a-system-model.md
  ```

---

# 模块 16：系统模型与现有环境兼容接入

## 概述

- 职责：把模块 15 的系统模型接入现有环境、reward、benchmark 与配置系统，同时保持 legacy 默认可运行。
- 前置依赖：模块 15 完成。
- 输出：
  - `src/mec_model/adapters.py`
  - 修改现有 `game_theory_env.py`
  - 修改 `src/trainer/base_trainer.py`
  - 修改 `scripts/benchmark.py`
  - 兼容性测试
- 预计步骤数：8

## Step 1：定位现有 env canonical owner

- **scope: auto**
- 操作：
  - 使用 `git ls-files "*game_theory_env.py"` 定位现有文件。
  - 预期 canonical 文件名：`game_theory_env.py`。
  - 若存在多个同名文件，只允许一个作为 canonical owner。
- 新增：`docs/references/env_owner_audit.md`
- 验证：
  ```bash
  git ls-files "*game_theory_env.py"
  test -f docs/references/env_owner_audit.md
  ```

## Step 2：创建 legacy-to-mainline-A adapter

- **scope: review**
- 新增：`src/mec_model/adapters.py`
- 必须实现：
  - `LegacyEnvSnapshot`
  - `SystemModelAdapter`
  - `build_system_state_from_legacy_env(env) -> SystemState`
  - `apply_system_decision_to_legacy_env(env, decision) -> None`
  - `extract_reward_components(env, system_state) -> dict`
- 设计要求：
  - adapter 不直接改训练器。
  - adapter 负责新旧 state/reward/action 字段映射。
- 验证：
  ```bash
  pytest tests/test_mec_model_adapters.py -q
  ```

## Step 3：给 env 增加可选 mainline-A 状态字段

- **scope: review**
- 修改：现有 `game_theory_env.py`
- 新增/调整方法：
  - `GameTheoryEnv._load_system_model_config(self)`
  - `GameTheoryEnv._build_system_state(self)`
  - `GameTheoryEnv._compute_mainline_a_metrics(self, actions)`
  - `GameTheoryEnv._compose_observation(self)`
- 设计要求：
  - 当 `system_model.enabled=false`，旧 observation shape 与旧 reward 逻辑不变。
  - 当 `system_model.enabled=true`，observation 增加 queue pressure、channel quality、migration state、edge heterogeneity features。
- 验证：
  ```bash
  pytest tests/test_env_legacy_compat.py tests/test_env_mainline_a_state.py -q
  ```

## Step 4：升级 reward components

- **scope: review**
- 修改：
  - 现有 `game_theory_env.py`
  - `src/trainer/base_trainer.py`
- 新增 reward components：
  - `delay_cost`
  - `energy_cost`
  - `queue_penalty`
  - `migration_penalty`
  - `deadline_violation_penalty`
  - `cooperation_gain`
  - `price_payment`
  - `provider_revenue`
  - `constraint_penalty`
- 要求：
  - 旧 `comm_score` 仍可计算。
  - 新 reward 必须可解释地输出每个 component。
- 验证：
  ```bash
  pytest tests/test_reward_components_mainline_a.py -q
  ```

## Step 5：升级 benchmark 配置入口

- **scope: auto**
- 修改：`scripts/benchmark.py`
- 新增参数：
  - `--system-model-config configs/system_model_mainline_a.yaml`
  - `--enable-mainline-a`
  - `--channel-model analytic|3gpp_lite|rayleigh|pathloss_only`
  - `--queue-model mm1|mmc|parallel|finite_capacity`
  - `--mobility-intensity low|medium|high`
- 验证：
  ```bash
  python scripts/benchmark.py --help
  python scripts/benchmark.py --enable-mainline-a --dry-run
  ```

## Step 6：新增 smoke benchmark

- **scope: auto**
- 新增：`configs/benchmark_mainline_a_smoke.yaml`
- 要求：
  - 2 algorithms
  - 2 seeds
  - 1000 steps
  - analytic channel
  - mm1 queue
  - no cooperation
- 验证：
  ```bash
  python scripts/benchmark.py --config configs/benchmark_mainline_a_smoke.yaml --dry-run
  ```

## Step 7：兼容旧训练入口

- **scope: auto**
- 操作：
  - 执行旧 help / dry-run / quick smoke。
  - 确认 `system_model.enabled=false` 时不改变旧路径。
- 验证：
  ```bash
  python scripts/benchmark.py --help
  pytest tests/test_active_entrypoints.py tests/test_env_legacy_compat.py -q
  ```

## Step 8：记录兼容性结论

- **scope: auto**
- 新增：`docs/mainline_a_compatibility_report.md`
- 内容：
  - legacy 默认兼容性。
  - 新模型启用方式。
  - 已知不兼容项。
  - dashboard 仍需外部复核。
- 验证：
  ```bash
  test -f docs/mainline_a_compatibility_report.md
  grep -n "legacy default" docs/mainline_a_compatibility_report.md
  ```

---

# 模块 17：状态依赖 Stackelberg 动态定价

## 概述

- 职责：把静态价格升级为队列/信道/迁移状态依赖的动态定价，并输出定理假设与可测试性质。
- 前置依赖：模块 15-16。
- 输出：
  - `src/game_pricing/`
  - `configs/pricing_dynamic_mainline_a.yaml`
  - `tests/test_dynamic_pricing.py`
  - `docs/theory/dynamic_stackelberg_pricing.md`
- 预计步骤数：8

## Step 1：创建 pricing 包

- **scope: auto**
- 新增：
  - `src/game_pricing/__init__.py`
  - `src/game_pricing/types.py`
  - `src/game_pricing/dynamic_pricing.py`
- 定义：
  - `PricingState`
  - `PriceVector`
  - `PricingBounds`
  - `FollowerDemand`
  - `ProviderCost`
- 验证：
  ```bash
  python - <<'PY'
  from src.game_pricing.dynamic_pricing import PricingState
  print("pricing import ok")
  PY
  ```

## Step 2：实现状态依赖价格函数

- **scope: review**
- 修改/新增：`src/game_pricing/dynamic_pricing.py`
- 必须实现：
  - `compute_state_dependent_price(pricing_state, bounds, params) -> PriceVector`
  - `compute_queue_price_component(queue_pressure, params)`
  - `compute_channel_price_component(channel_quality, params)`
  - `compute_migration_price_component(migration_risk, params)`
  - `clip_price_to_bounds(price, bounds)`
- 公式结构：
  - `p_i(t)=clip(p0_i + alpha_q Q_i(t) - alpha_h H_i(t) + alpha_m M_i(t), p_min, p_max)`
  - `Q`: queue pressure
  - `H`: channel quality
  - `M`: migration/handover risk
- 验证：
  ```bash
  pytest tests/test_dynamic_pricing.py -q
  ```

## Step 3：实现 follower response

- **scope: review**
- 新增：`src/game_pricing/follower_response.py`
- 必须实现：
  - `compute_best_response(user_state, price_vector, system_state)`
  - `compute_demand_elasticity(user_state, price_vector)`
  - `project_response_to_constraints(response, constraints)`
- 要求：
  - 需求对价格单调非增。
  - 预算约束、deadline constraint、local CPU constraint 都必须投影。
- 验证：
  ```bash
  pytest tests/test_follower_response.py -q
  ```

## Step 4：实现 leader objective

- **scope: review**
- 新增：`src/game_pricing/leader_objective.py`
- 必须实现：
  - `compute_provider_revenue(price_vector, demand)`
  - `compute_provider_cost(system_state, demand)`
  - `compute_leader_utility(price_vector, demand, system_state)`
  - `compute_social_welfare(user_utilities, provider_utility)`
- 验证：
  ```bash
  pytest tests/test_leader_objective.py -q
  ```

## Step 5：实现唯一性/单调性数值检查器

- **scope: review**
- 新增：`src/game_pricing/theory_checks.py`
- 必须实现：
  - `check_demand_price_monotonicity(samples)`
  - `check_strong_concavity_proxy(hessian_or_fd_matrix)`
  - `check_unique_best_response_grid(user_state, price_grid)`
  - `check_price_lipschitz_bound(pricing_policy, state_samples)`
- 输出：
  - `results/theory_checks_dynamic_pricing.json`（生成产物，不纳入 Git）
  - `docs/theory/dynamic_pricing_theory_checks.md`
- 验证：
  ```bash
  pytest tests/test_pricing_theory_checks.py -q
  ```

## Step 6：新增动态定价配置

- **scope: auto**
- 新增：`configs/pricing_dynamic_mainline_a.yaml`
- 内容：
  ```yaml
  dynamic_pricing:
    enabled: false
    base_price: 1.0
    bounds: {min: 0.05, max: 10.0}
    components:
      queue: {enabled: true, alpha: 0.4}
      channel: {enabled: true, alpha: 0.2}
      migration: {enabled: true, alpha: 0.3}
    constraints:
      budget_projection: true
      deadline_projection: true
      monotonicity_check: true
  ```
- 验证：
  ```bash
  python - <<'PY'
  import yaml
  cfg = yaml.safe_load(open("configs/pricing_dynamic_mainline_a.yaml", encoding="utf-8"))
  assert cfg["dynamic_pricing"]["enabled"] is False
  PY
  ```

## Step 7：接入 env pricing path

- **scope: review**
- 修改：现有 `game_theory_env.py`
- 新增方法：
  - `GameTheoryEnv._compute_dynamic_prices(self, system_state)`
  - `GameTheoryEnv._compute_follower_responses(self, price_vector, system_state)`
  - `GameTheoryEnv._compute_pricing_rewards(self, price_vector, responses)`
- 要求：
  - `dynamic_pricing.enabled=false` 时保留旧静态价格。
  - `dynamic_pricing.enabled=true` 时 action/reward 写入 dynamic price metadata。
- 验证：
  ```bash
  pytest tests/test_env_dynamic_pricing.py -q
  ```

## Step 8：编写理论说明

- **scope: review**
- 新增：`docs/theory/dynamic_stackelberg_pricing.md`
- 必含：
  - follower utility 假设。
  - leader utility 假设。
  - 均衡存在性条件。
  - 唯一性充分条件。
  - 需求-价格单调性。
  - 价格更新 Lipschitz 条件。
  - 与队列/信道/迁移状态耦合的边界。
- 验证：
  ```bash
  test -f docs/theory/dynamic_stackelberg_pricing.md
  grep -n "Uniqueness" docs/theory/dynamic_stackelberg_pricing.md
  grep -n "Monotonicity" docs/theory/dynamic_stackelberg_pricing.md
  ```

---

# 模块 18：game-aware critic 与 primal-dual 多智能体更新

## 概述

- 职责：把动态定价和约束信息注入 MARL 更新，形成主线A算法贡献。
- 前置依赖：模块 15-17。
- 输出：
  - `src/rl_algorithms/game_aware/`
  - `configs/algorithm_game_aware_pd_marl.yaml`
  - `tests/test_game_aware_critic.py`
  - `docs/references/ref-game-aware-constrained-marl.md`
- 预计步骤数：8

## Step 1：创建 game-aware algorithm 包

- **scope: auto**
- 新增：
  - `src/rl_algorithms/game_aware/__init__.py`
  - `src/rl_algorithms/game_aware/critic_features.py`
  - `src/rl_algorithms/game_aware/primal_dual.py`
  - `src/rl_algorithms/game_aware/reward_design.py`
- 验证：
  ```bash
  python - <<'PY'
  import src.rl_algorithms.game_aware.critic_features
  print("game-aware import ok")
  PY
  ```

## Step 2：实现 critic feature builder

- **scope: review**
- 新增：`src/rl_algorithms/game_aware/critic_features.py`
- 必须实现：
  - `GameAwareCriticFeatures`
  - `build_critic_features(system_state, price_vector, reward_components)`
  - `normalize_pricing_features(features, running_stats)`
  - `mask_unavailable_edges(features, edge_mask)`
- critic 输入必须包含：
  - queue pressure
  - channel quality
  - migration risk
  - price vector
  - follower demand elasticity
  - constraint residuals
- 验证：
  ```bash
  pytest tests/test_game_aware_critic.py -q
  ```

## Step 3：实现 primal-dual 更新器

- **scope: review**
- 新增：`src/rl_algorithms/game_aware/primal_dual.py`
- 必须实现：
  - `PrimalDualState`
  - `ConstraintResiduals`
  - `PrimalDualUpdater`
  - `update_dual_variables(residuals)`
  - `compute_lagrangian_reward(base_reward, residuals, dual_vars)`
- 约束项：
  - latency deadline
  - energy budget
  - queue stability
  - migration rate
  - budget feasibility
- 验证：
  ```bash
  pytest tests/test_primal_dual_update.py -q
  ```

## Step 4：实现可解释 reward 设计

- **scope: review**
- 新增：`src/rl_algorithms/game_aware/reward_design.py`
- 必须实现：
  - `RewardComponent`
  - `RewardBreakdown`
  - `compute_interpretable_reward(reward_components, dual_state, weights)`
  - `export_reward_explanation(record)`
- 要求：
  - 每个 episode 输出 reward component 均值。
  - 支持 ablation：no_price / no_queue / no_migration / no_dual / no_cooperation。
- 验证：
  ```bash
  pytest tests/test_reward_design_mainline_a.py -q
  ```

## Step 5：接入现有 trainer

- **scope: review**
- 修改：`src/trainer/base_trainer.py`
- 新增方法：
  - `BaseTrainer._build_game_aware_batch(self, batch)`
  - `BaseTrainer._apply_primal_dual_update(self, metrics)`
  - `BaseTrainer._log_reward_breakdown(self, reward_breakdown)`
- 要求：
  - `game_aware.enabled=false` 时旧训练流程不变。
  - `game_aware.enabled=true` 时记录 dual variables 与 constraint residuals。
- 验证：
  ```bash
  pytest tests/test_base_trainer_game_aware.py -q
  ```

## Step 6：新增算法配置

- **scope: auto**
- 新增：`configs/algorithm_game_aware_pd_marl.yaml`
- 内容：
  ```yaml
  game_aware:
    enabled: false
    critic_features:
      include_queue: true
      include_channel: true
      include_migration: true
      include_price: true
      include_elasticity: true
    primal_dual:
      enabled: true
      dual_lr: 0.01
      dual_clip: [0.0, 20.0]
    reward_explanation:
      enabled: true
      export_every_eval: true
  ```
- 验证：
  ```bash
  python - <<'PY'
  import yaml
  cfg = yaml.safe_load(open("configs/algorithm_game_aware_pd_marl.yaml", encoding="utf-8"))
  assert cfg["game_aware"]["enabled"] is False
  PY
  ```

## Step 7：新增算法 smoke test

- **scope: auto**
- 新增：`configs/benchmark_game_aware_smoke.yaml`
- 要求：
  - `--enable-mainline-a`
  - `dynamic_pricing.enabled=true`
  - `game_aware.enabled=true`
  - 2 algorithms, 2 seeds, 1000 steps
- 验证：
  ```bash
  python scripts/benchmark.py --config configs/benchmark_game_aware_smoke.yaml --dry-run
  ```

## Step 8：编写算法实现参考

- **scope: auto**
- 新增：`docs/references/ref-game-aware-constrained-marl.md`
- 内容：
  - critic feature schema。
  - primal-dual update equations。
  - reward components。
  - ablation switches。
- 验证：
  ```bash
  test -f docs/references/ref-game-aware-constrained-marl.md
  grep -n "primal-dual" docs/references/ref-game-aware-constrained-marl.md
  ```

---

# 模块 19：理论证明与数值验证资产

## 概述

- 职责：把动态定价与 primal-dual MARL 的理论叙述整理成可写入论文的证明资产，同时用数值 checker 防止公式与实现脱节。
- 前置依赖：模块 17-18。
- 输出：
  - `docs/theory/mainline_a_theory_appendix.md`
  - `src/analysis/theory_validation.py`
  - `tests/test_theory_validation.py`
- 预计步骤数：6

## Step 1：建立理论附录骨架

- **scope: review**
- 新增：`docs/theory/mainline_a_theory_appendix.md`
- 必含章节：
  - Problem formulation
  - State-dependent Stackelberg pricing
  - Existence of Stackelberg equilibrium
  - Uniqueness sufficient condition
  - Monotonicity of demand response
  - Primal-dual update and constraint residual
  - Constraint violation probability bound
  - Convergence-rate discussion under stochastic approximation assumptions
- 验证：
  ```bash
  test -f docs/theory/mainline_a_theory_appendix.md
  grep -n "Constraint violation probability bound" docs/theory/mainline_a_theory_appendix.md
  ```

## Step 2：写入唯一性充分条件

- **scope: review**
- 修改：`docs/theory/mainline_a_theory_appendix.md`
- 要求：
  - follower utility 对卸载量严格凹。
  - demand response 对 price Lipschitz。
  - leader revenue minus cost 在 price 上强凹或满足 contraction mapping。
  - 明确“充分条件，不是无条件唯一性”。
- 验证：
  ```bash
  grep -n "sufficient condition" docs/theory/mainline_a_theory_appendix.md
  grep -n "strong concavity" docs/theory/mainline_a_theory_appendix.md
  ```

## Step 3：写入单调性与价格状态依赖解释

- **scope: review**
- 修改：`docs/theory/mainline_a_theory_appendix.md`
- 要求：
  - demand 对 price 单调非增。
  - price 对 queue pressure 单调非降。
  - price 对 migration risk 单调非降。
  - price 对 channel quality 的方向必须解释：低质量抬价用于拥塞/风险抑制，或高质量抬价用于资源稀缺收益；二者只能选一种并与实现一致。
- 验证：
  ```bash
  grep -n "monotone non-increasing" docs/theory/mainline_a_theory_appendix.md
  grep -n "queue pressure" docs/theory/mainline_a_theory_appendix.md
  ```

## Step 4：写入约束违反概率界

- **scope: review**
- 修改：`docs/theory/mainline_a_theory_appendix.md`
- 要求：
  - 使用 Lyapunov / primal-dual residual 口径写概率界。
  - 明确 bounded reward、bounded gradients、bounded stochastic noise 假设。
  - 给出 `O(1/sqrt(T))` 或实现可支撑的保守速率，不写无法证明的强结论。
- 验证：
  ```bash
  grep -n "O(1/sqrt(T))" docs/theory/mainline_a_theory_appendix.md
  grep -n "bounded stochastic noise" docs/theory/mainline_a_theory_appendix.md
  ```

## Step 5：实现理论数值 checker

- **scope: auto**
- 新增：`src/analysis/theory_validation.py`
- 必须实现：
  - `validate_price_monotonicity(records)`
  - `validate_demand_elasticity(records)`
  - `validate_constraint_residual_trend(records)`
  - `export_theory_validation_report(records, output_path)`
- 新增：`tests/test_theory_validation.py`
- 验证：
  ```bash
  pytest tests/test_theory_validation.py -q
  ```

## Step 6：生成理论验证报告模板

- **scope: auto**
- 新增：`docs/theory/theory_validation_report_template.md`
- 内容：
  - price monotonicity check。
  - demand elasticity check。
  - constraint residual trend。
  - violation rate trend。
- 验证：
  ```bash
  test -f docs/theory/theory_validation_report_template.md
  grep -n "price monotonicity" docs/theory/theory_validation_report_template.md
  ```

---

# 模块 20：paper2 内置新模型实验矩阵、消融、oracle 与 OOD 泛化

## 概述

- 职责：在 paper2 项目内建立新模型专属实验链，替代旧 L2/L3 作为论文证据来源。
- 前置依赖：模块 15-19。
- 输出：
  - `configs/experiments/mainline_a_*.yaml`
  - `scripts/run_mainline_a_experiments.py`
  - `docs/mainline_a_experiment_protocol.md`
  - `docs/mainline_a_publication_gate.md`
- 预计步骤数：10

## Step 1：创建新实验协议

- **scope: review**
- 新增：`docs/mainline_a_experiment_protocol.md`
- 定义证据等级：
  - `N0`: smoke correctness
  - `N1`: small-scale oracle comparison
  - `N2`: controlled ablation
  - `N3`: OOD generalization
- 要求：
  - N1 必须有小规模最优值或枚举 oracle。
  - N2 必须覆盖 no_price/no_queue/no_migration/no_dual/no_cooperation。
  - N3 必须覆盖用户数、边缘数、移动性强度、信道模型变化。
- 验证：
  ```bash
  test -f docs/mainline_a_experiment_protocol.md
  grep -n "N1" docs/mainline_a_experiment_protocol.md
  grep -n "OOD" docs/mainline_a_experiment_protocol.md
  ```

## Step 2：实现小规模 oracle

- **scope: review**
- 新增：`src/analysis/small_scale_oracle.py`
- 必须实现：
  - `enumerate_offloading_assignments(instance)`
  - `solve_small_scale_optimum(instance, objective)`
  - `compare_policy_to_oracle(policy_result, oracle_result)`
  - `export_oracle_gap_report(records, output_path)`
- 约束：
  - 只用于小规模：`num_users <= 4`，`num_edges <= 3`，离散动作。
  - 输出 optimality gap、constraint violation、runtime。
- 验证：
  ```bash
  pytest tests/test_small_scale_oracle.py -q
  ```

## Step 3：新增 N0 smoke 配置

- **scope: auto**
- 新增：`configs/experiments/mainline_a_n0_smoke.yaml`
- 要求：
  - users: 4
  - edges: 2
  - seeds: [42]
  - steps: 1000
  - channel: analytic
  - queue: mm1
  - dynamic_pricing: true
  - game_aware: true
- 验证：
  ```bash
  python scripts/benchmark.py --config configs/experiments/mainline_a_n0_smoke.yaml --dry-run
  ```

## Step 4：新增 N1 oracle 配置

- **scope: review**
- 新增：`configs/experiments/mainline_a_n1_oracle.yaml`
- 要求：
  - users: [2, 3, 4]
  - edges: [2, 3]
  - seeds: [42, 43, 44]
  - algorithms: baseline_static_stackelberg, MAPPO, game_aware_pd_marl
  - 输出 oracle gap。
- 验证：
  ```bash
  python scripts/run_mainline_a_experiments.py --config configs/experiments/mainline_a_n1_oracle.yaml --dry-run
  ```

## Step 5：新增 N2 消融配置

- **scope: review**
- 新增：`configs/experiments/mainline_a_n2_ablation.yaml`
- 必须包含：
  - full_model
  - no_dynamic_price
  - no_queue_state
  - no_channel_state
  - no_migration_state
  - no_primal_dual
  - no_cooperation
  - analytic_channel_only
  - 3gpp_lite_channel
- 验证：
  ```bash
  python scripts/run_mainline_a_experiments.py --config configs/experiments/mainline_a_n2_ablation.yaml --dry-run
  ```

## Step 6：新增 N3 OOD 泛化配置

- **scope: review**
- 新增：`configs/experiments/mainline_a_n3_ood.yaml`
- 训练/测试变化：
  - train: users=20, edges=4, mobility=medium, channel=analytic
  - test: users=40, edges=6, mobility=high, channel=3gpp_lite
  - test: queue_model=parallel
  - test: cooperation_enabled=true
- 指标：
  - social welfare
  - average latency
  - p95 latency
  - energy
  - provider revenue
  - constraint violation rate
  - Jain fairness
  - oracle gap for small cases
- 验证：
  ```bash
  python scripts/run_mainline_a_experiments.py --config configs/experiments/mainline_a_n3_ood.yaml --dry-run
  ```

## Step 7：实现实验 runner

- **scope: review**
- 新增：`scripts/run_mainline_a_experiments.py`
- 必须支持：
  - `--config`
  - `--dry-run`
  - `--stage N0|N1|N2|N3|all`
  - `--resume`
  - `--output-root experiments/mainline_a`
  - `--results-root results/mainline_a`
- 要求：
  - 不修改旧 benchmark 语义。
  - 所有生成产物仍被 `.gitignore` 覆盖。
- 验证：
  ```bash
  python scripts/run_mainline_a_experiments.py --help
  pytest tests/test_mainline_a_experiment_runner.py -q
  ```

## Step 8：新增可解释报告生成

- **scope: review**
- 修改：`scripts/plot_results.py`
- 新增：
  - reward breakdown 图。
  - price vs queue/channel/migration 曲线。
  - dual variables 曲线。
  - oracle gap 表。
  - OOD generalization 表。
- 验证：
  ```bash
  pytest tests/test_mainline_a_plots.py -q
  python scripts/plot_results.py --help
  ```

## Step 9：新增新模型 publication gate

- **scope: auto**
- 新增：`docs/mainline_a_publication_gate.md`
- 规则：
  - N0 只证明管线可跑。
  - N1 进入“small-scale optimality comparison”。
  - N2 进入 ablation。
  - N3 进入 robustness/generalization。
  - 未完成 N1/N2/N3 不得写主线A完整实验结论。
- 验证：
  ```bash
  test -f docs/mainline_a_publication_gate.md
  grep -n "small-scale optimality" docs/mainline_a_publication_gate.md
  grep -n "generalization" docs/mainline_a_publication_gate.md
  ```

## Step 10：全实验 dry-run 验收

- **scope: auto**
- 操作：
  - 跑所有 N0-N3 dry-run。
  - 跑新测试。
- 验证：
  ```bash
  python scripts/run_mainline_a_experiments.py --stage all --dry-run
  pytest tests/test_mec_model_tasks.py tests/test_mec_model_queues.py tests/test_dynamic_pricing.py tests/test_game_aware_critic.py tests/test_primal_dual_update.py tests/test_small_scale_oracle.py tests/test_mainline_a_experiment_runner.py -q
  ```

---

# 模块 21：论文写作资产与差异化改写接口

## 概述

- 职责：只生成论文改写资产与待确认差异清单，不直接改论文正文；待用户给出“论文改写有所区别”的具体要求后再进入单独写作迭代。
- 前置依赖：模块 15-20 的接口和实验协议已确定。
- 输出：
  - `writing_ref/paper2_mainline_a_revision/`
  - `docs/paper_revision_manifest.md`
  - `docs/paper_revision_pending_questions.md`
- 预计步骤数：5

## Step 1：创建论文写作资产目录

- **scope: auto**
- 新增：
  - `writing_ref/paper2_mainline_a_revision/README.md`
  - `writing_ref/paper2_mainline_a_revision/model_change_inventory.md`
  - `writing_ref/paper2_mainline_a_revision/equation_inventory.md`
  - `writing_ref/paper2_mainline_a_revision/experiment_figure_inventory.md`
- 注意：
  - 不直接生成“最终论文改写版”。
  - 若 `writing_ref/` 当前不入 Git，执行端只本地生成并在 `docs/paper_revision_manifest.md` 记录 SHA-256。
- 验证：
  ```bash
  test -f writing_ref/paper2_mainline_a_revision/model_change_inventory.md
  ```

## Step 2：生成模型变更清单

- **scope: review**
- 修改：`writing_ref/paper2_mainline_a_revision/model_change_inventory.md`
- 必含：
  - MEC task model changes。
  - queue model changes。
  - heterogeneous cooperative edge changes。
  - energy model changes。
  - mobility model changes。
  - analytic vs 3GPP-lite communication model split。
  - dynamic pricing changes。
  - game-aware critic / primal-dual changes。
- 验证：
  ```bash
  grep -n "dynamic pricing" writing_ref/paper2_mainline_a_revision/model_change_inventory.md
  ```

## Step 3：生成公式与图表库存

- **scope: auto**
- 修改：
  - `writing_ref/paper2_mainline_a_revision/equation_inventory.md`
  - `writing_ref/paper2_mainline_a_revision/experiment_figure_inventory.md`
- 要求：
  - 只列“应该改哪些公式/图表”，不替用户决定最终论文段落措辞。
- 验证：
  ```bash
  grep -n "primal-dual" writing_ref/paper2_mainline_a_revision/equation_inventory.md
  grep -n "OOD" writing_ref/paper2_mainline_a_revision/experiment_figure_inventory.md
  ```

## Step 4：生成论文差异待确认清单

- **scope: auto**
- 新增：`docs/paper_revision_pending_questions.md`
- 必问项：
  - 论文源文件路径。
  - 目标期刊/会议模板。
  - 当前稿件是否保留原章节标题。
  - 用户所说“论文改写有所区别”的具体区别。
  - 是否允许重写 Introduction / System Model / Theory / Experiments。
  - 是否已有图表编号或公式编号必须保留。
- 验证：
  ```bash
  test -f docs/paper_revision_pending_questions.md
  grep -n "论文改写有所区别" docs/paper_revision_pending_questions.md
  ```

## Step 5：生成 paper revision manifest

- **scope: auto**
- 新增：`docs/paper_revision_manifest.md`
- 内容：
  - 资产路径。
  - 文件 SHA-256。
  - 是否进入 Git。
  - 与模块 15-20 的依赖。
  - 明确：论文正文改写需等待用户确认差异要求。
- 验证：
  ```bash
  test -f docs/paper_revision_manifest.md
  grep -n "pending_questions" docs/paper_revision_manifest.md
  ```

---

# 总体验收

## 必跑命令

```bash
pytest tests/test_mec_model_tasks.py tests/test_mec_model_queues.py tests/test_mec_model_edge_topology.py tests/test_mec_model_energy.py tests/test_mec_model_mobility.py tests/test_mec_model_channel.py -q
pytest tests/test_dynamic_pricing.py tests/test_follower_response.py tests/test_leader_objective.py tests/test_pricing_theory_checks.py -q
pytest tests/test_game_aware_critic.py tests/test_primal_dual_update.py tests/test_reward_design_mainline_a.py -q
pytest tests/test_small_scale_oracle.py tests/test_mainline_a_experiment_runner.py tests/test_theory_validation.py -q
python scripts/benchmark.py --help
python scripts/run_mainline_a_experiments.py --stage all --dry-run
```

## 成功判定

- paper2 与仿真实验已统一在一个计划内。
- 当前旧 L2/L3 不再阻塞新模型。
- `src/mec_model/` 可独立测试。
- `src/game_pricing/` 可独立测试。
- game-aware critic / primal-dual 更新可开关接入。
- legacy 默认路径仍可运行。
- 新模型 N0/N1/N2/N3 实验链有 dry-run 与 publication gate。
- 论文只生成写作资产与差异待确认清单，不直接按代码执行链改正文。
