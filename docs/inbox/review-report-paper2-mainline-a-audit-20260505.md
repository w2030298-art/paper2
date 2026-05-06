# 代码审查报告

## 元信息

- 审查日期：2026-05-05
- 审查范围：`w2030298-art/paper2` `main` 分支，重点审查 commit `fa741320632ee0ec9d5c9c9017320bb425f58372`
- 变更范围：`system-model-overhaul-v4.1`，涉及 docs、MEC system model、dynamic pricing、game-aware MARL、experiment runner、tests、配置文件
- 对照文档：`docs/plan.md`、`docs/progress.md`、`docs/report.md`
- 审查限制：未在本地 clone 运行测试；结论基于 GitHub 文件内容、compare metadata、关键文件抽样读取和缺失文件探测

## 总体结论

本次更新不能直接进入下一轮实验或论文主线。核心风险不是代码风格，而是执行闭环断裂：`progress.md` 宣称模块 14R-21 完成，但 clean checkout 下至少两个 runner/test 路径依赖未入仓文件；同时 Mainline-A 的若干关键路径仍停留在 metadata / proxy / dry-run 级实现，尚未满足 plan 中“可验证新模型实验链”的含义。

建议先做一轮 fix，不建议直接启动 N0/N1/N2/N3，也不建议把当前实现作为论文主结果依据。

---

## 问题清单

### 🔴 Critical

#### C-1：Mainline-A 默认实验 runner 引用未入仓配置，clean checkout 会直接失败

- 位置：
  - `scripts/run_mainline_a_experiments.py`
  - `configs/experiments/mainline_a_n0_smoke.yaml`
  - `configs/experiments/mainline_a_n1_oracle.yaml`
  - `configs/experiments/mainline_a_n2_ablation.yaml`
  - `configs/experiments/mainline_a_n3_ood.yaml`
  - `tests/test_mainline_a_experiment_runner.py`
- 类型：执行链断裂 / 缺失文件 / 测试不可运行
- 现象：
  - runner 的 `DEFAULT_STAGE_CONFIGS` 指向 `configs/experiments/mainline_a_n0_smoke.yaml` 等 4 个文件。
  - GitHub search 只找到这些文件名出现在 runner 和 plan 中，未找到对应配置文件。
  - 直接读取 `configs/experiments/mainline_a_n0_smoke.yaml` 返回 404。
  - 测试 `test_runner_resolves_all_stage_plans()` 调用 `resolve_plans(args)`，会触发 `load_experiment_config()` 读取上述默认文件。
- 影响：
  - clean checkout 下 `pytest tests/test_mainline_a_experiment_runner.py -q` 应失败。
  - `python scripts/run_mainline_a_experiments.py --stage all --dry-run` 应失败。
  - 模块 20 “paper2 内置新模型实验矩阵”不能视为完成。
- 修复建议：
  1. 新增目录 `configs/experiments/`。
  2. 创建：
     - `configs/experiments/mainline_a_n0_smoke.yaml`
     - `configs/experiments/mainline_a_n1_oracle.yaml`
     - `configs/experiments/mainline_a_n2_ablation.yaml`
     - `configs/experiments/mainline_a_n3_ood.yaml`
  3. 或修改 `DEFAULT_STAGE_CONFIGS` 指向已存在的 `configs/benchmark_mainline_a_smoke.yaml`，但必须补齐 N1/N2/N3。
  4. 为 `tests/test_mainline_a_experiment_runner.py` 增加文件存在断言和 CLI subprocess 测试。
- 验证：
  ```bash
  test -f configs/experiments/mainline_a_n0_smoke.yaml
  test -f configs/experiments/mainline_a_n1_oracle.yaml
  test -f configs/experiments/mainline_a_n2_ablation.yaml
  test -f configs/experiments/mainline_a_n3_ood.yaml
  python scripts/run_mainline_a_experiments.py --stage all --dry-run
  pytest tests/test_mainline_a_experiment_runner.py -q
  ```
- 关联：plan.md 模块 20

#### C-2：formal convergence runner 测试依赖 `results/` 下未入仓产物，clean checkout 下测试不可复现

- 位置：
  - `scripts/run_formal_convergence_protocol.py`
  - `tests/test_formal_convergence_runner.py`
  - `.gitignore`
  - `results/l1_baseline_convergence_assessment.json`
  - `results/convergence_event_audit.json`
- 类型：测试夹具缺失 / 生成产物依赖 / CI 不可复现
- 现象：
  - `.gitignore` 明确忽略 `results/`。
  - runner 的 `select_l2_algorithms()` 通过 `_l1_decisions()` 读取 `results/l1_baseline_convergence_assessment.json`，通过 `_event_audit_allows_l2()` 读取 `results/convergence_event_audit.json`。
  - `tests/test_formal_convergence_runner.py` 直接断言 COMA/MAPPO/TRPO 与 IQL/VDN/IPPO/MADDPG 被选中，但测试没有创建这些 JSON fixtures。
  - GitHub 上读取 `results/l1_baseline_convergence_assessment.json` 返回 404。
- 影响：
  - clean checkout 或 CI 中 `pytest tests/test_formal_convergence_runner.py -q` 应失败。
  - 模块 14 formal convergence 的测试不具备可复现性。
- 修复建议：
  1. 不要让单元测试依赖 `results/`。
  2. 在 `tests/fixtures/formal_convergence/` 添加最小 JSON fixtures。
  3. 将 `select_l2_algorithms()` 改为接收显式 `l1_decisions` / `event_allowed` 参数，文件读取只留在 CLI 层。
  4. 对缺失 runtime artifacts 的 CLI 行为给出清晰错误或 `not_run` 报告，不要让库函数隐式读仓库外状态。
- 验证：
  ```bash
  rm -rf results/
  pytest tests/test_formal_convergence_runner.py -q
  python scripts/run_formal_convergence_protocol.py --phase L2 --dry-run
  ```
- 关联：plan.md 模块 14 / 14R

---

### 🟠 High

#### H-1：`plan.md`、`progress.md`、`report.md` 状态互相矛盾，恢复上下文不可信

- 位置：
  - `docs/plan.md`
  - `docs/progress.md`
  - `docs/report.md`
- 类型：执行状态漂移 / 文档恢复契约破坏
- 现象：
  - `docs/plan.md` 的 Status 仍写“模块 14R Step 1 待执行”“模块 14R、15-21 待执行”。
  - `docs/progress.md` 写“模块 14R-21 已按 v4.1 执行；review scope 项等待用户/Web 审核”，并把模块 15-21 全部勾选完成。
  - `docs/report.md` 写 STATUS `NEEDS_REVIEW`，同时记录本地还有 18 个已跟踪 docs 删除未提交。
- 影响：
  - 执行端从 `plan.md` 恢复会认为 14R Step 1 尚未执行。
  - 从 `progress.md` 恢复会认为 14R-21 已完成。
  - Review 无法可靠判断哪些 Step 是实装完成、哪些只是 dry-run 或待审。
- 修复建议：
  1. 以 `progress.md` 的实际完成态为准，更新 `docs/plan.md` Status。
  2. 对模块 15-21 的每个 review scope Step 标注 `DONE_PENDING_REVIEW` 或 `DONE_REJECTED_BY_REVIEW`，不要只写模块级已完成。
  3. 在 `docs/report.md` 增加“本次 GitHub 审计结果”小节，明确 clean checkout 失败项。
- 验证：
  ```bash
  grep -n "当前阶段" docs/plan.md docs/progress.md docs/report.md
  grep -n "模块 14R" docs/plan.md docs/progress.md
  ```

#### H-2：Mainline-A smoke 配置 schema 与 canonical system model schema 不一致，配置可能被静默忽略

- 位置：
  - `configs/benchmark_mainline_a_smoke.yaml`
  - `configs/system_model_mainline_a.yaml`
  - `scripts/benchmark.py`
  - `src/environments/mec_v3/game_theory_env.py`
- 类型：配置契约不一致
- 现象：
  - `configs/system_model_mainline_a.yaml` 使用 `system_model.queue_model` 与 nested `system_model.channel_model.theory/simulation`。
  - `configs/benchmark_mainline_a_smoke.yaml` 使用 `system_model.queue` 与 `system_model.channel`。
  - plan 中要求 benchmark 参数为 `--channel-model`、`--queue-model`，但 smoke YAML 不是同一 schema。
- 影响：
  - smoke 配置即使能加载，也可能没有实际切换 queue/channel 模型。
  - N0 结果无法证明“analytic channel + mm1 queue”真的生效。
- 修复建议：
  1. 统一 schema：只允许 `queue_model` 与 `channel_model.{theory,simulation}`。
  2. 在配置加载处增加 strict validation，禁止未知字段 `queue`、`channel`。
  3. 在 dry-run 输出中打印 resolved config。
- 验证：
  ```bash
  python scripts/benchmark.py --config configs/benchmark_mainline_a_smoke.yaml --dry-run
  pytest tests/test_env_mainline_a_state.py tests/test_mec_model_channel.py tests/test_mec_model_queues.py -q
  ```

#### H-3：Mainline-A env/pricing/reward 接入仍偏 metadata/proxy，未形成真实闭环

- 位置：
  - `src/mec_model/adapters.py`
  - `src/game_pricing/follower_response.py`
  - `src/rl_algorithms/game_aware/primal_dual.py`
  - `src/trainer/base_trainer.py`
- 类型：架构合规偏差 / 研究功能未闭环
- 现象：
  - `apply_system_decision_to_legacy_env()` 只把 decision 存成 `env.last_mainline_a_decision`，没有驱动 legacy env 的 action/state/reward。
  - `extract_reward_components()` 中 `migration_penalty`、`deadline_violation_penalty`、`cooperation_gain`、`constraint_penalty` 恒为 0。
  - `compute_best_response()` 忽略 `system_state`，deadline 固定为 1.0。
  - `BaseTrainer._apply_primal_dual_update()` 只从 metrics 中提取/记录 dual mean，没有调用 `PrimalDualUpdater.update_dual_variables()`。
- 影响：
  - 动态定价、constraint reward、primal-dual MARL 很可能只出现在 metadata 和日志中，没有真正改变训练行为。
  - 模块 16-18 的 review scope 不应通过。
- 修复建议：
  1. `apply_system_decision_to_legacy_env()` 必须写入下一步 env 可消费字段，或删除该函数避免假接入。
  2. `extract_reward_components()` 必须由 env step 的真实 latency/energy/deadline/migration/cooperation 信息计算。
  3. `compute_best_response()` 必须使用 `system_state` 中的 deadline、budget、edge capacity、channel、queue。
  4. `BaseTrainer` 只做 hook 不够；应把 dual updater 放进 agent/trainer state，并在 update step 前后参与 reward shaping。
- 验证：
  ```bash
  pytest tests/test_env_dynamic_pricing.py tests/test_reward_components_mainline_a.py tests/test_primal_dual_update.py tests/test_base_trainer_game_aware.py -q
  ```

#### H-4：新增测试主要验证“存在/metadata”，缺少行为级断言

- 位置：
  - `tests/test_env_mainline_a_state.py`
  - `tests/test_env_legacy_compat.py`
  - `tests/test_env_dynamic_pricing.py`
  - `tests/test_mec_model_adapters.py`
  - `tests/test_mainline_a_experiment_runner.py`
  - `tests/test_active_entrypoints.py`
- 类型：测试完整性不足
- 现象：
  - env 测试只检查 observation 维度、info 字段、metadata。
  - dynamic pricing env 测试只检查 `info["dynamic_price_metadata"]`，不检查 reward/action/queue/price 对 env step 的影响。
  - adapter 测试只检查 state 数量和 decision attr。
  - active entrypoints 测试未覆盖新加的 `scripts/run_mainline_a_experiments.py` 与 `scripts/run_formal_convergence_protocol.py`。
- 影响：
  - 当前测试无法证明 Mainline-A 真的参与决策或训练。
  - 大量“通过测试”只能证明 import/reset 层面可用。
- 修复建议：
  1. 增加行为测试：同一 seed 下开启/关闭 dynamic pricing，至少一个 price/reward/action-path 指标应有可解释差异。
  2. 增加 `run_mainline_a_experiments.py --dry-run`、`run_formal_convergence_protocol.py --phase L2 --dry-run` 到 active entrypoints。
  3. 对 reward components 增加非零断言与方向性断言。
- 验证：
  ```bash
  pytest tests/test_active_entrypoints.py tests/test_env_dynamic_pricing.py tests/test_reward_components_mainline_a.py -q
  ```

---

### 🟡 Medium

#### M-1：3GPP-lite channel 随机性与可复现性设计有问题

- 位置：`src/mec_model/channel.py`
- 类型：模型质量 / 随机性
- 现象：
  - `ThreeGppLiteRateModel.pathloss_db()` 每次调用都创建 `random.Random(self.rng_seed)`。
  - `rng_seed` 固定时，每次调用的 LoS/shadowing 序列从头开始，导致采样模式不随调用推进。
  - `rng_seed=None` 时，每次调用可能重新用系统熵初始化，破坏可复现性。
- 影响：
  - OOD 和 channel ablation 结果可能不可复现或统计性质失真。
- 修复建议：
  1. 在 dataclass 中维护 RNG 实例，或显式要求调用方传入 rng。
  2. 对 fixed seed 添加多次调用不完全相同但跨 run 可复现的测试。
- 验证：
  ```bash
  pytest tests/test_mec_model_channel.py -q
  ```

#### M-2：dynamic pricing 对 channel quality 未做归一化，价格容易被 clip 掩盖

- 位置：`src/game_pricing/dynamic_pricing.py`
- 类型：数值稳定性 / 模型语义
- 现象：
  - `compute_channel_price_component()` 直接返回 `-alpha_channel * max(channel_quality, 0)`。
  - 如果 channel_quality 是 dB、Mbps 或 rate proxy，量纲不一致会导致价格被 clip 到下界。
- 影响：
  - queue/migration component 可能被 channel component 完全覆盖。
  - 价格敏感性测试可能通过，但真实实验退化为边界价格。
- 修复建议：
  1. 定义 `channel_quality ∈ [0,1]`，或显式 normalize。
  2. 增加 component-level contribution export。
  3. 测试价格对 queue/channel/migration 的单调性和非饱和区间。
- 验证：
  ```bash
  pytest tests/test_dynamic_pricing.py tests/test_pricing_theory_checks.py -q
  ```

#### M-3：legacy adapter 对 channel matrix 没有边界保护

- 位置：`src/mec_model/adapters.py`
- 类型：健壮性
- 现象：
  - 当 `env.channel_qualities` 存在时，代码直接访问 `channels[user_idx][edge_idx]`。
  - 若 legacy env 暴露 shape 不一致、单用户/多用户混合、list/np.ndarray 维度异常，会抛出 IndexError。
- 影响：
  - 多 env wrapper 或不同算法路径可能随机失败。
- 修复建议：
  1. 增加 `_safe_channel_quality(channels, user_idx, edge_idx, default=1.0)`。
  2. 测试 ragged / short / missing channel matrix。
- 验证：
  ```bash
  pytest tests/test_mec_model_adapters.py -q
  ```

#### M-4：`progress.md` 承认正式 N0/N1/N2/N3 未启动，但模块 20 标记完成，语义不清

- 位置：`docs/progress.md`
- 类型：项目状态表达
- 现象：
  - 模块 20 写 Step 1-10 完成。
  - 同一文件“已知问题”写“本轮仅执行 dry-run 和单元测试；N0/N1/N2/N3 正式实验未启动”。
- 影响：
  - “实验矩阵完成”容易被误读为“实验完成”。
- 修复建议：
  - 把模块 20 拆成：
    - `20A experiment infrastructure [DONE_PENDING_REVIEW]`
    - `20B N0/N1/N2/N3 execution [NOT_STARTED]`
- 验证：
  ```bash
  grep -n "N0" docs/progress.md docs/report.md
  ```

---

### 🟢 Low

#### L-1：`game_theory_env.py` 在模块级修改 `sys.path`

- 位置：`src/environments/mec_v3/game_theory_env.py`
- 类型：可维护性
- 现象：
  - 文件模块级执行 `sys.path.insert(...)`。
- 影响：
  - 可能掩盖 packaging/import 问题，影响测试与运行环境一致性。
- 修复建议：
  - 移除该写法，统一用 package import 和 editable install。
- 验证：
  ```bash
  python -m pytest tests/test_env_legacy_compat.py -q
  ```

---

## 计划偏差总览

| 模块 / Step | plan 要求 | 实际实装 | 偏差类型 | 严重度 |
|---|---|---|---|---|
| 模块 14 / formal convergence | L1/L2/L3 runner 与测试可复现 | 单元测试依赖 `results/` ignored artifact | 测试不可复现 | Critical |
| 模块 15 | MEC model 可替换、可测试 | 基础模块基本存在，但部分模型为简化 proxy | 模型深度不足 | Medium |
| 模块 16 | system model 接入 env/reward/benchmark，legacy 默认兼容 | adapter 多处只存 metadata，reward components 多项恒 0 | 接入未闭环 | High |
| 模块 17 | 状态依赖 Stackelberg 动态定价 | pricing 函数存在，但 channel 量纲/归一化不清 | 数值语义风险 | Medium |
| 模块 18 | game-aware critic + primal-dual MARL | trainer 只记录 dual metrics，不实际更新 dual state | 训练闭环缺失 | High |
| 模块 20 | N0/N1/N2/N3 runner/configs/protocol | 默认 config 缺失；正式实验未启动 | 执行链断裂 | Critical |
| 模块 21 | writing assets only, 不改论文正文 | 基本符合边界 | 无重大偏差 | Low |

## 执行端已知问题对照

| report.md 已记录 | 审查发现 | 状态 |
|---|---|---|
| 模块 14R Step 1 待审核 | 状态文档互相矛盾，plan 仍写待执行 | 未解决 |
| 模块 15-20 review scope 实现项待审核 | 发现 runner/test/接入闭环问题 | 需修复 |
| 未提交 18 个 docs 删除 | 说明本地工作区与 GitHub 状态不一致 | 需用户确认后处理 |
| 正式 N0/N1/N2/N3 实验未启动 | 模块 20 不应标为完成实验链 | 需改状态语义 |
| 外部 dashboard 兼容性未复核 | 本次未复核 | 保持 blocked |

## 正面发现

- `src/mec_model/` 的基础 dataclass 拆分清晰，任务、队列、能耗、移动性、channel 至少形成了独立模块边界。
- `configs/system_model_mainline_a.yaml` 默认 `enabled: false`，符合 legacy 默认不变的总体原则。
- `docs/report.md` 明确记录本地 18 个 docs 删除未提交，没有把未确认删除直接推送。
- `scripts/run_formal_convergence_protocol.py` 有拒绝 full-17 和 L2/L3 分级的防线设计，方向正确。

## 是否建议修复

建议立即修复 Critical + High。不要先修 Medium/Low，不要顺手重构旧 env。修复顺序：

1. 先修 C-1 / C-2，让 clean checkout 的 runner 和 tests 可复现。
2. 再修 H-1，同步 plan/progress/report。
3. 再修 H-2 / H-3 / H-4，把 Mainline-A 从 metadata 接入推进到行为闭环和测试闭环。
4. 最后处理 M/L 项。

