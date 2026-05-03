# 代码审查报告：paper2 核心功能导向二次瘦身审计

## 元信息

- 审查日期：2026-05-02
- 审查模式：Review
- 审查对象：`w2030298-art/paper2` GitHub `main`
- 当前可见最新基线：`85d247e020c47337ed8f167754ecfd0655929145`
- 审查目标：重新确认冗余范围；从“当前维护更新使用的核心功能”反推保留边界；所有不确定内容显式列出，等待用户确认，不自动忽略。
- 审查依据：
  - `README.md`
  - `.vscode/tasks.json`
  - `scripts/benchmark.py`
  - `scripts/train.py`
  - `scripts/experiment_manager.py`
  - `scripts/evaluate.py`
  - `scripts/generate_report.py`
  - `src/experiment/manager.py`
  - `src/trainer/base_trainer.py`
  - `src/environments/mec_v2/__init__.py`
  - `src/environments/mec_v2/base_env.py`
  - `docs/README.md`
  - `.gitignore`
  - `pyproject.toml`
  - `requirements.txt`
  - `graphify-out/GRAPH_REPORT.md`
- 审查限制：
  - 当前通过 GitHub connector 审查，无法运行本地 import graph、pytest、vulture、git ls-files。
  - 本报告把“可确定删除”和“需确认后删除”分开，不把不确定项默认为保留。

---

## 0. 当前核心功能边界

按项目现状，paper2 的“正在维护更新使用的核心功能”应收缩为 4 条链路：

### Core-1：17 算法 GameTheory MEC benchmark

保留：

```text
rl_algorithms/
configs/algorithms/
configs/scoring_profiles.yaml
src/environments/mec_v3/
src/environments/mec_v2/base_env.py
src/trainer/base_trainer.py
src/trainer/on_policy_trainer.py
src/trainer/off_policy_trainer.py
scripts/benchmark.py
scripts/train.py
src/utils/helpers.py
src/utils/composite_score.py
```

理由：

- `scripts/benchmark.py` 明确声明自己是 “GameTheory 环境唯一” benchmark 入口。
- `scripts/benchmark.py` 直接注册 17 个算法类、On/Off policy 分类、GameTheory env 映射和 composite score。
- `scripts/train.py` 复用 `benchmark.py` 中的创建逻辑，是单算法训练入口。
- `src/trainer/base_trainer.py` 仍是 trainer 主循环，`on_policy_trainer.py` / `off_policy_trainer.py` 继承它。

### Core-2：VSCode / CLI 实验编排

保留：

```text
scripts/experiment_manager.py
src/experiment/
.vscode/tasks.json
.vscode/launch.json
docs/vscode_experiment_usage.md
docs/frontend_monitoring_integration.md
```

理由：

- `.vscode/tasks.json` 当前大量任务直接调用 `scripts/experiment_manager.py`。
- `scripts/experiment_manager.py` 调用 `ExperimentManager`、`JsonStateStore`、`BenchmarkResultWriter`。
- `src/experiment/manager.py` 是实际状态机，支撑 `start/resume/stop/status/export/reset/list/rebuild-index`。

### Core-3：结果可视化、收敛质量、失败诊断

保留：

```text
scripts/plot_results.py
scripts/generate_report.py   # 暂保留，但需要参数化/去旧默认
docs/convergence_plot_quality.md
docs/convergence_failure_analysis.md  # 若模块 12 已生成
scripts/analyze_convergence_failures.py  # 若模块 12 已生成
```

理由：

- 近期主要维护内容是收敛曲线质量管线、未收敛算法诊断和报告输出。
- `scripts/plot_results.py` 是当前结果图核心入口。
- `generate_report.py` 虽然旧硬编码明显，但仍承接报告集成，需要改造后保留或替代。

### Core-4：执行端协作文档

保留：

```text
docs/README.md
docs/plan.md
docs/report.md
docs/progress.md
docs/issues.md
docs/inbox/
docs/archive/
docs/references/   # 只保留当前仍有参考价值的少量文件
```

理由：

- 用户工作流依赖 Web/Codex 双端计划、报告、inbox、archive。
- `docs/README.md` 已明确执行端只从 `docs/plan.md` 与 `docs/inbox/plan.md` 恢复。

---

## 1. 明确冗余：可以进入第一轮清理，不需要再确认语义

### 🔴 C-1：生成产物与实验目录仍不应进入 Git

- 位置：
  - `experiments/`
  - `results/`
  - `figures/`
  - `logs/`
  - `checkpoints/`
- 类型：生成产物 / 实验产物 / 仓库污染
- 证据：
  - `.gitignore` 已忽略 `figures/`、`results/`，并忽略 `experiments` 下的 lock/control/process/logs/checkpoint 产物。
- 判断：
  - 这些是运行结果，不是源码。
  - 需要从 Git tracking 移除，但不能删除用户本地文件。
- 处理：
  - `git rm --cached -r experiments results figures logs checkpoints`，存在即处理。
  - 保留本地文件。
  - 新增 `tests/test_repo_hygiene.py`，禁止这些目录中的结果文件重新入库。

### 🔴 C-2：`src/trainer/benchmark.py` 是废弃且损坏的旧入口

- 位置：`src/trainer/benchmark.py`
- 类型：废弃入口 / 运行不可用
- 证据：
  - 该文件仍尝试创建 `MECEnvDiscrete`、`MECEnvContinuous`、`MECEnvMultiAgent`。
  - `src/environments/mec_v2/__init__.py` 明确声明这些 env 已被移除，只保留 `base_env.py`。
  - 当前正式入口是 `scripts/benchmark.py`。
- 判断：
  - 直接删除，不应保留 stub。
- 处理：
  - 删除 `src/trainer/benchmark.py`。
  - 全仓搜索并删除 `src.trainer.benchmark` / `python -m src.trainer.benchmark` 文档引用。
  - 增加 `tests/test_active_entrypoints.py` 断言该路径不存在。

### 🔴 C-3：`docs/.archive/` 与 `docs/archive/` 双归档冲突

- 位置：
  - `docs/.archive/`
  - `docs/archive/`
- 类型：目录契约冲突
- 证据：
  - `docs/README.md` 规定唯一归档目录为 `docs/archive/`。
  - 搜索仍显示 `docs/.archive/plan-vpre-v3-20260502.md`。
- 判断：
  - 隐藏归档目录必须删除或合并。
- 处理：
  - 内容需要保留则移动到 `docs/archive/`。
  - 删除 `docs/.archive/`。
  - `tests/test_docs_contract.py` 增加禁止 `docs/.archive`。

### 🟠 H-1：`graphify-out/cache/` 是生成缓存，不能属于源码核心

- 位置：
  - `graphify-out/cache/`
  - `graphify-out/GRAPH_REPORT.md`
- 类型：生成物 / 分析缓存
- 证据：
  - `GRAPH_REPORT.md` 是 2026-04-28 的图谱报告，包含 1218 nodes、2620 edges、cache 导航信息。
- 判断：
  - `cache/` 必删。
  - `GRAPH_REPORT.md` 不是运行核心；若还要参考，可移动到 `docs/references/graph_report_20260428.md`。
- 处理：
  - 删除 `graphify-out/cache/`。
  - `GRAPH_REPORT.md` 迁移或删除，二选一。
  - `.gitignore` 增加 `graphify-out/` 或至少 `graphify-out/cache/`。

### 🟠 H-2：`requirements.txt` 包含未证明使用的 Dashboard 依赖

- 位置：
  - `requirements.txt`
  - `pyproject.toml`
- 类型：依赖膨胀 / 依赖源不一致
- 证据：
  - `requirements.txt` 含 `fastapi`、`uvicorn`。
  - `pyproject.toml` 没有这两个依赖。
  - 搜索未发现 FastAPI app。
- 判断：
  - 如果当前不维护 dashboard backend，删除 `fastapi` 和 `uvicorn`。
- 处理：
  - 第一轮删除 `requirements.txt` 中 Dashboard 依赖。
  - 后续统一依赖源，推荐 `requirements.txt` 与 `pyproject.toml` 一致。

---

## 2. 新增发现：上一轮未充分强调的冗余/边界问题

### 🟠 H-3：`src/utils/logger.py` 很可能已被 `BaseTrainer` 内置 TensorBoard 逻辑取代

- 位置：
  - `src/utils/logger.py`
  - `src/utils/__init__.py`
  - `src/trainer/base_trainer.py`
- 类型：重复日志系统
- 证据：
  - `src/utils/logger.py` 定义独立 `Logger`，带 TensorBoard 和 JSON metrics。
  - `src/trainer/base_trainer.py` 已直接使用 `SummaryWriter`，并维护 `train_logs` / `eval_logs` / `_save_logs()`。
  - 搜索未发现主入口直接使用 `Logger`。
- 判断：
  - 若没有外部脚本依赖 `from src.utils import Logger`，应删除。
- 建议处理：
  - 第一轮不直接删；加入引用扫描。
  - 若扫描仅 `src/utils/__init__.py` 引用，则删除 `src/utils/logger.py`，并从 `src/utils/__init__.py` 移除导出。
- 需要确认：
  - 你是否仍手动或外部 notebook 使用 `src.utils.Logger`？

### 🟠 H-4：`src/utils/config.py` 与当前 YAML 读取链路重复

- 位置：
  - `src/utils/config.py`
  - `scripts/benchmark.py`
  - `scripts/train.py`
- 类型：重复配置系统
- 证据：
  - `src/utils/config.py` 使用 OmegaConf dataclass 配置。
  - `scripts/benchmark.py` 当前使用 `yaml.safe_load` 的 `load_config(path)`。
  - `scripts/train.py` 通过 `benchmark.load_config` 加载算法 YAML。
  - 搜索未发现主链路使用 `src.utils.config.load_config`。
- 判断：
  - 这是旧配置系统或未完成迁移产物。
- 建议处理：
  - 若确认不再使用 OmegaConf 配置系统，删除 `src/utils/config.py`。
  - 同步评估是否可从依赖中移除 `omegaconf`。
- 需要确认：
  - 你是否还计划保留 OmegaConf 配置能力？如果不保留，可删。

### 🟠 H-5：`src/utils/action_utils.py` 可能已被环境适配器和 trainer 内部 action formatting 取代

- 位置：
  - `src/utils/action_utils.py`
  - `scripts/evaluate.py`
  - `src/trainer/base_trainer.py`
  - `src/environments/mec_v3/game_theory_adapters.py`
- 类型：重复动作处理工具
- 证据：
  - `ActionScaler` 仅提供 scale/unscale/discrete/sanitize。
  - `scripts/evaluate.py` 自己实现 `_format_action()`。
  - `BaseTrainer.evaluate()` 内也实现大量 action 格式处理。
  - GameTheory adapter 本身也做动作编码/解码。
- 判断：
  - 很可能是旧通用工具，当前主链路没有统一使用。
- 建议处理：
  - 若引用扫描无真实使用，删除 `src/utils/action_utils.py`。
  - 反向选择：也可以保留它并把 evaluate/base_trainer 重复逻辑收敛到这里；但这是重构，不是瘦身。
- 需要确认：
  - 本轮目标是“删除冗余”还是“合并重复 action formatting 后保留工具”？建议本轮删除未用工具，不做重构。

### 🟠 H-6：`src/trainer/callbacks.py` 是扩展点，但当前没有证据显示它被实际启用

- 位置：
  - `src/trainer/callbacks.py`
  - `src/trainer/base_trainer.py`
- 类型：未使用扩展点
- 证据：
  - `BaseTrainer` 支持 `callbacks` 参数并调用 callback hooks。
  - `scripts/benchmark.py` / `scripts/train.py` 创建 trainer 时没有传 callbacks。
  - callbacks 中包含 Logging/Checkpoint/EarlyStopping，但 BaseTrainer 已内置 logging/checkpoint。
- 判断：
  - 这是可选扩展点，不是当前核心功能。
- 建议处理：
  - 若你不打算近期使用 early stopping / custom hook，删除 `callbacks.py`，并从 `BaseTrainer` 移除 callbacks 参数和 hook 调用。
  - 如果保留，需要明确写入“扩展 API，不是冗余”。
- 需要确认：
  - 是否保留 callback 扩展机制？

### 🟠 H-7：`scripts/generate_report.py` 是旧报告脚本，硬编码旧实验，需二选一

- 位置：`scripts/generate_report.py`
- 类型：旧脚本 / 当前功能未完全适配
- 证据：
  - 默认输入仍是 `benchmark_full_500k_20260423_101541.json`。
  - 默认输出仍是 `benchmark_report_20260423.md` 和 `benchmark_analysis_20260423.xlsx`。
  - 内部写死 `benchmark_time = "2026-04-23 10:15:41"` 和 `total_timesteps = 500000`。
- 判断：
  - 它不是可直接删除项，因为近期已追加 convergence quality notes。
  - 但以当前形态不应算核心稳定功能。
- 建议处理：
  - 方案 A：保留并重构为当前标准报告入口。
  - 方案 B：删除，改用 `plot_results.py + docs/convergence_failure_analysis.md`。
- 需要确认：
  - 你是否还需要自动生成 Markdown/Excel benchmark 报告？如果只需要图和 JSON，可删或降级。

### 🟡 M-1：`scripts/evaluate.py` 是单 checkpoint 评估入口，但 VSCode/实验主流程未直接使用

- 位置：`scripts/evaluate.py`
- 类型：边缘入口
- 证据：
  - `scripts/evaluate.py` 复用 `benchmark.py` 的 create_agent/make_env/load_config。
  - `.vscode/tasks.json` 未展示 evaluate 任务。
  - benchmark/train/evaluate 三者有重复的 metric extraction/action formatting。
- 判断：
  - 若你需要手动评估 checkpoint，保留。
  - 若当前所有评估都由 trainer/evaluate 和 benchmark 管线完成，可删除。
- 需要确认：
  - 你是否还会手动跑 `python scripts/evaluate.py --checkpoint ...`？

### 🟡 M-2：`scripts/backup_experiment.py` 不是上一轮可直接删除项，因为当前仍被主流程引用

- 位置：
  - `scripts/backup_experiment.py`
  - `scripts/experiment_manager.py`
  - `.vscode/tasks.json`
- 类型：运维辅助入口
- 证据：
  - `.vscode/tasks.json` 有 `experiment: full17 backup` 和 `backup-with-plots` 两个任务直接调用它。
  - `scripts/experiment_manager.py start --fresh` 默认会调用 `manager.backup_experiment()`，而 `src/experiment/manager.py` 直接 import `scripts.backup_experiment`。
- 判断：
  - 不能在第一轮直接删除。
  - 如果想瘦身，需要先修改 fresh 语义和 VSCode tasks。
- 建议处理：
  - 选项 A：保留 backup，但只作为本地保护机制，并禁止 backup 目录入 Git。
  - 选项 B：删除 backup 功能，同时改 `--fresh` 为拒绝覆盖或只删除本地未追踪目录。
- 需要确认：
  - fresh clean run 前是否还需要自动备份？

### 🟡 M-3：`src/comm/` 不是单一模块，而是可分层瘦身对象

- 位置：`src/comm/**`
- 类型：大体量研究库 / 非核心候选
- 证据：
  - `src/comm/__init__.py` 暴露 channel、antenna、propagation、signal、modulation、queueing、utils。
  - `src/environments/mec_v2/base_env.py` 可选引用 `src.comm.channel.pathloss`。
  - 暂未看到主入口直接使用 antenna、beamforming、doa、mimo、modulation、signal、queueing、mobility 等。
- 判断：
  - 不能直接删除整个 `src/comm/`，因为 `mec_v2/base_env.py` 可选使用 `channel/pathloss.py`。
  - 但可考虑仅保留 `src/comm/channel/pathloss.py`，删除其他通信用研究模块。
- 建议处理：
  - 第一轮只做 import scan，不删。
  - 第二轮若确认 GameTheory 主环境不依赖这些模块，可删除：
    ```text
    src/comm/antenna/
    src/comm/modulation/
    src/comm/signal/
    src/comm/propagation/
    src/comm/queueing/
    src/comm/channel/mimo.py
    src/comm/channel/fading.py
    src/comm/utils/
    ```
  - 保留：
    ```text
    src/comm/channel/pathloss.py
    ```
    或把 pathloss 直接内联到 `mec_v2/base_env.py` 后删除整个 `src/comm/`。
- 需要确认：
  - 论文是否还需要 MIMO/OFDM/antenna/QAM 这些通信模块？如果只是 GameTheory MEC benchmark，不需要。

### 🟡 M-4：`docs_paper/` 是文档资产，不是运行核心

- 位置：`docs_paper/**`
- 类型：论文资料 / 文档膨胀
- 证据：
  - `README.md` 和搜索结果显示 `docs_paper/reporting/*`、`mec_v3_game_theory.md`、`systemmodeldraft.md`、`mec_improvements.md` 等大量论文/汇报资料。
- 判断：
  - 如果当前仓库目标是“代码复现实验”，`docs_paper/` 应该迁出到 paper/documentation 仓或压缩归档。
  - 如果当前仓库也承载论文写作，则不能删除。
- 建议处理：
  - 选项 A：保留，但移动为 `paper/`，只保留最终稿和必要图表说明。
  - 选项 B：整个 `docs_paper/` 移到 `docs/archive/paper/` 或外部文档仓。
- 需要确认：
  - 论文写作资料是否必须留在当前代码仓？

### 🟡 M-5：`src/utils/buffer.py` 与 `rl_algorithms/utils/buffers.py` 是重复基础设施

- 位置：
  - `src/utils/buffer.py`
  - `rl_algorithms/utils/buffers.py`
- 类型：重复实现
- 证据：
  - 搜索显示两个路径都定义/导出 ReplayBuffer/RolloutBuffer。
  - 多个算法文件命中 `rl_algorithms/utils/buffers.py`。
- 判断：
  - 不能盲删；需要引用扫描确定实际使用。
- 建议处理：
  - 优先保留 `rl_algorithms/utils/buffers.py`。
  - 若 `src/utils/buffer.py` 只被 `src/utils/__init__.py` 导出，则删除或改为 re-export 过渡。
- 需要确认：
  - 你是否允许把 buffer owner 收敛到 `rl_algorithms/utils/buffers.py`？

---

## 3. 当前建议清理分级

### P0：立即清理，低风险

```text
src/trainer/benchmark.py
docs/.archive/
graphify-out/cache/
experiments/     # Git tracking remove only
results/         # Git tracking remove only
figures/         # Git tracking remove only
logs/            # Git tracking remove only
checkpoints/     # Git tracking remove only
requirements.txt: fastapi, uvicorn
README.md 中旧入口/旧目录描述
```

### P1：确认后清理

```text
src/utils/logger.py
src/utils/config.py
src/utils/action_utils.py
src/trainer/callbacks.py
scripts/evaluate.py
scripts/generate_report.py
scripts/backup_experiment.py
docs_paper/
graphify-out/GRAPH_REPORT.md
src/comm/ 除 pathloss 外的大部分通信研究模块
src/utils/buffer.py
```

### P2：暂不清理

```text
rl_algorithms/
configs/algorithms/
src/environments/mec_v3/
src/environments/mec_v2/base_env.py
src/trainer/base_trainer.py
src/trainer/on_policy_trainer.py
src/trainer/off_policy_trainer.py
src/experiment/
scripts/benchmark.py
scripts/train.py
scripts/experiment_manager.py
scripts/plot_results.py
src/utils/helpers.py
src/utils/composite_score.py
.vscode/
```

---

## 4. 建议执行策略

### Phase 1：Repo hygiene + 坏入口删除

- scope: auto
- 目标：处理 P0，不碰任何训练逻辑。
- 验证：
  ```bash
  pytest tests/test_docs_contract.py tests/test_repo_hygiene.py tests/test_active_entrypoints.py -q
  python scripts/benchmark.py --help
  python scripts/train.py --help
  python scripts/experiment_manager.py --help
  python scripts/plot_results.py --help
  ```

### Phase 2：引用扫描审计

- scope: auto
- 新增：
  ```text
  scripts/audit_import_graph.py
  docs/import_usage_audit.md
  tests/test_import_usage_audit.py
  ```
- 扫描对象：
  ```text
  src/utils/logger.py
  src/utils/config.py
  src/utils/action_utils.py
  src/trainer/callbacks.py
  scripts/evaluate.py
  scripts/generate_report.py
  scripts/backup_experiment.py
  src/comm/
  src/utils/buffer.py
  docs_paper/
  ```
- 输出每个候选对象：
  - 被哪些 `.py/.json/.md` 引用
  - 是否属于 `.vscode` 入口
  - 是否属于主训练链路
  - 建议动作：keep / delete / move / refactor

### Phase 3：用户确认后删除 P1

- scope: review
- 不在没有确认的情况下删除：
  - `src/comm/`
  - `docs_paper/`
  - `scripts/evaluate.py`
  - `scripts/generate_report.py`
  - `scripts/backup_experiment.py`
  - callbacks/logger/config/action_utils/buffer

### Phase 4：收缩 README 与依赖

- scope: auto
- README 只描述：
  - GameTheory MEC benchmark
  - 17 algorithms
  - VSCode/CLI experiment manager
  - plotting/reporting
- 删除旧环境、旧 trainer benchmark、旧 docs/reporting 等描述。

---

## 5. 需要用户确认的问题

以下内容我不会自动忽视，也不会替你默认保留；需要你明确选择。

1. `src/comm/`：是否还需要 MIMO/antenna/QAM/propagation/queueing 这些通信研究模块？  
   - A：不需要，后续只保留 pathloss 或完全内联后删除整个 `src/comm/`
   - B：需要，作为论文模型资产保留

2. `docs_paper/`：论文写作资料是否必须留在当前代码仓？  
   - A：不需要，迁出或归档
   - B：需要，但整理为 `paper/`，只保留最终相关材料

3. `scripts/evaluate.py`：是否还会手动评估 checkpoint？  
   - A：不用，删除
   - B：需要，保留并减少与 benchmark/base_trainer 的重复逻辑

4. `scripts/generate_report.py`：是否需要 Markdown/Excel 报告生成？  
   - A：不用，删除
   - B：需要，重构为当前 benchmark JSON 通用入口

5. `scripts/backup_experiment.py`：fresh run 前是否需要自动备份旧实验？  
   - A：不需要，删除 backup 功能并改 VSCode tasks
   - B：需要，保留但禁止 backup 产物入 Git

6. `src/trainer/callbacks.py`：是否保留 callback 扩展机制？  
   - A：不用，删除 callbacks 机制
   - B：需要，保留并补测试/文档说明

7. `src/utils/logger.py` / `config.py` / `action_utils.py`：是否允许删除未被主链路引用的通用工具？  
   - A：允许，引用扫描后删除
   - B：保留，作为未来扩展工具

8. `src/utils/buffer.py`：是否允许将 buffer owner 收敛到 `rl_algorithms/utils/buffers.py`？  
   - A：允许，删除或 re-export `src/utils/buffer.py`
   - B：不允许，保留双 buffer 但需要说明用途差异

---

## 6. 是否建议继续修复

建议继续，但分两步：

1. 先执行 P0：无争议瘦身，降低仓库噪声。
2. 等你确认第 5 节 8 个问题后，再执行 P1：删除非核心功能。

不建议一次性删除 P1；这会把“瘦身”变成潜在破坏性重构。
