# docs/inbox/plan.md：paper2 项目瘦身 Phase 1-2

## 元信息

- 项目：`paper2`
- 计划版本：`slimming-plan-v1`
- 变更类型：`patch`
- 当前状态：**Phase 1-2 已执行，等待 review scope 审核**
- GitHub 仓库：`w2030298-art/paper2`
- 外部关联仓库：`w2030298-art/rl-mec-dashboard`
- 创建日期：2026-05-02

## Status

> 执行端读到此区块即可恢复上下文。

- 当前阶段：模块 13 Step 8 完成
- 当前模块：模块 13 Step 8
- 整体进度：`14 / 14`
- 状态：`NEEDS_REVIEW`
- 阻塞项：`C:\Users\22003\paper2\rl-mec-dashboard` 本机不存在，外部 dashboard 兼容性无法在本轮验证。
- 重要说明：本轮已完成 paper2 仓库内瘦身、`docs_paper/` 外迁、生成产物 Git tracking 移除、旧入口/旧工具删除；review scope 删除项等待人工审核。

## 用户已确认决策

1. `src/comm/`：保留，作为论文/通信模型资产。
2. `docs_paper/`：迁移到 `C:\Users\22003\paper2\writing_ref` 后从 paper2 仓库移除。
3. `scripts/evaluate.py`：删除。
4. `scripts/generate_report.py`：删除。
5. `scripts/backup_experiment.py`：保留。
6. `src/trainer/callbacks.py`：确认 `rl-mec-dashboard` 未使用后删除。
7. `src/utils/logger.py`、`src/utils/config.py`、`src/utils/action_utils.py`：确认 `rl-mec-dashboard` 未使用后删除。
8. buffer：保留 `src/utils/buffer.py` 作为 canonical owner；迁移旧引用后删除或短期保留 `rl_algorithms/utils/buffers.py` 兼容 wrapper。

## 必须保留边界

```text
rl_algorithms/
configs/algorithms/
configs/scoring_profiles.yaml
src/environments/mec_v3/
src/environments/mec_v2/base_env.py
src/trainer/base_trainer.py
src/trainer/on_policy_trainer.py
src/trainer/off_policy_trainer.py
src/experiment/
src/comm/
src/utils/helpers.py
src/utils/composite_score.py
src/utils/buffer.py
scripts/benchmark.py
scripts/train.py
scripts/experiment_manager.py
scripts/plot_results.py
scripts/backup_experiment.py
.vscode/
docs/
```

## 本轮待删除/迁移候选

```text
src/trainer/benchmark.py
scripts/evaluate.py
scripts/generate_report.py
src/trainer/callbacks.py
src/utils/logger.py
src/utils/config.py
src/utils/action_utils.py
docs_paper/
docs/.archive/
graphify-out/cache/
requirements.txt 中 fastapi/uvicorn
rl_algorithms/utils/buffers.py  # 仅在旧引用迁移完成后删除，否则短期保留兼容 wrapper
```

## 本轮不做

- 不删除 `rl_algorithms/`
- 不删除 `src/comm/`
- 不删除 `src/experiment/`
- 不删除 `.vscode/`
- 不改训练算法逻辑
- 不改 GameTheory 环境逻辑
- 不改 reward、metrics、convergence 计算逻辑
- 不删除本地 `experiments/`、`results/`、`figures/` 数据，只处理 Git tracking / ignore policy

---

# 模块 12：无争议仓库卫生清理

## Step 1：生成执行前审计快照

- **scope: auto**
- 新增：`docs/slimming_audit_phase2.md`
- 操作：
  - 在 paper2 根目录记录：
    ```bash
    git status --short
    git ls-files > .tmp_tracked_files.txt
    grep -R "src.trainer.benchmark\|scripts/evaluate.py\|scripts/generate_report.py\|TrainerCallback\|LoggingCallback\|EarlyStoppingCallback\|src.utils.logger\|src.utils.config\|src.utils.action_utils\|ActionScaler\|OmegaConf" -n . --exclude-dir=.git --exclude-dir=.venv --exclude-dir=experiments --exclude-dir=results --exclude-dir=figures || true
    ```
  - 在 dashboard 仓库记录：
    ```bash
    cd C:\Users\22003\paper2\rl-mec-dashboard
    grep -R "src.trainer.callbacks\|TrainerCallback\|LoggingCallback\|EarlyStoppingCallback\|src.utils.logger\|src.utils.config\|src.utils.action_utils\|ActionScaler\|OmegaConf\|scripts.evaluate\|generate_report" -n . --exclude-dir=.git --exclude-dir=.venv || true
    ```
  - 报告必须包含：`paper2 internal references`、`rl-mec-dashboard references`、处理结论表。

## Step 2：从 Git tracking 移除生成产物

- **scope: auto**
- 对象：`experiments/`、`results/`、`figures/`、`logs/`、`checkpoints/`
- 操作：
  ```bash
  git rm --cached -r experiments results figures logs checkpoints 2>nul || true
  ```
- 更新 `.gitignore` 至少包含：
  ```text
  experiments/.index.sqlite3
  experiments/**/*.lock
  experiments/**/control/
  experiments/**/process.json
  experiments/**/logs/
  experiments/**/artifacts/**/*.pt
  experiments/**/artifacts/**/*.pth
  experiments/**/artifacts/**/*.ckpt
  figures/
  results/
  logs/
  checkpoints/
  graphify-out/cache/
  ```

## Step 3：删除废弃坏入口 `src/trainer/benchmark.py`

- **scope: auto**
- 删除：`src/trainer/benchmark.py`
- 修改：`README.md`、`docs/README.md`、`.vscode/tasks.json` 中的旧引用。
- 不修改：`scripts/benchmark.py`

## Step 4：统一 docs 归档并清理 graphify cache

- **scope: auto**
- 若 `docs/.archive/` 存在，将其中 `.md` 移到 `docs/archive/`，再删除空目录。
- 删除 `graphify-out/cache/`。
- `graphify-out/GRAPH_REPORT.md` 若保留，移动为 `docs/references/graph_report_20260428.md`；否则从 Git 删除。
- 更新 `docs/README.md`，明确禁止 `docs/.archive/`。

## Step 5：移除未使用 dashboard 依赖

- **scope: auto**
- 从 paper2 `requirements.txt` 删除：
  ```text
  fastapi>=0.100.0
  uvicorn>=0.23.0
  ```
- 本 Step 不删除 `omegaconf`，等模块 13 删除 `src/utils/config.py` 后再处理。

## Step 6：新增仓库卫生测试

- **scope: auto**
- 新增：`tests/test_repo_hygiene.py`、`tests/test_active_entrypoints.py`
- 覆盖：
  - `docs/.archive` 不存在
  - `src/trainer/benchmark.py` 不存在
  - `.gitignore` 覆盖生成产物目录
  - `docs/slimming_audit_phase2.md` 记录 dashboard grep
  - 主入口 `--help` 可运行：`benchmark.py`、`train.py`、`experiment_manager.py`、`plot_results.py`

---

# 模块 13：已确认非核心功能删除

## Step 1：迁移 `docs_paper/` 到外部写作资料目录

- **scope: review**
- 源：`docs_paper/`
- 目标：`C:\Users\22003\paper2\writing_ref\docs_paper`
- 操作：
  1. 先复制到目标目录。
  2. 校验关键文件存在。
  3. 校验通过后 `git rm -r docs_paper`。
  4. 新增 `docs/references/writing_ref_migration.md` 记录迁移日期、源路径、目标路径。

## Step 2：删除 `scripts/evaluate.py`

- **scope: auto**
- 删除：`scripts/evaluate.py`
- 移除文档中的 `python scripts/evaluate.py`、`--checkpoint` 离线评估说明。
- 不修改 `BaseTrainer.evaluate()`。

## Step 3：删除 `scripts/generate_report.py`

- **scope: auto**
- 删除：`scripts/generate_report.py`
- 移除相关文档和测试。
- 报告职责改为：
  - `scripts/plot_results.py` 生成图和 convergence quality report。
  - `scripts/analyze_convergence_failures.py` 生成 failure analysis（若存在）。
  - 论文资料迁至 `C:\Users\22003\paper2\writing_ref`。

## Step 4：删除 callback 扩展机制

- **scope: review**
- 前置：`docs/slimming_audit_phase2.md` 证明 dashboard 无 callback 引用。
- 删除：`src/trainer/callbacks.py`
- 修改：`src/trainer/base_trainer.py`
- 操作：
  - 从 `BaseTrainer.__init__()` 删除 `callbacks` 参数。
  - 删除 `self.callbacks`。
  - 删除 `cb.on_train_begin`、`cb.on_step_end`、`cb.on_eval_end`、`cb.on_train_end`。
  - 若 `_stop_training` 只被 callback 使用，同步删除并简化训练循环。
  - 保留 BaseTrainer 内置 TensorBoard、checkpoint、eval logging。

## Step 5：删除旧 utils 工具

- **scope: review**
- 前置：`docs/slimming_audit_phase2.md` 证明 dashboard 无相关引用。
- 删除：
  ```text
  src/utils/logger.py
  src/utils/config.py
  src/utils/action_utils.py
  ```
- 修改：
  - `src/utils/__init__.py`
  - `requirements.txt`
  - `pyproject.toml`
- 删除 `src/utils/config.py` 后，若全仓无 `omegaconf` 引用，从 `requirements.txt` 和 `pyproject.toml` 删除 `omegaconf`。

## Step 6：buffer canonical owner 收敛

- **scope: review**
- 保留：`src/utils/buffer.py`
- 搜索：
  ```bash
  grep -R "rl_algorithms.utils.buffers" -n rl_algorithms src scripts tests || true
  ```
- 若存在旧引用，改为：
  ```python
  from src.utils.buffer import ReplayBuffer, RolloutBuffer
  ```
- 迁移完成后：
  - 无第三方兼容需要则删除 `rl_algorithms/utils/buffers.py`。
  - 若短期保留 wrapper，必须在 `docs/slimming_audit_phase2.md` 标记为 `compat-keep`。

## Step 7：更新 README 与 docs 契约

- **scope: auto**
- 修改：`README.md`、`docs/README.md`、`docs/vscode_experiment_usage.md`、`docs/frontend_monitoring_integration.md`（如存在）
- README 只保留当前核心链路：
  - 17 algorithms
  - GameTheory MEC environment
  - `scripts/benchmark.py`
  - `scripts/train.py`
  - `scripts/experiment_manager.py`
  - `scripts/plot_results.py`
  - VSCode tasks
  - dashboard 只读文件协议
- 删除说明：`scripts/evaluate.py`、`scripts/generate_report.py`、`src/trainer/benchmark.py`、`docs_paper/`、旧 `docs/reporting/`。

## Step 8：更新执行报告与进度

- **scope: auto**
- 修改：`docs/report.md`、`docs/progress.md`、`docs/issues.md`
- `docs/report.md` 写入：
  ```text
  STATUS: NEEDS_REVIEW
  Completed: Project Slimming Phase 1-2 execution
  In Review: 删除非核心入口和工具后的核心链路验证
  ```
- `docs/progress.md` 记录模块 12-13 的 14 个 Step 状态。
- 如果任何候选删除项因真实引用无法删除，必须写入 `docs/issues.md`，并标记 `NEEDS_REVIEW`。

---

# 总体验收

## 必跑命令

```bash
pytest -q
python scripts/benchmark.py --help
python scripts/train.py --help
python scripts/experiment_manager.py --help
python scripts/plot_results.py --help
```

## 禁止项检查

```bash
test ! -f src/trainer/benchmark.py
test ! -f scripts/evaluate.py
test ! -f scripts/generate_report.py
test ! -f src/trainer/callbacks.py
test ! -f src/utils/logger.py
test ! -f src/utils/config.py
test ! -f src/utils/action_utils.py
test ! -d docs/.archive
test ! -d graphify-out/cache
```

## 关键保留项检查

```bash
test -d rl_algorithms
test -d src/comm
test -d src/experiment
test -f scripts/benchmark.py
test -f scripts/train.py
test -f scripts/experiment_manager.py
test -f scripts/plot_results.py
test -f scripts/backup_experiment.py
test -f src/utils/buffer.py
```

## Dashboard 兼容性检查

```bash
cd C:\Users\22003\paper2\rl-mec-dashboard
python -m pytest -q
```

若 dashboard 测试失败，不直接回滚 paper2 删除项；先确认失败是否仅由测试环境路径造成。若失败来自 paper2 文件协议变化，记录到 `docs/issues.md` 并标记 `NEEDS_REVIEW`。

---

# 任务派发

## 类型:patch

执行 `docs/inbox/plan.md`：paper2 项目瘦身 Phase 1-2。注意：瘦身尚未开始，这是待执行计划，不是已完成状态。

本次范围:
- 先完成模块 12：repo hygiene、坏入口、生成产物 tracking、docs/archive、graphify cache、依赖清理
- 再完成模块 13：迁出 `docs_paper/`，删除已确认废弃入口和旧工具
- 保留 `src/comm/`、`scripts/backup_experiment.py`、`src/utils/buffer.py`
- 删除前必须 grep `C:\Users\22003\paper2\rl-mec-dashboard`，结果写入 `docs/slimming_audit_phase2.md`
- 不删除本地 `experiments/results/figures` 数据，只从 Git tracking 移除
- `docs_paper/` 必须先复制到 `C:\Users\22003\paper2\writing_ref\docs_paper` 并校验，再从 Git 删除
- 若任何候选删除项仍有真实引用，写 `docs/issues.md` 并标 `NEEDS_REVIEW`
- 修完跑 `pytest -q` 和四个主入口 `--help`
