# Plan Patch — Full 17 实验数据备份与 Fresh 自动保护

## 元信息

- 变更日期：2026-05-01
- 变更类型：新增功能 + 修改现有功能
- 关联原 plan.md：`docs/plan.md` — VSCode 一键化实验入口全量改造
- 基线项目：`paper2`
- 基线仓库：`w2030298-art/paper2`
- 涉及模块数：5
- 新增步骤数：7
- 修改步骤数：5
- 删除步骤数：0
- 是否需要微调研：否
- 核心目标：
  1. 用户可先点击 `💾 Backup Full 17 Data` 保存已完成 Full 17 实验数据。
  2. 用户点击 `🏁 Full 17 Fresh (Auto-Backup)` 时，系统自动检测旧数据，有则先备份再 fresh 删除重跑。
  3. 保留 `--no-backup` 作为显式跳过备份的危险操作入口。

## 变更背景

当前 `scripts/experiment_manager.py start --fresh` 的行为是：

1. 检测到旧 run 存在；
2. 直接调用 `manager.delete_experiment(run_id)`；
3. 重新创建并启动实验。

这个行为对 Quick smoke test 可接受，但对 `paper2_full_17_vscode` 这种长耗时全量实验风险过高。Full 17 完成后再次 fresh 会直接删除历史 `experiments/paper2_full_17_vscode/`，同时结果 JSON 和图表也没有统一归档入口。

本 patch 将 fresh 语义调整为：

```text
start --fresh 默认行为：
旧实验存在 → 自动备份 experiment dir + benchmark*.json → 删除旧实验 → 重新创建 → 启动

start --fresh --no-backup：
旧实验存在 → 直接删除旧实验 → 重新创建 → 启动
```

---

## 操作清单（Codex 必须严格按此处理）

# 模块 10：实验数据备份与 Fresh 自动保护

## 概述

- 职责：为 Full 17 实验增加显式备份入口，并将 `start --fresh` 改为默认自动备份后重跑。
- 前置依赖：
  - 原模块 2：`scripts/experiment_manager.py` 已支持 `--preset` 与 `--fresh`
  - 原模块 3：`.vscode/launch.json` 已支持 Full 17 Start/Resume
  - 原模块 4：`.vscode/tasks.json` 若本地存在则修改；若不存在则创建
- 预计步骤数：7

---

### [ADD] 模块 10 — Step 1：创建备份服务脚本 `scripts/backup_experiment.py`

- 操作：新建文件 `scripts/backup_experiment.py`
- 文件职责：
  1. 备份指定实验目录：`experiments/<run_id>/` → `experiments/<run_id>_<suffix>_<timestamp>/`
  2. 备份结果文件：`results/benchmark*.json` → `results/archive/<timestamp>/`
  3. 可选备份图表：`figures/*` → `figures/archive/<timestamp>/`
  4. 输出 JSON 结果，方便 VSCode task 和 CLI 自动化解析。
- 必须定义以下函数：

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse
import json
import shutil
import sys


@dataclass
class BackupResult:
    run_id: str
    timestamp: str
    experiment_backup_dir: str | None
    results_archive_dir: str | None
    figures_archive_dir: str | None
    copied_result_files: list[str]
    copied_figure_files: list[str]
    skipped: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    """Return timestamp string in YYYYMMDD_HHMMSS format."""


def assert_safe_run_id(run_id: str) -> None:
    """Allow only [A-Za-z0-9_.-]+ and reject path traversal."""


def backup_experiment(
    *,
    run_id: str,
    experiments_dir: Path = Path("experiments"),
    results_dir: Path = Path("results"),
    figures_dir: Path = Path("figures"),
    include_plots: bool = False,
    suffix: str = "backup",
    timestamp: str | None = None,
    require_existing: bool = False,
) -> BackupResult:
    """Backup experiment directory, benchmark result json files, and optional figures."""


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""


def main(argv: list[str] | None = None) -> int:
    """Run backup command and print BackupResult JSON."""
```

- `backup_experiment()` 实现细节：
  - `timestamp` 为 `None` 时调用 `utc_timestamp()`。
  - `suffix` 只允许：
    - `"backup"`：手动备份
    - `"auto"`：fresh 自动备份
  - `assert_safe_run_id(run_id)` 必须拒绝：
    - 空字符串
    - 包含 `/`
    - 包含 `\`
    - 包含 `..`
    - 不匹配正则 `^[A-Za-z0-9_.-]+$`
  - `experiment_dir = experiments_dir / run_id`
  - `experiment_backup_dir = experiments_dir / f"{run_id}_{suffix}_{timestamp}"`
  - 如果 `experiment_dir` 不存在：
    - 当 `require_existing=False`：返回 `BackupResult(skipped=True, reason="experiment directory not found")`
    - 当 `require_existing=True`：抛出 `FileNotFoundError`
  - 如果 `experiment_dir / "process.json"` 存在：
    - 抛出 `RuntimeError(f"Cannot backup running experiment: {run_id}")`
    - 不复制任何文件
  - 如果 `experiment_backup_dir` 已存在：
    - 抛出 `FileExistsError`
    - 不覆盖
  - 使用 `shutil.copytree(experiment_dir, experiment_backup_dir)` 复制实验目录。
  - 结果文件归档：
    - 创建 `results_archive_dir = results_dir / "archive" / timestamp`
    - 只复制 `results_dir.glob("benchmark*.json")`
    - 若没有匹配结果文件，允许 archive 目录不存在，`results_archive_dir=None`
    - 不递归复制 `results/archive/` 旧归档
  - 图表归档：
    - 仅当 `include_plots=True` 时执行
    - 创建 `figures_archive_dir = figures_dir / "archive" / timestamp`
    - 复制 `figures_dir` 顶层文件，不递归复制 `figures/archive/`
    - 只复制普通文件，不复制目录
  - 返回 `BackupResult`。
- CLI 参数：
  - `--run-id`：required
  - `--experiments-dir`：默认 `experiments`
  - `--results-dir`：默认 `results`
  - `--figures-dir`：默认 `figures`
  - `--include-plots`：flag
  - `--suffix`：choices `backup|auto`，默认 `backup`
  - `--require-existing`：flag
- CLI 行为：
  - 成功时 stdout 输出 `BackupResult.to_dict()` JSON，返回 0
  - 失败时 stderr 输出 `{"status": "error", "error": "..."}`
  - `FileNotFoundError` 返回 2
  - `RuntimeError` 返回 3
  - 其他异常返回 1
- 验证：

```bash
python -c "from scripts.backup_experiment import backup_experiment, BackupResult; print('OK')"
python scripts/backup_experiment.py --help
```

---

### [MODIFY] 模块 2 — Step 4：扩展 `ExperimentManager`，增加备份入口

**原步骤**（参考 `docs/plan.md` 模块 2 Step 4）：
> 增加 `--fresh` 支持 Quick clean run：如果 `args.fresh is True` 且 `manager.store.exists(run_id)`，调用 `manager.delete_experiment(run_id)` 后重新创建。

**修改后**：

- 文件：`src/experiment/manager.py`
- 操作：新增方法：

```python
from pathlib import Path

from scripts.backup_experiment import BackupResult, backup_experiment


class ExperimentManager:
    ...

    def backup_experiment(
        self,
        run_id: str,
        *,
        include_plots: bool = False,
        suffix: str = "backup",
    ) -> BackupResult:
        return backup_experiment(
            run_id=run_id,
            experiments_dir=self.store.root_dir,
            results_dir=Path("results"),
            figures_dir=Path("figures"),
            include_plots=include_plots,
            suffix=suffix,
            require_existing=True,
        )
```

- 注意：
  - 不要把 backup 逻辑重复写在 `ExperimentManager` 里。
  - Manager 只做 orchestration，真实复制逻辑必须在 `scripts/backup_experiment.py`。
  - 保留现有 `delete_experiment()` 不删除。
  - `results_dir` 与 `figures_dir` 当前固定为项目根目录下 `results` / `figures`，与现有 CLI 默认 `output_dir="results"` 保持一致。
- 验证：

```bash
python -c "from src.experiment.manager import ExperimentManager; assert hasattr(ExperimentManager, 'backup_experiment')"
```

---

### [MODIFY] 模块 2 — Step 4：修改 `scripts/experiment_manager.py start --fresh` 默认自动备份

**原步骤**（参考 `docs/plan.md` 模块 2 Step 4）：
> `start --fresh` 检测到旧实验存在时直接删除旧 run。

**修改后**：

- 文件：`scripts/experiment_manager.py`
- 操作 1：在 `start_parser` 中新增参数：

```python
start_parser.add_argument(
    "--no-backup",
    action="store_true",
    help="Skip automatic backup before --fresh deletes an existing experiment.",
)
```

- 操作 2：修改 `args.command == "start"` 分支中的 fresh 逻辑。
- 旧逻辑：

```python
if args.fresh and manager.store.exists(run_id):
    manager.delete_experiment(run_id)
if args.fresh or not manager.store.exists(run_id):
    manager.create_experiment(**options)
manager.start_or_resume(run_id)
```

- 新逻辑：

```python
if args.fresh and manager.store.exists(run_id):
    if not args.no_backup:
        manager.backup_experiment(run_id, include_plots=False, suffix="auto")
    manager.delete_experiment(run_id)

if args.fresh or not manager.store.exists(run_id):
    manager.create_experiment(**options)

manager.start_or_resume(run_id)
```

- 语义要求：
  - 默认：`--fresh` 会自动备份。
  - 显式危险操作：`--fresh --no-backup` 才跳过备份。
  - 自动备份的 suffix 必须为 `"auto"`，生成目录名 `experiments/<run_id>_auto_<timestamp>/`。
  - 自动备份不包含 plots，因为 fresh 重跑前 plots 可能来自旧 benchmark alias；plots 需要用户通过显式 `Backup Full 17 Data + Plots` 保留。
- 验证：

```bash
python scripts/experiment_manager.py start --help | grep -E -- "--no-backup"
```

---

### [ADD] 模块 10 — Step 2：新增备份脚本单元测试

- 操作：新建文件 `tests/test_backup_experiment.py`
- 测试函数必须包含：

```python
def test_backup_experiment_copies_experiment_dir_and_benchmark_files(tmp_path): ...
def test_backup_experiment_include_plots_copies_top_level_figures(tmp_path): ...
def test_backup_experiment_skips_missing_when_not_required(tmp_path): ...
def test_backup_experiment_requires_existing_when_requested(tmp_path): ...
def test_backup_experiment_refuses_running_experiment(tmp_path): ...
def test_backup_experiment_rejects_unsafe_run_id(tmp_path): ...
def test_backup_experiment_does_not_copy_existing_archives(tmp_path): ...
```

- 测试细节：
  - 使用 `tmp_path` 构造：
    - `experiments/paper2_full_17_vscode/run.json`
    - `experiments/paper2_full_17_vscode/state.json`
    - `results/benchmark.json`
    - `results/benchmark_paper2_full_17_vscode.json`
    - `results/archive/old/benchmark_old.json`
    - `figures/convergence_curves.png`
    - `figures/archive/old/old.png`
  - 调用 `backup_experiment(..., timestamp="20260501_150000")` 固定 timestamp，避免测试不稳定。
  - 断言：
    - `experiments/paper2_full_17_vscode_backup_20260501_150000/run.json` 存在
    - `results/archive/20260501_150000/benchmark.json` 存在
    - `results/archive/20260501_150000/benchmark_paper2_full_17_vscode.json` 存在
    - `results/archive/20260501_150000/archive` 不存在
    - include_plots 时 `figures/archive/20260501_150000/convergence_curves.png` 存在
    - `figures/archive/20260501_150000/archive` 不存在
    - 存在 `process.json` 时抛出 `RuntimeError`
    - run id 为 `"../x"`、`"a/b"`、`"a\\b"`、`""` 时抛出 `ValueError`
- 验证：

```bash
pytest tests/test_backup_experiment.py -v
```

---

### [MODIFY] 模块 2 — Step 5：更新 CLI fresh 测试，覆盖自动备份与 no-backup

**原步骤**（参考 `docs/plan.md` 模块 2 Step 5）：
> `test_cli_start_fresh_deletes_existing_before_create` 断言 fresh 事件顺序为 `delete → create → start`。

**修改后**：

- 文件：`tests/test_experiment_manager_cli.py`
- 操作 1：修改现有 `test_cli_start_fresh_deletes_existing_before_create`
- 新断言事件顺序：

```python
assert events == ["backup", "delete", "create", "start"]
```

- FakeManager 新增：

```python
def backup_experiment(self, run_id: str, *, include_plots: bool = False, suffix: str = "backup") -> None:
    assert run_id == "vscode_quick"
    assert include_plots is False
    assert suffix == "auto"
    events.append("backup")
```

- 操作 2：新增测试：

```python
def test_cli_start_fresh_no_backup_skips_backup(monkeypatch) -> None:
    ...
```

- 该测试断言：
  - 调用：`cli.main(["start", "--preset", "quick", "--fresh", "--no-backup"])`
  - 事件顺序为 `["delete", "create", "start"]`
  - FakeManager 的 `backup_experiment()` 如果被调用则 `raise AssertionError`
- 操作 3：新增测试：

```python
def test_cli_start_help_mentions_no_backup(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["start", "--help"])
    assert exc_info.value.code == 0
    assert "--no-backup" in capsys.readouterr().out
```

- 验证：

```bash
pytest tests/test_experiment_manager_cli.py -v
```

---

### [MODIFY] 模块 3 — Step 4：更新 `.vscode/launch.json`，增加 Full 17 备份与 fresh 自动备份入口

**原步骤**（参考 `docs/plan.md` 模块 3 Step 4）：
> 新增 Full 17 Start/Resume 入口，args 为 `start --preset full17`。

**修改后**：

- 文件：`.vscode/launch.json`
- 操作 1：保留现有入口：
  - `🏁 Experiment Full 17 Start/Resume`
- 操作 2：新增手动备份入口：

```json
{
  "name": "💾 Backup Full 17 Data",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/scripts/backup_experiment.py",
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
  },
  "args": [
    "--run-id",
    "paper2_full_17_vscode"
  ]
}
```

- 操作 3：新增手动备份含图表入口：

```json
{
  "name": "💾 Backup Full 17 Data + Plots",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/scripts/backup_experiment.py",
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
  },
  "args": [
    "--run-id",
    "paper2_full_17_vscode",
    "--include-plots"
  ]
}
```

- 操作 4：新增 Full 17 fresh 自动备份入口：

```json
{
  "name": "🏁 Full 17 Fresh (Auto-Backup)",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/scripts/experiment_manager.py",
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
  },
  "args": [
    "start",
    "--preset",
    "full17",
    "--fresh"
  ]
}
```

- 操作 5：新增危险入口，放在配置列表后部，不作为推荐入口：

```json
{
  "name": "⚠️ Full 17 Fresh (No Backup)",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/scripts/experiment_manager.py",
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
  },
  "args": [
    "start",
    "--preset",
    "full17",
    "--fresh",
    "--no-backup"
  ]
}
```

- 排序要求：
  1. `💾 Backup Full 17 Data`
  2. `💾 Backup Full 17 Data + Plots`
  3. `🏁 Full 17 Fresh (Auto-Backup)`
  4. `🏁 Experiment Full 17 Start/Resume`
  5. 其他原有入口
  6. `⚠️ Full 17 Fresh (No Backup)` 放在靠后位置
- 验证：

```bash
python -m json.tool .vscode/launch.json
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path(".vscode/launch.json").read_text(encoding="utf-8"))
names = [item["name"] for item in data["configurations"]]
assert "💾 Backup Full 17 Data" in names
assert "💾 Backup Full 17 Data + Plots" in names
assert "🏁 Full 17 Fresh (Auto-Backup)" in names
assert "⚠️ Full 17 Fresh (No Backup)" in names
fresh = next(item for item in data["configurations"] if item["name"] == "🏁 Full 17 Fresh (Auto-Backup)")
assert fresh["program"].endswith("/scripts/experiment_manager.py")
assert fresh["args"] == ["start", "--preset", "full17", "--fresh"]
PY
```

---

### [ADD] 模块 10 — Step 3：新增或合并 `.vscode/tasks.json` 备份任务

- 文件：`.vscode/tasks.json`
- 操作：
  - 如果文件不存在：创建标准 VSCode tasks JSON。
  - 如果文件已存在：只追加下面两个 task，不删除原有任务。
- 新增 task 1：

```json
{
  "label": "experiment: full17 backup",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/backup_experiment.py",
    "--run-id",
    "paper2_full_17_vscode"
  ],
  "group": "build",
  "problemMatcher": []
}
```

- 新增 task 2：

```json
{
  "label": "experiment: full17 backup-with-plots",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/backup_experiment.py",
    "--run-id",
    "paper2_full_17_vscode",
    "--include-plots"
  ],
  "group": "build",
  "problemMatcher": []
}
```

- 如果创建新文件，完整结构为：

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "experiment: full17 backup",
      "type": "shell",
      "command": "python",
      "args": [
        "scripts/backup_experiment.py",
        "--run-id",
        "paper2_full_17_vscode"
      ],
      "group": "build",
      "problemMatcher": []
    },
    {
      "label": "experiment: full17 backup-with-plots",
      "type": "shell",
      "command": "python",
      "args": [
        "scripts/backup_experiment.py",
        "--run-id",
        "paper2_full_17_vscode",
        "--include-plots"
      ],
      "group": "build",
      "problemMatcher": []
    }
  ]
}
```

- 验证：

```bash
python -m json.tool .vscode/tasks.json
python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path(".vscode/tasks.json").read_text(encoding="utf-8"))
labels = [task["label"] for task in data["tasks"]]
assert "experiment: full17 backup" in labels
assert "experiment: full17 backup-with-plots" in labels
PY
```

---

### [MODIFY] 模块 1 — Step 1：扩展 VSCode 配置测试，覆盖 backup/fresh 入口

**原步骤**（参考 `docs/plan.md` 模块 1 Step 1）：
> 创建 `tests/test_vscode_config.py`，检查 launch/tasks JSON 结构与核心入口。

**修改后**：

- 文件：`tests/test_vscode_config.py`
- 操作：新增测试函数：

```python
def test_launch_json_contains_full17_backup_entries() -> None: ...
def test_launch_json_contains_full17_fresh_auto_backup_entry() -> None: ...
def test_tasks_json_contains_full17_backup_tasks() -> None: ...
```

- 断言要求：
  - `💾 Backup Full 17 Data` 存在
  - `💾 Backup Full 17 Data + Plots` 存在
  - `🏁 Full 17 Fresh (Auto-Backup)` 存在
  - `⚠️ Full 17 Fresh (No Backup)` 存在
  - 两个 backup 入口的 `program` 均为 `${workspaceFolder}/scripts/backup_experiment.py`
  - `🏁 Full 17 Fresh (Auto-Backup)` 的 args 精确等于 `["start", "--preset", "full17", "--fresh"]`
  - `⚠️ Full 17 Fresh (No Backup)` 的 args 精确等于 `["start", "--preset", "full17", "--fresh", "--no-backup"]`
  - `.vscode/tasks.json` 包含：
    - `experiment: full17 backup`
    - `experiment: full17 backup-with-plots`
- 验证：

```bash
pytest tests/test_vscode_config.py -v
```

---

### [ADD] 模块 10 — Step 4：全流程验收

- 操作：执行以下命令：

```bash
# 1. 新备份脚本可导入
python -c "from scripts.backup_experiment import backup_experiment, BackupResult; print('OK')"

# 2. CLI help 正常
python scripts/backup_experiment.py --help
python scripts/experiment_manager.py start --help

# 3. 单元测试
pytest tests/test_backup_experiment.py tests/test_experiment_manager_cli.py tests/test_vscode_config.py -v

# 4. JSON 配置合法
python -m json.tool .vscode/launch.json
python -m json.tool .vscode/tasks.json

# 5. 无训练副作用 help 检查
python scripts/experiment_manager.py --help
python scripts/benchmark.py --help
```

- 如果本地有可安全备份的已完成 Full 17 数据，可额外执行人工验收：

```bash
python scripts/backup_experiment.py --run-id paper2_full_17_vscode --include-plots
python scripts/experiment_manager.py start --preset full17 --fresh --no-backup
```

- 注意：
  - 第二条 `--no-backup` 是危险验收命令，只能在确认旧数据已备份后执行。
  - 常规用户操作应使用 `start --preset full17 --fresh`，不带 `--no-backup`。
- 验证：全部命令返回 0。

---

## 受影响但无需修改的模块

- 模块 7：Reward + comm_score 权重修正
  - 不修改。其产出的 `results/benchmark*.json` 会被备份脚本归档。
- 模块 8：收敛曲线数据收集与可视化
  - 不修改。其产出的 `figures/convergence_curves.*` 仅在 `--include-plots` 时归档。
- 模块 9：Composite Score 综合评分体系
  - 不修改。其产出的综合评分 JSON 字段与图表文件按通用 result/figure 备份规则处理。

---

## patch 自检

1. 每条 [MODIFY] 均标明了原模块和原 Step。
2. 每条 [ADD] 均指定了文件、函数、CLI 参数、测试和验证命令。
3. 本 patch 不包含 [DELETE]。
4. 会影响已完成的模块 1/2/3/4 测试，但这是对已完成功能的增量增强；progress.md 中对应步骤应标记 `[MODIFIED]`。
5. 不重跑模块 7-9，不修改算法、reward、comm_score、composite score。
6. `--fresh` 语义改变是有意破坏性增强：默认自动备份，只有 `--no-backup` 才保持旧危险行为。

---

## 验收标准

- [ ] `python scripts/backup_experiment.py --run-id paper2_full_17_vscode` 能生成 `experiments/paper2_full_17_vscode_backup_<timestamp>/`
- [ ] `python scripts/backup_experiment.py --run-id paper2_full_17_vscode --include-plots` 能同时归档 `figures/` 顶层图表
- [ ] `python scripts/experiment_manager.py start --preset full17 --fresh` 检测到旧 run 时先生成 `experiments/paper2_full_17_vscode_auto_<timestamp>/`，再删除旧 run 并重建
- [ ] `python scripts/experiment_manager.py start --preset full17 --fresh --no-backup` 跳过自动备份，直接 fresh
- [ ] `.vscode/launch.json` 包含 4 个新增入口：
  - `💾 Backup Full 17 Data`
  - `💾 Backup Full 17 Data + Plots`
  - `🏁 Full 17 Fresh (Auto-Backup)`
  - `⚠️ Full 17 Fresh (No Backup)`
- [ ] `.vscode/tasks.json` 包含 2 个新增任务：
  - `experiment: full17 backup`
  - `experiment: full17 backup-with-plots`
- [ ] 以下测试通过：

```bash
pytest tests/test_backup_experiment.py tests/test_experiment_manager_cli.py tests/test_vscode_config.py -v
```

- [ ] 以下配置校验通过：

```bash
python -m json.tool .vscode/launch.json
python -m json.tool .vscode/tasks.json
```
