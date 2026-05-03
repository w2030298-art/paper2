# 开发计划：VSCode 一键化实验入口全量改造

## 元信息

- 项目：`paper2`
- 需求名称：VSCode Run and Debug 面板覆盖实验系统全部核心能力
- 技术栈：Python、argparse、VSCode `launch.json`、VSCode `tasks.json`、debugpy、pytest、JSON
- 路线：简单路线，跳过技术调研与架构设计，直接进入计划制定
- 总模块数：6
- 预计步骤总数：33
- 建议开发顺序：
  1. 现状锁定与 Quick 入口 bug 复现
  2. 实验预设与 CLI 稳定性增强
  3. VSCode `launch.json` 全入口改造
  4. VSCode `tasks.json` 任务入口补齐
  5. 使用文档更新
  6. 自动化测试与最终验收

## 固定需求边界

### 用户确认的默认 Full 17 benchmark 配置

- `run_id`: `paper2_full_17_vscode`
- `name`: `Paper2 Full 17 Algorithms VSCode Benchmark`
- `algorithms`:
  - `GRPO`
  - `PPO`
  - `SAC`
  - `DDQN`
  - `DDPG`
  - `TD3`
  - `A3C`
  - `TRPO`
  - `SimPO`
  - `MAPPO`
  - `QMIX`
  - `COMA`
  - `IPPO`
  - `VDN`
  - `MADDPG`
  - `IQL`
  - `MATD3`
- `timesteps`: `100000`
- `seed`: `42`
- `device`: `auto`
- `eval_episodes`: `10`
- `env`: `auto`
- `output_dir`: `results`

### 保留 Quick 入口

保留 Quick smoke test 入口，但必须修复当前用户反馈的 Quick 入口报错问题。Quick 默认配置如下：

- `run_id`: `vscode_quick`
- `name`: `VSCode Quick Benchmark`
- `algorithms`: `GRPO PPO SAC`
- `timesteps`: `5000`
- `seed`: `42`
- `device`: `auto`
- `eval_episodes`: `3`
- `env`: `auto`
- `output_dir`: `results`

### 必须覆盖的 VSCode 点击能力

VSCode 必须至少覆盖以下能力，用户无需手输命令：

1. Quick smoke test 启动/恢复
2. Quick fresh clean run，用于排查 Quick 入口旧状态导致的问题
3. Full 17 algorithms benchmark 启动/恢复
4. 当前实验恢复
5. 请求停止
6. 查看状态
7. 查看所有实验
8. 导出结果
9. 重置失败算法
10. 重建本地实验索引
11. 直接调用 `scripts/benchmark.py --all` 的全量 benchmark 入口

---

# 模块 1：现状锁定与 Quick 入口 bug 复现

## 概述

- 职责：先把当前 VSCode 入口、CLI 入口、Quick 报错路径固定下来，避免后续只改配置但不解决真实故障。
- 前置依赖：无
- 预计步骤数：5

## Step 1：创建 VSCode 配置测试文件

- 操作：创建文件：
  - `tests/test_vscode_config.py`
- 在文件中实现以下测试函数：
  - `test_launch_json_is_valid_json()`
  - `test_launch_json_contains_required_debug_entries()`
  - `test_quick_entry_uses_experiment_manager_start()`
  - `test_full_17_entry_contains_all_algorithms()`
  - `test_tasks_json_is_valid_json_if_present()`
  - `test_tasks_json_contains_required_tasks_if_present()`
- 测试实现要求：
  - 使用 `json.loads(Path(".vscode/launch.json").read_text(encoding="utf-8"))`
  - 不允许使用注释 JSON 解析器；本项目 `.vscode/launch.json` 必须保持标准 JSON
  - 所有断言只检查结构，不执行真实训练
- 验证：
  - 运行 `pytest tests/test_vscode_config.py -q`
  - 预期：新增测试可以运行；如果 Full 17 入口尚未实现，则对应测试先失败，作为后续开发目标。

## Step 2：复现 Quick 入口的等价 CLI 路径

- 操作：在 Codex 执行环境中运行现有 Quick 入口等价命令，捕获 stdout/stderr：
  ```bash
  python scripts/experiment_manager.py start --run-id vscode_quick --name "VSCode Quick Benchmark" --algorithms GRPO PPO SAC --timesteps 5000 --seed 42 --device auto --eval-episodes 3
  ```
- 如果命令返回非 0：
  - 打开并记录：
    - `experiments/vscode_quick/state.json`
    - `experiments/vscode_quick/run.json`
    - `experiments/vscode_quick/artifacts/GRPO/stdout.log`
    - `experiments/vscode_quick/artifacts/GRPO/stderr.log`
    - `experiments/vscode_quick/artifacts/PPO/stdout.log`
    - `experiments/vscode_quick/artifacts/PPO/stderr.log`
    - `experiments/vscode_quick/artifacts/SAC/stdout.log`
    - `experiments/vscode_quick/artifacts/SAC/stderr.log`
  - 不要先修改 `.vscode/launch.json` 掩盖问题。
- 验证：
  - 能明确定位 Quick 报错属于以下哪一类：
    1. CLI 参数或 run 状态问题
    2. Python import / `PYTHONPATH` 问题
    3. `scripts/train.py` 未写出 result JSON
    4. 训练依赖缺失
    5. 算法本身训练异常

## Step 3：修复 `scripts/train.py` 的 import 稳定性

- 操作：修改 `scripts/train.py` 顶部路径注入逻辑。
- 当前逻辑只插入 `project_root`。改为同时插入：
  - `project_root`
  - `project_root / "scripts"`
  - `project_root / "src"`
  - `project_root / "rl_algorithms"`
- 具体实现位置：`scripts/train.py` 中 `project_root = Path(__file__).resolve().parents[1]` 之后。
- 关键代码指引：
  ```python
  project_root = Path(__file__).resolve().parents[1]
  for path in [
      project_root,
      project_root / "scripts",
      project_root / "src",
      project_root / "rl_algorithms",
  ]:
      path_str = str(path)
      if path_str not in sys.path:
          sys.path.insert(0, path_str)
  ```
- 保留现有延迟导入 `_load_training_dependencies()` 设计，不能把 `torch`、trainer 或 `benchmark` 相关导入移回顶层。
- 验证：
  - `python scripts/train.py --help` 不应因为 `torch` 缺失而失败。
  - `python -m pytest tests/test_experiment_manager_cli.py -q` 通过。

## Step 4：补齐 result JSON 写出异常的诊断信息

- 操作：修改 `src/experiment/process_runner.py`。
- 目标：当训练子进程 exit code 为 0 但 `result.json` 不存在时，错误信息必须包含：
  - algorithm name
  - expected result json path
  - stdout log path
  - stderr log path
- 修改位置：`ProcessRunner.run_algorithm()` 中：
  ```python
  if not interrupted and exit_code == 0 and not paths["result_json"].exists():
      error = "Result JSON not found"
  ```
- 替换为：
  ```python
  if not interrupted and exit_code == 0 and not paths["result_json"].exists():
      error = (
          f"Result JSON not found for algorithm {spec.name}: "
          f"expected={paths['result_json']}; "
          f"stdout={paths['stdout_log']}; stderr={paths['stderr_log']}"
      )
  ```
- 验证：
  - 新增或更新单元测试，模拟 exit code 0 但无 `result.json`，断言错误信息包含 `algorithm`、`expected`、`stdout`、`stderr`。
  - 运行 `pytest tests/test_experiment_manager_cli.py -q` 通过。

## Step 5：记录 Quick bug 修复结论

- 操作：创建或更新文档：
  - `docs/vscode_experiment_usage.md`
- 在“常见问题”中新增小节：
  - `Quick 入口报错如何定位`
- 内容必须列出：
  - 等价 CLI 命令
  - 状态文件路径
  - stdout/stderr 路径
  - 若只是旧实验状态污染，应使用 VSCode 的 `Quick Fresh Clean Run` 或 `Reset Failed Algorithm`
- 验证：
  - 文档中包含字符串 `Quick Fresh Clean Run`
  - 文档中包含路径 `experiments/vscode_quick/state.json`

---

# 模块 2：实验预设与 CLI 稳定性增强

## 概述

- 职责：把 Quick 与 Full 17 的算法列表和默认参数集中管理，减少 `.vscode/launch.json` 中长参数列表造成的维护风险。
- 前置依赖：模块 1
- 预计步骤数：8

## Step 1：新增实验预设模块

- 操作：创建文件：
  - `src/experiment/presets.py`
- 文件中定义以下常量：
  ```python
  QUICK_RUN_ID = "vscode_quick"
  QUICK_NAME = "VSCode Quick Benchmark"
  QUICK_ALGORITHMS = ["GRPO", "PPO", "SAC"]
  QUICK_TIMESTEPS = 5000
  QUICK_SEED = 42
  QUICK_DEVICE = "auto"
  QUICK_EVAL_EPISODES = 3

  FULL_17_RUN_ID = "paper2_full_17_vscode"
  FULL_17_NAME = "Paper2 Full 17 Algorithms VSCode Benchmark"
  FULL_17_ALGORITHMS = [
      "GRPO", "PPO", "SAC", "DDQN", "DDPG", "TD3", "A3C", "TRPO",
      "SimPO", "MAPPO", "QMIX", "COMA", "IPPO", "VDN", "MADDPG",
      "IQL", "MATD3",
  ]
  FULL_17_TIMESTEPS = 100000
  FULL_17_SEED = 42
  FULL_17_DEVICE = "auto"
  FULL_17_EVAL_EPISODES = 10

  PRESETS = {
      "quick": {...},
      "full17": {...},
  }
  ```
- `PRESETS["quick"]` 和 `PRESETS["full17"]` 的值必须是普通 `dict`，字段包括：
  - `run_id`
  - `name`
  - `algorithms`
  - `timesteps`
  - `seed`
  - `device`
  - `eval_episodes`
  - `env`
  - `output_dir`
- 不引入第三方依赖。
- 验证：
  - `python -c "from src.experiment.presets import FULL_17_ALGORITHMS; assert len(FULL_17_ALGORITHMS) == 17"`

## Step 2：让算法注册表复用 Full 17 常量

- 操作：修改 `src/experiment/registry.py`。
- 将 `AlgorithmRegistry.SUPPORTED_ALGORITHMS` 改为引用 `FULL_17_ALGORITHMS` 的浅拷贝：
  ```python
  from .presets import FULL_17_ALGORITHMS

  class AlgorithmRegistry:
      SUPPORTED_ALGORITHMS = list(FULL_17_ALGORITHMS)
  ```
- 验证：
  - `python -c "from src.experiment.registry import AlgorithmRegistry; assert len(AlgorithmRegistry.SUPPORTED_ALGORITHMS) == 17"`

## Step 3：为 `experiment_manager.py start` 增加 `--preset`

- 操作：修改 `scripts/experiment_manager.py`。
- 在 `start_parser` 中新增：
  ```python
  start_parser.add_argument("--preset", choices=["quick", "full17"], default=None)
  ```
- 修改 `main()` 的 `args.command == "start"` 分支：
  - 如果 `args.preset is None`，保持现有行为。
  - 如果 `args.preset == "quick"`：
    - 使用 `PRESETS["quick"]` 填充默认值。
  - 如果 `args.preset == "full17"`：
    - 使用 `PRESETS["full17"]` 填充默认值。
  - 用户显式传入的参数优先级高于 preset 默认值。
- 为了支持“显式传参覆盖 preset”，需要把 start parser 的默认值调整为 `None`：
  - `--run-id` 从 `required=True` 改为 `default=None`
  - `--name` 保持 `default=None`
  - `--algorithms` 从默认 `["GRPO", "PPO", "SAC", "DDQN"]` 改为 `default=None`
  - `--timesteps` 从 `default=5000` 改为 `default=None`
  - `--seed` 从 `default=42` 改为 `default=None`
  - `--device` 从 `default="auto"` 改为 `default=None`
  - `--eval-episodes` 从 `default=3` 改为 `default=None`
  - `--env` 从 `default="auto"` 改为 `default=None`
  - `--output-dir` 从 `default="results"` 改为 `default=None`
- 如果 `args.preset is None` 且 `args.run_id is None`，抛出 `ValueError("start requires --run-id or --preset")`。
- 验证：
  - `python scripts/experiment_manager.py start --help` 显示 `--preset {quick,full17}`。
  - `pytest tests/test_experiment_manager_cli.py -q` 通过。

## Step 4：增加 `--fresh` 支持 Quick clean run

- 操作：修改 `scripts/experiment_manager.py`、`src/experiment/manager.py`、`src/experiment/state_store.py`。
- 目标：VSCode 可以点击“Quick Fresh Clean Run”，删除旧 `vscode_quick` 实验目录后重新创建，解决旧状态污染导致的 Quick 入口问题。
- `scripts/experiment_manager.py`：
  ```python
  start_parser.add_argument("--fresh", action="store_true")
  ```
- `src/experiment/state_store.py`：
  - 新增方法：
    ```python
    def delete(self, run_id: str) -> None:
        ...
    ```
  - 实现要求：
    - 只允许删除 `self.root_dir / run_id`
    - 如果目录不存在，直接返回
    - 使用 `shutil.rmtree`
    - 如果存在 `process.json`，抛出 `ExperimentStateError("Cannot delete running experiment: {run_id}")`
    - 删除前先清理 stale lock 文件不做特殊处理；如目录内存在 `process.json`，绝对不能删除。
- `src/experiment/manager.py`：
  - 新增方法：
    ```python
    def delete_experiment(self, run_id: str) -> None:
        self.store.delete(run_id)
        self.index.initialize()
        self.index.rebuild_from_store(self.store)
    ```
- `scripts/experiment_manager.py` 的 start 分支：
  - 如果 `args.fresh is True` 且 `manager.store.exists(run_id)`：
    - 调用 `manager.delete_experiment(run_id)`
    - 然后按正常 start 逻辑重新创建
- 验证：
  - 新增测试 `test_cli_start_fresh_deletes_existing_before_create`
  - 运行 `pytest tests/test_experiment_manager_cli.py -q`

## Step 5：增加 `preset` CLI 单元测试

- 操作：更新 `tests/test_experiment_manager_cli.py`。
- 新增测试：
  - `test_cli_start_quick_preset_uses_quick_defaults`
  - `test_cli_start_full17_preset_uses_full17_defaults`
  - `test_cli_start_preset_allows_explicit_run_id_override`
  - `test_cli_start_requires_run_id_without_preset`
- 断言要求：
  - quick preset 创建实验时 algorithms 等于 `["GRPO", "PPO", "SAC"]`
  - full17 preset 创建实验时 algorithms 长度为 17，且包含 `MATD3`
  - 显式 `--run-id custom_run` 覆盖 preset run id
  - 无 `--run-id` 且无 `--preset` 时返回非 0
- 验证：
  - `pytest tests/test_experiment_manager_cli.py -q`

## Step 6：增加 `delete/fresh` state store 单元测试

- 操作：创建文件或更新现有测试：
  - `tests/test_experiment_state_store.py`
- 新增测试：
  - `test_state_store_delete_missing_run_is_noop`
  - `test_state_store_delete_existing_run_removes_directory`
  - `test_state_store_delete_refuses_running_experiment_with_process_json`
- 使用 `tmp_path` 构造临时 `JsonStateStore`。
- 验证：
  - `pytest tests/test_experiment_state_store.py -q`

## Step 7：保持原 CLI 兼容性

- 操作：确认以下旧命令仍可解析：
  ```bash
  python scripts/experiment_manager.py start --run-id demo --algorithms GRPO PPO --timesteps 5000 --seed 42 --device auto --eval-episodes 3
  python scripts/experiment_manager.py resume --run-id demo
  python scripts/experiment_manager.py stop --run-id demo
  python scripts/experiment_manager.py status --run-id demo
  python scripts/experiment_manager.py list
  python scripts/experiment_manager.py export --run-id demo
  python scripts/experiment_manager.py reset --run-id demo --algorithm GRPO
  python scripts/experiment_manager.py rebuild-index
  ```
- 验证：
  - `pytest tests/test_experiment_manager_cli.py -q`
  - `python scripts/experiment_manager.py --help` 正常输出

## Step 8：确认 Full 17 预设不触发真实训练的结构测试

- 操作：新增测试：
  - `tests/test_experiment_presets.py`
- 测试函数：
  - `test_full17_preset_has_exactly_registered_algorithms`
  - `test_quick_preset_is_subset_of_full17`
  - `test_preset_values_are_json_serializable`
- 验证：
  - `pytest tests/test_experiment_presets.py -q`

---

# 模块 3：VSCode `launch.json` 全入口改造

## 概述

- 职责：让 VSCode Run and Debug 面板覆盖实验系统核心操作，重点支持 Full 17 benchmark 一键运行。
- 前置依赖：模块 2
- 预计步骤数：7

## Step 1：重写 `.vscode/launch.json` 的公共规则

- 操作：修改 `.vscode/launch.json`。
- 所有 Python 调试配置必须满足：
  - `"type": "debugpy"`
  - `"request": "launch"`
  - `"cwd": "${workspaceFolder}"`
  - `"console": "integratedTerminal"`
  - `"justMyCode": false`
  - `"env"` 包含：
    ```json
    {
      "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    }
    ```
- 所有实验编排入口的 `"program"` 必须是：
  ```json
  "${workspaceFolder}/scripts/experiment_manager.py"
  ```
- 验证：
  - `python -m json.tool .vscode/launch.json` 通过
  - `pytest tests/test_vscode_config.py -q`

## Step 2：保留并修复 Quick Start/Resume 入口

- 操作：在 `.vscode/launch.json` 中配置：
  ```json
  {
    "name": "🧪 Experiment Quick Start/Resume",
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
      "quick"
    ]
  }
  ```
- 删除或替换旧名称：
  - `🧪 Experiment Start/Resume (Quick)`
- 验证：
  - VSCode 入口名称中能看到 `🧪 Experiment Quick Start/Resume`
  - `pytest tests/test_vscode_config.py::test_quick_entry_uses_experiment_manager_start -q`

## Step 3：新增 Quick Fresh Clean Run 入口

- 操作：在 `.vscode/launch.json` 中新增：
  ```json
  {
    "name": "🧪 Experiment Quick Fresh Clean Run",
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
      "quick",
      "--fresh"
    ]
  }
  ```
- 用途：当 `vscode_quick` 的历史状态污染导致 Quick 入口异常时，点击该入口重建 Quick 实验。
- 验证：
  - `pytest tests/test_vscode_config.py -q`
  - 手动点击该入口后，应重新生成 `experiments/vscode_quick/run.json` 和 `experiments/vscode_quick/state.json`

## Step 4：新增 Full 17 Start/Resume 入口

- 操作：在 `.vscode/launch.json` 中新增：
  ```json
  {
    "name": "🏁 Experiment Full 17 Start/Resume",
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
      "full17"
    ]
  }
  ```
- 该入口是本需求的最低验收核心：一键启动全部 17 算法对比实验。
- 验证：
  - `pytest tests/test_vscode_config.py::test_full_17_entry_contains_all_algorithms -q`
  - 点击后 `experiments/paper2_full_17_vscode/run.json` 中 `algorithms` 数量为 17

## Step 5：保留常规管理入口

- 操作：在 `.vscode/launch.json` 中确保存在以下入口：

### Resume Full 17

```json
{
  "name": "▶️ Experiment Full 17 Resume",
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
    "resume",
    "--run-id",
    "paper2_full_17_vscode"
  ]
}
```

### Stop Full 17

```json
{
  "name": "⏹ Experiment Full 17 Stop",
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
    "stop",
    "--run-id",
    "paper2_full_17_vscode"
  ]
}
```

### Status Full 17

```json
{
  "name": "📊 Experiment Full 17 Status",
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
    "status",
    "--run-id",
    "paper2_full_17_vscode"
  ]
}
```

### List All Experiments

```json
{
  "name": "📋 Experiment List All",
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
    "list"
  ]
}
```

### Export Full 17 Results

```json
{
  "name": "📦 Experiment Full 17 Export Results",
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
    "export",
    "--run-id",
    "paper2_full_17_vscode",
    "--output",
    "results/benchmark_paper2_full_17_vscode.json"
  ]
}
```

- 验证：
  - `python -m json.tool .vscode/launch.json`
  - `pytest tests/test_vscode_config.py -q`

## Step 6：新增 Reset 与 Rebuild Index 入口

- 操作：在 `.vscode/launch.json` 中新增：

### Reset GRPO for Quick

```json
{
  "name": "♻️ Experiment Quick Reset GRPO",
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
    "reset",
    "--run-id",
    "vscode_quick",
    "--algorithm",
    "GRPO"
  ]
}
```

### Reset current failed algorithm for Full 17

由于 VSCode `launch.json` 不能在调试配置中动态读取当前 failed algorithm，先提供 17 个固定入口，每个入口的 `--algorithm` 分别为：

- `GRPO`
- `PPO`
- `SAC`
- `DDQN`
- `DDPG`
- `TD3`
- `A3C`
- `TRPO`
- `SimPO`
- `MAPPO`
- `QMIX`
- `COMA`
- `IPPO`
- `VDN`
- `MADDPG`
- `IQL`
- `MATD3`

每个入口命名格式：

```text
♻️ Experiment Full 17 Reset <ALGORITHM>
```

对应 args：

```json
[
  "reset",
  "--run-id",
  "paper2_full_17_vscode",
  "--algorithm",
  "<ALGORITHM>"
]
```

### Rebuild Index

```json
{
  "name": "🧱 Experiment Rebuild Index",
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
    "rebuild-index"
  ]
}
```

- 验证：
  - `pytest tests/test_vscode_config.py -q`
  - `python scripts/experiment_manager.py rebuild-index` 输出 `{"status": "ok"}`

## Step 7：新增 Direct Benchmark All 入口

- 操作：在 `.vscode/launch.json` 中新增直接调用 `scripts/benchmark.py` 的入口：
  ```json
  {
    "name": "🏁 Benchmark Direct All 17",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/scripts/benchmark.py",
    "cwd": "${workspaceFolder}",
    "console": "integratedTerminal",
    "justMyCode": false,
    "env": {
      "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    },
    "args": [
      "--all",
      "--episodes",
      "10",
      "--timesteps",
      "100000",
      "--device",
      "auto",
      "--output",
      "results/benchmark_direct_all_17_vscode.json"
    ]
  }
  ```
- 如果 `scripts/benchmark.py` 当前参数名不是 `--output` 而是其他结果路径参数：
  - Codex 必须先运行 `python scripts/benchmark.py --help`
  - 以 `--help` 中实际存在的结果输出参数为准
  - 更新测试使其断言实际参数
  - 不得保留无效参数
- 验证：
  - `python scripts/benchmark.py --help` 成功
  - VSCode 入口点击后能够进入 benchmark 主流程
  - 直接 benchmark 与 experiment manager benchmark 分别写入不同结果文件，避免覆盖

---

# 模块 4：VSCode `tasks.json` 任务入口补齐

## 概述

- 职责：为“命令型”操作提供 VSCode Tasks 面板入口。`launch.json` 保持 Debug 面板体验，`tasks.json` 提供更轻量的一键任务。
- 前置依赖：模块 3
- 预计步骤数：5

## Step 1：创建 `.vscode/tasks.json`

- 操作：创建文件：
  - `.vscode/tasks.json`
- 基础结构：
  ```json
  {
    "version": "2.0.0",
    "tasks": []
  }
  ```
- 所有任务必须满足：
  - `"type": "shell"`
  - `"options.cwd": "${workspaceFolder}"`
  - `"options.env.PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"`
  - `"problemMatcher": []`
- 验证：
  - `python -m json.tool .vscode/tasks.json`

## Step 2：添加 Full 17 核心任务

- 操作：在 `.vscode/tasks.json` 中添加任务：

### Full 17 Start/Resume

```json
{
  "label": "experiment: full17 start-resume",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/experiment_manager.py",
    "start",
    "--preset",
    "full17"
  ],
  "options": {
    "cwd": "${workspaceFolder}",
    "env": {
      "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    }
  },
  "problemMatcher": []
}
```

### Full 17 Status

```json
{
  "label": "experiment: full17 status",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/experiment_manager.py",
    "status",
    "--run-id",
    "paper2_full_17_vscode"
  ],
  "options": {
    "cwd": "${workspaceFolder}",
    "env": {
      "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    }
  },
  "problemMatcher": []
}
```

### Full 17 Stop

```json
{
  "label": "experiment: full17 stop",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/experiment_manager.py",
    "stop",
    "--run-id",
    "paper2_full_17_vscode"
  ],
  "options": {
    "cwd": "${workspaceFolder}",
    "env": {
      "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    }
  },
  "problemMatcher": []
}
```

### Full 17 Export

```json
{
  "label": "experiment: full17 export",
  "type": "shell",
  "command": "python",
  "args": [
    "scripts/experiment_manager.py",
    "export",
    "--run-id",
    "paper2_full_17_vscode",
    "--output",
    "results/benchmark_paper2_full_17_vscode.json"
  ],
  "options": {
    "cwd": "${workspaceFolder}",
    "env": {
      "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
    }
  },
  "problemMatcher": []
}
```

- 验证：
  - `pytest tests/test_vscode_config.py::test_tasks_json_contains_required_tasks_if_present -q`

## Step 3：添加 Quick 任务

- 操作：添加任务：
  - `experiment: quick start-resume`
  - `experiment: quick fresh`
  - `experiment: quick status`
  - `experiment: quick stop`
  - `experiment: quick export`
- 参数分别对应：
  - `start --preset quick`
  - `start --preset quick --fresh`
  - `status --run-id vscode_quick`
  - `stop --run-id vscode_quick`
  - `export --run-id vscode_quick --output results/benchmark_vscode_quick.json`
- 验证：
  - `python -m json.tool .vscode/tasks.json`
  - `pytest tests/test_vscode_config.py -q`

## Step 4：添加索引与列表任务

- 操作：添加任务：
  - `experiment: list`
  - `experiment: rebuild-index`
- 参数：
  - `scripts/experiment_manager.py list`
  - `scripts/experiment_manager.py rebuild-index`
- 验证：
  - VSCode `Terminal > Run Task...` 中能看到以上任务
  - `pytest tests/test_vscode_config.py -q`

## Step 5：添加 Direct Benchmark All 任务

- 操作：添加任务：
  - `benchmark: direct all17`
- 参数必须与 `launch.json` 的 `🏁 Benchmark Direct All 17` 保持一致。
- 验证：
  - `pytest tests/test_vscode_config.py -q`

---

# 模块 5：使用文档更新

## 概述

- 职责：把新的 VSCode 使用方式写成用户可直接照做的文档。
- 前置依赖：模块 3、模块 4
- 预计步骤数：4

## Step 1：更新 `docs/vscode_experiment_usage.md` 的功能目标

- 操作：更新文档开头功能目标。
- 必须写明：
  - 用户无需输入命令行
  - VSCode Run and Debug 覆盖核心入口
  - VSCode Tasks 覆盖轻量命令入口
  - Full 17 benchmark 的默认配置
- 验证：
  - 文档包含 `paper2_full_17_vscode`
  - 文档包含 `17`

## Step 2：新增“推荐入口优先级”

- 操作：在文档中新增章节：
  - `推荐入口优先级`
- 内容：
  1. 正式全量对比实验：`🏁 Experiment Full 17 Start/Resume`
  2. 查看进度：`📊 Experiment Full 17 Status`
  3. 中断：`⏹ Experiment Full 17 Stop`
  4. 导出结果：`📦 Experiment Full 17 Export Results`
  5. 快速排错：`🧪 Experiment Quick Fresh Clean Run`
  6. 直接 benchmark：`🏁 Benchmark Direct All 17`
- 验证：
  - 文档包含上述 6 个入口名称。

## Step 3：新增“Full 17 一键运行说明”

- 操作：新增章节：
  - `Full 17 algorithms benchmark 一键运行`
- 内容必须包含：
  - 入口名称：`🏁 Experiment Full 17 Start/Resume`
  - 默认 run id：`paper2_full_17_vscode`
  - 17 算法列表
  - 结果导出路径：`results/benchmark_paper2_full_17_vscode.json`
  - 恢复规则：从第一个未完成算法继续
- 验证：
  - 文档包含所有 17 个算法名。

## Step 4：新增“Quick 入口报错处理”

- 操作：新增章节：
  - `Quick 入口报错处理`
- 内容：
  - 优先点击 `🧪 Experiment Quick Fresh Clean Run`
  - 若只想重跑某个失败算法，点击 `♻️ Experiment Quick Reset GRPO` 后再点击 `🧪 Experiment Quick Start/Resume`
  - 若需要定位日志，查看：
    - `experiments/vscode_quick/state.json`
    - `experiments/vscode_quick/artifacts/<ALGORITHM>/stdout.log`
    - `experiments/vscode_quick/artifacts/<ALGORITHM>/stderr.log`
- 验证：
  - 文档包含 `Quick Fresh Clean Run`
  - 文档包含 `stdout.log`
  - 文档包含 `stderr.log`

---

# 模块 6：自动化测试与最终验收

## 概述

- 职责：确保改造不会破坏现有 CLI，且 VSCode 入口实际覆盖用户目标。
- 前置依赖：模块 1 至模块 5
- 预计步骤数：4

## Step 1：运行单元测试

- 操作：运行：
  ```bash
  pytest tests/test_experiment_manager_cli.py tests/test_experiment_presets.py tests/test_experiment_state_store.py tests/test_vscode_config.py -q
  ```
- 验证：
  - 全部测试通过
  - 无 JSON 解析错误
  - 无 CLI 参数解析错误

## Step 2：验证帮助命令无训练依赖副作用

- 操作：运行：
  ```bash
  python scripts/train.py --help
  python scripts/benchmark.py --help
  python scripts/experiment_manager.py --help
  ```
- 验证：
  - 三个命令都能正常输出帮助信息
  - `train.py --help` 不因 `torch`、trainer、环境模块缺失而失败
  - `experiment_manager.py --help` 展示 `--preset` 和 `--fresh`

## Step 3：验证 Quick smoke test 入口

- 操作：
  1. 通过 VSCode 点击 `🧪 Experiment Quick Fresh Clean Run`
  2. 等待命令启动并进入训练
  3. 如不想跑完，点击 `⏹ Experiment Quick Stop`
  4. 点击 `📊 Experiment Quick Status`
- 验证：
  - `experiments/vscode_quick/run.json` 存在
  - `experiments/vscode_quick/state.json` 存在
  - `state.json` 中 algorithms 数量为 3
  - 状态为 `running`、`stopped`、`completed` 或 `failed` 中的一个合法值
  - 如果 failed，错误信息指向具体 stdout/stderr 路径

## Step 4：验证 Full 17 benchmark 入口

- 操作：
  1. 通过 VSCode 点击 `🏁 Experiment Full 17 Start/Resume`
  2. 等待第一个算法进入训练后，可点击 `📊 Experiment Full 17 Status`
  3. 如需中断，点击 `⏹ Experiment Full 17 Stop`
  4. 点击 `📦 Experiment Full 17 Export Results`
- 验证：
  - `experiments/paper2_full_17_vscode/run.json` 存在
  - `run.json` 中 algorithms 数量为 17
  - algorithms 顺序严格为：
    ```text
    GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3
    ```
  - `experiments/paper2_full_17_vscode/state.json` 存在
  - `results/benchmark_paper2_full_17_vscode.json` 可由 export 命令生成
  - 中断后再次点击 `🏁 Experiment Full 17 Start/Resume` 会从第一个未完成算法继续

---

# 验收标准

## 功能验收

- [ ] 用户可以在 VSCode Run and Debug 面板点击启动 Quick 实验。
- [ ] 用户可以在 VSCode Run and Debug 面板点击启动 Full 17 algorithms benchmark。
- [ ] Full 17 benchmark 的 run id 固定为 `paper2_full_17_vscode`。
- [ ] Full 17 benchmark 覆盖全部 17 个算法。
- [ ] 用户可以点击停止、恢复、查看状态、列出实验、导出结果、重置失败算法、重建索引。
- [ ] Quick 入口当前报错问题已被定位并修复；如果训练本身失败，错误信息必须指向具体日志。
- [ ] 用户可以通过 VSCode Tasks 面板执行同等轻量命令入口。
- [ ] 用户无需手输任何命令即可完成主要实验流程。

## 工程验收

- [ ] `.vscode/launch.json` 是合法 JSON。
- [ ] `.vscode/tasks.json` 是合法 JSON。
- [ ] `scripts/experiment_manager.py --help` 正常。
- [ ] `scripts/train.py --help` 正常且不强制加载训练依赖。
- [ ] `pytest tests/test_experiment_manager_cli.py tests/test_experiment_presets.py tests/test_experiment_state_store.py tests/test_vscode_config.py -q` 通过。
- [ ] 没有新增第三方依赖。
- [ ] 没有破坏旧 CLI 用法。

## 不做事项

- 不改算法训练逻辑。
- 不改奖励函数、环境定义或论文实验指标。
- 不新增 dashboard 写能力。
- 不引入数据库。
- 不做 PyTorch DDP。
- 不改变实验最小恢复单元，仍以“算法完成状态”为恢复单元。

