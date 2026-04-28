# 技术调研报告

## 1. 调研范围

### 1.1 项目与需求背景

项目：`w2030298-art/paper2`  
项目性质：已有 Python 强化学习算法对比框架，在现有 VSCode 调试训练平台上新增“可中断、可恢复、可跨设备接力执行”的算法对比实验能力。  
目标用户：在两台 Windows 设备之间进行算法对比实验；一台有 GPU，一台无 GPU；不要求多个算法并发训练。

### 1.2 已确认需求

| 维度 | 已确认结论 |
|---|---|
| VSCode 平台范围 | 以 VSCode 调试配置为核心，一键启动 |
| 停止语义 | 优雅停止：停止前保存阶段状态 |
| 恢复粒度 | benchmark 队列级恢复，恢复到上次中断的算法 |
| 分布式含义 | 不做 PyTorch DDP；不同算法阶段可在不同时间、不同设备上分布执行 |
| 设备环境 | 两台 Windows；一台有 GPU，一台无 GPU |
| 依赖策略 | 允许新增依赖，但优先保持轻量 |
| 状态保存粒度 | 最小保存单元为“算法完成状态”；正在运行的算法不强制保存 |
| 结果兼容性 | 保持现有 `results/benchmark*.json` 兼容 |
| 并行要求 | 无需并行多个算法；一次只跑一个算法 |
| 跨设备协作 | 使用 Git 同步实验状态 |

### 1.3 调研关键词

- VSCode debug launch for Python scripts
- Windows graceful stop for long-running Python process
- resumable benchmark queue
- SQLite local experiment state
- Git-friendly JSON manifest for cross-device state sync
- atomic file write on Windows
- file lock on Windows
- subprocess-based training orchestration
- RL benchmark checkpoint and final result compatibility

### 1.4 覆盖范围

本调研覆盖工程实现方案，不覆盖算法论文。原因：

1. 需求不改变 GRPO/PPO/SAC/DDQN 等算法本身。
2. 需求核心是实验编排、状态持久化、启动脚本和 VSCode 调试入口。
3. 用户明确确认“不需要论文/学术资料检索”。

---

## 2. 当前项目可利用基础

### 2.1 已有单算法训练入口

项目已有 `scripts/train.py`，用于训练单个 RL 算法，并支持如下关键参数：

- `--config`
- `--algorithm`
- `--env`
- `--timesteps`
- `--seed`
- `--device`
- `--save-dir`
- `--eval-episodes`

这说明新增实验编排层无需重写训练逻辑，可以通过 subprocess 调用现有 `scripts/train.py`，并为每个算法提供独立参数和保存目录。

### 2.2 已有 benchmark 入口

项目已有 `scripts/benchmark.py`，并且其中已经维护：

- `ALGORITHM_CLASSES`
- `ON_POLICY`
- `OFF_POLICY`
- `ALL_ALGOS`
- `ALGO_ENV_MAP`
- `SCALE_PRESETS`
- `make_env(...)`
- `create_agent(...)`
- benchmark JSON 输出逻辑

这说明新增“可恢复 benchmark 队列”应优先复用 `scripts/benchmark.py` 中的算法映射与参数解析规则，而不是另建一套算法注册表。

### 2.3 已有训练器保存能力

项目已有 `src/trainer/base_trainer.py`：

- `BaseTrainer.save(path)`
- `BaseTrainer.load(path)`
- `KeyboardInterrupt` 时保存 `ckpt_interrupt_<steps>.pt`
- 正常完成或 `_stop_training` 时保存 `final.pt`
- `train_logs.json` 输出
- `callbacks` 扩展点
- `_stop_training` 标志

这说明已有基础支持“训练器内部保存模型”。不过本需求的保存粒度是“算法完成状态”，并不要求恢复算法内部 timestep。因此新增编排层只需在每个算法完成后将该算法标记为 `completed`，并记录其结果路径即可。

### 2.4 已有 VSCode 调试文档

项目已有 `docs/vscode_debug_guide.md`，并明确 `.vscode/launch.json` 是调试启动配置，且已有 Train / Benchmark / Evaluate / Plot / Pytest 等调试配置说明。新增功能应继续沿用此结构，增加：

- 一键启动可恢复 benchmark
- 停止当前实验
- 查看实验状态
- 继续上次实验
- CPU / GPU 两套设备配置

---

## 3. 技术方案对比

| 方案 | 来源 | 核心原理 | 优势 | 劣势 | 适用场景 | 可行性评分 |
|---|---|---|---|---|---|---:|
| A. 直接改造 `scripts/benchmark.py` 内部循环 | 当前项目已有 benchmark 脚本 | 在每个算法完成后写入 progress 文件；启动时跳过已完成算法 | 改动少；最贴近现有入口 | 状态管理、停止控制、VSCode 控制命令会和 benchmark 逻辑耦合；后续维护较差 | 快速临时方案 | 3/5 |
| B. 新增 `scripts/experiment_manager.py` 编排器，内部 subprocess 调用 `scripts/train.py` | 当前项目已有单算法入口与 VSCode 调试模式 | 编排器维护 run manifest、算法状态、设备信息、停止标志；每次只启动一个算法子进程 | 与训练代码解耦；适合 VSCode 一键启动；易做 stop/status/resume；符合一次只跑一个算法 | 需要新增状态模型与 CLI；需要处理 Windows 子进程停止 | 推荐方案 | 5/5 |
| C. 引入 Ray 进行任务调度 | 分布式 Python 任务调度通用方案 | Ray head/worker 分发不同算法任务 | 真分布式、可扩展 | Windows 跨设备部署复杂；用户不需要并行；Ray 状态与 Git 接力不匹配；引入重依赖 | 多机并发训练、多 GPU 队列 | 2/5 |
| D. 使用 SQLite 作为唯一状态源并提交到 Git | SQLite 本地嵌入式数据库 | 所有 run/algorithm 状态写入 `.db` 文件 | 查询方便；事务可靠 | SQLite 是二进制文件，Git 合并冲突差；跨设备接力容易误提交锁文件或冲突 | 单机本地任务追踪 | 3/5 |
| E. Git-friendly JSON manifest 作为权威状态，SQLite 作为本地可重建索引 | 文件状态机 + 可选 SQLite 索引 | `run.json`/`state.json` 进入 Git；SQLite 只做本地索引或缓存 | 跨设备同步友好；可读可审计；冲突可手动解决；兼容 Windows | 查询能力弱于纯 SQLite；需要原子写入 | 推荐用于两台 Windows + Git 接力 | 5/5 |
| F. 训练器内部精确 checkpoint/resume | 当前 `BaseTrainer.save/load` 和 agent save/load | 保存 agent、optimizer、buffer、random state、当前 timestep，恢复正在训练的算法 | 最完整 | 用户明确不要求正在运行算法恢复；不同算法保存格式不一致，改造面大 | 未来高级需求 | 2/5 |

---

## 4. 推荐方案

### 4.1 首选方案

采用 **“轻量实验编排器 + Git-friendly 状态清单 + VSCode 调试入口”**：

1. 新增 `scripts/experiment_manager.py` 作为统一入口。
2. 每个 benchmark run 生成独立目录：

```text
runs/
└── <run_id>/
    ├── run.json
    ├── state.json
    ├── lock.json
    ├── stop.request
    ├── stdout.log
    ├── stderr.log
    ├── results/
    │   ├── benchmark_<run_id>.json
    │   └── per_algorithm/
    │       ├── GRPO.json
    │       ├── PPO.json
    │       └── ...
    └── checkpoints/
        ├── GRPO/
        ├── PPO/
        └── ...
```

3. `state.json` 作为跨设备 Git 同步的权威状态文件。
4. SQLite 可作为本地索引缓存，存放在 `.experiment/experiment_state.sqlite`，默认不作为跨设备同步源。
5. 每个算法完成后立即写入：
   - 算法状态：`completed`
   - 完成时间
   - 设备信息
   - seed
   - config path
   - save dir
   - result summary
   - compatible benchmark JSON 更新
6. 当前正在运行的算法如果被中断，不标记为 completed；下次 resume 重新运行该算法。
7. VSCode 增加一键启动配置：
   - Start/Resume benchmark run
   - Stop current run
   - Status current run
   - Start GPU run
   - Start CPU run
   - Debug single algorithm through manager

### 4.2 推荐理由

该方案最符合用户约束：

- 用户不需要并行多个算法，因此不引入 Ray。
- 用户最小保存单元是“算法”，因此不做算法内部精确 checkpoint/resume。
- 用户使用两台 Windows 设备 + Git 同步，因此 JSON manifest 比 SQLite 二进制文件更适合作为跨设备状态源。
- 项目已有 `scripts/train.py`，支持单算法训练参数，编排器可直接复用。
- 项目已有 `scripts/benchmark.py` 的算法注册表和环境映射，编排器可导入复用，避免重复维护。
- 项目已有 VSCode 调试文档，新增 launch 配置符合现有使用习惯。

### 4.3 备选方案

如后续用户要求“同一个算法训练到一半也能精确恢复”，再进入第二阶段增强：

1. 扩展 `BaseTrainer.save_checkpoint_state(...)`。
2. 为每个 agent 增加统一 `state_dict()` / `load_state_dict()` 协议。
3. 保存 optimizer、scheduler、replay buffer、rollout buffer、random state。
4. 在 `scripts/train.py` 增加 `--resume-from`。
5. 在 `experiment_manager.py` 中允许 `algorithm.status = interrupted` 后从 checkpoint 恢复。

当前不建议实现该增强，因为它明显扩大改造面，并且超出用户已确认需求。

---

## 5. 状态模型建议

### 5.1 `run.json`

`run.json` 保存不可变或低频变化的实验配置：

```json
{
  "schema_version": 1,
  "run_id": "benchmark_20260428_001",
  "created_at": "2026-04-28T00:00:00",
  "project": "paper2",
  "mode": "resumable_benchmark",
  "algorithms": ["GRPO", "PPO", "SAC"],
  "timesteps": 100000,
  "seed": 42,
  "device_policy": "auto",
  "output_compatibility": {
    "write_existing_results_json": true,
    "results_path": "results/benchmark_20260428_001.json"
  }
}
```

### 5.2 `state.json`

`state.json` 保存当前可恢复状态：

```json
{
  "schema_version": 1,
  "run_id": "benchmark_20260428_001",
  "status": "running",
  "current_algorithm": "SAC",
  "current_index": 2,
  "completed_algorithms": ["GRPO", "PPO"],
  "failed_algorithms": [],
  "algorithm_states": {
    "GRPO": {
      "status": "completed",
      "started_at": "2026-04-28T10:00:00",
      "finished_at": "2026-04-28T10:30:00",
      "device": "cuda",
      "result_path": "runs/benchmark_20260428_001/results/per_algorithm/GRPO.json",
      "checkpoint_dir": "runs/benchmark_20260428_001/checkpoints/GRPO"
    },
    "PPO": {
      "status": "completed",
      "started_at": "2026-04-28T10:31:00",
      "finished_at": "2026-04-28T11:10:00",
      "device": "cuda",
      "result_path": "runs/benchmark_20260428_001/results/per_algorithm/PPO.json",
      "checkpoint_dir": "runs/benchmark_20260428_001/checkpoints/PPO"
    },
    "SAC": {
      "status": "running",
      "started_at": "2026-04-28T11:11:00",
      "device": "cuda",
      "pid": 12345
    }
  },
  "updated_at": "2026-04-28T11:11:00"
}
```

### 5.3 中断语义

当用户执行 stop：

1. `experiment_manager.py stop <run_id>` 写入 `runs/<run_id>/stop.request`。
2. 正在运行的 manager 在算法边界或轮询点检测到 stop request。
3. 若当前算法已经自然完成，则写入 `completed`。
4. 若当前算法尚未完成，则不写入 completed；状态变为 `stopped`，下次 resume 重新执行该算法。
5. `stop.request` 被消费后删除或重命名为 `stop.consumed.json`。

### 5.4 Git 同步语义

跨设备操作规则：

1. 设备 A 运行到第 N 个算法完成后，提交：
   - `runs/<run_id>/run.json`
   - `runs/<run_id>/state.json`
   - `runs/<run_id>/results/per_algorithm/*.json`
   - `results/benchmark_<run_id>.json`
2. 设备 B `git pull` 后执行：
   - `python scripts/experiment_manager.py resume --run-id <run_id>`
3. 设备 B 读取 `state.json`，跳过已完成算法，从第一个未完成算法开始。
4. 不建议提交大体积 checkpoint 到 Git；checkpoint 可按 `.gitignore` 策略排除，除非用户确实需要跨设备复制模型文件。

---

## 6. 关键参考资料

### 6.1 项目内部参考

| 文件 | 核心贡献 | 后续设计用途 |
|---|---|---|
| `scripts/train.py` | 单算法训练入口，已有算法、环境、设备、seed、save-dir 参数 | `experiment_manager.py` 用 subprocess 调用它完成单算法训练 |
| `scripts/benchmark.py` | 维护算法注册表、环境映射、结果 JSON 输出逻辑 | 编排器复用算法列表、环境映射和结果兼容策略 |
| `src/trainer/base_trainer.py` | 已有 trainer 保存、日志、callback、KeyboardInterrupt 处理 | 后续可用 callback 或现有 save 机制保证算法完成后产物落盘 |
| `docs/vscode_debug_guide.md` | 已有 VSCode 调试入口规范 | 新增 launch 配置时沿用现有文档结构 |
| `requirements.txt` / `pyproject.toml` | 当前依赖与项目脚本定义 | 控制新增依赖，避免破坏已有安装方式 |

### 6.2 外部工程概念参考

| 主题 | 关键结论 | 本项目采用方式 |
|---|---|---|
| SQLite 本地状态 | 适合本地事务和查询，不适合 Git 合并 | 作为本地可重建索引，不作为跨设备唯一权威源 |
| JSON manifest | 可读、可 diff、适合 Git 同步 | `run.json` 和 `state.json` 作为权威状态 |
| 原子写入 | Windows 上应写临时文件后 `os.replace` | 所有状态文件更新必须使用 atomic write |
| 文件锁 | Windows 需要跨平台库或简化单进程约束 | 当前一次只跑一个算法；用 `lock.json` + PID + hostname 做软锁 |
| subprocess 编排 | 子进程隔离训练，父进程负责状态 | `experiment_manager.py` 作为父进程 |
| VSCode launch | 通过 `program` / `module` / `args` 固化常用入口 | `.vscode/launch.json` 增加 start/resume/stop/status |

---

## 7. 需要蒸馏给 Codex 的技术要点

### 7.1 总体原则

Codex 实现时必须遵守：

1. 不实现 PyTorch DDP。
2. 不实现算法内部精确断点恢复。
3. 不并行启动多个算法。
4. 保持现有 `scripts/train.py`、`scripts/benchmark.py` 的原有用法兼容。
5. 新功能通过新增文件和少量非破坏性扩展实现。
6. 权威跨设备状态使用 JSON 文件，必须可提交到 Git。
7. Windows 兼容优先，不依赖 POSIX signal。
8. 每个算法完成后才标记为 completed。
9. 正在运行算法被中断时，下次 resume 从该算法重新开始。
10. 保留 `results/benchmark*.json` 兼容输出。

### 7.2 推荐新增文件

```text
scripts/
├── experiment_manager.py

src/
└── experiment/
    ├── __init__.py
    ├── models.py
    ├── state_store.py
    ├── runner.py
    ├── git_sync.py
    └── vscode.py

tests/
└── experiment/
    ├── test_state_store.py
    ├── test_runner_resume.py
    ├── test_stop_request.py
    └── test_compat_results.py

.vscode/
└── launch.json

docs/
└── resumable_experiments.md
```

### 7.3 数据模型

在 `src/experiment/models.py` 中定义：

```python
from dataclasses import dataclass, field
from typing import Literal

RunStatus = Literal["created", "running", "stopped", "completed", "failed"]
AlgorithmStatus = Literal["pending", "running", "completed", "failed"]

@dataclass
class AlgorithmState:
    name: str
    status: AlgorithmStatus = "pending"
    index: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    device: str | None = None
    pid: int | None = None
    hostname: str | None = None
    config_path: str | None = None
    checkpoint_dir: str | None = None
    result_path: str | None = None
    error: str | None = None

@dataclass
class RunState:
    schema_version: int
    run_id: str
    status: RunStatus
    algorithms: list[str]
    current_index: int
    current_algorithm: str | None
    completed_algorithms: list[str] = field(default_factory=list)
    failed_algorithms: list[str] = field(default_factory=list)
    algorithm_states: dict[str, AlgorithmState] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
```

### 7.4 状态存储

在 `src/experiment/state_store.py` 中实现：

- `class ExperimentStateStore`
  - `__init__(self, runs_dir: Path)`
  - `create_run(...) -> RunState`
  - `load(run_id: str) -> RunState`
  - `save(state: RunState) -> None`
  - `mark_algorithm_running(run_id, algorithm, pid, device, hostname) -> RunState`
  - `mark_algorithm_completed(run_id, algorithm, result_path, checkpoint_dir) -> RunState`
  - `mark_algorithm_failed(run_id, algorithm, error) -> RunState`
  - `request_stop(run_id: str) -> None`
  - `is_stop_requested(run_id: str) -> bool`
  - `clear_stop_request(run_id: str) -> None`
  - `next_pending_algorithm(state: RunState) -> str | None`

实现要求：

1. `save(...)` 必须先写入 `state.json.tmp`，再用 `os.replace(tmp, state_path)`。
2. `run.json` 保存创建时配置；`state.json` 保存动态进度。
3. `state.json` 中只把算法状态标记为 completed，不保存未完成算法的半成品状态。
4. `completed_algorithms` 必须从 `algorithm_states` 中同步生成，避免两处不一致。
5. 写入 JSON 时使用 `ensure_ascii=False` 和 `indent=2`。
6. Windows 路径统一转为 POSIX 风格相对路径，便于 Git diff。

### 7.5 编排器 Runner

在 `src/experiment/runner.py` 中实现：

- `class ExperimentRunner`
  - `__init__(self, state_store, project_root: Path)`
  - `start_new_run(args) -> str`
  - `resume_run(run_id: str) -> int`
  - `run_next_algorithm(run_id: str) -> int`
  - `_build_train_command(...) -> list[str]`
  - `_run_subprocess(command: list[str], stdout_path: Path, stderr_path: Path) -> int`
  - `_write_compat_benchmark_json(run_id: str) -> None`

关键行为：

1. `start_new_run` 创建 run 后立即进入 `resume_run`。
2. `resume_run` 循环查找第一个 pending 或 failed-but-retry-allowed 算法。
3. 每次只运行一个算法。
4. 算法开始前写入 `running`。
5. 子进程返回码为 0 才写入 `completed`。
6. 返回码非 0 时写入 `failed` 并停止整个 run。
7. 若检测到 `stop.request`，当前算法完成后停止；如果未来需要强制停止，则不得标记当前算法 completed。
8. 兼容结果 JSON 每完成一个算法更新一次，路径为：
   - `runs/<run_id>/results/benchmark_<run_id>.json`
   - 同步写入或复制到 `results/benchmark_<run_id>.json`
9. 不覆盖旧的 `results/benchmark.json`，除非现有项目已有该行为且用户明确接受；当前建议新增 timestamp/run_id 文件。

### 7.6 CLI

在 `scripts/experiment_manager.py` 中实现 argparse 子命令：

```bash
python scripts/experiment_manager.py start --run-id demo --algorithms GRPO PPO SAC --timesteps 100000 --seed 42 --device auto
python scripts/experiment_manager.py resume --run-id demo
python scripts/experiment_manager.py stop --run-id demo
python scripts/experiment_manager.py status --run-id demo
python scripts/experiment_manager.py list
```

命令语义：

- `start`：创建新 run；如果同名 run 已存在则报错。
- `resume`：读取已有 run，从第一个未完成算法开始。
- `stop`：写入 stop request 文件，不直接 kill 训练进程。
- `status`：打印当前 run 状态、已完成算法、当前算法、失败算法。
- `list`：列出 `runs/` 下所有 run。

### 7.7 VSCode 配置

`.vscode/launch.json` 增加配置：

1. `🟣 Experiment: Start/Resume Default Benchmark`
2. `🟣 Experiment: Resume by Run ID`
3. `🟣 Experiment: Stop Current Run`
4. `🟣 Experiment: Status Current Run`
5. `🟣 Experiment: Start GPU Benchmark`
6. `🟣 Experiment: Start CPU Benchmark`

实现要求：

- 使用 `"type": "debugpy"`。
- 使用 `"program": "${workspaceFolder}/scripts/experiment_manager.py"`。
- 使用 `"console": "integratedTerminal"`。
- 使用 `"cwd": "${workspaceFolder}"`。
- 设置 `PYTHONPATH` 包含：
  - `${workspaceFolder}`
  - `${workspaceFolder}/src`
  - `${workspaceFolder}/scripts`
  - `${workspaceFolder}/rl_algorithms`

### 7.8 Git 同步约束

新增或调整 `.gitignore`：

建议纳入 Git：

```text
!runs/
!runs/**/run.json
!runs/**/state.json
!runs/**/results/**/*.json
```

建议排除 Git：

```text
runs/**/checkpoints/
runs/**/stdout.log
runs/**/stderr.log
runs/**/lock.json
runs/**/stop.request
.experiment/
*.sqlite
*.sqlite-shm
*.sqlite-wal
```

说明：

1. 状态和结果进入 Git，便于设备 A/B 接力。
2. checkpoint 默认不进 Git，避免大文件。
3. SQLite 默认不进 Git，避免二进制冲突。
4. 如果用户之后需要跨设备复用训练好的模型，再单独设计 artifact 同步方式。

### 7.9 测试策略

必须新增 pytest 测试：

1. `tests/experiment/test_state_store.py`
   - 创建 run 后生成 `run.json` 和 `state.json`
   - `mark_algorithm_completed` 后 `completed_algorithms` 更新
   - 原子写入不会留下损坏 JSON

2. `tests/experiment/test_runner_resume.py`
   - 已完成前两个算法时，resume 从第三个算法开始
   - 当前 running 但未 completed 的算法，下次 resume 重新运行该算法
   - 所有算法 completed 后 run 状态为 completed

3. `tests/experiment/test_stop_request.py`
   - `stop` 创建 `stop.request`
   - runner 检测 stop 后在算法边界停止
   - stop 后 run 状态为 stopped

4. `tests/experiment/test_compat_results.py`
   - 每个 completed 算法被写入兼容 benchmark JSON
   - 不破坏现有 `results/benchmark*.json` 格式
   - JSON 可被现有 plot/report 脚本读取

### 7.10 验证命令

Codex 实装后必须运行：

```bash
python -m pytest tests/experiment -v
python scripts/experiment_manager.py start --run-id smoke_test --algorithms GRPO PPO --timesteps 10 --seed 42 --device cpu
python scripts/experiment_manager.py status --run-id smoke_test
python scripts/experiment_manager.py resume --run-id smoke_test
```

如果 Windows + GPU 设备可用，再运行：

```bash
python scripts/experiment_manager.py start --run-id gpu_smoke --algorithms GRPO --timesteps 10 --seed 42 --device cuda
```

---

## 8. 风险与约束

| 风险 | 影响 | 应对 |
|---|---|---|
| VSCode 的红色停止按钮可能直接终止进程 | 无法保证优雅保存当前算法状态 | 提供 `Experiment: Stop Current Run` 配置，使用 stop request；文档中提示优雅停止应使用该入口 |
| Git 同步状态时两台设备同时运行同一 run | state.json 冲突或重复训练 | 明确不支持同一 run 并发运行；用 `lock.json` + hostname/pid 做提示 |
| SQLite 文件提交到 Git 产生冲突 | 跨设备恢复失败 | SQLite 只作为本地索引，默认 gitignore |
| 当前算法中途终止后丢失该算法训练进度 | 该算法需重跑 | 符合用户已确认的最小保存单元：算法级 |
| checkpoint 文件过大 | Git 仓库膨胀 | 默认不提交 checkpoint，只提交状态与结果 |
| 不同设备 CPU/GPU 结果不完全一致 | 可复现实验存在差异 | `state.json` 记录 hostname/device/seed；结果按算法和设备留痕 |
| 现有 benchmark 输出格式被破坏 | plot/report 脚本不可用 | 新增兼容 JSON 写入测试，避免改坏现有结果结构 |

---

## 9. 阶段 1 结论

推荐进入架构设计阶段，并采用以下明确技术路线：

1. 新增独立实验编排模块 `src/experiment/`。
2. 新增 CLI 入口 `scripts/experiment_manager.py`。
3. 使用 `run.json` + `state.json` 作为跨设备 Git 同步的权威状态。
4. 使用 SQLite 作为可选本地索引，不作为 Git 同步源。
5. 每个算法完成后才写入 completed 状态。
6. 中断后从第一个未 completed 算法重新开始。
7. 不实现算法内部精确 checkpoint/resume。
8. 不引入 Ray，不实现 DDP，不并行训练多个算法。
9. 保留现有 `scripts/train.py` 与 `scripts/benchmark.py` 的原有用法。
10. 在 `.vscode/launch.json` 中新增 start/resume/stop/status 一键调试入口。
