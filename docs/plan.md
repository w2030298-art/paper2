# 开发计划

## 元信息

- 项目：paper2 / grpo-mec
- 仓库：`w2030298-art/paper2`
- 技术栈：Python 3.10+、PyTorch、Gymnasium、PyYAML、TensorBoard、pytest、VSCode debugpy
- 需求目标：为 VSCode 测试训练平台增加“算法对比实验可启动、可停止、跨设备接力恢复”的轻量实验编排能力，并优化 VSCode 启动配置。
- 复杂度路线：复杂路线，已完成阶段 1 技术调研与阶段 2 架构设计。
- 总模块数：9
- 预计步骤总数：49
- 建议开发顺序：模块 1 → 模块 2 → 模块 3 → 模块 4 → 模块 5 → 模块 6 → 模块 7 → 模块 8 → 模块 9

## 全局技术决策

1. 不实现 PyTorch DDP、不实现 Ray、不实现多算法并行。
2. 分布式语义限定为：算法对比实验的不同算法阶段可以在不同时间、不同设备上接力运行。
3. 权威状态使用 Git-friendly JSON 文件：
   - `experiments/<run_id>/run.json`
   - `experiments/<run_id>/state.json`
4. 本地 SQLite 只作为可重建索引缓存：
   - `experiments/.index.sqlite3`
   - 该文件不进入 Git。
5. 最小保存单元为“算法”：
   - 某算法完整训练并导出结果后，状态标记为 `completed`。
   - 正在运行的算法被中断时，不标记完成；下次 resume 从该算法重新开始。
6. 保持现有 `results/benchmark*.json` 兼容：
   - 新增导出逻辑只追加生成新的 benchmark JSON，不破坏旧文件。
   - `results/benchmark.json` 继续作为最新结果别名。
7. VSCode 一键入口为主：
   - `Start/Resume Experiment`
   - `Stop Experiment`
   - `Status Experiment`
   - `List Experiments`
   - `Export Experiment Results`

## 模块 1：实验编排包与数据模型

### 概述

- 职责：建立 `src/experiment/` 包，定义实验、算法任务、运行状态、结果记录等数据结构。
- 前置依赖：无
- 预计步骤数：5

### Step 1：创建实验编排包文件结构

- 操作：创建以下文件。
  - `src/experiment/__init__.py` — 导出核心类与版本号。
  - `src/experiment/models.py` — 数据模型与枚举。
  - `src/experiment/errors.py` — 自定义异常。
  - `tests/test_experiment_models.py` — 数据模型测试。
- `src/experiment/__init__.py` 内容要求：
  - 定义 `__all__`，至少导出：
    - `ExperimentStatus`
    - `AlgorithmStatus`
    - `AlgorithmSpec`
    - `AlgorithmRunRecord`
    - `ExperimentManifest`
    - `ExperimentState`
- 验证：
  - 运行 `python -c "from src.experiment import ExperimentManifest, ExperimentState"` 无报错。

### Step 2：实现状态枚举

- 操作：在 `src/experiment/models.py` 中定义以下枚举，全部继承 `str, Enum`。
  - `class ExperimentStatus(str, Enum)`
    - `INITIALIZED = "initialized"`
    - `RUNNING = "running"`
    - `STOP_REQUESTED = "stop_requested"`
    - `STOPPED = "stopped"`
    - `COMPLETED = "completed"`
    - `FAILED = "failed"`
  - `class AlgorithmStatus(str, Enum)`
    - `PENDING = "pending"`
    - `RUNNING = "running"`
    - `COMPLETED = "completed"`
    - `INTERRUPTED = "interrupted"`
    - `FAILED = "failed"`
    - `SKIPPED = "skipped"`
- 特别注意：
  - 不要使用 `auto()`，字符串值必须固定，保证 JSON 兼容与跨版本稳定。
- 验证：
  - 在 `tests/test_experiment_models.py` 中新增 `test_status_enum_values_are_stable`，断言每个枚举值等于上述字符串。
  - 运行 `pytest tests/test_experiment_models.py -v` 通过。

### Step 3：实现算法任务模型

- 操作：在 `src/experiment/models.py` 中定义 `@dataclass`：
  - `class AlgorithmSpec`
    - 字段：
      - `name: str`
      - `config_path: str`
      - `timesteps: int`
      - `seed: int`
      - `device: str = "auto"`
      - `env: str = "auto"`
      - `eval_episodes: int = 10`
      - `extra_args: list[str] = field(default_factory=list)`
    - 方法：
      - `to_dict(self) -> dict[str, Any]`
      - `@classmethod from_dict(cls, data: Mapping[str, Any]) -> "AlgorithmSpec"`
- 规范：
  - `name` 在 `from_dict` 中统一转为大写，但保留 `SimPO` 这类混合大小写算法时通过注册表再 canonicalize；本模型只做 `str.strip()`。
  - `timesteps`、`seed`、`eval_episodes` 必须转为 `int`。
  - `extra_args` 必须为 `list[str]`；如果传入 `None`，转换为空列表。
- 验证：
  - 新增 `test_algorithm_spec_roundtrip`：
    - 构造 `AlgorithmSpec(name="GRPO", config_path="configs/algorithms/grpo.yaml", timesteps=5000, seed=42)`。
    - `to_dict()` 后再 `from_dict()`。
    - 断言对象字段一致。

### Step 4：实现运行记录模型

- 操作：在 `src/experiment/models.py` 中定义 `@dataclass`：
  - `class AlgorithmRunRecord`
    - 字段：
      - `name: str`
      - `status: AlgorithmStatus = AlgorithmStatus.PENDING`
      - `started_at: str | None = None`
      - `finished_at: str | None = None`
      - `exit_code: int | None = None`
      - `attempts: int = 0`
      - `device: str = "auto"`
      - `result_path: str | None = None`
      - `checkpoint_dir: str | None = None`
      - `stdout_log: str | None = None`
      - `stderr_log: str | None = None`
      - `error: str | None = None`
    - 方法：
      - `to_dict(self) -> dict[str, Any]`
      - `@classmethod from_dict(cls, data: Mapping[str, Any]) -> "AlgorithmRunRecord"`
      - `mark_running(self, *, started_at: str, device: str) -> None`
      - `mark_completed(self, *, finished_at: str, exit_code: int, result_path: str, checkpoint_dir: str) -> None`
      - `mark_interrupted(self, *, finished_at: str, exit_code: int | None, error: str | None = None) -> None`
      - `mark_failed(self, *, finished_at: str, exit_code: int | None, error: str) -> None`
- 规范：
  - `mark_running` 必须 `attempts += 1`。
  - `mark_completed` 必须设置 `status = AlgorithmStatus.COMPLETED` 且清空 `error`。
  - `mark_interrupted` 不得清空 `result_path`，但不得将算法加入 completed 列表。
- 验证：
  - 新增 `test_algorithm_run_record_transitions`。
  - 断言 `mark_running` 后 `attempts == 1` 且状态为 `running`。
  - 断言 `mark_completed` 后状态为 `completed`。

### Step 5：实现实验清单与状态模型

- 操作：在 `src/experiment/models.py` 中定义两个 `@dataclass`。
  - `class ExperimentManifest`
    - 字段：
      - `schema_version: int`
      - `run_id: str`
      - `name: str`
      - `created_at: str`
      - `updated_at: str`
      - `algorithms: list[AlgorithmSpec]`
      - `project_root: str = "."`
      - `output_dir: str = "results"`
      - `experiment_dir: str = ""`
      - `metadata: dict[str, Any] = field(default_factory=dict)`
    - 方法：
      - `to_dict(self) -> dict[str, Any]`
      - `@classmethod from_dict(cls, data: Mapping[str, Any]) -> "ExperimentManifest"`
  - `class ExperimentState`
    - 字段：
      - `schema_version: int`
      - `run_id: str`
      - `status: ExperimentStatus`
      - `current_index: int`
      - `records: list[AlgorithmRunRecord]`
      - `completed_algorithms: list[str] = field(default_factory=list)`
      - `stop_requested: bool = False`
      - `last_error: str | None = None`
      - `updated_at: str = ""`
    - 方法：
      - `to_dict(self) -> dict[str, Any]`
      - `@classmethod from_dict(cls, data: Mapping[str, Any]) -> "ExperimentState"`
      - `next_pending_index(self) -> int | None`
      - `get_record(self, algorithm_name: str) -> AlgorithmRunRecord`
      - `request_stop(self, updated_at: str) -> None`
      - `clear_stop(self, updated_at: str) -> None`
- 规范：
  - `next_pending_index()` 必须返回第一个 `status != COMPLETED` 的记录索引。
  - 当所有记录均 `COMPLETED` 时，返回 `None`。
  - `completed_algorithms` 顺序必须与 manifest.algorithms 顺序一致。
- 验证：
  - 新增 `test_experiment_state_next_pending_index`。
  - 构造三个记录，前两个 completed，第三个 pending，断言返回 `2`。
  - 三个都 completed 后断言返回 `None`。

## 模块 2：JSON 状态存储、文件锁与本地索引

### 概述

- 职责：实现跨设备 Git 同步友好的权威 JSON 状态读写，并提供本地 SQLite 索引用于快速 list/status。
- 前置依赖：模块 1
- 预计步骤数：7

### Step 1：创建状态存储文件结构

- 操作：创建以下文件。
  - `src/experiment/time_utils.py`
  - `src/experiment/state_store.py`
  - `src/experiment/local_index.py`
  - `tests/test_experiment_state_store.py`
  - `tests/test_experiment_local_index.py`
- 验证：
  - 运行 `python -c "from src.experiment.state_store import JsonStateStore"` 无报错。

### Step 2：实现时间工具

- 操作：在 `src/experiment/time_utils.py` 中实现：
  - `def utc_now_iso() -> str`
    - 返回 UTC ISO 字符串。
    - 格式示例：`2026-04-28T12:34:56Z`。
  - `def safe_timestamp() -> str`
    - 返回适合文件名的时间戳。
    - 格式：`YYYYMMDD_HHMMSS`。
- 验证：
  - 在 `tests/test_experiment_state_store.py` 中新增 `test_utc_now_iso_suffix`。
  - 断言 `utc_now_iso().endswith("Z")`。

### Step 3：实现原子 JSON 写入

- 操作：在 `src/experiment/state_store.py` 中实现：
  - `def read_json(path: Path) -> dict[str, Any]`
  - `def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None`
- 规范：
  - `write_json_atomic` 必须：
    - 先创建父目录。
    - 写入临时文件：`<name>.tmp`。
    - 使用 `json.dump(..., indent=2, ensure_ascii=False)`。
    - 使用 `os.replace(tmp_path, path)` 原子替换。
  - `read_json` 必须使用 `encoding="utf-8"`。
- 验证：
  - 新增 `test_write_json_atomic_roundtrip`。
  - 写入中文字段，读取后内容一致。

### Step 4：实现跨平台简单锁

- 操作：在 `src/experiment/state_store.py` 中实现：
  - `class FileLock`
    - `__init__(self, lock_path: Path, stale_seconds: int = 86400)`
    - `acquire(self) -> None`
    - `release(self) -> None`
    - `__enter__(self) -> "FileLock"`
    - `__exit__(self, exc_type, exc, tb) -> None`
- 锁策略：
  - 使用 `os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` 创建锁文件。
  - 锁文件内容写入 JSON：
    - `pid`
    - `created_at`
  - 如果锁文件已存在且修改时间超过 `stale_seconds`，删除旧锁后重试。
  - 默认最多重试 50 次，每次 sleep 0.1 秒。
  - 失败后抛出 `ExperimentLockError`。
- 需要在 `src/experiment/errors.py` 中新增：
  - `class ExperimentError(Exception)`
  - `class ExperimentLockError(ExperimentError)`
  - `class ExperimentNotFoundError(ExperimentError)`
  - `class ExperimentStateError(ExperimentError)`
- 验证：
  - 新增 `test_file_lock_exclusive`。
  - 同一个路径连续 acquire 两次，第二次应抛出 `ExperimentLockError`。
  - 第一次 release 后第二次 acquire 成功。

### Step 5：实现 JsonStateStore

- 操作：在 `src/experiment/state_store.py` 中实现：
  - `class JsonStateStore`
    - `__init__(self, root_dir: Path | str = "experiments")`
    - `experiment_dir(self, run_id: str) -> Path`
    - `manifest_path(self, run_id: str) -> Path`
    - `state_path(self, run_id: str) -> Path`
    - `control_dir(self, run_id: str) -> Path`
    - `stop_request_path(self, run_id: str) -> Path`
    - `create(self, manifest: ExperimentManifest, state: ExperimentState) -> None`
    - `load_manifest(self, run_id: str) -> ExperimentManifest`
    - `load_state(self, run_id: str) -> ExperimentState`
    - `save_manifest(self, manifest: ExperimentManifest) -> None`
    - `save_state(self, state: ExperimentState) -> None`
    - `with_lock(self, run_id: str) -> ContextManager[FileLock]`
    - `exists(self, run_id: str) -> bool`
    - `list_run_ids(self) -> list[str]`
    - `write_stop_request(self, run_id: str) -> Path`
    - `clear_stop_request(self, run_id: str) -> None`
    - `has_stop_request(self, run_id: str) -> bool`
- 规范：
  - `create` 若 run 已存在，抛出 `ExperimentStateError`。
  - `load_manifest` 或 `load_state` 找不到文件时抛出 `ExperimentNotFoundError`。
  - `write_stop_request` 写入 `experiments/<run_id>/control/stop.request`，内容为 JSON。
  - `clear_stop_request` 删除该文件；文件不存在时不报错。
- 验证：
  - 新增 `test_json_state_store_create_load_update`。
  - 创建 manifest/state，保存后重新加载，字段一致。
  - 修改 state.status 后保存，再加载确认状态更新。

### Step 6：实现本地 SQLite 索引

- 操作：在 `src/experiment/local_index.py` 中实现：
  - `class LocalExperimentIndex`
    - `__init__(self, db_path: Path | str = "experiments/.index.sqlite3")`
    - `initialize(self) -> None`
    - `upsert(self, manifest: ExperimentManifest, state: ExperimentState) -> None`
    - `rebuild_from_store(self, store: JsonStateStore) -> None`
    - `list_runs(self) -> list[dict[str, Any]]`
    - `get_run(self, run_id: str) -> dict[str, Any] | None`
- SQLite 表结构：
  - 表名：`experiment_runs`
  - 字段：
    - `run_id TEXT PRIMARY KEY`
    - `name TEXT NOT NULL`
    - `status TEXT NOT NULL`
    - `current_index INTEGER NOT NULL`
    - `total_algorithms INTEGER NOT NULL`
    - `completed_count INTEGER NOT NULL`
    - `updated_at TEXT NOT NULL`
    - `experiment_dir TEXT NOT NULL`
- 规范：
  - SQLite 只作为缓存；任何异常不得破坏 `run.json` / `state.json`。
  - `rebuild_from_store` 先 `DELETE FROM experiment_runs`，再从 JSON 逐个 upsert。
- 验证：
  - 新增 `test_local_index_rebuild_from_store`。
  - 创建两个实验，rebuild 后 `list_runs()` 返回两个 run。

### Step 7：更新 `.gitignore` 以支持 Git 同步

- 操作：修改 `.gitignore`，追加以下内容到文件末尾：

```gitignore
# Experiment orchestration
experiments/.index.sqlite3
experiments/**/*.lock
experiments/**/control/
experiments/**/process.json
experiments/**/logs/
experiments/**/artifacts/**/*.pt
experiments/**/artifacts/**/*.pth
experiments/**/artifacts/**/*.ckpt
```

- 规范：
  - 不得忽略 `experiments/**/run.json`。
  - 不得忽略 `experiments/**/state.json`。
  - 不得忽略 `experiments/**/artifacts/*/result.json`，因为跨设备接力需要已完成算法结果。
- 验证：
  - 运行 `git check-ignore experiments/demo/run.json`，应返回非 0。
  - 运行 `git check-ignore experiments/demo/control/stop.request`，应返回 0。
  - 运行 `git check-ignore experiments/demo/artifacts/GRPO/result.json`，应返回非 0。

## 模块 3：算法注册表与训练命令构建

### 概述

- 职责：统一管理支持的算法、配置文件路径、训练命令参数，避免 CLI 和 VSCode 配置重复硬编码。
- 前置依赖：模块 1、模块 2
- 预计步骤数：5

### Step 1：创建注册表文件与测试

- 操作：创建：
  - `src/experiment/registry.py`
  - `tests/test_experiment_registry.py`
- 验证：
  - 运行 `python -c "from src.experiment.registry import AlgorithmRegistry"` 无报错。

### Step 2：实现算法名称规范化

- 操作：在 `src/experiment/registry.py` 中实现：
  - `class AlgorithmRegistry`
    - 常量 `SUPPORTED_ALGORITHMS`
      - 必须包含：
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
    - `def canonicalize(self, name: str) -> str`
    - `def validate(self, names: Sequence[str]) -> list[str]`
- 规范：
  - `canonicalize("simpo") == "SimPO"`。
  - `canonicalize("mappo") == "MAPPO"`。
  - 未知算法抛出 `ValueError("Unsupported algorithm: <name>")`。
- 验证：
  - `test_canonicalize_algorithm_names` 覆盖大小写输入。
  - `test_validate_rejects_unknown_algorithm` 覆盖未知算法。

### Step 3：实现配置路径解析

- 操作：在 `AlgorithmRegistry` 中实现：
  - `def config_path_for(self, algorithm_name: str) -> str`
- 规则：
  - 将 canonical 名转小写拼接到 `configs/algorithms/<lower>.yaml`。
  - 例如：
    - `GRPO` → `configs/algorithms/grpo.yaml`
    - `SimPO` → `configs/algorithms/simpo.yaml`
    - `MADDPG` → `configs/algorithms/maddpg.yaml`
- 验证：
  - `test_config_path_for_known_algorithms` 断言上述三个路径。

### Step 4：实现 AlgorithmSpec 列表构建

- 操作：在 `AlgorithmRegistry` 中实现：
  - `def build_specs(self, algorithms: Sequence[str], *, timesteps: int, seed: int, device: str, eval_episodes: int, env: str = "auto") -> list[AlgorithmSpec]`
- 规范：
  - 输出顺序必须与输入算法顺序一致。
  - 每个 `AlgorithmSpec.config_path` 来自 `config_path_for`。
  - `timesteps`、`seed`、`device`、`eval_episodes` 使用传入值。
- 验证：
  - `test_build_specs_preserves_order`：
    - 输入 `["ppo", "grpo", "sac"]`。
    - 输出 name 为 `["PPO", "GRPO", "SAC"]`。

### Step 5：实现训练命令构建器

- 操作：在 `src/experiment/registry.py` 中实现：
  - `class TrainCommandBuilder`
    - `__init__(self, project_root: Path | str = ".")`
    - `build(self, *, run_id: str, spec: AlgorithmSpec, experiment_dir: Path, python_executable: str | None = None) -> list[str]`
- 命令格式固定为：

```text
<python> scripts/train.py
  --config <spec.config_path>
  --algorithm <spec.name>
  --env <spec.env>
  --timesteps <spec.timesteps>
  --seed <spec.seed>
  --device <spec.device>
  --save-dir experiments/<run_id>/artifacts/<algorithm>/checkpoints
  --eval-episodes <spec.eval_episodes>
  --result-json experiments/<run_id>/artifacts/<algorithm>/result.json
```

- 规范：
  - `<python>` 默认为 `sys.executable`。
  - 算法名目录使用 canonical 名，例如 `GRPO`。
  - 如果 `spec.extra_args` 非空，追加到命令末尾。
- 验证：
  - `test_train_command_builder_contains_result_json`：
    - 构建 GRPO 命令。
    - 断言包含 `--result-json`。
    - 断言结果路径以 `experiments/demo/artifacts/GRPO/result.json` 结尾。

## 模块 4：增强 `scripts/train.py` 的机器可读结果输出

### 概述

- 职责：在不破坏现有用法的前提下，让单算法训练入口输出稳定 JSON，供实验编排器判断算法是否完成。
- 前置依赖：模块 3
- 预计步骤数：4

### Step 1：为 `scripts/train.py` 增加 CLI 参数

- 操作：修改 `scripts/train.py` 的 `parse_args()`，新增参数：
  - `--result-json`
    - `type=str`
    - `default=None`
    - `help="Optional path to write machine-readable final training result JSON"`
- 规范：
  - 未传 `--result-json` 时，现有行为完全不变。
- 验证：
  - 运行 `python scripts/train.py --help`，输出包含 `--result-json`。
  - 运行现有最小命令帮助不报错。

### Step 2：实现 JSON 可序列化工具

- 操作：在 `scripts/train.py` 中新增函数：
  - `def _to_jsonable(value):`
- 规则：
  - `dict`：递归转换 value。
  - `list` / `tuple`：递归转换元素。
  - `np.integer`：转 `int`。
  - `np.floating`：转 `float`。
  - `np.ndarray`：转 `.tolist()`。
  - `torch.Tensor`：先 `.detach().cpu().tolist()`。
  - 其他对象：保持原样；如果不能 JSON 序列化，转 `str(value)`。
- 注意：
  - 需要在文件顶部添加 `import json`。
  - 当前 `scripts/train.py` 未直接导入 `numpy`，如使用 `np.integer` 需新增 `import numpy as np`。
- 验证：
  - 新建测试 `tests/test_train_result_json.py`。
  - 使用 `importlib.util.spec_from_file_location` 加载 `scripts/train.py`。
  - 调用 `_to_jsonable({"x": np.float32(1.5)})`，断言结果为 Python `float`。

### Step 3：实现原子写 result JSON

- 操作：在 `scripts/train.py` 中新增函数：
  - `def _write_result_json(path: str | os.PathLike[str], payload: dict) -> None`
- 规范：
  - 创建父目录。
  - 写入 `<path>.tmp`。
  - `json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)`。
  - `os.replace(tmp_path, path)`。
- 验证：
  - 在 `tests/test_train_result_json.py` 中新增 `test_write_result_json_atomic`。
  - 写入临时目录，读取 JSON 后字段一致。

### Step 4：训练完成后写入 result JSON

- 操作：修改 `scripts/train.py` 的 `main()`。
- 在 `final_eval = trainer.evaluate()` 后、打印 final evaluation 前后均可，构造 payload：

```python
result_payload = {
    "algorithm": algo,
    "environment": env_name,
    "seed": seed,
    "device": str(device),
    "train_timesteps": total_timesteps,
    "checkpoint_dir": save_dir,
    "final_eval": final_eval,
    "status": "success",
}
```

- 如果 `args.result_json` 不为 `None`，调用 `_write_result_json(args.result_json, result_payload)`。
- 规范：
  - 只有 `trainer.train()` 和最终 `trainer.evaluate()` 均成功，才写 `status: success`。
  - 异常场景不在 `scripts/train.py` 捕获，由调用方 `ProcessRunner` 根据退出码标记失败。
- 验证：
  - `pytest tests/test_train_result_json.py -v` 通过。
  - `python scripts/train.py --help` 通过。

## 模块 5：子进程运行器与停止控制

### 概述

- 职责：由父进程启动单算法训练子进程，记录日志，轮询 stop request，在 Windows 下尽量发送 Ctrl-Break/中断信号。
- 前置依赖：模块 2、模块 3、模块 4
- 预计步骤数：6

### Step 1：创建进程运行器文件与测试

- 操作：创建：
  - `src/experiment/process_runner.py`
  - `tests/test_experiment_process_runner.py`
- 验证：
  - 运行 `python -c "from src.experiment.process_runner import ProcessRunner"` 无报错。

### Step 2：定义运行结果模型

- 操作：在 `src/experiment/process_runner.py` 中定义：
  - `@dataclass class ProcessResult`
    - `exit_code: int | None`
    - `interrupted: bool`
    - `stdout_log: str`
    - `stderr_log: str`
    - `result_json: str`
    - `checkpoint_dir: str`
    - `error: str | None = None`
- 方法：
  - `def succeeded(self) -> bool`
    - 返回 `self.exit_code == 0 and not self.interrupted`
- 验证：
  - `test_process_result_succeeded` 覆盖成功、失败、中断三种情况。

### Step 3：实现日志路径规划

- 操作：在 `ProcessRunner` 中实现：
  - `__init__(self, store: JsonStateStore, command_builder: TrainCommandBuilder | None = None)`
  - `def algorithm_paths(self, *, run_id: str, algorithm_name: str) -> dict[str, Path]`
- 返回字典必须包含：
  - `artifact_dir`
  - `checkpoint_dir`
  - `result_json`
  - `stdout_log`
  - `stderr_log`
  - `process_json`
- 路径规则：
  - `artifact_dir = experiments/<run_id>/artifacts/<algorithm_name>`
  - `checkpoint_dir = artifact_dir / "checkpoints"`
  - `result_json = artifact_dir / "result.json"`
  - `stdout_log = artifact_dir / "stdout.log"`
  - `stderr_log = artifact_dir / "stderr.log"`
  - `process_json = experiments/<run_id>/process.json`
- 验证：
  - `test_algorithm_paths_are_under_experiment_dir` 断言所有路径位于 `experiments/demo/` 下。

### Step 4：实现跨平台中断发送

- 操作：在 `ProcessRunner` 中实现：
  - `def _send_interrupt(self, process: subprocess.Popen) -> None`
- 规则：
  - Windows：
    - 优先调用 `process.send_signal(signal.CTRL_BREAK_EVENT)`。
    - 如果失败，调用 `process.terminate()`。
  - 非 Windows：
    - 优先 `process.send_signal(signal.SIGINT)`。
    - 如果失败，调用 `process.terminate()`。
- 特别注意：
  - 文件顶部导入 `signal`、`sys`、`subprocess`、`time`。
  - Windows 启动进程时需要 `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP`。
- 验证：
  - 单元测试中使用 fake process：
    - 有 `send_signal` 方法并记录参数。
    - 在 Windows 分支难以稳定测试时，只测试异常 fallback 调用 `terminate()`。

### Step 5：实现 run_algorithm

- 操作：在 `ProcessRunner` 中实现：
  - `def run_algorithm(self, *, run_id: str, spec: AlgorithmSpec) -> ProcessResult`
- 流程：
  1. 通过 `algorithm_paths()` 创建目录。
  2. 使用 `TrainCommandBuilder.build()` 生成命令。
  3. 将命令、pid、started_at 写入 `process.json`。
  4. 以 `subprocess.Popen` 启动子进程：
     - `cwd` 为项目根目录。
     - `stdout` 写入 `stdout.log`。
     - `stderr` 写入 `stderr.log`。
     - Windows 使用 `CREATE_NEW_PROCESS_GROUP`。
  5. 每 1 秒轮询：
     - 子进程是否退出。
     - `store.has_stop_request(run_id)` 是否为真。
  6. 若检测到 stop request：
     - 调用 `_send_interrupt(process)`。
     - 最多等待 30 秒。
     - 若仍未退出，调用 `process.kill()`。
     - 返回 `ProcessResult(interrupted=True, exit_code=<code>)`。
  7. 若正常退出：
     - 返回 `ProcessResult(interrupted=False, exit_code=process.returncode)`。
  8. 退出后删除 `process.json`。
- 规范：
  - 即使失败，也必须关闭 stdout/stderr 文件句柄。
  - 如果 `result.json` 不存在但 exit_code 为 0，返回 `error="Result JSON not found"` 并视为失败。
- 验证：
  - `test_run_algorithm_success_with_fake_subprocess`：
    - monkeypatch `subprocess.Popen` 为 fake。
    - fake returncode 为 0。
    - 预先创建 result.json。
    - 断言 `ProcessResult.succeeded()` 为真。
  - `test_run_algorithm_interrupted_when_stop_requested`：
    - monkeypatch `store.has_stop_request` 返回 True。
    - 断言 `interrupted is True`。

### Step 6：实现 process.json 状态清理

- 操作：在 `ProcessRunner` 中实现：
  - `def cleanup_process_file(self, run_id: str) -> None`
- 规范：
  - 删除 `experiments/<run_id>/process.json`。
  - 文件不存在时不报错。
  - `run_algorithm` 的 `finally` 块必须调用该方法。
- 验证：
  - `test_cleanup_process_file_is_idempotent`。
  - 连续调用两次不报错。

## 模块 6：实验管理器核心逻辑

### 概述

- 职责：实现 init/start/resume/stop/status/list/export 的核心业务逻辑，保证算法级恢复。
- 前置依赖：模块 1-5
- 预计步骤数：8

### Step 1：创建管理器文件与测试

- 操作：创建：
  - `src/experiment/manager.py`
  - `tests/test_experiment_manager.py`
- 验证：
  - 运行 `python -c "from src.experiment.manager import ExperimentManager"` 无报错。

### Step 2：实现 ExperimentManager 初始化

- 操作：在 `src/experiment/manager.py` 中实现：
  - `class ExperimentManager`
    - `__init__(self, store: JsonStateStore | None = None, registry: AlgorithmRegistry | None = None, runner: ProcessRunner | None = None, index: LocalExperimentIndex | None = None)`
- 默认依赖：
  - `store = JsonStateStore("experiments")`
  - `registry = AlgorithmRegistry()`
  - `runner = ProcessRunner(store=store)`
  - `index = LocalExperimentIndex("experiments/.index.sqlite3")`
- 验证：
  - `test_manager_default_init` 创建对象不报错。

### Step 3：实现 create_experiment

- 操作：在 `ExperimentManager` 中实现：
  - `def create_experiment(self, *, run_id: str, name: str, algorithms: Sequence[str], timesteps: int, seed: int, device: str, eval_episodes: int, env: str = "auto", output_dir: str = "results", metadata: dict[str, Any] | None = None) -> ExperimentManifest`
- 流程：
  1. 使用 `registry.build_specs()` 构造算法列表。
  2. 创建同长度的 `AlgorithmRunRecord`，初始状态 `PENDING`。
  3. 创建 `ExperimentManifest(schema_version=1, ...)`。
  4. 创建 `ExperimentState(schema_version=1, status=INITIALIZED, current_index=0, ...)`。
  5. 调用 `store.create()`。
  6. 调用 `index.initialize()` 与 `index.upsert()`。
- 规范：
  - `run_id` 只能包含 `[A-Za-z0-9_.-]`，非法则抛出 `ValueError`。
  - `algorithms` 不能为空。
  - `timesteps > 0`。
  - `eval_episodes > 0`。
- 验证：
  - `test_create_experiment_writes_manifest_and_state`。
  - 加载 JSON 后断言：
    - manifest.algorithms 长度正确。
    - state.records 长度正确。
    - state.current_index == 0。

### Step 4：实现 start_or_resume

- 操作：在 `ExperimentManager` 中实现：
  - `def start_or_resume(self, run_id: str) -> ExperimentState`
- 流程：
  1. 使用 `store.with_lock(run_id)` 加锁。
  2. 加载 manifest/state。
  3. 清除 stop request。
  4. 如果 state 已 `COMPLETED`，直接返回 state。
  5. 设置 state.status = RUNNING，保存。
  6. 释放锁后进入算法循环。
  7. 每轮重新加锁并加载最新 state。
  8. 如果 `store.has_stop_request(run_id)` 或 `state.stop_requested`：
     - 设置 state.status = STOPPED。
     - 保存并返回。
  9. 找到 `state.next_pending_index()`。
  10. 若为 None：
      - 设置 state.status = COMPLETED。
      - 保存并返回。
  11. 将该算法 record 标为 RUNNING 并保存。
  12. 调用 `runner.run_algorithm(run_id=run_id, spec=manifest.algorithms[index])`。
  13. 根据结果更新 record：
      - succeeded → `mark_completed`
      - interrupted → `mark_interrupted`
      - failed → `mark_failed`
  14. succeeded 时：
      - 将算法名追加到 `completed_algorithms`。
      - `current_index = index + 1`。
  15. interrupted 时：
      - `current_index = index`。
      - `status = STOPPED`。
      - 保存并返回。
  16. failed 时：
      - `current_index = index`。
      - `status = FAILED`。
      - `last_error = result.error`。
      - 保存并返回。
  17. 所有算法 completed 后状态为 COMPLETED。
- 规范：
  - 正在运行算法被停止时不得加入 `completed_algorithms`。
  - resume 必须从第一个未完成算法开始。
  - 每个算法结束后都必须保存 state，便于 Git 同步。
- 验证：
  - `test_start_or_resume_completes_two_algorithms`：
    - fake runner 每次返回 success。
    - 断言最终 completed_algorithms 为两个算法。
  - `test_resume_restarts_first_unfinished_algorithm`：
    - 初始 state 中第一个 completed，第二个 interrupted。
    - 调用 start_or_resume。
    - 断言 runner 首次运行第二个算法。

### Step 5：实现 request_stop

- 操作：在 `ExperimentManager` 中实现：
  - `def request_stop(self, run_id: str) -> ExperimentState`
- 流程：
  1. 加锁。
  2. 加载 state。
  3. `store.write_stop_request(run_id)`。
  4. `state.request_stop(updated_at=utc_now_iso())`。
  5. 保存 state。
  6. 更新 index。
  7. 返回 state。
- 规范：
  - 如果实验已经 COMPLETED，仍可写入 stop request 但状态不得从 COMPLETED 改成 STOP_REQUESTED。
  - 如果实验不存在，抛出 `ExperimentNotFoundError`。
- 验证：
  - `test_request_stop_sets_stop_requested`。
  - 断言 `state.stop_requested is True` 且 stop.request 文件存在。

### Step 6：实现 status 与 list_runs

- 操作：在 `ExperimentManager` 中实现：
  - `def get_status(self, run_id: str) -> dict[str, Any]`
  - `def list_runs(self, rebuild_index: bool = True) -> list[dict[str, Any]]`
- `get_status` 返回字典必须包含：
  - `run_id`
  - `name`
  - `status`
  - `current_index`
  - `total_algorithms`
  - `completed_count`
  - `completed_algorithms`
  - `next_algorithm`
  - `updated_at`
- `list_runs`：
  - 默认先调用 `index.rebuild_from_store(store)`。
  - 返回按 `updated_at` 倒序排列。
- 验证：
  - `test_get_status_reports_next_algorithm`。
  - `test_list_runs_sorted_by_updated_at`。

### Step 7：实现 reset_failed_algorithm

- 操作：在 `ExperimentManager` 中实现：
  - `def reset_failed_algorithm(self, run_id: str, algorithm_name: str) -> ExperimentState`
- 用途：
  - 某算法失败后，用户可以手动修复代码再 resume。
- 流程：
  1. 加锁加载 state。
  2. 找到 record。
  3. 仅允许 `FAILED` 或 `INTERRUPTED` 状态重置。
  4. 设置 record.status = PENDING。
  5. 清空 `error`、`exit_code`、`started_at`、`finished_at`。
  6. 将 `current_index` 设为该算法索引。
  7. 清理 `completed_algorithms` 中该算法之后的所有条目，保证顺序一致。
  8. 保存。
- 验证：
  - `test_reset_failed_algorithm_moves_current_index_back`。

### Step 8：实现 rebuild_index

- 操作：在 `ExperimentManager` 中实现：
  - `def rebuild_index(self) -> None`
- 行为：
  - 调用 `self.index.initialize()`。
  - 调用 `self.index.rebuild_from_store(self.store)`。
- 验证：
  - `test_rebuild_index_after_manual_json_copy`：
    - 模拟手工复制 JSON 到 experiments 目录。
    - rebuild 后 list_runs 能看到该实验。

## 模块 7：CLI 入口 `scripts/experiment_manager.py`

### 概述

- 职责：提供 VSCode 和命令行均可调用的稳定入口。
- 前置依赖：模块 6
- 预计步骤数：6

### Step 1：创建 CLI 文件

- 操作：创建：
  - `scripts/experiment_manager.py`
  - `tests/test_experiment_manager_cli.py`
- 文件头部要求：
  - 添加项目根目录到 `sys.path`，模式与现有 `scripts/train.py` 一致。
  - 导入 `ExperimentManager`。
- 验证：
  - 运行 `python scripts/experiment_manager.py --help` 成功。

### Step 2：实现 argparse 根解析器

- 操作：在 `scripts/experiment_manager.py` 中实现：
  - `def build_parser() -> argparse.ArgumentParser`
  - `def main(argv: list[str] | None = None) -> int`
- 子命令：
  - `start`
  - `resume`
  - `stop`
  - `status`
  - `list`
  - `export`
  - `reset`
  - `rebuild-index`
- 规范：
  - `main()` 不直接调用 `sys.exit()`，而是返回 int。
  - `if __name__ == "__main__": raise SystemExit(main())`。
- 验证：
  - `test_cli_help` 调用 `main(["--help"])` 可捕获 `SystemExit`，退出码 0。

### Step 3：实现 start 子命令

- 操作：`start` 参数：
  - `--run-id` 必填。
  - `--name` 默认等于 run_id。
  - `--algorithms` 支持多个值，默认 `GRPO PPO SAC DDQN`。
  - `--timesteps` 默认 `5000`。
  - `--seed` 默认 `42`。
  - `--device` 默认 `auto`。
  - `--eval-episodes` 默认 `3`。
  - `--env` 默认 `auto`。
  - `--output-dir` 默认 `results`。
- 行为：
  - 如果 run_id 不存在，先调用 `create_experiment()`。
  - 然后调用 `start_or_resume()`。
  - 输出最终 status JSON 到 stdout。
- 规范：
  - stdout 必须是 JSON，便于 VSCode 终端复制与后续脚本解析。
  - JSON 使用 `ensure_ascii=False`。
- 验证：
  - `test_cli_start_creates_when_missing` 使用 fake manager，断言 create 和 start_or_resume 都被调用。
  - `test_cli_start_resumes_when_existing` 使用 fake manager，断言 run 已存在时不重复 create。

### Step 4：实现 resume、stop、status、list 子命令

- 操作：
  - `resume --run-id <id>` → `manager.start_or_resume(run_id)`。
  - `stop --run-id <id>` → `manager.request_stop(run_id)`。
  - `status --run-id <id>` → `manager.get_status(run_id)`。
  - `list` → `manager.list_runs(rebuild_index=True)`。
- 输出：
  - 全部输出 JSON 到 stdout。
- 验证：
  - `test_cli_stop_calls_request_stop`。
  - `test_cli_status_prints_json`。
  - `test_cli_list_prints_array`。

### Step 5：实现 reset 与 rebuild-index 子命令

- 操作：
  - `reset --run-id <id> --algorithm <name>` → `manager.reset_failed_algorithm(...)`。
  - `rebuild-index` → `manager.rebuild_index()`。
- 输出：
  - reset 输出 state JSON。
  - rebuild-index 输出 `{"status": "ok"}`。
- 验证：
  - `test_cli_reset_calls_manager`。
  - `test_cli_rebuild_index_outputs_ok`。

### Step 6：实现 CLI 异常处理

- 操作：在 `main()` 中捕获：
  - `ExperimentNotFoundError`
  - `ExperimentStateError`
  - `ExperimentLockError`
  - `ValueError`
- 行为：
  - 向 stderr 输出错误 JSON：
    - `{"status": "error", "error": "..."}`
  - 返回退出码：
    - Not found：2
    - 状态错误/锁错误：3
    - 参数错误：4
    - 未知异常：1
- 验证：
  - `test_cli_returns_nonzero_on_value_error`。
  - fake manager 抛出 ValueError，断言返回 4。

## 模块 8：结果导出与旧 benchmark JSON 兼容

### 概述

- 职责：将实验状态与每个算法的 `result.json` 汇总为现有 benchmark 风格 JSON，并更新 `results/benchmark.json` 最新别名。
- 前置依赖：模块 4、模块 6、模块 7
- 预计步骤数：4

### Step 1：创建结果写入模块

- 操作：创建：
  - `src/experiment/result_writer.py`
  - `tests/test_experiment_result_writer.py`
- 验证：
  - 运行 `python -c "from src.experiment.result_writer import BenchmarkResultWriter"` 无报错。

### Step 2：实现 result.json 到 benchmark 条目的转换

- 操作：在 `src/experiment/result_writer.py` 中实现：
  - `class BenchmarkResultWriter`
    - `__init__(self, store: JsonStateStore)`
    - `def load_algorithm_result(self, result_path: Path) -> dict[str, Any]`
    - `def to_benchmark_entry(self, *, record: AlgorithmRunRecord, result_payload: Mapping[str, Any]) -> dict[str, Any]`
- 输出字段必须包含：
  - `algorithm`
  - `environment`
  - `seed`
  - `device`
  - `train_timesteps`
  - `status`
  - `final_reward_mean`
  - `final_reward_std`
  - `final_latency_mean`
  - `final_energy_mean`
  - `final_comm_score`
  - `checkpoint_dir`
- 映射规则：
  - 从 `result_payload["final_eval"]` 中读取：
    - `eval/reward_mean` → `final_reward_mean`
    - `eval/reward_std` → `final_reward_std`
    - `eval/latency_mean` → `final_latency_mean`
    - `eval/energy_mean` → `final_energy_mean`
    - `eval/comm_score` → `final_comm_score`
  - 不存在的指标填 `None`，不得抛异常。
- 验证：
  - `test_to_benchmark_entry_maps_final_eval_fields`。

### Step 3：实现 export_run

- 操作：在 `BenchmarkResultWriter` 中实现：
  - `def export_run(self, run_id: str, output_path: Path | str | None = None) -> Path`
- 行为：
  1. 加载 manifest/state。
  2. 只导出 `AlgorithmStatus.COMPLETED` 的记录。
  3. 按 manifest.algorithms 顺序输出 list。
  4. 默认输出路径：
     - `results/benchmark_<run_id>.json`
  5. 使用原子写。
  6. 同步写 `results/benchmark.json` 作为最新别名。
- 规范：
  - 不导出 running/interrupted/failed 算法。
  - 如果某个 completed 记录缺少 result_path 或文件不存在，导出时将该算法写为：
    - `status: "failed"`
    - `error: "Completed record result JSON not found: <path>"`
- 验证：
  - `test_export_run_writes_benchmark_json_and_latest_alias`。
  - 断言两个文件均存在。

### Step 4：连接 CLI export 子命令

- 操作：修改 `scripts/experiment_manager.py` 的 `export` 子命令。
- 参数：
  - `export --run-id <id> [--output results/custom.json]`
- 行为：
  - 创建 `BenchmarkResultWriter(JsonStateStore("experiments"))`。
  - 调用 `export_run()`。
  - stdout 输出：
    - `{"status": "ok", "output": "<path>"}`
- 验证：
  - `test_cli_export_calls_writer`。
  - 可通过 monkeypatch 替换 writer，避免真实训练。

## 模块 9：VSCode 配置、文档与最终验证

### 概述

- 职责：提供一键启动/恢复/停止/查看状态的 VSCode 调试入口，补充文档，并完成全链路测试。
- 前置依赖：模块 1-8
- 预计步骤数：4

### Step 1：创建 `.vscode/launch.json`

- 操作：创建 `.vscode/launch.json`。
- 内容必须包含以下配置：

1. `🧪 Experiment Start/Resume (Quick)`
   - `type`: `debugpy`
   - `request`: `launch`
   - `program`: `${workspaceFolder}/scripts/experiment_manager.py`
   - `cwd`: `${workspaceFolder}`
   - `console`: `integratedTerminal`
   - `justMyCode`: `false`
   - `args`：
     - `start`
     - `--run-id`
     - `vscode_quick`
     - `--name`
     - `VSCode Quick Benchmark`
     - `--algorithms`
     - `GRPO`
     - `PPO`
     - `SAC`
     - `--timesteps`
     - `5000`
     - `--seed`
     - `42`
     - `--device`
     - `auto`
     - `--eval-episodes`
     - `3`

2. `🧪 Experiment Resume`
   - 调用 `resume --run-id vscode_quick`

3. `⏹ Experiment Stop`
   - 调用 `stop --run-id vscode_quick`

4. `📊 Experiment Status`
   - 调用 `status --run-id vscode_quick`

5. `📋 Experiment List`
   - 调用 `list`

6. `📦 Experiment Export Results`
   - 调用 `export --run-id vscode_quick`

- `env` 必须包含：

```json
{
  "PYTHONPATH": "${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms"
}
```

- 验证：
  - `python -m json.tool .vscode/launch.json` 成功。
  - VSCode 调试面板能看到上述 6 个配置。

### Step 2：修改 `.gitignore` 以允许提交 `.vscode/launch.json`

- 操作：当前 `.gitignore` 已忽略 `.vscode/`。修改为：
  - 保留忽略 `.vscode/*`
  - 追加例外：
    - `!.vscode/`
    - `!.vscode/launch.json`
- 推荐替换片段：

```gitignore
# IDEs
.idea/
.vscode/*
!.vscode/
!.vscode/launch.json
*.swp
*.swo
*~
```

- 规范：
  - 不要提交 `.vscode/settings.json`，避免污染用户本地解释器配置。
  - 只允许提交 `.vscode/launch.json`。
- 验证：
  - `git check-ignore .vscode/settings.json` 返回 0。
  - `git check-ignore .vscode/launch.json` 返回非 0。

### Step 3：更新 VSCode 调试文档

- 操作：修改 `docs/vscode_debug_guide.md`。
- 新增章节：`## 🧪 可恢复实验编排`
- 必须包含：
  - Quick start：
    - 打开 VSCode。
    - 选择 `🧪 Experiment Start/Resume (Quick)`。
    - 按 F5。
  - 停止：
    - 选择 `⏹ Experiment Stop`。
    - 当前运行算法会被中断，不标记完成。
    - 下次 resume 会从该算法重跑。
  - 跨设备接力：
    - 设备 A 运行并完成部分算法。
    - `git add experiments/<run_id>/run.json experiments/<run_id>/state.json experiments/<run_id>/artifacts/*/result.json results/benchmark_<run_id>.json`
    - `git commit -m "sync experiment <run_id>"`
    - 设备 B `git pull`
    - 设备 B 运行 `🧪 Experiment Resume`
  - 结果导出：
    - 选择 `📦 Experiment Export Results`
    - 查看 `results/benchmark_vscode_quick.json`
- 验证：
  - 文档中包含字符串 `Experiment Start/Resume`。
  - 文档中包含字符串 `跨设备接力`。

### Step 4：全链路 smoke test

- 操作：执行以下命令：

```bash
python scripts/experiment_manager.py start --run-id smoke_test --name SmokeTest --algorithms GRPO --timesteps 1 --seed 42 --device cpu --eval-episodes 1
python scripts/experiment_manager.py status --run-id smoke_test
python scripts/experiment_manager.py export --run-id smoke_test
```

- 预期：
  - 第一条命令可能因算法对 timesteps 过小而失败；如果失败，必须保证：
    - `experiments/smoke_test/run.json` 存在。
    - `experiments/smoke_test/state.json` 存在。
    - `status` 命令能输出 JSON。
  - 若训练成功，则：
    - `experiments/smoke_test/artifacts/GRPO/result.json` 存在。
    - `results/benchmark_smoke_test.json` 存在。
- 额外测试命令：

```bash
pytest tests/test_experiment_models.py -v
pytest tests/test_experiment_state_store.py -v
pytest tests/test_experiment_registry.py -v
pytest tests/test_train_result_json.py -v
pytest tests/test_experiment_process_runner.py -v
pytest tests/test_experiment_manager.py -v
pytest tests/test_experiment_manager_cli.py -v
pytest tests/test_experiment_result_writer.py -v
ruff check src/experiment scripts/experiment_manager.py scripts/train.py
```

- 验收标准：
  - 所有新增单元测试通过。
  - `scripts/train.py` 原有命令不受影响。
  - `.vscode/launch.json` 是合法 JSON。
  - `state.json` 中 completed 算法可被 Git 同步。
  - 中断后 resume 从第一个未完成算法重新开始。
  - 不引入 Ray 或 DDP 相关依赖。
  - 不破坏现有 `scripts/benchmark.py` 和 `results/benchmark*.json` 使用方式。

## 最终验收清单

- [ ] `src/experiment/` 包可导入。
- [ ] `scripts/train.py --help` 包含 `--result-json`。
- [ ] `scripts/experiment_manager.py --help` 可正常显示。
- [ ] `python scripts/experiment_manager.py start ...` 能创建 `experiments/<run_id>/run.json` 与 `state.json`。
- [ ] `python scripts/experiment_manager.py stop --run-id <run_id>` 能生成 stop request，并使运行中实验停止在算法级边界或中断当前算法。
- [ ] `python scripts/experiment_manager.py resume --run-id <run_id>` 从第一个未完成算法继续。
- [ ] `python scripts/experiment_manager.py export --run-id <run_id>` 能生成 `results/benchmark_<run_id>.json` 和 `results/benchmark.json`。
- [ ] `.vscode/launch.json` 包含 start/resume/stop/status/list/export 六个入口。
- [ ] `.gitignore` 允许提交 `.vscode/launch.json`，允许提交实验 JSON 状态，忽略本地索引、控制文件、日志与模型权重。
- [ ] 新增 pytest 全部通过。
- [ ] `ruff check src/experiment scripts/experiment_manager.py scripts/train.py` 通过。
