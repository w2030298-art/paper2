# 前端训练监控对接手册

本文用于同步网页监控应用在新训练架构下的数据读取方式。当前训练系统已从“单次 benchmark 结果文件 + 零散日志”改为“实验编排状态机 + 每算法 artifact 目录 + 可导出的 benchmark 兼容结果”。

## 1. 对接目标

前端需要支持：

- 展示实验列表、实验详情、整体进度、当前算法、每个算法状态。
- 实时查看 `stdout.log` / `stderr.log`。
- 读取每个算法的 `result.json`，展示最终指标。
- 读取导出的 `results/benchmark_<run_id>.json`，兼容旧 benchmark 图表。
- 对 Full 17 benchmark 和 Quick smoke test 使用固定 run id。
- 在训练失败时，把错误定位到具体算法和具体 stdout/stderr 日志。

前端不应再把某一个 `results/*.json` 当作唯一真实状态。真实状态以 `experiments/<run_id>/state.json` 为准。

## 2. 新架构数据流

```text
VSCode / CLI
  -> scripts/experiment_manager.py
  -> experiments/<run_id>/run.json
  -> experiments/<run_id>/state.json
  -> experiments/<run_id>/process.json           # 运行中临时文件
  -> experiments/<run_id>/artifacts/<ALGORITHM>/
       -> stdout.log
       -> stderr.log
       -> result.json                            # 算法完成后生成
       -> checkpoints/
  -> scripts/experiment_manager.py export
  -> results/benchmark_<run_id>.json             # 兼容旧前端图表
  -> results/benchmark.json                      # latest alias
```

## 3. 固定实验入口

### Full 17 正式实验

```text
run_id: paper2_full_17_vscode
name: Paper2 Full 17 Algorithms VSCode Benchmark
algorithms: GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3
timesteps: 100000
seed: 42
device: auto
eval_episodes: 10
env: auto
output_dir: results
```

前端默认首页推荐展示这个 run：

```text
experiments/paper2_full_17_vscode/run.json
experiments/paper2_full_17_vscode/state.json
results/benchmark_paper2_full_17_vscode.json
```

### Quick smoke test

```text
run_id: vscode_quick
name: VSCode Quick Benchmark
algorithms: GRPO, PPO, SAC
timesteps: 5000
seed: 42
device: auto
eval_episodes: 3
env: auto
output_dir: results
```

Quick 用于入口连通性和失败定位，不应作为正式论文结果展示。

## 4. 文件契约

### 4.1 `experiments/<run_id>/run.json`

用途：实验清单。创建后基本不变，前端用它构建算法列表和实验元信息。

关键字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | number | 当前为 `1` |
| `run_id` | string | 实验 ID |
| `name` | string | 展示名称 |
| `created_at` | string | UTC 时间，格式如 `2026-04-28T10:42:29Z` |
| `updated_at` | string | 清单更新时间 |
| `algorithms` | array | 算法执行规格，顺序就是执行顺序 |
| `project_root` | string | 项目根目录，通常为 `.` |
| `output_dir` | string | 结果输出目录，通常为 `results` |
| `experiment_dir` | string | 实验目录 |
| `metadata` | object | 预留扩展字段 |

`algorithms[]` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 算法名 |
| `config_path` | string | 算法配置文件 |
| `timesteps` | number | 训练步数 |
| `seed` | number | 随机种子 |
| `device` | string | `auto` / `cpu` / `cuda` |
| `env` | string | `auto` 或具体环境名 |
| `eval_episodes` | number | 评估 episode 数 |
| `extra_args` | array | 额外训练参数 |

前端建议：

- 用 `run.json.algorithms` 决定表格行顺序。
- 不要从目录名猜算法顺序。
- Full 17 的算法顺序必须保持：`GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3`。

### 4.2 `experiments/<run_id>/state.json`

用途：实验实时状态。前端主轮询文件。

关键字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | number | 当前为 `1` |
| `run_id` | string | 实验 ID |
| `status` | string | 实验整体状态 |
| `current_index` | number | 当前算法索引 |
| `records` | array | 每个算法的运行记录 |
| `completed_algorithms` | array | 已完成算法名 |
| `stop_requested` | boolean | 是否请求停止 |
| `last_error` | string/null | 最近一次实验级错误 |
| `updated_at` | string | 状态更新时间 |

实验整体状态枚举：

| 状态 | UI 文案建议 | 含义 |
|---|---|---|
| `initialized` | 已创建 | run/state 已生成，尚未训练 |
| `running` | 运行中 | 至少一个算法正在或即将执行 |
| `stop_requested` | 停止中 | 已请求停止，等待子进程响应 |
| `stopped` | 已停止 | 用户中断，可恢复 |
| `completed` | 已完成 | 全部算法完成 |
| `failed` | 失败 | 当前算法失败，需查看日志 |

`records[]` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 算法名 |
| `status` | string | 算法状态 |
| `started_at` | string/null | 本算法开始时间 |
| `finished_at` | string/null | 本算法结束时间 |
| `exit_code` | number/null | 子进程退出码 |
| `attempts` | number | 尝试次数 |
| `device` | string | 实际记录的 device |
| `result_path` | string/null | 完成后 result JSON 路径 |
| `checkpoint_dir` | string/null | checkpoint 目录 |
| `stdout_log` | string/null | 预留字段，当前失败时主要从 `error` 或固定路径拿 |
| `stderr_log` | string/null | 预留字段，当前失败时主要从 `error` 或固定路径拿 |
| `error` | string/null | 算法错误信息 |

算法状态枚举：

| 状态 | UI 文案建议 | 含义 |
|---|---|---|
| `pending` | 待运行 | 尚未开始 |
| `running` | 运行中 | 当前正在训练 |
| `completed` | 已完成 | 有效完成，通常应有 `result_path` |
| `interrupted` | 已中断 | 用户停止或进程中断 |
| `failed` | 失败 | 训练失败或结果缺失 |
| `skipped` | 已跳过 | 保留状态，当前主流程较少使用 |

前端必须遵守：

- UI 状态以 `record.status` 为准。
- `record.status == "running"` 时，即使 `error`、`exit_code`、`finished_at` 有旧值，也应当按运行中展示；这些字段可能来自上一次 attempt 的残留。
- 只有 `record.status in {"failed", "interrupted"}` 时，才把 `record.error` 作为当前错误展示。
- 进度百分比建议使用 `completed_algorithms.length / records.length`。
- 当前算法建议使用：优先取 `records[current_index]`；如果越界，则取第一个 `status != "completed"` 的 record。

### 4.3 `experiments/<run_id>/process.json`

用途：运行中子进程标记。训练开始时生成，子进程结束后清理。

字段示例：

```json
{
  "command": ["python", "scripts/train.py", "--algorithm", "GRPO"],
  "pid": 38844,
  "started_at": "2026-04-28T10:46:37Z"
}
```

前端建议：

- 有后端代理时，可以用 `pid` 校验进程是否还活着。
- 纯静态文件读取时，不要仅凭 `process.json` 判断一定在运行；它可能因为异常退出或跨 Windows/WSL 视角不同而变成 stale marker。
- `process.json` 存在且 `state.status == "running"` 时，可以显示“运行中”。
- `process.json` 存在但 `state.updated_at` 长时间不变、日志也不增长时，显示“可能卡住/需人工确认”。
- 不要由前端删除 `process.json`；控制入口应走 CLI 或 VSCode task。

### 4.4 `experiments/<run_id>/artifacts/<ALGORITHM>/stdout.log`

用途：训练标准输出。

内容包括：

- 算法名、环境、device、seed、timesteps。
- 训练进度条或 trainer 输出。
- 最终 evaluation 文本输出。

路径规则：

```text
experiments/<run_id>/artifacts/<ALGORITHM>/stdout.log
```

前端建议：

- 使用增量读取或 range request，避免每次全量读取大日志。
- 日志显示按纯文本处理，不要执行其中内容。
- 进度条包含 `\r` 回车更新，UI 可把 `\r` 归一成换行或只保留最后一行。

### 4.5 `experiments/<run_id>/artifacts/<ALGORITHM>/stderr.log`

用途：训练标准错误。

内容包括：

- Python traceback。
- warning。
- tqdm 进度条在某些环境下也可能写入 stderr。

路径规则：

```text
experiments/<run_id>/artifacts/<ALGORITHM>/stderr.log
```

前端建议：

- `stderr.log` 不为空不一定代表失败；很多进度条和 warning 会写 stderr。
- 是否失败仍以 `state.json.records[].status` 为准。
- 当 `record.status == "failed"` 时，默认打开 stderr tab，并提供 stdout tab。

### 4.6 `experiments/<run_id>/artifacts/<ALGORITHM>/result.json`

用途：单算法机器可读训练结果。只有算法成功完成后才应存在。

当前写出字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `algorithm` | string | 算法名 |
| `environment` | string | 实际环境名 |
| `seed` | number | 随机种子 |
| `device` | string | 实际 device |
| `train_timesteps` | number | 训练步数 |
| `checkpoint_dir` | string | checkpoint 目录 |
| `final_eval` | object | trainer.evaluate() 返回的指标 |
| `status` | string | 成功时为 `success` |

`final_eval` 指标 key 当前由 trainer 输出决定，常见 key：

```text
eval/reward_mean
eval/reward_std
eval/latency_mean
eval/energy_mean
eval/comm_score
```

前端建议：

- 只有 `record.status == "completed"` 且 `record.result_path` 非空时读取。
- 如果文件缺失，显示“结果文件缺失”，不要把算法显示为成功。
- 指标 key 需要容错；没有的 key 显示 `N/A`。

### 4.7 `results/benchmark_<run_id>.json`

用途：兼容旧 benchmark 图表的数据文件，由 export 命令生成。

生成命令：

```bash
python scripts/experiment_manager.py export --run-id <run_id> --output results/benchmark_<run_id>.json
```

Full 17 固定导出路径：

```text
results/benchmark_paper2_full_17_vscode.json
```

导出行为：

- 只导出 `completed` 算法。
- 同时更新 `results/benchmark.json` 作为 latest alias。
- 如果当前没有完成算法，导出文件是空数组 `[]`。

导出 entry 字段：

| 字段 | 说明 |
|---|---|
| `algorithm` | 算法名 |
| `environment` | 环境名 |
| `seed` | 随机种子 |
| `device` | device |
| `train_timesteps` | 训练步数 |
| `status` | 算法结果状态 |
| `final_reward_mean` | 平均 reward |
| `final_reward_std` | reward 标准差 |
| `final_latency_mean` | 平均 latency |
| `final_energy_mean` | 平均 energy |
| `final_comm_score` | 通信评分 |
| `checkpoint_dir` | checkpoint 目录 |

前端建议：

- 图表页优先读 `results/benchmark_paper2_full_17_vscode.json`。
- 实时运行页优先读 `state.json` + per-algorithm `result.json`。
- 不要用 `results/benchmark.json` 判断当前实验状态；它只是 latest export alias。

## 5. 推荐轮询策略

### 5.1 文件读取优先级

实时页每轮读取：

1. `experiments/<run_id>/state.json`
2. `experiments/<run_id>/run.json`，只在首次加载或 `schema_version/run_id` 变化时重读
3. 当前算法的 `stdout.log` / `stderr.log`
4. 当前算法完成后读取 `result.json`
5. 用户打开图表页或点击刷新结果时读取 `results/benchmark_<run_id>.json`

### 5.2 轮询间隔

| 场景 | 建议间隔 |
|---|---:|
| `state.status == "running"` | 2s |
| `state.status == "stop_requested"` | 1s |
| `state.status in {"failed", "stopped"}` | 5s 或停止自动轮询 |
| `state.status == "completed"` | 停止轮询 |
| 日志 tail | 1s 到 2s |

### 5.3 原子写保护

后端使用临时文件 + `os.replace` 写 JSON。前端仍应容错：

- JSON 解析失败时，保留上一帧数据，下一轮重试。
- 文件不存在时显示“等待生成”。
- 空文件或部分写入按 transient error 处理，不立刻报失败。

## 6. UI 状态合成规则

推荐前端把原始状态合成为以下视图模型：

```ts
type ExperimentView = {
  runId: string;
  name: string;
  status: 'initialized' | 'running' | 'stop_requested' | 'stopped' | 'completed' | 'failed';
  total: number;
  completed: number;
  progress: number;
  currentAlgorithm: string | null;
  lastError: string | null;
  rows: AlgorithmRow[];
};

type AlgorithmRow = {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'interrupted' | 'failed' | 'skipped';
  attempts: number;
  startedAt: string | null;
  finishedAt: string | null;
  stdoutPath: string;
  stderrPath: string;
  resultPath: string;
  error: string | null;
};
```

合成规则：

```ts
const artifactBase = `experiments/${runId}/artifacts/${algorithmName}`;
const stdoutPath = `${artifactBase}/stdout.log`;
const stderrPath = `${artifactBase}/stderr.log`;
const resultPath = record.result_path ?? `${artifactBase}/result.json`;
const error = ['failed', 'interrupted'].includes(record.status) ? record.error : null;
```

进度计算：

```ts
const total = state.records.length;
const completed = state.completed_algorithms.length;
const progress = total === 0 ? 0 : completed / total;
```

当前算法：

```ts
const current = state.records[state.current_index]
  ?? state.records.find((record) => record.status !== 'completed')
  ?? null;
```

## 7. 操作入口映射

如果前端只有静态文件读取能力，本节可作为按钮说明，不直接执行命令。

如果前端有后端代理，可以把按钮映射到以下 CLI：

| UI 操作 | CLI |
|---|---|
| Full 17 启动/恢复 | `python scripts/experiment_manager.py start --preset full17` |
| Full 17 状态 | `python scripts/experiment_manager.py status --run-id paper2_full_17_vscode` |
| Full 17 停止 | `python scripts/experiment_manager.py stop --run-id paper2_full_17_vscode` |
| Full 17 导出 | `python scripts/experiment_manager.py export --run-id paper2_full_17_vscode --output results/benchmark_paper2_full_17_vscode.json` |
| Quick 启动/恢复 | `python scripts/experiment_manager.py start --preset quick` |
| Quick fresh | `python scripts/experiment_manager.py start --preset quick --fresh` |
| Quick 状态 | `python scripts/experiment_manager.py status --run-id vscode_quick` |
| Quick 停止 | `python scripts/experiment_manager.py stop --run-id vscode_quick` |
| Quick 导出 | `python scripts/experiment_manager.py export --run-id vscode_quick --output results/benchmark_vscode_quick.json` |
| 重建索引 | `python scripts/experiment_manager.py rebuild-index` |
| 列出实验 | `python scripts/experiment_manager.py list` |
| 重置失败算法 | `python scripts/experiment_manager.py reset --run-id <run_id> --algorithm <ALGORITHM>` |

安全要求：

- 前端不要直接删除 `experiments/<run_id>/`。
- Fresh clean run 必须走 `--fresh`，因为它会拒绝删除存在 `process.json` 的运行中实验。
- Reset 只允许对 `failed` / `interrupted` 算法使用。

## 8. 旧前端迁移清单

### 必改

- 从“直接读取单个 benchmark JSON 判断训练状态”改为读取 `state.json`。
- 算法列表从 `run.json.algorithms` 获取。
- 日志路径改为 `experiments/<run_id>/artifacts/<ALGORITHM>/stdout.log` 和 `stderr.log`。
- 结果路径改为每算法 `result.json`，聚合图表再读 `results/benchmark_<run_id>.json`。
- 错误展示从 `state.last_error` 和 `records[].error` 读取；失败时链接到 stdout/stderr。

### 建议改

- 增加 Full 17 默认入口卡片：`paper2_full_17_vscode`。
- 增加 Quick 诊断入口卡片：`vscode_quick`。
- 增加“导出结果”按钮或提示用户点击 VSCode `📦 Experiment Full 17 Export Results`。
- 对 `result.json` 和 benchmark export 做 schema 容错，未知指标保留但不阻断页面。

### 不建议

- 不读取 `experiments/.index.sqlite3` 作为浏览器端数据源；它适合 CLI/backend，不适合直接网页读取。
- 不用 `results/benchmark.json` 作为当前实验状态源。
- 不用 stderr 是否为空判断失败。
- 不通过前端直接 kill 进程或删除目录。

## 9. 前端验收用例

### 用例 1：Full 17 初始/运行中

输入：

```text
experiments/paper2_full_17_vscode/run.json
experiments/paper2_full_17_vscode/state.json
```

期望：

- 页面显示 17 个算法。
- 算法顺序固定。
- 当前算法高亮为 `records[current_index]`。
- 进度按 completed 数计算。
- 当前算法可打开 stdout/stderr。

### 用例 2：Quick 失败

输入：

```text
experiments/vscode_quick/state.json
experiments/vscode_quick/artifacts/GRPO/stdout.log
experiments/vscode_quick/artifacts/GRPO/stderr.log
```

期望：

- 页面显示 Quick 共有 3 个算法。
- GRPO 失败时显示错误摘要。
- 错误详情提供 stdout/stderr 链接。
- PPO/SAC 保持 pending。

### 用例 3：导出结果为空

输入：

```json
[]
```

期望：

- 图表页显示“暂无已完成算法结果”。
- 不把空数组视为加载失败。

### 用例 4：运行中 record 带旧 error

输入：

```json
{
  "name": "GRPO",
  "status": "running",
  "exit_code": 1,
  "error": "previous failed attempt"
}
```

期望：

- UI 显示 GRPO 运行中。
- 不把旧 `error` 当作当前失败。
- 可以在“历史尝试/调试信息”区域弱提示 attempts > 1。

## 10. 建议交付边界

前端团队本轮只需要对接文件读取和状态展示，不需要实现训练调度后端。

推荐交付顺序：

1. 状态页：读取 `run.json` + `state.json`。
2. 日志页：tail 当前算法 stdout/stderr。
3. 结果页：读取 per-algorithm `result.json`。
4. 图表页：读取 `results/benchmark_<run_id>.json`。
5. 可选操作页：通过后端代理调用 `experiment_manager.py`。

## 11. 联系后端时需要确认的问题

如果网页部署在浏览器沙箱中，通常不能直接读取本地文件。需要网页团队确认以下一点：

- 当前网页是否已有本地文件代理/API？

如果没有，需要新增一个只读后端接口，把上述文件以 HTTP JSON/text 形式暴露给前端。接口建议保持薄封装，不改变训练目录结构。
