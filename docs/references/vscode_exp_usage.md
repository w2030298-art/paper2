# VSCode 算法对比实验使用说明

本文说明如何在 `paper2` 项目中使用 VSCode 一键启动、停止、恢复和导出算法对比实验。

本功能基于 `scripts/experiment_manager.py` 实现，VSCode 调试入口定义在 `.vscode/launch.json`，当前已包含 `start/resume/stop/status/list/export` 六类入口。
CLI 支持的命令包括 `start`、`resume`、`stop`、`status`、`list`、`export`、`reset`、`rebuild-index`。

## 1. 功能目标

该实验编排功能用于管理多算法对比实验，例如：

```text
GRPO → PPO → SAC
```

它的核心语义是：

* 一次只运行一个算法。
* 每个算法完成后写入实验状态。
* 如果中断，下次从第一个未完成算法继续。
* 不做 PyTorch DDP。
* 不要求保存正在运行算法的中间 checkpoint。
* 最小恢复单元是“算法完成状态”。
* 两台 Windows 设备可以通过 Git 同步 `experiments/<run_id>/` 状态后接力运行。

---

## 2. 使用前准备

### 2.1 安装依赖

在项目根目录执行：

```bash
uv venv .venv
uv pip install -r requirements.txt
```

或使用普通 `venv`：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果要真正训练算法，需要当前 Python 环境已安装 `torch`。
如果只是查看帮助命令，例如：

```bash
python scripts/train.py --help
python scripts/experiment_manager.py --help
```

则不应强制依赖 `torch`。

### 2.2 在 VSCode 中选择解释器

在 VSCode 中：

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ 选择 .venv/Scripts/python.exe
```

### 2.3 确认调试配置存在

项目中应存在：

```text
.vscode/launch.json
```

其中包含以下配置：

```text
🧪 Experiment Start/Resume (Quick)
🧪 Experiment Resume
⏹ Experiment Stop
📊 Experiment Status
📋 Experiment List
📦 Experiment Export Results
```

---

## 3. VSCode 一键运行流程

### 3.1 启动快速算法对比实验

在 VSCode 左侧选择：

```text
Run and Debug
```

选择配置：

```text
🧪 Experiment Start/Resume (Quick)
```

点击运行。

默认会启动一个实验：

```text
run_id: vscode_quick
name: VSCode Quick Benchmark
algorithms: GRPO PPO SAC
timesteps: 5000
seed: 42
device: auto
eval_episodes: 3
```

等价命令为：

```bash
python scripts/experiment_manager.py start ^
  --run-id vscode_quick ^
  --name "VSCode Quick Benchmark" ^
  --algorithms GRPO PPO SAC ^
  --timesteps 5000 ^
  --seed 42 ^
  --device auto ^
  --eval-episodes 3
```

Linux / WSL 写法：

```bash
python3 scripts/experiment_manager.py start \
  --run-id vscode_quick \
  --name "VSCode Quick Benchmark" \
  --algorithms GRPO PPO SAC \
  --timesteps 5000 \
  --seed 42 \
  --device auto \
  --eval-episodes 3
```

---

## 4. 查看实验状态

VSCode 中选择：

```text
📊 Experiment Status
```

等价命令：

```bash
python scripts/experiment_manager.py status --run-id vscode_quick
```

输出是 JSON，通常包含：

```json
{
  "run_id": "vscode_quick",
  "status": "...",
  "current_index": 0,
  "completed_algorithms": [],
  "records": {}
}
```

你主要关注：

| 字段                     | 含义        |
| ---------------------- | --------- |
| `run_id`               | 当前实验 ID   |
| `status`               | 实验整体状态    |
| `current_index`        | 当前算法索引    |
| `completed_algorithms` | 已完成算法     |
| `records`              | 每个算法的运行记录 |
| `stop_requested`       | 是否请求停止    |

---

## 5. 停止实验

VSCode 中选择：

```text
⏹ Experiment Stop
```

等价命令：

```bash
python scripts/experiment_manager.py stop --run-id vscode_quick
```

注意：该功能是“请求停止”，不是强制 kill 进程。
实验状态会写入：

```text
experiments/vscode_quick/state.json
```

由于当前设计的最小恢复单元是“算法”，正在运行但未完成的算法不会被视为完成。下次 resume 时会从第一个未完成算法继续。

---

## 6. 恢复实验

VSCode 中选择：

```text
🧪 Experiment Resume
```

等价命令：

```bash
python scripts/experiment_manager.py resume --run-id vscode_quick
```

恢复规则：

```text
从第一个未完成算法开始继续
```

例如：

```text
GRPO completed
PPO failed / interrupted / pending
SAC pending
```

则 resume 会从：

```text
PPO
```

开始。

---

## 7. 查看所有实验

VSCode 中选择：

```text
📋 Experiment List
```

等价命令：

```bash
python scripts/experiment_manager.py list
```

该命令会扫描本地实验状态，列出已有 run。

---

## 8. 导出 benchmark 结果

VSCode 中选择：

```text
📦 Experiment Export Results
```

等价命令：

```bash
python scripts/experiment_manager.py export --run-id vscode_quick
```

默认输出：

```text
results/benchmark_vscode_quick.json
```

该结果文件用于保持与既有 benchmark 结果格式兼容，方便后续绘图、报告生成或 dashboard 读取。

如果想自定义输出路径：

```bash
python scripts/experiment_manager.py export ^
  --run-id vscode_quick ^
  --output results/my_custom_benchmark.json
```

---

## 9. 两台 Windows 设备接力运行

假设你有：

```text
设备 A：有 GPU
设备 B：无 GPU
```

推荐流程：

### 9.1 设备 A 启动实验

```bash
python scripts/experiment_manager.py start ^
  --run-id paper2_main ^
  --name "Paper2 Main Benchmark" ^
  --algorithms GRPO PPO SAC DDQN IPPO ^
  --timesteps 50000 ^
  --seed 42 ^
  --device cuda ^
  --eval-episodes 10
```

某些算法完成后，状态会写入：

```text
experiments/paper2_main/run.json
experiments/paper2_main/state.json
```

### 9.2 提交实验状态

```bash
git add experiments/paper2_main
git commit -m "[实验状态] 更新 paper2_main 运行进度"
git push origin main
```

### 9.3 设备 B 拉取状态

```bash
git pull origin main
```

然后恢复：

```bash
python scripts/experiment_manager.py resume --run-id paper2_main
```

如果设备 B 没有 GPU，建议创建新实验或手动指定 CPU 运行；已经创建好的实验会沿用创建时记录的算法配置和设备参数。若要切换设备策略，优先新建一个 `run_id`，避免混淆结果。

---

## 10. 推荐 run_id 命名

建议不要一直使用 `vscode_quick` 做正式实验。

推荐格式：

```text
paper2_main_YYYYMMDD
paper2_ablation_small_YYYYMMDD
paper2_grpo_vs_baselines_YYYYMMDD
```

示例：

```bash
python scripts/experiment_manager.py start ^
  --run-id paper2_grpo_vs_baselines_20260428 ^
  --name "GRPO vs Baselines 2026-04-28" ^
  --algorithms GRPO PPO SAC DDQN IPPO ^
  --timesteps 100000 ^
  --seed 42 ^
  --device auto ^
  --eval-episodes 10
```

---

## 11. 常见操作速查

| 目标     | VSCode 配置                            | 命令                                                                     |
| ------ | ------------------------------------ | ---------------------------------------------------------------------- |
| 启动快速实验 | `🧪 Experiment Start/Resume (Quick)` | `python scripts/experiment_manager.py start --run-id vscode_quick ...` |
| 恢复实验   | `🧪 Experiment Resume`               | `python scripts/experiment_manager.py resume --run-id vscode_quick`    |
| 请求停止   | `⏹ Experiment Stop`                  | `python scripts/experiment_manager.py stop --run-id vscode_quick`      |
| 查看状态   | `📊 Experiment Status`               | `python scripts/experiment_manager.py status --run-id vscode_quick`    |
| 查看所有实验 | `📋 Experiment List`                 | `python scripts/experiment_manager.py list`                            |
| 导出结果   | `📦 Experiment Export Results`       | `python scripts/experiment_manager.py export --run-id vscode_quick`    |

---

## 12. 常见问题

### 12.1 `start` 返回 failed，是否一定是编排系统坏了？

不一定。

如果你使用极小的 `timesteps`，例如：

```text
timesteps=1
```

算法训练可能因训练步数不足、环境初始化或算法约束返回 failed。
这类情况需要区分：

| 现象                          | 判断                  |
| --------------------------- | ------------------- |
| `run.json` 存在               | 编排创建成功              |
| `state.json` 存在             | 状态写入成功              |
| `status` 可输出 JSON           | 状态查询正常              |
| `export` 可生成 benchmark JSON | 导出链路正常              |
| 算法本身 failed                 | 训练逻辑或参数问题，不一定是编排器问题 |

### 12.2 `python scripts/train.py --help` 报 torch 缺失怎么办？

正常情况下不应该报错。
`--help` 不应加载训练依赖。如果仍然出现该问题，说明 `scripts/train.py` 顶层又引入了 `torch` 或 trainer 相关模块，需要恢复延迟导入设计。

### 12.3 如何清理一次实验？

如果确认不再需要某个实验，可以删除：

```text
experiments/<run_id>/
results/benchmark_<run_id>.json
```

例如：

```bash
rmdir /s /q experiments\vscode_quick
del results\benchmark_vscode_quick.json
```

Linux / WSL：

```bash
rm -rf experiments/vscode_quick
rm -f results/benchmark_vscode_quick.json
```

### 12.4 是否应该提交 experiments 目录？

正式论文实验状态可以提交，尤其是需要两台设备接力时。
临时 smoke test 不建议提交，例如：

```text
experiments/smoke_test/
experiments/smoke_acceptance_*/
results/benchmark_smoke*.json
```

---

## 13. 推荐日常流程

### 快速测试

```text
VSCode → 🧪 Experiment Start/Resume (Quick)
VSCode → 📊 Experiment Status
VSCode → 📦 Experiment Export Results
```

### 正式实验

```bash
python scripts/experiment_manager.py start ^
  --run-id paper2_main_20260428 ^
  --name "Paper2 Main Benchmark" ^
  --algorithms GRPO PPO SAC DDQN IPPO VDN QMIX ^
  --timesteps 100000 ^
  --seed 42 ^
  --device auto ^
  --eval-episodes 10
```

然后：

```bash
python scripts/experiment_manager.py status --run-id paper2_main_20260428
python scripts/experiment_manager.py export --run-id paper2_main_20260428
```

### 中断后继续

```bash
python scripts/experiment_manager.py resume --run-id paper2_main_20260428
```

### 跨设备接力

```bash
git add experiments/paper2_main_20260428
git commit -m "[实验状态] 更新 paper2_main_20260428"
git push origin main
```

另一台设备：

```bash
git pull origin main
python scripts/experiment_manager.py resume --run-id paper2_main_20260428
```
