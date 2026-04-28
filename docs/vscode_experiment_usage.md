# VSCode 实验入口使用说明

本文说明如何在 `paper2` 项目中通过 VSCode Run and Debug / Tasks 完成主要实验流程。目标是用户无需输入命令行，就能启动、停止、恢复、查看状态、导出结果、重置失败算法、重建索引，并运行 Full 17 algorithms benchmark。

## 功能目标

- VSCode Run and Debug 覆盖核心入口：Quick smoke test、Full 17 benchmark、状态管理、结果导出、失败算法重置、索引重建、Direct Benchmark。
- VSCode Tasks 覆盖轻量命令入口：Full 17、Quick、list、rebuild-index、Direct Benchmark。
- Full 17 benchmark 默认 run id 为 `paper2_full_17_vscode`。
- Full 17 benchmark 固定覆盖 17 个算法，顺序为：`GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3`。
- 最小恢复单元仍是“算法完成状态”：中断后从第一个未完成算法继续。

默认 Full 17 配置：

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

## 推荐入口优先级

1. 正式全量对比实验：`🏁 Experiment Full 17 Start/Resume`
2. 查看进度：`📊 Experiment Full 17 Status`
3. 中断：`⏹ Experiment Full 17 Stop`
4. 导出结果：`📦 Experiment Full 17 Export Results`
5. 快速排错：`🧪 Experiment Quick Fresh Clean Run`
6. 直接 benchmark：`🏁 Benchmark Direct All 17`

## Full 17 algorithms benchmark 一键运行

在 VSCode 左侧打开 Run and Debug，选择并点击：

```text
🏁 Experiment Full 17 Start/Resume
```

该入口会调用：

```bash
python scripts/experiment_manager.py start --preset full17
```

默认 run id：

```text
paper2_full_17_vscode
```

17 算法列表和顺序固定为：

```text
GRPO, PPO, SAC, DDQN, DDPG, TD3, A3C, TRPO, SimPO, MAPPO, QMIX, COMA, IPPO, VDN, MADDPG, IQL, MATD3
```

运行状态文件：

```text
experiments/paper2_full_17_vscode/run.json
experiments/paper2_full_17_vscode/state.json
```

导出结果路径：

```text
results/benchmark_paper2_full_17_vscode.json
```

恢复规则：如果运行被中断，再次点击 `🏁 Experiment Full 17 Start/Resume` 会读取 `state.json`，从第一个未完成算法继续，不会重新开始已完成算法。

## Quick 入口报错处理

Quick smoke test 默认配置：

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

如果 Quick 入口报错，优先点击：

```text
🧪 Experiment Quick Fresh Clean Run
```

该入口会删除旧的 `vscode_quick` 实验状态并重新创建，适合处理旧状态污染。

若只想重跑某个失败算法，点击：

```text
♻️ Experiment Quick Reset GRPO
```

然后再点击：

```text
🧪 Experiment Quick Start/Resume
```

Quick 入口对应的等价 CLI 命令：

```bash
python scripts/experiment_manager.py start --run-id vscode_quick --name "VSCode Quick Benchmark" --algorithms GRPO PPO SAC --timesteps 5000 --seed 42 --device auto --eval-episodes 3
```

WSL 中如果没有 `python` 命令，可使用：

```bash
python3 scripts/experiment_manager.py start --run-id vscode_quick --name "VSCode Quick Benchmark" --algorithms GRPO PPO SAC --timesteps 5000 --seed 42 --device auto --eval-episodes 3
```

定位日志时查看：

```text
experiments/vscode_quick/state.json
experiments/vscode_quick/run.json
experiments/vscode_quick/artifacts/<ALGORITHM>/stdout.log
experiments/vscode_quick/artifacts/<ALGORITHM>/stderr.log
```

如果训练子进程失败，`state.json` 的错误信息会包含 stdout/stderr 日志路径；优先打开对应算法目录下的 `stdout.log` 和 `stderr.log`。

## VSCode Tasks 入口

打开 VSCode：

```text
Terminal -> Run Task...
```

常用任务：

```text
experiment: full17 start-resume
experiment: full17 status
experiment: full17 stop
experiment: full17 export
experiment: quick start-resume
experiment: quick fresh
experiment: quick status
experiment: quick stop
experiment: quick export
experiment: list
experiment: rebuild-index
benchmark: direct all17
```

Tasks 面板用于轻量命令执行；Run and Debug 面板用于需要调试体验的入口。
