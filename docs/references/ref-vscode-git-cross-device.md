# VSCode 一键入口与 Git 跨设备接力 — 实现参考

## 来源

- 项目调研结论：用户主要通过 VSCode 调试面板一键启动、停止、恢复、查看状态和导出结果。
- 核心贡献：把跨设备“分布式”语义落地为 Git 同步算法级完成状态。

## 关键实现要点

1. `.vscode/launch.json` 必须包含 6 个配置：
   - `🧪 Experiment Start/Resume (Quick)`
   - `🧪 Experiment Resume`
   - `⏹ Experiment Stop`
   - `📊 Experiment Status`
   - `📋 Experiment List`
   - `📦 Experiment Export Results`
2. 所有 VSCode 配置都调用：
   - `${workspaceFolder}/scripts/experiment_manager.py`
3. 所有配置应设置：
   - `type`: `debugpy`
   - `request`: `launch`
   - `cwd`: `${workspaceFolder}`
   - `console`: `integratedTerminal`
   - `justMyCode`: `false`
4. `PYTHONPATH` 固定为：
   - `${workspaceFolder};${workspaceFolder}/src;${workspaceFolder}/scripts;${workspaceFolder}/rl_algorithms`
5. `.gitignore` 处理：
   - 保留忽略 `.vscode/*`
   - 允许提交 `.vscode/launch.json`
   - 不提交 `.vscode/settings.json`
6. Git 同步范围：
   - 必须同步 `experiments/<run_id>/run.json`
   - 必须同步 `experiments/<run_id>/state.json`
   - 必须同步已完成算法的 `experiments/<run_id>/artifacts/<algorithm>/result.json`
   - 可以同步 `results/benchmark_<run_id>.json`
7. 不应同步：
   - `experiments/.index.sqlite3`
   - `experiments/**/control/`
   - `experiments/**/process.json`
   - `experiments/**/logs/`
   - 模型权重：`*.pt`、`*.pth`、`*.ckpt`

## 对应模块

- 使用方：plan.md 中的 模块 2、模块 7、模块 9
- 集成方式：`.gitignore` 定义可同步状态文件；`.vscode/launch.json` 提供一键入口；`docs/vscode_debug_guide.md` 说明跨设备接力流程。

## 注意事项

- `experiments/**/run.json` 和 `experiments/**/state.json` 不能被 `.gitignore` 忽略。
- VSCode 配置只提交 `launch.json`，不要提交用户本地解释器路径。
- 跨设备接力前，应先 commit/push 完成算法的状态和 result JSON，再在另一台设备 pull 后 resume。
