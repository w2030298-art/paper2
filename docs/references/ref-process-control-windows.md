# Windows 子进程停止控制 — 实现参考

## 来源

- 项目调研结论：用 `subprocess` 调用现有 `scripts/train.py`，由父进程轮询 stop request 并中断子进程。
- 核心贡献：在 Windows + VSCode 调试场景下提供可控停止语义，同时保持训练脚本原有行为。

## 关键实现要点

1. 父进程入口：
   - `scripts/experiment_manager.py start --run-id <id> ...`
   - `scripts/experiment_manager.py resume --run-id <id>`
2. 子进程命令固定调用：
   - `<python> scripts/train.py`
   - `--config <config_path>`
   - `--algorithm <algorithm>`
   - `--env <env>`
   - `--timesteps <timesteps>`
   - `--seed <seed>`
   - `--device <device>`
   - `--save-dir experiments/<run_id>/artifacts/<algorithm>/checkpoints`
   - `--eval-episodes <eval_episodes>`
   - `--result-json experiments/<run_id>/artifacts/<algorithm>/result.json`
3. Windows 启动子进程时使用：
   - `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP`
4. 中断发送策略：
   - Windows：优先 `process.send_signal(signal.CTRL_BREAK_EVENT)`。
   - Windows fallback：`process.terminate()`。
   - 非 Windows：优先 `process.send_signal(signal.SIGINT)`。
   - 非 Windows fallback：`process.terminate()`。
5. 中断等待策略：
   - 发送中断后最多等待 30 秒。
   - 若仍未退出，调用 `process.kill()`。
6. 日志路径固定：
   - `experiments/<run_id>/artifacts/<algorithm>/stdout.log`
   - `experiments/<run_id>/artifacts/<algorithm>/stderr.log`
7. 进程记录固定：
   - `experiments/<run_id>/process.json`
   - 记录 command、pid、started_at。
   - 进程结束后在 `finally` 中删除。

## 对应模块

- 使用方：plan.md 中的 模块 5、模块 6、模块 7
- 集成方式：`src/experiment/process_runner.py` 负责子进程生命周期；`src/experiment/manager.py` 负责 stop/resume 状态迁移；`scripts/experiment_manager.py` 暴露 CLI。

## 注意事项

- 不要在 `scripts/train.py` 内实现全局 stop 逻辑；stop 应由编排器控制。
- 不要依赖 POSIX-only 信号；项目运行环境包含 Windows。
- 不要跳过 `finally` 清理 `process.json`。
- 如果 exit code 为 0 但 result JSON 不存在，应视为失败。
