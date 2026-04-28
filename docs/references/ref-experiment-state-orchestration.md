# 实验状态机与算法级恢复 — 实现参考

## 来源

- 项目调研结论：轻量实验编排器 + Git-friendly JSON 状态清单 + VSCode 调试入口
- 核心贡献：将“分布式训练”限定为算法阶段级跨时间、跨设备接力，不实现 PyTorch DDP、Ray 或算法内部精确 checkpoint/resume。

## 关键实现要点

1. 权威状态文件固定为：
   - `experiments/<run_id>/run.json`
   - `experiments/<run_id>/state.json`
2. `run.json` 保存实验清单：
   - `schema_version`
   - `run_id`
   - `name`
   - `created_at`
   - `updated_at`
   - `algorithms`
   - `project_root`
   - `output_dir`
   - `experiment_dir`
   - `metadata`
3. `state.json` 保存运行状态：
   - `schema_version`
   - `run_id`
   - `status`
   - `current_index`
   - `records`
   - `completed_algorithms`
   - `stop_requested`
   - `last_error`
   - `updated_at`
4. Resume 规则固定：
   - 找到第一个 `status != "completed"` 的算法记录。
   - 从该算法重新开始运行。
   - 不尝试恢复该算法内部 timestep、optimizer、replay buffer 或 random state。
5. 算法完成规则固定：
   - 子进程退出码为 `0`。
   - `experiments/<run_id>/artifacts/<algorithm>/result.json` 存在。
   - 只有同时满足以上条件，才能将该算法记录为 `completed`。
6. 中断规则固定：
   - stop 请求到达时，当前算法标记为 `interrupted` 或保持未完成。
   - 当前算法不得加入 `completed_algorithms`。
   - 下次 resume 从该算法重跑。
7. `completed_algorithms` 顺序必须与 `run.json.algorithms` 顺序一致，避免跨设备 Git 同步后恢复顺序不稳定。

## 对应模块

- 使用方：plan.md 中的 模块 1、模块 2、模块 6
- 集成方式：`src/experiment/models.py` 定义状态模型；`src/experiment/state_store.py` 持久化 JSON；`src/experiment/manager.py` 执行状态迁移。

## 注意事项

- 不要把 SQLite 当作权威状态；SQLite 只作为可重建本地索引。
- 不要把正在运行的算法视为已完成。
- 不要使用随机 run_id 覆盖用户指定的 run_id。
- JSON 写入必须原子化，防止 VSCode 停止或 Windows 文件同步造成半写入。
