# 结果导出与旧 benchmark JSON 兼容 — 实现参考

## 来源

- 项目调研结论：新增实验编排能力不得破坏现有 `results/benchmark*.json` 用法。
- 核心贡献：每个算法输出独立 result JSON，实验结束或中途可导出为旧 benchmark 风格汇总文件。

## 关键实现要点

1. 修改 `scripts/train.py` 时必须保持默认行为不变。
2. 新增 `--result-json` 参数：
   - 未传该参数时，现有训练命令输出行为不变。
   - 传入时，训练完成后写入机器可读 JSON。
3. `result.json` payload 固定包含：
   - `algorithm`
   - `environment`
   - `seed`
   - `device`
   - `train_timesteps`
   - `checkpoint_dir`
   - `final_eval`
   - `status`
4. `final_eval` 指标映射到 benchmark 条目：
   - `eval/reward_mean` → `final_reward_mean`
   - `eval/reward_std` → `final_reward_std`
   - `eval/latency_mean` → `final_latency_mean`
   - `eval/energy_mean` → `final_energy_mean`
   - `eval/comm_score` → `final_comm_score`
5. 缺失指标填 `None`，不得抛异常。
6. 导出路径：
   - 默认：`results/benchmark_<run_id>.json`
   - 最新别名：`results/benchmark.json`
7. 导出规则：
   - 只导出 `completed` 算法。
   - 不导出 running/interrupted/failed 算法。
   - 若 completed 记录缺失 result JSON，导出条目中写入 `status: "failed"` 与错误信息。

## 对应模块

- 使用方：plan.md 中的 模块 4、模块 8
- 集成方式：`scripts/train.py` 写 per-algorithm result JSON；`src/experiment/result_writer.py` 汇总输出 benchmark JSON；CLI `export` 子命令调用导出器。

## 注意事项

- 不要修改旧 `scripts/benchmark.py` 的输出结构来适配新功能。
- 不要要求旧结果文件迁移。
- 写 JSON 必须使用原子写，避免 Windows 或 Git 同步时生成半截文件。
