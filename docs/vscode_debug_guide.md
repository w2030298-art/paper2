# VSCode 调试指南

## 🧪 可恢复实验编排

### Quick start
1. 打开 VSCode。
2. 在 Run and Debug 中选择 `🧪 Experiment Start/Resume (Quick)`。
3. 按 `F5` 启动。

### 停止
1. 选择 `⏹ Experiment Stop`。
2. 当前运行算法会被中断，不标记完成。
3. 下次 resume 会从该算法重跑。

### 跨设备接力
1. 设备 A 运行并完成部分算法。
2. 提交状态文件：
   - `git add experiments/<run_id>/run.json experiments/<run_id>/state.json experiments/<run_id>/artifacts/*/result.json results/benchmark_<run_id>.json`
   - `git commit -m "sync experiment <run_id>"`
3. 设备 B 执行 `git pull`。
4. 设备 B 选择 `🧪 Experiment Resume` 继续运行。

### 结果导出
1. 选择 `📦 Experiment Export Results`。
2. 查看 `results/benchmark_vscode_quick.json`。
