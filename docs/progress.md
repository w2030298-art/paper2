# 开发进度

## 当前状态
- 当前阶段：Patch 模块 7-9 全部完成
- 最后更新：2026-05-01
- 状态：completed

## 模块进度

### 模块 1：现状锁定与 Quick 入口 bug 复现
- [x] Step 1: 创建 VSCode 配置测试文件 ✅ 2026-04-28
- [x] Step 2: 复现 Quick 入口的等价 CLI 路径 ✅ 2026-04-28
- [x] Step 3: 修复 `scripts/train.py` 的 import 稳定性 ✅ 2026-04-28
- [x] Step 4: 补齐 result JSON 写出异常的诊断信息 ✅ 2026-04-28
- [x] Step 5: 记录 Quick bug 修复结论 ✅ 2026-04-28

### 模块 2：实验预设与 CLI 稳定性增强
- [x] Step 1: 新增实验预设模块 ✅ 2026-04-28
- [x] Step 2: 让算法注册表复用 Full 17 常量 ✅ 2026-04-28
- [x] Step 3: 为 `experiment_manager.py start` 增加 `--preset` ✅ 2026-04-28
- [x] Step 4: 增加 `--fresh` 支持 Quick clean run ✅ 2026-04-28
- [x] Step 5: 增加 `preset` CLI 单元测试 ✅ 2026-04-28
- [x] Step 6: 增加 `delete/fresh` state store 单元测试 ✅ 2026-04-28
- [x] Step 7: 保持原 CLI 兼容性 ✅ 2026-04-28
- [x] Step 8: 确认 Full 17 预设不触发真实训练的结构测试 ✅ 2026-04-28

### 模块 3：VSCode `launch.json` 全入口改造
- [x] Step 1: 重写 `.vscode/launch.json` 的公共规则 ✅ 2026-04-28
- [x] Step 2: 保留并修复 Quick Start/Resume 入口 ✅ 2026-04-28
- [x] Step 3: 新增 Quick Fresh Clean Run 入口 ✅ 2026-04-28
- [x] Step 4: 新增 Full 17 Start/Resume 入口 ✅ 2026-04-28
- [x] Step 5: 保留常规管理入口 ✅ 2026-04-28
- [x] Step 6: 新增 Reset 与 Rebuild Index 入口 ✅ 2026-04-28
- [x] Step 7: 新增 Direct Benchmark All 入口 ✅ 2026-04-28

### 模块 4：VSCode `tasks.json` 任务入口补齐
- [x] Step 1: 创建 `.vscode/tasks.json` ✅ 2026-04-28
- [x] Step 2: 添加 Full 17 核心任务 ✅ 2026-04-28
- [x] Step 3: 添加 Quick 任务 ✅ 2026-04-28
- [x] Step 4: 添加索引与列表任务 ✅ 2026-04-28
- [x] Step 5: 添加 Direct Benchmark All 任务 ✅ 2026-04-28

### 模块 5：使用文档更新
- [x] Step 1: 更新 `docs/vscode_experiment_usage.md` 的功能目标 ✅ 2026-04-28
- [x] Step 2: 新增"推荐入口优先级" ✅ 2026-04-28
- [x] Step 3: 新增"Full 17 一键运行说明" ✅ 2026-04-28
- [x] Step 4: 新增"Quick 入口报错处理" ✅ 2026-04-28

### 模块 6：自动化测试与最终验收
- [x] Step 1: 运行单元测试 ✅ 2026-04-28
- [x] Step 2: 验证帮助命令无训练依赖副作用 ✅ 2026-04-28
- [x] Step 3: 验证 Quick smoke test 入口 ✅ 2026-04-28
- [x] Step 4: 验证 Full 17 benchmark 入口 ✅ 2026-04-28

### 模块 7：Reward + comm_score 权重修正 (Patch)
- [x] Step 1: 修改 r_imm 权重 (game_theory_env.py) ✅ 2026-05-01
- [x] Step 2: 修改 _communication_cost 权重 ✅ 2026-05-01
- [x] Step 3: 修改 base_trainer.py comm_score 公式 ✅ 2026-05-01
- [x] Step 4: 同步 benchmark.py comm_score ✅ 2026-05-01
- [x] Step 5: 同步 evaluate.py comm_score ✅ 2026-05-01
- [x] Step 6: 编写权重变更单元测试 ✅ 2026-05-01

### 模块 8：收敛曲线数据收集与可视化 (Patch)
- [x] Step 1: benchmark_single 收集 train_logs ✅ 2026-05-01
- [x] Step 2: run_benchmark 汇总收敛数据 ✅ 2026-05-01
- [x] Step 3: 新增收敛曲线绘图函数 ✅ 2026-05-01
- [x] Step 4: 集成到 CLI ✅ 2026-05-01
- [x] Step 5: 收敛曲线单元测试 ✅ 2026-05-01

### 模块 9：Composite Score 综合评分体系 (Patch)
- [x] Step 1: 创建 CompositeScorer 类 ✅ 2026-05-01
- [x] Step 2: 创建 scoring_profiles.yaml ✅ 2026-05-01
- [x] Step 3: 集成到 benchmark.py ✅ 2026-05-01
- [x] Step 4: 新增可视化图表函数 ✅ 2026-05-01
- [x] Step 5: 更新 generate_report.py ✅ 2026-05-01
- [x] Step 6: 综合评分单元测试 ✅ 2026-05-01
- [x] Step 7: 全流程验收 ✅ 2026-05-01

## 已知问题

（暂无）
