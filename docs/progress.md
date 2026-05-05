# 开发进度

## 当前状态
- 当前阶段：slimming-plan-v3 模块 14 formal convergence verification protocol
- 最后更新：2026-05-04
- 状态：IN_PROGRESS，L2 100k x 3 seeds 长时验证已启动；L1 仍仅为预筛，无算法进入论文主图或主结论

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

### 模块 10：Convergence Quality Pipeline (Patch v3)
- [x] Step 1: 收敛数据 schema 归一化 ✅ 2026-05-02 [auto]
- [x] Step 2: 新增质量诊断与数据清洗层 ✅ 2026-05-02 [review]
- [x] Step 3: 实现多 seed 稳健聚合 ✅ 2026-05-02 [review]
- [x] Step 4: 实现方向感知的收敛判定 ✅ 2026-05-02 [review]
- [x] Step 5: 拆分 diagnostic raw 图与 publication clean 图 ✅ 2026-05-02 [review]
- [x] Step 6: 实现稳健坐标轴策略与异常算法标注 ✅ 2026-05-02 [review]
- [x] Step 7: 补齐 CLI 参数 ✅ 2026-05-02 [auto]
- [x] Step 8: 补齐 benchmark convergence schema 元数据 ✅ 2026-05-02 [review]
- [x] Step 9: 强化收敛曲线回归测试 ✅ 2026-05-02 [auto]
- [x] Step 10: 报告与文档集成 ✅ 2026-05-02 [auto]

### 模块 11：docs 目录整理 (Patch v3)
- [x] Step 1: 建立 docs 目录索引 ✅ 2026-05-02 [auto]
- [x] Step 2: 创建标准子目录并归档明显历史文件 ✅ 2026-05-02 [auto]
- [x] Step 3: 补齐执行端契约文档占位 ✅ 2026-05-02 [auto]
- [x] Step 4: 增加 docs 契约回归测试 ✅ 2026-05-02 [auto]

### 模块 12：未收敛算法诊断与稳定化计划 (Patch v4)
- [x] Step 1: 修复提交卫生与统一 `docs/archive/` ✅ 2026-05-02 [auto]
- [x] Step 2: 新增未收敛诊断脚本与测试 ✅ 2026-05-02 [auto]
- [x] Step 3: 扩展收敛判定为 bad_plateau / oscillating / diverging / catastrophic_outlier ✅ 2026-05-02 [review]
- [x] Step 4: 新增稳定化候选配置且默认不启用 ✅ 2026-05-02 [review]
- [x] Step 5: 记录 targeted rerun 计划，不重跑 full 17 ✅ 2026-05-02 [review]
- [x] Step 6: 产出 `docs/convergence_failure_analysis.md` ✅ 2026-05-02 [auto]

### 模块 12（Slimming）：无争议仓库卫生清理 (slimming-plan-v1)
- [x] Step 1: 生成执行前审计快照 `docs/slimming_audit_phase2.md` ✅ 2026-05-02 [auto]
- [x] Step 2: 从 Git tracking 移除生成产物，保留本地数据 ✅ 2026-05-02 [auto]
- [x] Step 3: 删除废弃坏入口 `src/trainer/benchmark.py` ✅ 2026-05-02 [auto]
- [x] Step 4: 统一 docs/archive 并清理 graphify cache ✅ 2026-05-02 [auto]
- [x] Step 5: 移除 dashboard 依赖 `fastapi/uvicorn` ✅ 2026-05-02 [auto]
- [x] Step 6: 新增仓库卫生与主入口帮助测试 ✅ 2026-05-02 [auto]

### 模块 13：已确认非核心功能删除 (slimming-plan-v1)
- [x] Step 1: 迁移 `docs_paper/` 到外部写作资料目录并校验 ✅ 2026-05-02 [review]
- [x] Step 2: 删除 `scripts/evaluate.py` ✅ 2026-05-02 [auto]
- [x] Step 3: 删除 `scripts/generate_report.py` ✅ 2026-05-02 [auto]
- [x] Step 4: 删除 callback 扩展机制并简化 `BaseTrainer` ✅ 2026-05-02 [review]
- [x] Step 5: 删除旧 utils 工具并移除 `omegaconf` ✅ 2026-05-02 [review]
- [x] Step 6: 删除旧 buffer wrapper，保留 `src/utils/buffer.py` ✅ 2026-05-02 [review]
- [x] Step 7: 更新 README 与 docs 契约 ✅ 2026-05-02 [auto]
- [x] Step 8: 更新执行报告与进度 ✅ 2026-05-02 [auto]

### 模块 14：formal convergence verification protocol (slimming-plan-v3)
- [x] Step 1: 固化正式收敛验证协议 ✅ 2026-05-04 [auto]
- [x] Step 2: 把 50k baseline 转换为 L1 评估报告 ✅ 2026-05-04 [auto]
- [x] Step 3: 补强收敛判定器 ✅ 2026-05-04 [review]
- [x] Step 4: 建立正式实验矩阵配置 ✅ 2026-05-04 [auto]
- [x] Step 5: 异常事件审计 gate ✅ 2026-05-04 [review]
- [x] Step 6: 单变量修复矩阵，不污染默认配置 ✅ 2026-05-04 [review]
- [ ] Step 7: 执行 L2 candidate validation -> running：`l2_20260504_171744` / PID `26860`，等待 100k x 3 seeds 完成 [review]
- [x] Step 8: L2 失败分流规则与当前 routing 已记录 ✅ 2026-05-04 [auto]
- [ ] Step 9: 执行 L3 formal verification -> blocked：依赖 L2 通过算法与 200k x 5 seeds 长时训练 [review]
- [x] Step 10: 生成 publication gate（当前无 L3 通过算法）✅ 2026-05-04 [auto]
- [x] Step 11: 更新绘图和质量报告绑定 ✅ 2026-05-04 [review]
- [x] Step 12: 更新 docs 状态 ✅ 2026-05-04 [auto]
- [x] Step 13: 最终验收与防误报检查（代码/测试通过；长时训练与全目录指令文本 grep 见 issues）✅ 2026-05-04 [auto]

## 已知问题

- `C:\Users\22003\paper2\rl-mec-dashboard` 本机不存在，外部 dashboard 兼容性需在有该仓库的环境复核。
- 模块 14 v3 L2 实际训练已启动；当前等待 `logs/formal_convergence/l2_20260504_171744.out.log` 与 `results/l2_candidate_convergence_report.json` 更新。
