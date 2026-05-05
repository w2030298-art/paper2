# Iter Stage 1 — Convergence Validation Reassessment

## 结论

不能把当前测试实验数据作为“算法改进已经保证收敛”的证明。当前数据只达到 `L1_test_run_only`：能验证 benchmark 管线可运行、异常是否复现、短程趋势是否改善；不能给出理论保证，也不能给出最终工程收敛验收。

要让下一步工作“直接验证算法改进是否达到工程收敛”，应把目标改写为：在固定环境、固定指标、固定 seeds 和固定步数预算下，通过分层门禁判定 `verified_converged_under_protocol`。如果目标是数学意义的“保证收敛”，则必须新增理论证明任务，实验数据本身无法证明。

## 输入数据

- baseline JSON: `convergence_validation_baseline_50k.json`
- stdout log: `benchmark_20260503_005009.log`
- stderr log: `benchmark_20260503_005009.err.log`
- run id: `20260503_005009`
- benchmark completed successfully: `True`
- stderr diagnostic hits: `{'ERROR': 0, 'Exception': 0, 'Traceback': 0, 'nan': 0, 'inf': 0, 'failed': 0}`

## 50k 单 seed 评估结果

| Algorithm | Status | Seeds | Steps | Eval Points | Min Reward | Final Reward | Tail Rel Change | Best-Tail Gap | Tail Volatility | Catastrophic Points |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IPPO | catastrophic_outlier | 1 | 51200 | 12 | -657.70 | -54.85 | -0.018 | 0.791 | 0.313 | 4 |
| IQL | catastrophic_outlier | 1 | 51200 | 12 | -6948.06 | -378.78 | 0.536 | 5.511 | 0.558 | 10 |
| MADDPG | catastrophic_outlier | 1 | 51200 | 12 | -4414.05 | -55.67 | -1.630 | 1.242 | 0.444 | 2 |
| VDN | catastrophic_outlier | 1 | 51200 | 12 | -31449.95 | -340.19 | -5.158 | 1.947 | 0.790 | 7 |
| A3C | diverging | 1 | 50000 | 5 | -21.91 | -21.91 | -0.210 | 0.105 | 0.095 | 0 |
| MATD3 | oscillating | 1 | 51200 | 12 | -74.74 | -38.04 | -0.897 | 1.073 | 0.249 | 0 |
| GRPO | bad_plateau | 1 | 51200 | 6 | -5.28 | -4.95 | 0.029 | 0.111 | 0.027 | 0 |
| SAC | bad_plateau | 1 | 51200 | 6 | -38.21 | -37.41 | -0.008 | 1.568 | 0.012 | 0 |
| COMA | L1_converged_candidate | 1 | 51200 | 12 | -24.37 | -14.80 | -0.003 | 0.097 | 0.016 | 0 |
| MAPPO | L1_converged_candidate | 1 | 51200 | 6 | -24.16 | -15.68 | -0.019 | 0.077 | 0.008 | 0 |
| TRPO | L1_converged_candidate | 1 | 51200 | 6 | -14.86 | -14.78 | -0.020 | 0.037 | 0.027 | 0 |

## 影响范围分析

### 直接受影响模块

- `scripts/analyze_convergence_failures.py`：需要从诊断脚本升级为验证判定器，输出 evidence level、pass/fail reason、per-seed gate。
- `scripts/benchmark.py`：需要支持正式验证矩阵，包括 seeds、steps、override id、run id、config hash、artifact manifest。
- `scripts/plot_results.py`：需要保证 raw/clean 图和 quality report 与验证结论绑定，不允许只凭 clean 图判定。
- `configs/stability_overrides.yaml` / `configs/targeted_stability_reruns.yaml`：必须保持默认不启用；只允许显式验证配置。
- `docs/report.md` / `docs/progress.md` / `docs/issues.md`：必须记录每个算法的 evidence level，不能写模糊的“已收敛”。

### 不受影响模块

- 模块 12-13 仓库瘦身结果不应重跑。
- `src/comm/`、`src/experiment/`、dashboard 文件协议不应在本轮改动。

## 对“保证收敛”的重新定义

| 目标 | 是否可由当前数据直接证明 | 下一步可达成方式 |
|---|---:|---|
| 数学/理论保证收敛 | 否 | 新增理论证明任务：列出 MDP/函数逼近/步长/探索/噪声假设，并证明或引用对应收敛定理。 |
| 工程协议下验证收敛 | 否，当前只是 L1 | 设计 `verified_converged_under_protocol`：100k+200k、多 seed、raw/clean/quality report、无 catastrophic、无指标退化。 |
| 调试验证改进方向 | 是，部分可用 | 当前 50k 单 seed 可筛出仍失败算法和候选算法。 |

## 下一步工作建议

新增模块 14 应从“跑更多实验”改为“收敛保证验证协议”。核心任务如下：

1. 先把 50k 单 seed 结果固化为 `L1`，禁止写 `verified_converged`。
2. 对 `catastrophic_outlier` 算法先排查异常事件：IQL、VDN、IPPO、MADDPG。
3. 对 `bad_plateau` / `diverging` 算法先做单变量修复验证，不一次性开启全部 override。
4. 对候选算法执行正式门禁：`seeds=[42,43,44,45,46]`，`100k -> 200k`，保留 raw/clean/quality report。
5. 只有通过正式门禁的算法才能标记 `verified_converged_under_protocol`。
6. 如果用户要求数学意义的“保证收敛”，必须新增理论证明模块；实验结果只能作为佐证。

## 阶段 1 检查点

建议进入阶段 3：生成新版完整 `docs/inbox/plan.md`，版本命名 `slimming-plan-v3`，追加模块 14：`formal convergence verification protocol`。

计划生成前需要用户确认：目标采用“工程协议下验证收敛”，还是追加“理论收敛证明”任务。
