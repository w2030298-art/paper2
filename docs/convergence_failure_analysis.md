# Convergence Failure Analysis

本报告由 `scripts/analyze_convergence_failures.py` 生成，只做诊断与 formal convergence protocol 输入，不触发 full 17 重跑。

## Failure Matrix

| Algorithm | Overall | Reward | Failed Seeds | Extreme Outliers | Tail Change | Best-Tail Gap | Recommendation |
|-----------|---------|--------|--------------|------------------|-------------|---------------|----------------|
| A3C | diverging | diverging | 0 | 0 | -0.0568 | 0.1049 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| COMA | diverging | bad_plateau | 0 | 0 | -0.0326 | 0.1039 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| GRPO | diverging | bad_plateau | 0 | 0 | 0.0136 | 0.1124 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| IPPO | catastrophic_outlier | catastrophic_outlier | 0 | 1 | 0.0045 | 0.6034 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |
| IQL | catastrophic_outlier | catastrophic_outlier | 0 | 1 | 0.9110 | 3.0488 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |
| MADDPG | catastrophic_outlier | catastrophic_outlier | 0 | 1 | -0.9135 | 1.6714 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |
| MAPPO | diverging | converged_good | 0 | 0 | -0.0378 | 0.0830 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| MATD3 | diverging | diverging | 0 | 0 | -0.7995 | 1.4117 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| SAC | bad_plateau | bad_plateau | 0 | 0 | -0.0189 | 1.5841 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| TRPO | oscillating | converged_good | 0 | 0 | 0.0244 | 0.0330 | 优先检查方差、advantage/critic 诊断和 eval episode 数量。 |
| VDN | catastrophic_outlier | catastrophic_outlier | 0 | 1 | -0.3965 | 2.9147 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |

## Status Definitions

- `converged_good`: 尾部稳定且接近历史最佳。
- `bad_plateau`: 尾部稳定但明显差于历史最佳或初始表现。
- `oscillating`: 尾部波动超过阈值。
- `diverging`: 尾部方向持续变坏。
- `catastrophic_outlier`: 出现极端异常 reward，先按异常事件排查。

## Next Step

当前 50k single-seed 数据只进入 L1 预筛。下一步是按 `configs/formal_convergence_matrix.yaml` 执行 L2 100k multi-seed candidate validation；只有 L2 通过算法才能进入 L3 200k formal validation，未通过 L3 的算法不得进入论文主图或主结论。
