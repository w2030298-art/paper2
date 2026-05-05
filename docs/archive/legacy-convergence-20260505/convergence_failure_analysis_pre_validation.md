# Convergence Failure Analysis

本报告由 `scripts/analyze_convergence_failures.py` 生成，只做诊断与稳定化计划，不触发 full 17 重跑。

## Failure Matrix

| Algorithm | Overall | Reward | Failed Seeds | Extreme Outliers | Tail Change | Best-Tail Gap | Recommendation |
|-----------|---------|--------|--------------|------------------|-------------|---------------|----------------|
| A3C | diverging | diverging | 0 | 0 | -0.1000 | 0.3558 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| COMA | bad_plateau | bad_plateau | 0 | 0 | -0.0269 | 0.2486 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| GRPO | bad_plateau | converged_good | 0 | 0 | 0.0012 | 0.0690 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| IPPO | catastrophic_outlier | catastrophic_outlier | 0 | 1 | 0.4521 | 1.4972 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |
| IQL | catastrophic_outlier | catastrophic_outlier | 0 | 1 | 0.6387 | 4.8840 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |
| MADDPG | bad_plateau | bad_plateau | 0 | 0 | 0.0080 | 0.4098 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| MAPPO | bad_plateau | bad_plateau | 0 | 0 | 0.0023 | 0.3067 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| MATD3 | diverging | diverging | 0 | 0 | -0.0647 | 2.6364 | 优先检查学习率、target update、replay 或多智能体非平稳问题。 |
| SAC | bad_plateau | bad_plateau | 0 | 0 | -0.0095 | 1.4116 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| TRPO | bad_plateau | converged_good | 0 | 0 | -0.0262 | 0.0565 | 检查是否停在坏平台；进入 targeted rerun 前先确认 best-tail gap。 |
| VDN | catastrophic_outlier | catastrophic_outlier | 0 | 1 | -0.8708 | 1.6700 | 先排查 result/stdout/stderr/train_logs 的 episode-level failure，不直接调参掩盖。 |

## Status Definitions

- `converged_good`: 尾部稳定且接近历史最佳。
- `bad_plateau`: 尾部稳定但明显差于历史最佳或初始表现。
- `oscillating`: 尾部波动超过阈值。
- `diverging`: 尾部方向持续变坏。
- `catastrophic_outlier`: 出现极端异常 reward，先按异常事件排查。

## Next Step

先做 targeted rerun 计划：每个异常算法短程 50k steps 验证异常是否复现，再决定是否扩大到 100k/200k；不直接重跑 full 17。
