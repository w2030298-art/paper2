# 收敛曲线质量管线说明

本文档说明 `scripts/plot_results.py` 中 convergence quality pipeline 的输出含义和使用边界。

## raw diagnostic 图与 clean publication 图

- `convergence_curves_raw_all.{fmt}` 是诊断图。它保留原始异常值、`nan`、`inf` 转换后的缺失情况和 failed seed 的可见证据，用于排查训练或评估数据问题。
- `convergence_curves_clean_all.{fmt}` 是论文/报告候选图。它默认使用 winsorize 清洗、按 seed 插值对齐，并以 median 作为主线、q25-q75 作为区间。
- 推荐论文和正式报告引用 `convergence_curves_clean_all.{fmt}`。
- debug 和回归排查使用 `convergence_curves_raw_all.{fmt}`。

## 为什么不能直接删除异常算法

异常算法可能暴露训练不稳定、评估指标方向错误、日志 schema 不一致或 seed 失败等问题。直接删除会掩盖实验质量问题，也会让不同报告之间不可复现。

本项目不硬编码排除 `IQL`、`QMIX`、`VDN` 或其他任何算法。是否异常由数据质量规则判定，并写入 `convergence_quality_report.json` 与 `convergence_quality_report.md`。

## 质量报告字段

`convergence_quality_report.json` 每条记录对应一个 algorithm、seed、metric 组合，核心字段如下：

| 字段 | 含义 |
|------|------|
| `algorithm` | 算法名称 |
| `seed` | seed 标识 |
| `metric` | `reward`、`latency`、`energy` 或 `comm_score` |
| `run_status` | `success`、`failed` 或 `excluded` |
| `failure_reason` | failed seed 的错误原因 |
| `raw_count` | 原始序列长度 |
| `finite_count` | 有效有限数值数量 |
| `nan_count` | 非数值、`inf`、`-inf` 或缺失数量 |
| `raw_min` / `raw_max` | 清洗前有效值范围 |
| `clean_min` / `clean_max` | 清洗后有效值范围 |
| `outlier_count` | 被裁剪或 mask 的点数 |
| `outlier_ratio` | 异常点占有效值比例 |
| `outlier_policy` | `none`、`winsorize` 或 `iqr-mask` |
| `severe_outlier` | 是否触发严重异常标记 |
| `skipped_from_clean_plot` | 是否从 clean 图跳过 |
| `excluded_from_clean_plot` | 是否由 CLI exclude 规则排除 |

`convergence_quality_report.md` 是同一内容的人工阅读版本，便于快速检查 warning 和 skip 原因。

## 指标方向与收敛判定

- `reward` 与 `comm_score` 越高越好。
- `latency` 与 `energy` 越低越好。
- 收敛状态基于尾部窗口相对变化和波动率计算，可能为 `converged`、`improving`、`degrading`、`unstable` 或 `insufficient`。

legend 使用单条算法记录，例如：

```text
GRPO [converged]
IQL [unstable] ⚠
```

其中 `⚠` 表示该算法在对应 metric 上存在严重异常，具体原因以 quality report 为准。
