# 导师汇报文档（可维护版本）

本目录用于沉淀“可复用、可追踪、低维护成本”的进度汇报材料。

## 建议使用方式

1. 每次汇报前先更新 [`status_snapshot.md`](./status_snapshot.md)。
2. 架构或流程变化时，只更新 `diagrams/` 下对应图，不要在多个文档重复画图。
3. 复制 [`mentor_report_template.md`](./mentor_report_template.md) 生成当期汇报稿。
4. 汇报结论中的关键数字，均引用 `results/`、`checkpoints/`、`tests/` 的真实产物。

## 目录说明

- `status_snapshot.md`: 当前状态快照（统一事实来源）。
- `mentor_report_template.md`: 面向导师的周报/月报模板。
- `implementation_matrix.md`: 实现状态矩阵 + 消融映射 + 符号一致性清单。
- `diagrams/system_architecture.md`: 系统结构图（模块关系）。
- `diagrams/training_flow.md`: 训练主流程图。
- `diagrams/benchmark_flow.md`: Benchmark 流程图（含 `--env` 分支）。
- `diagrams/environment_models.md`: 环境模型图（拓扑/时序/状态转移，导师汇报优先）。

## 维护原则

- 先改代码，再改图和结论，避免“图与代码不一致”。
- 图仅保留主干链路，细节放在文字中，避免过度复杂。
- 每次汇报只新增“增量变化”，不要重复抄历史内容。
