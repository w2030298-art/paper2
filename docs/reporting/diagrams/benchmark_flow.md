# Benchmark 流程图（含 `--env` 分支）

> 适用场景: 汇报“对比实验如何保证公平、参数如何生效”

```mermaid
flowchart TD
    A["解析参数 --algorithms --env --seeds"] --> B{"env 是否为 auto?"}

    B -- "是" --> C["按 ALGO_ENV_MAP 为每个算法分配默认环境"]
    B -- "否" --> D["所有算法使用用户指定环境 --env"]

    C --> E["按 (算法, seed, 环境) 循环 benchmark_single"]
    D --> E

    E --> F["make_env(env_name)"]
    F --> G["create_agent(algorithm, env, config)"]
    G --> H["trainer.train() + trainer.evaluate()"]
    H --> I["聚合多 seed 统计均值/方差"]
    I --> J["输出 results/benchmark.json 与 summary"]
```

## 与代码对齐点

- `run_benchmark` 将 `env_name` 传入 `benchmark_single`。
- `benchmark_single` 优先使用传入环境，否则回落到 `ALGO_ENV_MAP`。

## 维护说明

- 每次改 benchmark 参数语义时，先改代码，再改此图中判定节点。
