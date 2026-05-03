# docs/inbox/plan.md：paper2 收敛曲线质量管线定向改进 + docs 目录整理

## 元信息

- 项目：`paper2`
- 版本：`v3`
- 基线：
  - `docs/plan.md`：VSCode 一键化实验入口全量改造，模块 1-6
  - `docs/plan-patch.md`：算法评估指标体系优化，模块 7-9
  - GitHub 仓库：`w2030298-art/paper2`
  - 分支：`main`
- 变更类型：`patch`
- 本轮目标：把当前“能画曲线”升级为“可解释、可诊断、可用于论文/报告的收敛质量管线”，并顺手整理 `docs/` 目录契约
- 总模块数：11
- 既有步骤数：51（模块 1-9，作为当前仓库基线锁定）
- 本轮新增步骤数：14
- 预计步骤总数：65
- 建议开发顺序：模块 10 Step 1 → Step 10 → 模块 11 Step 1 → Step 4
- 创建日期：2026-05-02
- 最后更新：2026-05-02

### 变更记录

| 版本 | 日期 | 变更摘要 |
|------|------|---------|
| v1 | 2026-04-23 | 模块 1-9：VSCode 实验入口、评估指标、收敛曲线、Composite Score |
| v2 | 2026-05-02 | 新增模块 10：Convergence Quality Pipeline，修复异常值压缩、seed 聚合、收敛判定、重复输出和报告可信度问题 |
| v3 | 2026-05-02 | 追加模块 11：整理 `docs/` 目录，建立活跃文档/归档文档/执行 inbox 的边界 |

---

## Status

> 任何 agent 读到此区块即可恢复完整上下文。

- 当前阶段：模块 10 Step 1
- 整体进度：51 / 65 步骤完成（模块 1-9 视为当前仓库基线，模块 10-11 尚未执行）
- 状态：变更后待执行
- 阻塞项：
  - `docs/report.md` 当前仓库未发现；无法恢复执行端精确完成记录。
  - 本轮可直接执行；若真实 benchmark JSON schema 与当前测试 mock 不一致，执行端在 Step 1/8 内补齐兼容层。

### Last Iteration Summary

Review 结论：当前收敛图不是单纯“画得不好”，而是数据质量管线缺失。主要问题包括：

1. IQL/QMIX/VDN 等异常值把 reward y 轴拉爆，正常算法曲线被压扁。
2. 当前图没有区分 raw diagnostic 和 publication figure。
3. 多 seed 数据没有稳健聚合；均值容易被异常 seed 支配。
4. 没有对 latency/energy 的“越低越好”方向做收敛判定。
5. `convergence_all.png` 与 `convergence_curves.png` 内容重复，说明输出命名/保存职责不清。
6. 测试只覆盖“不崩溃 + 文件存在”，不能防止劣质曲线再次出现。

### Pending Decisions

无。本轮不引入新算法、不改训练超参、不改 reward 公式，只改收敛曲线数据清洗、聚合、诊断与报告集成。

---

## 继承基线

### 模块 1-6：VSCode 一键化实验入口全量改造

- 状态：`[DONE / BASELINE LOCKED]`
- 来源：`docs/plan.md`
- 本轮处理：不重做、不改入口语义；仅当 Step 10 需要文档补充时可追加说明。
- scope：`auto`

### 模块 7：Reward + comm_score 权重修正

- 状态：`[DONE / BASELINE LOCKED]`
- 来源：`docs/plan-patch.md`
- 本轮处理：不修改 reward、communication cost、comm_score 公式。
- scope：`auto`

### 模块 8：收敛曲线数据收集与可视化

- 状态：`[DONE / NEEDS TARGETED PATCH]`
- 来源：`docs/plan-patch.md`
- 本轮处理：保留已实现的 `convergence_by_seed` 输入约定，但替换当前绘图质量管线。
- scope：`review`

### 模块 9：Composite Score 综合评分体系

- 状态：`[DONE / BASELINE LOCKED]`
- 来源：`docs/plan-patch.md`
- 本轮处理：不重写评分体系；只在报告中可引用 convergence quality warning。
- scope：`auto`

---

# 模块 10：Convergence Quality Pipeline

## 概述

- 职责：将 `scripts/plot_results.py::plot_convergence_curves()` 从直接画 raw series 改造成完整质量管线：
  1. schema 归一化
  2. 数据清洗
  3. 多 seed 稳健聚合
  4. 指标方向感知收敛判定
  5. raw diagnostic 与 publication figure 分离
  6. 质量报告输出
  7. 回归测试覆盖异常值与重复输出
- 前置依赖：模块 8 已有 `convergence_by_seed` 或 `train_logs.json` 兼容读取逻辑。
- 不做事项：
  - 不硬编码排除 `IQL/QMIX/VDN`。
  - 不删除 raw 图。
  - 不用视觉裁剪掩盖失败算法；所有裁剪/排除必须写入 quality report。
  - 不修改训练算法网络、环境物理模型、reward 公式、Composite Score 权重。

---

## Step 1：收敛数据 schema 归一化

- **scope：auto**
- 文件：`scripts/plot_results.py`
- 操作：
  - 新增常量 `CONVERGENCE_METRIC_SPECS`，统一定义四类曲线：
    ```python
    CONVERGENCE_METRIC_SPECS = {
        "reward": {
            "title": "Reward",
            "ylabel": "Reward",
            "aliases": ["eval/reward_mean", "eval_eval/reward_mean"],
            "higher_is_better": True,
        },
        "latency": {
            "title": "Latency / Task",
            "ylabel": "Latency / Task",
            "aliases": ["eval/latency_mean", "eval_eval/latency_mean"],
            "higher_is_better": False,
        },
        "energy": {
            "title": "Energy / Task",
            "ylabel": "Energy / Task",
            "aliases": ["eval/energy_mean", "eval_eval/energy_mean"],
            "higher_is_better": False,
        },
        "comm_score": {
            "title": "Comm Score",
            "ylabel": "Comm Score",
            "aliases": ["eval/comm_score", "eval_eval/comm_score"],
            "higher_is_better": True,
        },
    }
    ```
  - 新增函数：
    ```python
    def _series_to_float_array(series: Any) -> np.ndarray:
        ...
    ```
    要求：
    - 将 list/tuple/np.ndarray 转为 `np.ndarray(dtype=float)`。
    - 非数值、`inf`、`-inf` 转为 `np.nan`。
    - 空序列返回空数组。
  - 新增函数：
    ```python
    def _pick_metric_series(seed_data: dict, spec: dict) -> np.ndarray:
        ...
    ```
    要求：
    - 按 `spec["aliases"]` 顺序查找。
    - 返回 `_series_to_float_array(...)` 结果。
  - 新增函数：
    ```python
    def _build_timestep_axis(seed_data: dict, length: int) -> np.ndarray:
        ...
    ```
    要求：
    - 优先使用 `seed_data["timesteps"]`。
    - 否则使用 `eval_interval * np.arange(length)`。
    - 若 `eval_interval` 缺失，默认 `1000`。
    - 返回长度必须与 series 对齐。
  - 修改 `plot_convergence_curves()`，先调用上述归一化函数，不再在绘图循环内散落硬编码 key。
- 验证：
  ```bash
  python -c "from scripts.plot_results import CONVERGENCE_METRIC_SPECS, _series_to_float_array, _pick_metric_series, _build_timestep_axis; print('OK')"
  pytest tests/test_convergence_plot.py tests/test_docs_contract.py -q
  ```

---

## Step 2：新增质量诊断与数据清洗层

- **scope：review**
- 文件：`scripts/plot_results.py`
- 操作：
  - 新增函数：
    ```python
    def _sanitize_metric_series(
        values: np.ndarray,
        *,
        outlier_policy: str = "winsorize",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> tuple[np.ndarray, dict]:
        ...
    ```
  - 支持策略：
    - `none`：不裁剪，只统计质量。
    - `winsorize`：按分位数裁剪到 `[q01, q99]`。
    - `iqr-mask`：超出 `[Q1 - 3*IQR, Q3 + 3*IQR]` 的点置为 `np.nan`。
  - 返回：
    - 清洗后数组。
    - 质量统计 dict，字段必须包括：
      - `raw_count`
      - `finite_count`
      - `nan_count`
      - `raw_min`
      - `raw_max`
      - `clean_min`
      - `clean_max`
      - `outlier_count`
      - `outlier_ratio`
      - `outlier_policy`
  - 新增函数：
    ```python
    def _write_convergence_quality_report(report: list[dict], output_dir: Path) -> None:
        ...
    ```
    输出：
    - `convergence_quality_report.json`
    - `convergence_quality_report.md`
  - 每个算法、每个 seed、每个 metric 都必须写一条 quality record。
- 验证：
  ```bash
  python -c "from scripts.plot_results import _sanitize_metric_series, _write_convergence_quality_report; print('OK')"
  pytest tests/test_convergence_plot.py::TestConvergenceCurves -q
  ```

---

## Step 3：实现多 seed 稳健聚合

- **scope：review**
- 文件：`scripts/plot_results.py`
- 操作：
  - 新增函数：
    ```python
    def _aggregate_metric_by_seed(
        seed_curves: list[tuple[np.ndarray, np.ndarray]],
        *,
        aggregate: str = "median",
    ) -> dict:
        ...
    ```
  - 输入：`[(x_seed_1, y_seed_1), (x_seed_2, y_seed_2), ...]`
  - 处理规则：
    - 丢弃有效点少于 2 的 seed。
    - 构造 common x-grid：
      - 起点：所有 seed 有效 x 的最小共同起点。
      - 终点：所有 seed 有效 x 的最大共同终点。
      - 步长：所有 seed 正向 x 差分的中位数；若无法计算，使用 `1000`。
    - 用 `np.interp` 对每个 seed 插值到 common grid。
    - 默认输出 `median/q25/q75`，同时保留 `mean/std/n_seeds`。
  - 返回字段：
    ```python
    {
        "x": np.ndarray,
        "median": np.ndarray,
        "q25": np.ndarray,
        "q75": np.ndarray,
        "mean": np.ndarray,
        "std": np.ndarray,
        "n_seeds": int,
    }
    ```
  - `plot_convergence_curves()` 默认使用 `median` 画主线，用 `q25-q75` 画带状区间。
  - 禁止默认使用 mean/std 作为 publication 主线。
- 验证：
  ```bash
  python -c "from scripts.plot_results import _aggregate_metric_by_seed; print('OK')"
  pytest tests/test_convergence_plot.py -q
  ```

---

## Step 4：实现方向感知的收敛判定

- **scope：review**
- 文件：`scripts/plot_results.py`
- 操作：
  - 新增函数：
    ```python
    def compute_convergence_status(
        x: np.ndarray,
        y: np.ndarray,
        *,
        higher_is_better: bool,
        window_fraction: float = 0.10,
        rel_change_threshold: float = 0.05,
        volatility_threshold: float = 0.10,
        min_points: int = 5,
    ) -> dict:
        ...
    ```
  - 返回字段：
    ```python
    {
        "status": "converged" | "improving" | "degrading" | "unstable" | "insufficient",
        "tail_mean": float | None,
        "prev_tail_mean": float | None,
        "relative_change": float | None,
        "tail_volatility": float | None,
        "higher_is_better": bool,
    }
    ```
  - 判定规则：
    - 有效点 `< min_points`：`insufficient`
    - 最后 10% 窗口相对变化 `< 5%` 且波动 `< 10%`：`converged`
    - 对 reward/comm_score，tail 变好：`improving`，变差：`degrading`
    - 对 latency/energy，tail 变低：`improving`，变高：`degrading`
    - 波动超阈值：`unstable`
  - Legend 只显示一次算法名，格式：
    ```text
    GRPO [converged]
    IQL [unstable]
    ```
    不允许当前实现里用额外空 plot 造成重复 legend。
- 验证：
  ```bash
  python -c "from scripts.plot_results import compute_convergence_status; print('OK')"
  pytest tests/test_convergence_plot.py -q
  ```

---

## Step 5：拆分 diagnostic raw 图与 publication clean 图

- **scope：review**
- 文件：`scripts/plot_results.py`
- 操作：
  - 修改函数签名：
    ```python
    def plot_convergence_curves(
        results: list[dict],
        output_dir: Path,
        fmt: str = "png",
        *,
        mode: str = "both",
        aggregate: str = "median",
        outlier_policy: str = "winsorize",
        include_algorithms: set[str] | None = None,
        exclude_algorithms: set[str] | None = None,
    ) -> None:
        ...
    ```
  - `mode` 支持：
    - `raw`：只输出 raw diagnostic。
    - `clean`：只输出 cleaned publication。
    - `both`：同时输出 raw + clean。
  - 输出文件固定为：
    - `convergence_curves_raw_all.{fmt}`
    - `convergence_curves_clean_all.{fmt}`
    - `convergence_quality_report.json`
    - `convergence_quality_report.md`
  - 删除旧的重复保存路径逻辑。
  - 保留兼容：如果调用方不传新参数，默认输出 `raw_all + clean_all + quality_report`。
- 验证：
  ```bash
  pytest tests/test_convergence_plot.py -q
  python scripts/plot_results.py --help
  ```

---

## Step 6：实现稳健坐标轴策略与异常算法标注

- **scope：review**
- 文件：`scripts/plot_results.py`
- 操作：
  - 新增函数：
    ```python
    def _robust_ylim(values: list[np.ndarray], *, lower: float = 0.01, upper: float = 0.99) -> tuple[float, float] | None:
        ...
    ```
  - clean 图使用 robust y-limit：
    - 汇总该 metric 下所有 clean series。
    - 使用 q01/q99 作为候选范围。
    - 添加 5% padding。
    - 若有效值少于 3 个，使用 Matplotlib 默认范围。
  - raw 图不启用 robust y-limit，必须展示原始异常。
  - 对严重异常算法加标注：
    - 若某算法某 metric 的 `outlier_ratio >= 0.20` 或 `raw_max/raw_min` 跨度异常，legend 后追加 `⚠`。
    - 具体异常原因写入 quality report，图上只做轻量标识。
  - 不允许静默排除算法；若 `exclude_algorithms` 生效，必须写入 report：
    ```json
    {"algorithm": "IQL", "excluded_from_clean_plot": true, "reason": "..."}
    ```
- 验证：
  ```bash
  pytest tests/test_convergence_plot.py -q
  ```

---

## Step 7：补齐 CLI 参数

- **scope：auto**
- 文件：`scripts/plot_results.py`
- 操作：
  - 在 `main()` 中新增参数：
    ```python
    parser.add_argument("--convergence-mode", choices=["raw", "clean", "both"], default="both")
    parser.add_argument("--convergence-aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--convergence-outlier-policy", choices=["none", "winsorize", "iqr-mask"], default="winsorize")
    parser.add_argument("--convergence-include", nargs="*", default=None)
    parser.add_argument("--convergence-exclude", nargs="*", default=None)
    ```
  - 调用：
    ```python
    plot_convergence_curves(
        results,
        output_dir,
        args.format,
        mode=args.convergence_mode,
        aggregate=args.convergence_aggregate,
        outlier_policy=args.convergence_outlier_policy,
        include_algorithms=set(args.convergence_include) if args.convergence_include else None,
        exclude_algorithms=set(args.convergence_exclude) if args.convergence_exclude else None,
    )
    ```
- 验证：
  ```bash
  python scripts/plot_results.py --help | grep convergence
  pytest tests/test_convergence_plot.py -q
  ```

---

## Step 8：补齐 benchmark convergence schema 元数据

- **scope：review**
- 文件：`scripts/benchmark.py`
- 操作：
  - 在 `benchmark_single()` 生成 `convergence_data` 时补齐以下字段：
    ```python
    convergence_data["schema_version"] = 2
    convergence_data["seed"] = seed
    convergence_data["algorithm"] = name
    convergence_data["run_status"] = "success"
    convergence_data["failure_reason"] = None
    ```
  - 若训练失败但外层仍保留 result，应写：
    ```python
    convergence_data["run_status"] = "failed"
    convergence_data["failure_reason"] = str(exc)
    ```
  - `run_benchmark()` 汇总 `avg["convergence_by_seed"]` 时保留这些元数据。
  - `plot_results.py` 中读取时：
    - diagnostic raw 图可显示 failed seed。
    - clean 图默认跳过 `run_status != "success"` 的 seed，并在 quality report 中记录。
- 验证：
  ```bash
  python scripts/benchmark.py --help
  python -c "import scripts.benchmark; print('OK')"
  pytest tests/test_convergence_plot.py -q
  ```

---

## Step 9：强化收敛曲线回归测试

- **scope：auto**
- 文件：`tests/test_convergence_plot.py`
- 操作：
  - 保留现有两个 smoke tests。
  - 新增测试：
    ```python
    def test_quality_report_flags_extreme_outlier(tmp_path): ...
    def test_clean_plot_uses_robust_axis_without_dropping_raw(tmp_path): ...
    def test_output_names_are_not_duplicate(tmp_path): ...
    def test_direction_aware_convergence_for_lower_is_better_metric(): ...
    def test_seed_aggregation_interpolates_mismatched_timesteps(): ...
    def test_failed_seed_excluded_from_clean_but_reported(tmp_path): ...
    def test_nan_and_inf_do_not_crash_plotting(tmp_path): ...
    ```
  - mock 数据必须覆盖：
    - reward 中单点 `-65000`
    - latency 中 `inf`
    - energy 中 `nan`
    - 三个 seed eval interval 不一致
    - 一个 failed seed
    - 正常算法与异常算法同时存在
  - 每个测试只生成临时文件，不依赖真实 `results/` 或 `experiments/`。
- 验证：
  ```bash
  pytest tests/test_convergence_plot.py -v
  ```

---

## Step 10：报告与文档集成

- **scope：auto**
- 文件：
  - `scripts/generate_report.py`
  - `docs/convergence_plot_quality.md`
- 操作：
  - 新建文档 `docs/convergence_plot_quality.md`，必须包含：
    - raw diagnostic 图与 clean publication 图的区别。
    - 为什么异常算法不能直接删除。
    - `convergence_quality_report.json/md` 字段说明。
    - 推荐论文/报告使用 `convergence_curves_clean_all.{fmt}`。
    - debug 使用 `convergence_curves_raw_all.{fmt}`。
  - 修改 `scripts/generate_report.py`：
    - 如果存在 `figures/convergence_curves_clean_all.png` 或 `.pdf`，优先引用 clean 图。
    - 如果存在 `convergence_quality_report.md`，在报告中新增 `## Convergence Quality Notes`。
    - 对存在严重异常的算法输出 warning 表。
  - 不删除旧报告章节；只追加质量说明。
- 验证：
  ```bash
  python scripts/generate_report.py --help
  test -f docs/convergence_plot_quality.md
  grep -n "convergence_curves_clean_all" docs/convergence_plot_quality.md
  grep -n "convergence_quality_report" docs/convergence_plot_quality.md
  ```

---

# 模块 11：docs 目录整理

## 概述

- 职责：整理 `docs/` 目录结构，建立“活跃执行文档 / 历史归档 / 参考资料 / inbox patch”的边界，避免后续执行端读错旧计划或备份文件。
- 前置依赖：模块 10 Step 10（先产出新的收敛质量文档，再整理目录索引）
- 预计步骤数：4
- 本模块原则：
  - 只整理文档，不改源码逻辑。
  - 不删除历史文档，只移动到 `docs/archive/`。
  - 不移动当前有效文件：`docs/plan.md`、`docs/plan-patch.md`、`docs/inbox/plan.md`、`docs/report.md`、`docs/progress.md`、`docs/issues.md`。

## Step 1：建立 docs 目录索引

- **scope: auto**
- 文件：
  - 新建或更新：`docs/README.md`
- 操作：
  - 在 `docs/README.md` 中写清楚当前目录契约：
    ```text
    docs/
    ├── README.md
    ├── plan.md
    ├── plan-patch.md
    ├── progress.md
    ├── issues.md
    ├── report.md
    ├── vscode_experiment_usage.md
    ├── convergence_plot_quality.md
    ├── inbox/
    ├── references/
    └── archive/
    ```
  - 每个条目必须说明用途：
    - `plan.md`：主开发计划
    - `plan-patch.md`：当前已合入或待合入的增量计划基线
    - `inbox/`：执行端 merge-back 输入区
    - `archive/`：历史计划、备份、过期 patch
    - `references/`：实现参考和调研资料
    - `convergence_plot_quality.md`：本轮收敛曲线质量规则说明
  - 明确写入规则：执行端只从 `docs/plan.md` 与 `docs/inbox/plan.md` 恢复计划，不从 `archive/` 恢复。
- 验证：
  ```bash
  test -f docs/README.md
  grep -n "docs/inbox/plan.md" docs/README.md
  grep -n "archive" docs/README.md
  ```

## Step 2：创建标准子目录并归档明显历史文件

- **scope: auto**
- 文件/目录：
  - `docs/archive/`
  - `docs/references/`
  - `docs/inbox/`
- 操作：
  - 确保以下目录存在：
    ```bash
    mkdir -p docs/archive docs/references docs/inbox
    ```
  - 若存在以下文件，移动到 `docs/archive/`：
    - `docs/plan-patch-full17-backup.md`
    - `docs/plan-patch-*-backup.md`
    - `docs/*backup*.md`
  - 不移动：
    - `docs/plan.md`
    - `docs/plan-patch.md`
    - `docs/inbox/plan.md`
    - `docs/report.md`
    - `docs/progress.md`
    - `docs/issues.md`
    - `docs/convergence_plot_quality.md`
  - 移动前后都要保持 Git 可追踪，不使用删除重建代替移动。
- 验证：
  ```bash
  test -d docs/archive
  test -d docs/references
  test -d docs/inbox
  find docs -maxdepth 1 -iname "*backup*.md" | wc -l
  ```

## Step 3：补齐执行端契约文档占位

- **scope: auto**
- 文件：
  - `docs/report.md`
  - `docs/progress.md`
  - `docs/issues.md`
- 操作：
  - 若 `docs/report.md` 不存在，创建最小模板：
    ```markdown
    # Execution Report

    ## STATUS: PENDING

    > 上次更新: 2026-05-02 | plan.md 版本:v3

    ## Last Execution
    尚未执行模块 10-11。

    ## Completed
    无

    ## In Review
    无

    ## Blocked
    无

    ## Discovered Issues
    无

    ## Recommendations
    无
    ```
  - 若 `docs/progress.md` 不存在，创建最小进度表，至少包含模块 10 和模块 11 的 14 个未完成 Step。
  - 若 `docs/issues.md` 不存在，创建标题和空白问题列表。
  - 若文件已存在，不覆盖内容，只追加缺失的模块 10/11 状态信息。
- 验证：
  ```bash
  test -f docs/report.md
  test -f docs/progress.md
  test -f docs/issues.md
  grep -n "模块 10" docs/progress.md
  grep -n "模块 11" docs/progress.md
  ```

## Step 4：增加 docs 契约回归测试

- **scope: auto**
- 文件：
  - 新建或更新：`tests/test_docs_contract.py`
- 测试函数：
  - `test_docs_required_files_exist()`
  - `test_docs_required_directories_exist()`
  - `test_docs_readme_defines_active_and_archive_contract()`
  - `test_docs_root_has_no_backup_markdown_files()`
- 断言要求：
  - `docs/README.md`、`docs/plan.md`、`docs/report.md`、`docs/progress.md`、`docs/issues.md` 存在。
  - `docs/inbox/`、`docs/references/`、`docs/archive/` 存在。
  - `docs/README.md` 包含 `docs/inbox/plan.md`、`archive`、`references`。
  - `docs/` 根目录不能存在文件名包含 `backup` 的 `.md` 文件。
- 验证：
  ```bash
  pytest tests/test_docs_contract.py -q
  ```

---

# 验收标准

## 单元测试

```bash
pytest tests/test_convergence_plot.py -v
```

必须通过。

## 导入检查

```bash
python -c "from scripts.plot_results import plot_convergence_curves, compute_convergence_status, _aggregate_metric_by_seed, _sanitize_metric_series; print('OK')"
python -c "import scripts.benchmark; print('OK')"
```

必须通过。

## CLI 检查

```bash
python scripts/plot_results.py --help | grep convergence
python scripts/benchmark.py --help
python scripts/generate_report.py --help
```

必须通过。

## 输出文件检查

使用 mock benchmark JSON 或最近一次真实 JSON 运行：

```bash
python scripts/plot_results.py --input results/benchmark.json --output figures --format png --convergence-mode both
```

期望至少生成：

```text
figures/convergence_curves_raw_all.png
figures/convergence_curves_clean_all.png
figures/convergence_quality_report.json
figures/convergence_quality_report.md
```

## 回归风险检查

- `convergence_curves_raw_all.png` 必须保留异常值，不做视觉美化。
- `convergence_curves_clean_all.png` 必须不被单个极端 outlier 压扁。
- `convergence_quality_report.md` 必须记录被裁剪、被 mask、被跳过的 seed/算法。
- `tests/test_convergence_plot.py` 必须能防止“只生成空图也算通过”的假阳性。

---

# 执行端注意事项

1. 先改测试，再改实现；本轮目标是防止劣质图回归。
2. 不要硬编码 `IQL/QMIX/VDN` 为坏算法；坏数据由 quality rules 判定。
3. raw 图是诊断证据，clean 图是论文/报告候选图，两者都必须保留。
4. 不要在 Step 2/6 中静默裁剪；所有处理必须写入 quality report。
5. 若真实 JSON 字段与 mock 不一致，在 Step 1 schema 兼容层内解决，不要在绘图主循环里堆 if。
6. docs 整理只做目录契约和历史文件归档，不顺手改正文档内容语义。
