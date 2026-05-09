# paper2 Mainline-A full-17 migration plan v4.3

## 元信息

- 项目：`paper2`
- 计划版本：`system-model-overhaul-v4.3`
- 变更类型：`patch / docs replacement`
- 日期：2026-05-06
- 上一版本：`system-model-overhaul-v4.2`
- 当前主线：Mainline-A
- 默认训练环境：`mainline-a`
- fallback：`legacy` 仅显式指定；禁止 silent fallback。
- 目标：
  1. 用 slim docs 替换旧长文档。
  2. 统一 `plan.md` / `progress.md` / `report.md` 状态源。
  3. 将 full-17 benchmark、experiment manager、direct benchmark、VSCode 入口默认迁移到 Mainline-A。
  4. 保留旧环境为显式 fallback。
  5. 建立 full-17 dry-run → quick smoke → full benchmark 门禁。

## Status

> 执行端读到此区块即可恢复当前上下文。

- 当前阶段：模块 25 Mainline-A full-17 migration completed；模块 26 execution gate pending。
- 当前状态：`NEEDS_REVIEW`
- v4.2 输入状态：Mainline-A N0/N1/N2/N3 final review 已按 `ACCEPTED_WITH_BOUNDARIES` 关闭。
- v4.3 变更原因：
  - 当前 GitHub `docs/plan.md` 与 `docs/report.md` 状态口径不一致。
  - 旧 docs 包含大量历史流水账，影响恢复上下文效率。
  - full-17 默认链路已迁移到 Mainline-A environment profile。
- 阻塞项：dashboard 兼容性为外部复核项，不阻塞本迁移。
- 决策：
  - 默认 profile：`mainline-a`
  - fallback profile：`legacy`
  - Mainline-A 初始化失败必须 fail fast；错误提示可用 `--environment-profile legacy` 复现旧环境。

### Last Iteration Summary

- v4.2 已关闭 Mainline-A experiment final review。
- N0/N1/N2/N3 证据等级保持：smoke / small-scale oracle / deterministic controlled probe / OOD formal execution。
- `paper2` 不维护论文正文、写作资产或论文主结论。
- v4.3 主线：在 Mainline-A 下运行 full-17，对算法重新筛选。

### Pending Decisions

- full-17 正式 benchmark 训练规模：默认沿用现有 `100000 timesteps / 10 eval episodes`，正式运行前可下调。
- 是否恢复 per-algorithm VSCode reset：默认不恢复；用 CLI `reset --algorithm`。

---

# 模块 24：docs directory slimming

## 概述

- 职责：用本 slim docs 包替换旧 `docs/`，删除历史流水账式长文档。
- 输出：新的短版 `docs/`。
- 预计步骤数：4。

## Step 1：替换 docs 目录

- **scope: auto**
- 操作：在仓库根目录删除旧 `docs/`，解压本包生成新 `docs/`；不复制旧 `docs/archive/*` 长文档。
- 验证：
  ```bash
  test -f docs/README.md
  test -f docs/plan.md
  test -f docs/progress.md
  test -f docs/report.md
  test -f docs/issues.md
  test -f docs/mainline_a_experiment_gate_review.md
  ```

## Step 2：验证 docs 轻量化边界

- **scope: auto**
- 操作：检查活跃文件行数，确认没有旧长流水账。
- 验证：
  ```bash
  python - <<'PY'
  from pathlib import Path
  limits={"docs/plan.md":320,"docs/progress.md":160,"docs/report.md":160,"docs/issues.md":160}
  for name, limit in limits.items():
      n=len(Path(name).read_text(encoding="utf-8").splitlines())
      assert n <= limit, f"{name} too long: {n}>{limit}"
  PY
  ```

## Step 3：验证状态源一致

- **scope: auto**
- 操作：所有活跃状态文件指向 v4.3；v4.2 只作为历史输入。
- 验证：
  ```bash
  grep -n "system-model-overhaul-v4.3" docs/plan.md docs/progress.md docs/report.md
  grep -n "CHANGE_PENDING_EXECUTION" docs/plan.md docs/report.md
  grep -n "ACCEPTED_WITH_BOUNDARIES" docs/mainline_a_experiment_gate_review.md
  ```

## Step 4：提交 docs reset

- **scope: auto**
- 操作：`git add -A docs`。
- 验证：
  ```bash
  git status --short docs
  git diff --cached --stat docs
  ```

---

# 模块 25：Mainline-A environment profile 与 full-17 默认迁移

## 概述

- 职责：把 full-17 默认训练环境切到 Mainline-A，legacy 只保留为显式 fallback。
- 参考：`docs/references/mainline_a_migration_notes.md`
- 预计步骤数：8。

## Step 1：新增环境 profile 定义

- **scope: review**
- 新增：`src/experiment/environment_profiles.py`
- 操作：
  - 定义 `EnvironmentProfile(name, enable_mainline_a, system_model_config, dynamic_pricing_config, description)`。
  - 定义 `DEFAULT_ENVIRONMENT_PROFILE="mainline-a"`、`MAINLINE_A_PROFILE`、`LEGACY_PROFILE`。
  - 实现 `resolve_environment_profile()` 与 `profile_to_train_extra_args()`。
  - Mainline-A 指向 `configs/system_model_mainline_a.yaml` 与 `configs/pricing_dynamic_mainline_a.yaml`。
- 验证：`pytest tests/test_environment_profiles.py -q`

## Step 2：扩展 train.py 支持 profile

- **scope: review**
- 修改：`scripts/train.py`
- 操作：
  - `parse_args()` 增加 `--environment-profile {mainline-a,legacy}`，默认 `mainline-a`。
  - 增加 `--enable-mainline-a`、`--system-model-config`、`--dynamic-pricing-config` 兼容参数。
  - 在 `main()` 中将 profile 注入 `env_overrides`。
  - Mainline-A 失败必须 fail fast，不能自动切 legacy。
- 验证：
  ```bash
  python scripts/train.py --help | grep -n "environment-profile"
  pytest tests/test_train_environment_profile_args.py -q
  ```

## Step 3：扩展 registry/spec 注入 profile

- **scope: review**
- 修改：`src/experiment/registry.py`、`src/experiment/models.py`
- 操作：
  - 保持 `AlgorithmSpec.extra_args` 为注入通道。
  - `AlgorithmRegistry.build_specs()` 增加 `environment_profile="mainline-a"`。
  - 每个 spec 追加 `["--environment-profile", profile]`。
- 验证：
  ```bash
  pytest tests/test_experiment_mainline_a_default.py -q
  python - <<'PY'
  from src.experiment.registry import AlgorithmRegistry
  spec=AlgorithmRegistry().build_specs(["GRPO"], timesteps=1, seed=42, device="cpu", eval_episodes=1)[0]
  assert spec.extra_args == ["--environment-profile", "mainline-a"]
  PY
  ```

## Step 4：扩展 experiment_manager CLI

- **scope: review**
- 修改：`scripts/experiment_manager.py`、`src/experiment/manager.py`
- 操作：
  - `start` 增加 `--environment-profile {mainline-a,legacy}`。
  - `_resolve_start_options()` 返回 `environment_profile`。
  - `create_experiment()` 传入 profile，并写入 `manifest.metadata["environment_profile"]`。
- 验证：
  ```bash
  python scripts/experiment_manager.py start --help | grep -n "environment-profile"
  pytest tests/test_experiment_manager_profiles.py -q
  ```

## Step 5：重写 presets 默认值

- **scope: auto**
- 修改：`src/experiment/presets.py`
- 操作：
  - `PRESETS["quick"]["environment_profile"]="mainline-a"`。
  - `PRESETS["full17"]["environment_profile"]="mainline-a"`。
  - `FULL_17_ALGORITHMS` 保持 17 个算法。
  - 不新增 `legacy_full17` preset。
- 验证：
  ```bash
  python - <<'PY'
  from src.experiment.presets import PRESETS, FULL_17_ALGORITHMS
  assert len(FULL_17_ALGORITHMS)==17
  assert PRESETS["full17"]["environment_profile"]=="mainline-a"
  PY
  ```

## Step 6：统一 direct benchmark profile 参数

- **scope: review**
- 修改：`scripts/benchmark.py`
- 操作：
  - 增加 `--environment-profile {mainline-a,legacy}`，默认 `mainline-a`。
  - `--all` 默认走 Mainline-A。
  - 输出 JSON 增加 `environment_profile`。
  - `legacy` 必须显式关闭 Mainline-A 配置注入。
- 验证：
  ```bash
  python scripts/benchmark.py --help | grep -n "environment-profile"
  python scripts/benchmark.py --all --environment-profile mainline-a --dry-run
  python scripts/benchmark.py --all --environment-profile legacy --dry-run
  pytest tests/test_benchmark_environment_profile.py -q
  ```

## Step 7：重写 VSCode 调试入口

- **scope: auto**
- 修改：`.vscode/launch.json`
- 操作：删除 per-algorithm reset 入口，保留不超过 8 个入口：
  `Mainline-A Full 17 Fresh`、`Resume`、`Status`、`Export Results`、`Direct Benchmark Dry Run`、`Plot Latest`、`Legacy Full 17 Fresh (Explicit Fallback)`、`Experiment List`。
- 验证：`pytest tests/test_vscode_launch_mainline_a.py -q`

## Step 8：回归测试与边界检查

- **scope: auto**
- 操作：运行 profile、manager、train、benchmark、VSCode 测试；不启动正式 full-17 长训练。
- 验证：
  ```bash
  pytest tests/test_environment_profiles.py tests/test_train_environment_profile_args.py tests/test_experiment_mainline_a_default.py tests/test_experiment_manager_profiles.py tests/test_benchmark_environment_profile.py tests/test_vscode_launch_mainline_a.py -q
  git status --short experiments results figures logs checkpoints | grep -E "^[AM]" && exit 1 || true
  ```

---

# 模块 26：Mainline-A full-17 benchmark execution gate

## 概述

- 职责：迁移完成后按 Mainline-A 环境筛选 full-17 算法。
- 前置依赖：模块 25 全部通过。
- 预计步骤数：5。

## Step 1：为 experiment_manager 增加 dry-run

- **scope: review**
- 修改：`scripts/experiment_manager.py`、`src/experiment/manager.py`、`src/experiment/registry.py`
- 操作：`start --dry-run` 只输出 manifest、17 个算法、profile、train commands，不写 `experiments/`，不启动训练。
- 验证：
  ```bash
  python scripts/experiment_manager.py start --preset full17 --environment-profile mainline-a --dry-run
  pytest tests/test_experiment_manager_dry_run.py -q
  ```

## Step 2：Mainline-A full-17 preflight

- **scope: auto**
- 操作：执行 dry-run，确认 17 个 command 都包含 `--environment-profile mainline-a`。
- 验证：
  ```bash
  python scripts/experiment_manager.py start --preset full17 --environment-profile mainline-a --dry-run > /tmp/full17_mainline_a_dryrun.json
  python - <<'PY'
  import json
  p=json.load(open("/tmp/full17_mainline_a_dryrun.json", encoding="utf-8"))
  assert p["dry_run"] is True and p["environment_profile"]=="mainline-a"
  assert p["algorithm_count"]==17
  for cmd in p["commands"]: assert "--environment-profile" in cmd and "mainline-a" in cmd
  PY
  ```

## Step 3：Mainline-A smoke benchmark

- **scope: review**
- 操作：只跑 quick preset；验证链路，不筛选算法优劣。
- 命令：
  ```bash
  python scripts/experiment_manager.py start --preset quick --environment-profile mainline-a --fresh --timesteps 5000 --eval-episodes 3
  python scripts/experiment_manager.py export --run-id vscode_quick --output results/benchmark_vscode_quick_mainline_a.json
  ```

## Step 4：Mainline-A full-17 正式运行

- **scope: review**
- 操作：用户确认训练规模后运行；默认 `100000 timesteps / 10 eval episodes`。
- 命令：
  ```bash
  python scripts/experiment_manager.py start --preset full17 --run-id paper2_full_17_mainline_a --environment-profile mainline-a --fresh
  python scripts/experiment_manager.py export --run-id paper2_full_17_mainline_a --output results/benchmark_paper2_full_17_mainline_a.json
  ```

## Step 5：legacy fallback 只做对照复现

- **scope: review**
- 操作：仅 Mainline-A 链路失败或需复现旧结果时执行；不得混入 Mainline-A ranking。
- 命令：
  ```bash
  python scripts/experiment_manager.py start --preset full17 --run-id paper2_full_17_legacy_fallback --environment-profile legacy --fresh
  ```

---

# 执行纪律

- 先完成 docs replacement，再改代码。
- Mainline-A 是默认主线；legacy 只能显式指定。
- 不做 silent fallback。
- 不恢复旧 `.vscode/launch.json` 的 per-algorithm reset 膨胀入口。
- 不把 generated artifacts 纳入 Git tracking。
- full-17 正式训练前必须先 dry-run，再 quick smoke。
