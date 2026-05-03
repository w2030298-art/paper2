# paper2 未收敛算法影响分析与下一步计划

## 0. 当前判断

本次截图显示：上一次完整 17 算法对比实验中，以下算法的 Reward 曲线没有形成可信的单调改善或稳定收敛趋势：

- GRPO
- SAC
- A3C
- TRPO
- MAPPO
- COMA
- IPPO
- VDN
- MADDPG
- IQL
- MATD3

其余算法按用户描述已收敛。本报告只基于截图做第一轮诊断；精确归因仍需要对应 `train_logs.json`、`result.json`、stderr/stdout 与 convergence quality report。

---

## 1. 分组结论

| 分组 | 算法 | 现象 | 初步判断 |
|---|---|---|---|
| A：尾部稳定但不代表有效学习 | GRPO、SAC、TRPO、MAPPO | tail relative change 小于 5%，但整体没有明显持续上升趋势，部分算法先崩后平台 | 当前“尾部变化率 < 5%”会把“停在坏平台”误判为收敛 |
| B：边界震荡型 | A3C、COMA | tail relative change 接近或略高于 5%，曲线存在周期性回落 | 高方差策略梯度 / 多智能体信用分配噪声 |
| C：明显不稳定型 | IPPO、VDN、MADDPG、IQL、MATD3 | 大幅尖峰、断崖式下降、长期漂移，tail relative change 很高 | 多智能体非平稳、off-policy bootstrap 不稳、探索/回放/目标网络参数不匹配 |
| D：疑似环境或指标异常型 | IQL、VDN、IPPO | 出现极端负 reward，IQL 约 -65000，VDN/IPPO 有异常深坑 | 不能只按算法不收敛处理，需要先确认是否存在 episode-level failure 或 metric 采集异常 |

---

## 2. 单算法初步原因

### GRPO

现象：Reward 在 -4.6 到 -5.3 之间波动，尾部变化率约 1.1%，但不是明显持续改善。

可能原因：
- 尾部稳定但停在局部平台，当前收敛判定只检查尾部变化率，未检查相对最优值恢复程度。
- policy update 可能过于保守或 reward scale 太窄，导致改善信号弱。
- eval 采样少，reward 曲线受任务生成波动影响明显。

建议：
- 增加 `best_tail_gap` 指标：尾部均值必须接近历史最佳值才算收敛。
- 对 GRPO 检查 `group_size`、`eps_clip`、advantage normalization 和 entropy 系数。

### SAC

现象：前期从约 -15 快速掉到 -37 后长期平台，尾部变化率约 4.5%，但停在明显坏平台。

可能原因：
- SAC 可能“收敛到坏策略”，不是有效收敛。
- entropy/temperature、reward scale、critic target 或 replay warmup 不匹配。
- 连续动作映射可能过早推向低质量 offload 策略。

建议：
- 检查 `alpha` / automatic entropy tuning、replay warmup、critic loss。
- 增加“坏平台检测”：若尾部均值显著差于初始/历史最佳，不标记为 converged。

### A3C

现象：震荡明显，尾部变化率约 6.7%，后段没有稳定平台。

可能原因：
- A3C 高方差、异步更新和 advantage 估计噪声导致曲线振荡。
- 学习率或 entropy schedule 可能偏激。
- eval episode 数量不足会放大波动。

建议：
- 降低 actor/critic learning rate。
- 加强 advantage normalization 与 gradient clipping。
- 增加 eval episodes 或使用 moving median 评估。

### TRPO

现象：尾部变化率约 0.4%，曲线在 -13.7 到 -15.2 间震荡，无强上升趋势。

可能原因：
- KL 约束使更新保守，表现为窄幅周期震荡。
- 可能已到弱局部平台，但没有证据证明达到最优附近。
- 当前判定会把“窄幅震荡”视为收敛。

建议：
- 增加 slope-based 判定：尾部斜率接近 0 且 volatility 低才算稳定。
- 检查 `max_kl`、CG iterations、value function fit quality。

### MAPPO

现象：早期大跌后恢复，后段再下滑，尾部变化率约 3.4%。

可能原因：
- 多智能体 centralized critic / decentralized actors 不稳定。
- shared reward / credit assignment 导致策略互相追逐。
- value loss 与 policy loss 可能不同步。

建议：
- 检查 centralized critic 输入、advantage normalization、value loss coefficient。
- 降低 policy lr，提高 minibatch 稳定性。
- 增加 per-agent reward/advantage diagnostics。

### COMA

现象：前段大幅下跌后回升，后段继续下滑，tail relative change 约 5.6%。

可能原因：
- counterfactual baseline 估计不稳。
- credit assignment 噪声较大。
- discrete multi-agent action space 下 variance 高。

建议：
- 记录 advantage/baseline variance。
- 降低 critic lr，增加 baseline smoothing。
- 检查 COMA critic 是否正确使用 joint action 与 agent id。

### IPPO

现象：大幅振荡，约 39130 step 附近出现深坑，tail relative change 约 32.9%。

可能原因：
- independent learners 在多智能体环境中面对非平稳对手策略，天然容易不稳定。
- 各 agent 共享或独立策略的状态归一化可能不一致。
- reward/advantage scale 波动过大。

建议：
- 增加 per-agent reward/entropy/KL 记录。
- 降低 lr，启用或加强 gradient clipping。
- 若 IPPO 仍不稳，作为 baseline 保留，不强行调到论文主力算法。

### VDN

现象：早期和中段出现极端负 reward 深坑，tail relative change 约 34.1%。

可能原因：
- value decomposition 对环境非线性协作收益表达不足。
- TD bootstrap 与 epsilon exploration 可能造成 replay 中 bad episodes 传播。
- 极端值可能来自 episode-level failure 或 metric 采集异常。

建议：
- 先检查 result/stdout/stderr 与 episode reward min。
- 调慢 epsilon decay，增加 replay warmup。
- 检查 target network update interval / tau 与 TD loss 是否爆炸。

### MADDPG

现象：连续多次尖峰和回落，尾部仍大幅变化，tail relative change 约 18.2%。

可能原因：
- 多智能体连续控制 off-policy actor-critic 容易不稳定。
- exploration noise、target network、critic overestimation 可能不匹配。
- centralized critic 对 joint action 输入尺度敏感。

建议：
- 降低 actor/critic lr。
- 增加 replay warmup。
- 采用更慢 target update，检查 action noise decay。
- 优先与 MATD3 对比 TD3-style stabilization 是否生效。

### IQL

现象：出现约 -65000 极端负 reward，tail relative change 约 52.6%。

可能原因：
- 这不是普通“不收敛”，更像严重 episode failure、环境惩罚爆炸、状态/动作映射错误或 metric 聚合异常。
- independent Q-learning 在多智能体非平稳环境中高度不稳定。
- TD error 可能出现爆炸，或 replay 中异常 transition 被反复学习。

建议：
- 第一优先级检查 `stdout.log`、`stderr.log`、`result.json` 和 raw train logs。
- 加 TD error clipping / reward clipping 诊断，不建议直接改最终 reward 公式。
- 检查动作编码、epsilon schedule、target update、Q value range。

### MATD3

现象：中后期持续下滑且多处深坑，tail relative change 约 32.8%。

可能原因：
- 多智能体 TD3 虽有 delayed policy update，但 joint critic 仍可能受 replay distribution shift 影响。
- target smoothing / policy delay / exploration noise 可能不适配当前 MEC 动作空间。
- 可能与 MADDPG 一样存在 centralized critic 输入归一化问题。

建议：
- 检查 actor/critic loss、Q value min/max、target Q range。
- 降低 actor lr，增加 policy delay，减小 target policy noise。
- 与 MADDPG 共用一套连续 MARL 稳定化实验。

---

## 3. 需要你上传的数据

截图足够做方向判断，不足以证明根因。请优先上传以下文件，路径从项目根目录 `C:/Users/22003/paper2/paper2/` 开始找。

### 必传：完整实验汇总

优先找最近一次完整 17 算法实验：

```text
results/benchmark.json
results/benchmark_paper2_full_17_vscode.json
results/benchmark_full_*.json
figures/convergence_quality_report.json
figures/convergence_quality_report.md
```

如果 `results/` 被 .gitignore 或本地未保留，继续看 `experiments/`。

### 必传：完整 17 算法实验目录

优先目录：

```text
experiments/paper2_full_17_vscode/
```

如果它不是截图对应的那次实验，就找最近的 backup：

```text
experiments/paper2_full_17_vscode_backup_*/
```

每个异常算法至少上传：

```text
experiments/paper2_full_17_vscode/artifacts/<ALGO>/result.json
experiments/paper2_full_17_vscode/artifacts/<ALGO>/stdout.log
experiments/paper2_full_17_vscode/artifacts/<ALGO>/stderr.log
experiments/paper2_full_17_vscode/artifacts/<ALGO>/checkpoints/train_logs.json
```

其中 `<ALGO>` 替换为：

```text
GRPO SAC A3C TRPO MAPPO COMA IPPO VDN MADDPG IQL MATD3
```

同时上传：

```text
experiments/paper2_full_17_vscode/run.json
experiments/paper2_full_17_vscode/state.json
```

### 如果文件太多

只打包这些即可：

```text
results/benchmark*.json
figures/convergence_quality_report.*
experiments/paper2_full_17_vscode/run.json
experiments/paper2_full_17_vscode/state.json
experiments/paper2_full_17_vscode/artifacts/{GRPO,SAC,A3C,TRPO,MAPPO,COMA,IPPO,VDN,MADDPG,IQL,MATD3}/{result.json,stdout.log,stderr.log}
experiments/paper2_full_17_vscode/artifacts/{GRPO,SAC,A3C,TRPO,MAPPO,COMA,IPPO,VDN,MADDPG,IQL,MATD3}/checkpoints/train_logs.json
```

---

## 4. 下一轮计划方案

### 模块 12：未收敛算法诊断与稳定化实验计划

#### Step 1：修复当前提交卫生问题

- **scope: auto**
- 范围：
  - 清理 Git 中误提交或误删除的 `experiments/`、`results/`、`figures/` 产物变更。
  - 保留 `.gitignore` 对 `figures/`、`results/` 的规则。
  - 统一 `docs/archive/` 与 `docs/.archive/`。
  - 修正 `docs/plan.md` / `docs/report.md` / `docs/progress.md` 状态一致性。
- 验证：
  ```bash
  git status --short
  pytest tests/test_docs_contract.py -q
  ```

#### Step 2：新增未收敛诊断脚本

- **scope: auto**
- 文件：
  - 新建 `scripts/analyze_convergence_failures.py`
  - 新建 `tests/test_analyze_convergence_failures.py`
- 功能：
  - 读取 `results/benchmark*.json`
  - 读取 `figures/convergence_quality_report.json`
  - 可选读取 `experiments/<run_id>/artifacts/<ALGO>/checkpoints/train_logs.json`
  - 输出：
    - `results/convergence_failure_matrix.json`
    - `docs/convergence_failure_analysis.md`
- 诊断指标：
  - tail relative change
  - tail slope
  - best-tail gap
  - volatility
  - extreme outlier count
  - failed seed count
  - plateau-badness
  - reward regression from initial / from best
- 验证：
  ```bash
  python scripts/analyze_convergence_failures.py --help
  pytest tests/test_analyze_convergence_failures.py -q
  ```

#### Step 3：修正收敛判定，不再只看 tail relative change

- **scope: review**
- 文件：
  - `scripts/plot_results.py`
  - `scripts/analyze_convergence_failures.py`
- 新判定：
  - `converged_good`：尾部稳定，且 tail mean 接近历史最佳。
  - `bad_plateau`：尾部稳定，但明显差于历史最佳或初始值。
  - `oscillating`：tail volatility 高。
  - `diverging`：尾部方向持续变坏。
  - `catastrophic_outlier`：出现极端异常 reward。
- 验证：
  ```bash
  pytest tests/test_convergence_plot.py tests/test_analyze_convergence_failures.py -q
  ```

#### Step 4：按算法族生成稳定化配置候选

- **scope: review**
- 文件：
  - 新建 `configs/stability_overrides.yaml`
  - 更新 `scripts/benchmark.py`，支持读取稳定化配置覆盖项。
- 分组：
  - `on_policy_high_variance`: A3C、COMA、IPPO、MAPPO
  - `continuous_actor_critic`: SAC、MADDPG、MATD3
  - `value_decomposition`: VDN、IQL
  - `conservative_policy_update`: GRPO、TRPO
- 候选动作：
  - 降低 actor/critic lr。
  - 增加 gradient clipping。
  - 调整 entropy / exploration decay。
  - 增加 replay warmup。
  - 调整 target update tau 或 interval。
  - 增加 eval episodes。
- 验证：
  ```bash
  python scripts/benchmark.py --help
  python -c "import yaml; yaml.safe_load(open('configs/stability_overrides.yaml', encoding='utf-8'))"
  ```

#### Step 5：先做小规模 targeted rerun，不重跑全 17

- **scope: review**
- 目标算法：
  ```text
  GRPO SAC A3C TRPO MAPPO COMA IPPO VDN MADDPG IQL MATD3
  ```
- 运行策略：
  - 第一轮：每算法 50k steps，确认不再出现明显异常。
  - 第二轮：每算法 100k 或 200k steps，确认趋势。
  - 最后才考虑完整 17 算法重跑。
- 输出：
  - `results/benchmark_stability_<timestamp>.json`
  - `figures/convergence_curves_raw_all.png`
  - `figures/convergence_curves_clean_all.png`
  - `figures/convergence_quality_report.json`
- 验证：
  ```bash
  python scripts/plot_results.py --input results/benchmark_stability_<timestamp>.json --output figures --format png --convergence-mode both
  python scripts/analyze_convergence_failures.py --input results/benchmark_stability_<timestamp>.json --quality-report figures/convergence_quality_report.json
  ```

#### Step 6：产出下一版决策报告

- **scope: auto**
- 文件：
  - `docs/convergence_failure_analysis.md`
- 内容：
  - 每个未收敛算法的证据链。
  - 是否为算法问题、配置问题、环境/指标异常。
  - 是否进入下一轮调参。
  - 是否建议从论文主结果中降级为 baseline 或 ablation。
- 验证：
  ```bash
  test -f docs/convergence_failure_analysis.md
  grep -n "IQL" docs/convergence_failure_analysis.md
  grep -n "bad_plateau" docs/convergence_failure_analysis.md
  ```

---

## 5. 推荐派发方式

把“提交卫生修复”作为 fix 放在最前，随后执行模块 12 的诊断与计划落地。不要直接调参重跑全量实验；先补诊断脚本和分类规则，再用 targeted rerun 验证。
