# Review Report — 2026-05-14

## Scope
- Reviewed:
  - `.ai/ledger.json`
  - `.ai/checkpoint.json`
  - `ref/paper2_v49_execution_preflight.md`
  - `ref/paper2_final_claim_boundary.md`
  - `ref/paper2_l2_candidate_validation_report.md`
  - `scripts/run_paper2_main_matrix.py`
  - `scripts/benchmark.py`
  - `src/environments/mec_v3/game_theory_adapters.py`
  - `src/experiment/presets.py`
  - `ref/mainline_a_final_algorithm_adaptation.md`
- Latest Change Focus:
  - v4.9 工具链修复与 formal closeout 阻断状态。
  - 新研究语义修正：单智能体算法不应继续以 1-agent 环境代表多用户 MEC 博弈场景。

## Verdict
BLOCKED

v4.9 工具链修复本身可接受：N2 ablation 展开、ablation 配置生效、GAM-COMA fallback critic、strict collector、多指标统计和 claim boundary 都已进入可验证状态。但 v4.9 formal closeout 目标没有完成：真实 L2/L3 未执行，checkpoint 未更新，ledger 正确保持 `NEEDS_WEB` / `PARTIAL`。此外，用户本轮指出的实验语义问题会使旧 single-agent full17 评估失效，因此下一轮不能继续推进 CL-PPO，而应先重建单智能体算法的 3-user shared-policy 对比证据。

## Findings
| Severity | Area | Finding | Evidence | Recommendation |
|---|---|---|---|---|
| Critical | Formal evidence | v4.9 没有真实 L2/L3 strict results，不能生成论文级统计结论。 | ledger `state=NEEDS_WEB`, `validation.status=PARTIAL`, open_items 明确无真实 L2/L3 strict collections。 | 保持 `NEEDS_WEB`，不要进入 final paper claims。 |
| Critical | Experiment semantics | 旧单智能体算法评估使用 1-agent 接口，与“agent≈用户终端、多个 agent≈多个用户同时博弈”的研究语义冲突。 | `scripts/benchmark.py` 中 `_resolve_num_agents()` 对非 multi-agent/heuristic 算法返回 1；full17 preset 是单 seed 100K 工程证据。 | 停止使用旧 single-agent full17 结果作为研究证据，改为 single-policy / 3-user shared-control 重跑。 |
| High | CL-PPO branch | CL-PPO 以 PPO 旧 1-agent 评估为基础，当前应暂停。 | 用户明确要求 CL-PPO 暂停；旧 PPO 口径与多用户博弈场景不一致。 | 写 `ref/paper2_cl_ppo_branch_freeze.md`，新 single-policy 3-user 评估完成并 review 后再决策。 |
| High | Frozen core | frozen Mainline-A core 在 v4.9 preflight 中存在 pre-patch dirty file。 | `ref/paper2_v49_execution_preflight.md` 记录 `game_theory_env.py` 1570 insertions/deletions diff。 | v5.0 实验前必须做 semantic diff；若不是 line-ending-only，停止并 NEEDS_WEB。 |
| Medium | Architecture | 适配层已具备多智能体动作输入容错，但 benchmark 当前不会让单智能体算法默认进入 3-user 场景。 | adapters 支持单/多智能体 action payload；benchmark `_resolve_num_agents()` 对 PPO/SAC/DDPG 等返回 1。 | 新增 single-policy multi-user runner，不修改 frozen core。 |

## Architecture Notes
- 正确方向不是把 PPO 改成 MARL，也不是把 CL-PPO 继续扩展；应先实现“一个共享策略控制 3 个用户终端”的接口层。
- 该接口应保留单智能体算法的 policy/replay/rollout 语义，但环境 step 必须是 3-user simultaneous action。
- MAPPO/COMA/GAM-COMA 等多智能体算法路径不应被这次修复改写。
- Mainline-A core 仍应冻结；修改点应限制在 adapter、benchmark、preset、analysis、VSCode、tests。

## Validation Notes
- v4.9 full pytest 通过，但只证明工具链和阻断报告有效，不证明 formal L2/L3 结论有效。
- 新 v5.0 必须增加 single-policy multi-user adapter/runner 单测，并用 PPO/DDQN/DDPG 至少覆盖 on-policy、discrete off-policy、continuous off-policy 三类路径。
- 新 full17-equivalent 对比仍是 single seed 100K artifact-level evidence，不应被包装成统计显著性结论。

## Ledger Recommendation
```json
{
  "state": "NEEDS_WEB",
  "open_items": [
    "paper2-v4.9 formal L2/L3 remains blocked: no real strict L2/L3 result collections exist.",
    "paper2-v5.0 required: old single-agent full17 results are suspended for research claims because the 1-agent interface conflicts with the multi-user MEC game interpretation.",
    "paper2-v5.0 required: implement single-policy 3-user Mainline-A interface and rerun full17-equivalent single-agent comparison before deciding whether to resume CL-PPO."
  ]
}
```
