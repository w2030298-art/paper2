# Issues

## Open

### DASHBOARD-EXTERNAL-COMPAT

- 严重度：Medium
- 状态：external
- 现象：dashboard 兼容性需要在外部 dashboard 仓库环境复核。
- 处理：不阻塞 paper2 v4.3。

## Fixed This Round

### [Fixed] V4.3-STATE-SOURCE-MISMATCH — 2026-05-06

- 严重度：High
- 处理：slim docs replacement 后，`plan.md` / `progress.md` / `report.md` 统一为 `system-model-overhaul-v4.3`；模块 24/25 状态已同步。

### [Fixed] V4.3-FULL17-NOT-MAINLINE-A-DEFAULT — 2026-05-06

- 严重度：High
- 处理：新增 `src/experiment/environment_profiles.py`；full17 / quick preset、registry、manager、train、direct benchmark 默认使用 `mainline-a`。

### [Fixed] V4.3-VSCODE-ENTRYPOINT-SPRAWL — 2026-05-06

- 严重度：Medium
- 处理：`.vscode/launch.json` 已收敛为 8 个入口；per-algorithm reset 不再放入 launch.json。

## Closed / Historical Index

- v4.2 boundary cleanup：closed。
- Mainline-A N0/N1/N2/N3 final review：closed with `ACCEPTED_WITH_BOUNDARIES`。
- 论文写作资产误入 paper2：closed；模块 21 `REMOVED_OUT_OF_SCOPE`。
- legacy L2/L3 convergence：retired as legacy baseline。
