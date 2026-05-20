# paper2 v5.0 GPU Artifact Inventory

Generated: 2026-05-20 11:50:39 +08:00

Scope: v5.0 corrected `single_policy_multi_user` 3-user Mainline-A full17-equivalent GPU benchmark on the RTX 4060 Laptop GPU platform.

Important sync note: `results/` and `figures/` are ignored by Git in this workspace. Manually sync the raw result, summary CSV, figure files, and `figure_manifest.json` listed below to the Web side before review.

Official benchmark command:

```text
python scripts/benchmark.py --config configs/benchmark_mainline_a_single_policy_3user_full17.yaml --output results/benchmark_mainline_a_single_policy_3user_full17.json --no-latest-alias
```

Official raw result validation: 9 / 9 algorithms present (`GRPO`, `PPO`, `SAC`, `DDQN`, `DDPG`, `TD3`, `A3C`, `TRPO`, `SimPO`); all records have `status=ok`, `interface=single_policy_multi_user`, `num_agents=3`, `single_policy_num_users=3`, and `device=cuda`.

| Category | Path | Size bytes | SHA256 | Note |
| --- | --- | ---: | --- | --- |
| Raw results | `results/benchmark_mainline_a_single_policy_3user_full17.json` | 742254 | `1cc0627943f64dcfb9f637b6e4c6e5dd1ad2c69e4b04a70e6ae202c2f0933c0f` | Official v5.0 corrected 3-user full17-equivalent benchmark output; ignored by Git |
| Analysis results | `results/single_policy_3user_full17_summary.csv` | 958 | `3aa18dcfb7f6c3de42c5d8b02a57019acf22c9a10c65d1834eff35482ea4ccbf` | Analysis summary CSV; ignored by Git |
| Figures | `figures/single_policy_3user_full17/figure_manifest.json` | 101 | `6c0206e460644b575138bb25ef41bd954bbe3c9ee74960467b6ef23855d38058` | Figure manifest; ignored by Git |
| Figures | `figures/single_policy_3user_full17/reward_ranking.png` | 37305 | `5e018968bbea34d48345c7e1cf23e8307bb23741b13d06ff827efee95a5bb3fe` | Reward ranking figure; ignored by Git |
| Figures | `figures/single_policy_3user_full17/comm_score_ranking.png` | 40105 | `925a8707b74b8c2f134d99c35b4cb09eb627f7d29fb2ae426608e146ea6335a6` | Communication score ranking figure; ignored by Git |
| Official benchmark logs | `logs/benchmark_20260519_154138.log` | 5130 | `dfade0a7df1d812beeb1aada5c30a98762d006aeac1701e74cebe7c507babe65` | Benchmark logger stdout-equivalent summary |
| Official benchmark logs | `logs/benchmark_20260519_154138.err.log` | 1047651 | `b414cd483fa302e919b18075b1edaf7c9bee745242267dbcee85f3e9fe54ba19` | Benchmark progress/stderr log |
| Official benchmark logs | `logs/benchmark_gpu_full17_20260519_154138.stdout.log` | 5130 | `dfade0a7df1d812beeb1aada5c30a98762d006aeac1701e74cebe7c507babe65` | Start-Process stdout capture |
| Official benchmark logs | `logs/benchmark_gpu_full17_20260519_154138.stderr.log` | 1013974 | `a6c4e0345b4a6776d3cb5dfaa3ff1e5723fd839e0840be2e1a5ac1a600c0eb27` | Start-Process stderr capture |
| Audit logs | `logs/benchmark_20260519_143708.log` | 16740 | `4395062bdf4a21ad7728e7f97c3aabcf8bbdb25af0b8d8d047bce0cc1af04c56` | Superseded foreground attempt log; parent pipe timeout caused tqdm flush errors |
| Audit logs | `logs/benchmark_20260519_143708.err.log` | 8278 | `13abe4502b19889ca0e492082bcabc6dc9bea1b36be9a59f7cb509653d5a911b` | Superseded foreground attempt stderr |
| Audit logs | `logs/benchmark_20260519_143708.pipe_failure_result.json` | 24024 | `e017af3b1f64cd8472651dd38c3af13a20f11cbf69d34c09e8c1ce1ff1e947fb` | Quarantined pipe-failure result, not used as official evidence |
| Reports | `ref/paper2_single_policy_3user_full17_report.md` | 2822 | `b56487fa537fe4535d11f1190c5af76954b4cb4596070e5f7e6e05c63a0e53a9` | Regenerated analysis report from official GPU result |
| Reports | `ref/paper2_single_agent_reassessment_decision.md` | 434 | `1a5d94ac37777783b01a482c0e8afe94f6172a4e0ed7e00ad35d8395d7d8e661` | Regenerated reassessment decision from official GPU result |
| Context | `configs/benchmark_mainline_a_single_policy_3user_full17.yaml` | 1373 | `2e74b7ef021d315d1d51fbd306459fe97b278cffb73f3063ec1d0765ef953b1b` | Benchmark config: 9 algorithms, seed 42, steps 100000, 3-user single_policy_multi_user |
