#!/usr/bin/env python3
"""
Plot Benchmark Results — 从 JSON 生成对比图表

用法:
    python scripts/plot_results.py --input results/benchmark.json --output figures/
    python scripts/plot_results.py --input results/benchmark.json --output figures/ --format pdf
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt


# 配色方案
ALGO_COLORS = {
    "GRPO": "#e74c3c",
    "PPO": "#3498db",
    "SAC": "#2ecc71",
    "DDQN": "#f39c12",
    "DDPG": "#9b59b6",
    "TD3": "#1abc9c",
    "A3C": "#e67e22",
    "TRPO": "#34495e",
    "SimPO": "#e91e63",
    "MAPPO": "#00bcd4",
    "QMIX": "#795548",
    "COMA": "#FF5722",
    "IPPO": "#607D8B",
    "VDN": "#8BC34A",
    "MADDPG": "#FF9800",
    "IQL": "#03A9F4",
    "MATD3": "#9C27B0",
}


def load_results(path: str) -> List[Dict[str, Any]]:
    """加载 benchmark JSON 结果"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_float(value):
    try:
        if value is None:
            return None
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _fmt_or_na(value, fmt: str) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "NA"
    return format(numeric, fmt)


def _pick_first_numeric(record: Dict[str, Any], keys: List[str]) -> float | None:
    for key in keys:
        value = _to_float(record.get(key))
        if value is not None:
            return value
    return None


def _pick_latency_metric(record: Dict[str, Any]) -> float | None:
    return _pick_first_numeric(
        record,
        [
            "final_latency_per_task_mean_mean",
            "final_latency_per_task_mean",
            "final_latency_total_mean_mean",
            "final_latency_mean_mean",
            "final_latency_total_mean",
            "final_latency_mean",
        ],
    )


def _pick_energy_metric(record: Dict[str, Any]) -> float | None:
    return _pick_first_numeric(
        record,
        [
            "final_energy_per_task_mean_mean",
            "final_energy_per_task_mean",
            "final_energy_total_mean_mean",
            "final_energy_mean_mean",
            "final_energy_total_mean",
            "final_energy_mean",
        ],
    )


def plot_reward_comparison(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """柱状图: 各算法平均奖励对比 (按环境分组)"""
    # 按环境分组
    env_groups: Dict[str, List[Dict]] = {}
    for r in results:
        env_name = r.get("environment", "unknown")
        env_groups.setdefault(env_name, []).append(r)

    for env_name, group in env_groups.items():
        algos = []
        rewards = []
        stds = []
        colors = []

        for r in group:
            algo = r["algorithm"]
            # 多 seed 取 mean_mean，单 seed 取 final_reward_mean
            rm = _to_float(r.get("final_reward_mean_mean", r.get("final_reward_mean", 0)))
            rs = _to_float(r.get("final_reward_mean_std", r.get("final_reward_std", 0)))
            if rm is None:
                continue
            algos.append(algo)
            rewards.append(rm)
            stds.append(0.0 if rs is None else rs)
            colors.append(ALGO_COLORS.get(algo, "#95a5a6"))

        if not algos:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(algos))
        bars = ax.bar(x, rewards, yerr=stds, capsize=5,
                      color=colors, edgecolor="white", linewidth=0.8)

        ax.set_ylabel("Mean Reward", fontsize=12)
        ax.set_title(f"Benchmark: {env_name}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # 在柱顶标注数值
        for bar, val in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        safe_env = env_name.replace("/", "_")
        fig.savefig(output_dir / f"reward_comparison_{safe_env}.{fmt}", dpi=150)
        plt.close(fig)


def plot_training_time(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """柱状图: 训练耗时对比"""
    algos = []
    times = []
    colors = []

    for r in results:
        algo = r["algorithm"]
        tm = _to_float(r.get("train_time_seconds_mean", r.get("train_time_seconds", 0)))
        if tm is None:
            continue
        algos.append(algo)
        times.append(tm)
        colors.append(ALGO_COLORS.get(algo, "#95a5a6"))

    if not algos:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(algos))
    ax.bar(x, times, color=colors, edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Training Time (s)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"training_time.{fmt}", dpi=150)
    plt.close(fig)


def plot_latency_energy(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """散点图: Latency/task vs Energy/task (气泡大小 = reward)"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for r in results:
        algo = r["algorithm"]
        lat = _pick_latency_metric(r)
        eng = _pick_energy_metric(r)
        rew = _to_float(r.get("final_reward_mean_mean", r.get("final_reward_mean", 0)))
        if lat is None or eng is None or rew is None:
            continue
        color = ALGO_COLORS.get(algo, "#95a5a6")

        # 气泡大小: abs(reward) 映射到 [50, 500]
        size = max(50, min(500, abs(rew) * 200 + 50))
        ax.scatter(lat, eng, s=size, c=color, alpha=0.7, edgecolors="white",
                   linewidth=1.5, label=algo, zorder=3)
        ax.annotate(algo, (lat, eng), fontsize=9, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points")

    ax.set_xlabel("Mean Latency per Task", fontsize=12)
    ax.set_ylabel("Mean Energy per Task", fontsize=12)
    ax.set_title("Latency vs Energy per Task (bubble size ∝ |reward|)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    if ax.collections:
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / f"latency_vs_energy.{fmt}", dpi=150)
    plt.close(fig)


def plot_summary_table(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """表格图: 所有指标汇总"""
    # 收集列
    columns = ["Algorithm", "Reward", "Latency/task", "Energy/task", "Time (s)"]
    rows = []
    for r in results:
        algo = r["algorithm"]
        rew = r.get("final_reward_mean_mean", r.get("final_reward_mean", 0))
        lat = _pick_latency_metric(r)
        eng = _pick_energy_metric(r)
        tm = r.get("train_time_seconds_mean", r.get("train_time_seconds", 0))
        rows.append(
            [
                algo,
                _fmt_or_na(rew, ".4f"),
                _fmt_or_na(lat, ".2f"),
                _fmt_or_na(eng, ".2f"),
                _fmt_or_na(tm, ".1f"),
            ]
        )

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.5 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center",
                     cellLoc="center", colColours=["#3498db"] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 表头样式
    for j in range(len(columns)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Benchmark Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / f"summary_table.{fmt}", dpi=150)
    plt.close(fig)


def plot_convergence_curves(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """4 子图: 各算法收敛曲线 (reward / latency / energy / comm_score)"""
    # 收集有 convergence_by_seed 数据的算法
    algo_data: Dict[str, List[Dict]] = {}
    for r in results:
        algo = r["algorithm"]
        seeds = r.get("convergence_by_seed")
        if seeds and isinstance(seeds, dict) and len(seeds) > 0:
            # dict format: {seed_str: convergence_data_dict}
            algo_data[algo] = list(seeds.values())
        elif seeds and isinstance(seeds, list) and len(seeds) > 0:
            algo_data[algo] = seeds

    # 如果没有 convergence_by_seed 数据，尝试从 train_logs.json 文件读取
    if not algo_data:
        for r in results:
            algo = r["algorithm"]
            checkpoint_dir = r.get("checkpoint_dir", "")
            if checkpoint_dir:
                # 构建 train_logs.json 路径
                train_logs_path = Path(checkpoint_dir) / "train_logs.json"
                if not train_logs_path.exists():
                    # 尝试从 experiments 目录查找
                    for exp_dir in Path("experiments").iterdir():
                        if exp_dir.is_dir():
                            candidate = exp_dir / "artifacts" / algo / "checkpoints" / "train_logs.json"
                            if candidate.exists():
                                train_logs_path = candidate
                                break
                
                if train_logs_path.exists():
                    try:
                        with open(train_logs_path, encoding="utf-8") as f:
                            logs = json.load(f)
                        algo_data[algo] = [logs]  # 包装成列表以兼容原有逻辑
                    except Exception:
                        pass

    if not algo_data:
        return

    # train_logs.json 中的 metric keys（与 convergence_by_seed 不同）
    metric_keys_train_logs = [
        ("eval_eval/reward_mean", "eval/reward_mean", "Reward"),
        ("eval_eval/latency_mean", "eval/latency_mean", "Latency / Task"),
        ("eval_eval/energy_mean", "eval/energy_mean", "Energy / Task"),
        ("eval_eval/comm_score", "eval/comm_score", "Comm Score"),
    ]
    
    metric_keys = [
        ("eval/reward_mean", "Reward"),
        ("eval/latency_mean", "Latency / Task"),
        ("eval/energy_mean", "Energy / Task"),
        ("eval/comm_score", "Comm Score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, ((eval_key, train_log_key, metric_label)) in enumerate(metric_keys_train_logs):
        ax = axes[idx]
        for algo, seeds in algo_data.items():
            color = ALGO_COLORS.get(algo, "#95a5a6")
            all_series = []
            for seed_data in seeds:
                # 先尝试 convergence_by_seed 格式的 key，再尝试 train_logs.json 格式
                series = seed_data.get(eval_key)
                if not series:
                    series = seed_data.get(train_log_key)
                if series and isinstance(series, list) and len(series) > 0:
                    all_series.append(series)

            if not all_series:
                continue

            # 对齐长度: 取最短
            min_len = min(len(s) for s in all_series)
            arr = np.array([s[:min_len] for s in all_series], dtype=float)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)

            # x 轴: eval_interval * index
            eval_interval = 1000  # 默认间隔
            for r in results:
                if r["algorithm"] == algo:
                    # eval_interval may be in convergence data or top level
                    seeds_data = r.get("convergence_by_seed", {})
                    if isinstance(seeds_data, dict):
                        for seed_data in seeds_data.values():
                            if isinstance(seed_data, dict) and "eval_interval" in seed_data:
                                eval_interval = seed_data["eval_interval"]
                                break
                    elif "eval_interval" in r:
                        eval_interval = r["eval_interval"]
                    break
            x = np.arange(min_len) * eval_interval

            ax.plot(x, mean, color=color, label=algo, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

            # 滑动窗口收敛检测: 最后 10% 窗口相对变化 < 5%
            window = max(1, min_len // 10)
            tail_mean = np.nanmean(mean[-window:])
            head_mean = np.nanmean(mean[-2 * window:-window]) if 2 * window <= min_len else mean[0]
            if head_mean != 0 and abs(tail_mean - head_mean) / abs(head_mean) < 0.05:
                # 收敛: 在 legend 中标记 ✓
                ax.plot([], [], color=color, marker="o", linestyle="",
                        label=f"{algo} ✓")

        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"{metric_label} Convergence", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / f"convergence_curves.{fmt}", dpi=150)
    plt.close(fig)


def plot_composite_ranking(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """3 子图: 各 profile 下综合排名 (balanced / latency_critical / energy_constrained)"""
    profiles = ["balanced", "latency_critical", "energy_constrained"]
    profile_labels = ["Balanced", "Latency-Critical", "Energy-Constrained"]

    # 收集有 composite_scores 数据的算法
    algo_scores: Dict[str, Dict] = {}
    for r in results:
        algo = r["algorithm"]
        cs = r.get("composite_scores")
        if cs and isinstance(cs, dict):
            algo_scores[algo] = cs

    if not algo_scores:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (profile, label) in enumerate(zip(profiles, profile_labels)):
        ax = axes[idx]
        # 收集各算法在该 profile 下的 composite_score
        entries = []
        for algo, cs in algo_scores.items():
            score = cs.get(profile)
            if score is not None and isinstance(score, (int, float)):
                entries.append((algo, float(score)))

        if not entries:
            ax.set_title(f"{label}\n(no data)", fontsize=12, fontweight="bold")
            ax.axis("off")
            continue

        # 按 composite_score 降序排列
        entries.sort(key=lambda x: x[1], reverse=True)
        algos = [e[0] for e in entries]
        scores = [e[1] for e in entries]
        colors = [ALGO_COLORS.get(a, "#95a5a6") for a in algos]

        x = np.arange(len(algos))
        bars = ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.8)

        ax.set_ylabel("Composite Score", fontsize=11)
        ax.set_title(f"{label}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # 柱顶标注数值
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"composite_ranking.{fmt}", dpi=150)
    plt.close(fig)


def plot_radar_chart(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """雷达图: Top-6 算法在 balanced profile 下的多维对比"""
    # 收集有 composite_scores 的算法
    algo_scores: Dict[str, Dict] = {}
    for r in results:
        algo = r["algorithm"]
        cs = r.get("composite_scores")
        if cs and isinstance(cs, dict):
            algo_scores[algo] = cs

    if not algo_scores:
        return

    # 按 balanced profile 排名，取 top 6
    balanced_scores = []
    for algo, cs in algo_scores.items():
        score = cs.get("balanced")
        if score is not None and isinstance(score, (int, float)):
            balanced_scores.append((algo, float(score)))

    if not balanced_scores:
        return

    balanced_scores.sort(key=lambda x: x[1], reverse=True)
    top_algos = [e[0] for e in balanced_scores[:6]]

    # 4 个维度
    dimensions = ["reward_norm", "latency_norm", "energy_norm", "stability_norm"]
    dim_labels = ["Reward", "Latency", "Energy", "Stability"]
    n_dims = len(dimensions)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for algo in top_algos:
        cs = algo_scores[algo]
        values = []
        for dim in dimensions:
            val = cs.get(dim)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))
            else:
                values.append(0.0)
        values += values[:1]  # 闭合

        color = ALGO_COLORS.get(algo, "#95a5a6")
        ax.plot(angles, values, color=color, linewidth=1.8, label=algo)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_title("Top-6 Algorithm Radar (Balanced Profile)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / f"radar_top6.{fmt}", dpi=150)
    plt.close(fig)


def plot_weight_sensitivity(results: List[Dict], output_dir: Path, fmt: str = "png"):
    """折线图: 各算法在不同 profile 下的排名变化 (权重敏感性)"""
    profiles = ["balanced", "latency_critical", "energy_constrained"]
    profile_labels = ["Balanced", "Latency-Critical", "Energy-Constrained"]

    # 收集有 composite_scores 的算法
    algo_scores: Dict[str, Dict] = {}
    for r in results:
        algo = r["algorithm"]
        cs = r.get("composite_scores")
        if cs and isinstance(cs, dict):
            algo_scores[algo] = cs

    if not algo_scores:
        return

    # 计算每个 profile 下的排名
    algo_ranks: Dict[str, List[int]] = {}
    for profile in profiles:
        entries = []
        for algo, cs in algo_scores.items():
            score = cs.get(profile)
            if score is not None and isinstance(score, (int, float)):
                entries.append((algo, float(score)))

        entries.sort(key=lambda x: x[1], reverse=True)
        for rank, (algo, _) in enumerate(entries, start=1):
            algo_ranks.setdefault(algo, []).append(rank)

    if not algo_ranks:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(profiles))

    for algo, ranks in algo_ranks.items():
        if len(ranks) != len(profiles):
            continue
        color = ALGO_COLORS.get(algo, "#95a5a6")
        ax.plot(x, ranks, color=color, marker="o", linewidth=1.8,
                markersize=6, label=algo)

    ax.set_xticks(x)
    ax.set_xticklabels(profile_labels, fontsize=11)
    ax.set_ylabel("Rank", fontsize=12)
    ax.set_title("Weight Sensitivity: Algorithm Rank Across Profiles",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()  # rank 1 在顶部
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / f"weight_sensitivity.{fmt}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Benchmark Results")
    parser.add_argument("--input", type=str, required=True, help="benchmark JSON path")
    parser.add_argument("--output", type=str, default="figures", help="output directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.input)
    print(f"Loaded {len(results)} results from {args.input}")

    plot_reward_comparison(results, output_dir, args.format)
    plot_training_time(results, output_dir, args.format)
    plot_latency_energy(results, output_dir, args.format)
    plot_summary_table(results, output_dir, args.format)
    plot_convergence_curves(results, output_dir, args.format)
    plot_composite_ranking(results, output_dir, args.format)
    plot_radar_chart(results, output_dir, args.format)
    plot_weight_sensitivity(results, output_dir, args.format)

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
