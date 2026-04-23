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

    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
