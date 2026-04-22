#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local DP noise sensitivity visualization.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "save"
OUTPUT_DIR = SAVE_DIR

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def moving_average(data, window_size=10):
    result = np.array(data, dtype=float)
    for i in range(len(result)):
        start = max(0, i - window_size + 1)
        result[i] = np.mean(data[start:i + 1])
    return result


def load_dp_results():
    results = []
    for folder in sorted(SAVE_DIR.glob("sens_dp_sigma*")):
        pkl_files = sorted(folder.glob("*.pkl"))
        if not pkl_files:
            continue
        with open(pkl_files[0], "rb") as f:
            data = pickle.load(f)
        args = data.get("args", {})
        sigma = float(args.get("dp_noise_multiplier", 0.0))
        acc = data.get("test_accuracy", [])
        if not acc:
            continue
        results.append(
            {
                "sigma": sigma,
                "accuracy": acc,
                "final_acc": acc[-1] * 100.0,
                "last5_acc": np.mean(acc[-5:]) * 100.0,
                "best_acc": np.max(acc) * 100.0,
                "folder": folder.name,
            }
        )
    results.sort(key=lambda x: x["sigma"])
    return results


def plot_dp_sensitivity(output_path=None):
    if output_path is None:
        output_path = OUTPUT_DIR / "dp_sensitivity.png"

    results = load_dp_results()
    if not results:
        print("未找到 DP 敏感性结果。")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    for item in results:
        rounds = range(1, len(item["accuracy"]) + 1)
        smoothed = moving_average([a * 100.0 for a in item["accuracy"]], window_size=10)
        ax1.plot(rounds, smoothed, linewidth=2.2, label=f'σ={item["sigma"]}')

    ax1.set_xlabel("Training Round", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("DP Noise Sensitivity Curves", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    sigmas = [item["sigma"] for item in results]
    last5 = [item["last5_acc"] for item in results]
    best = [item["best_acc"] for item in results]

    ax2.plot(sigmas, last5, marker='o', linewidth=2.2, label='Last-5 Avg Acc')
    ax2.plot(sigmas, best, marker='s', linewidth=2.0, label='Best Acc')
    for x, y in zip(sigmas, last5):
        ax2.text(x, y, f'{y:.2f}', fontsize=9, ha='center', va='bottom')

    ax2.set_xlabel("DP Noise Multiplier σ_dp", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy vs DP Noise", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.suptitle("Sensitivity Analysis: Local DP Noise Multiplier", fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图已保存: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_dp_sensitivity()
