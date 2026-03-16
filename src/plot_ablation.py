#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验可视化脚本
- SV消融：准确率曲线对比（Ours vs w/o SV）
- Energy+Lyapunov消融：能耗 + Q(t) 曲线对比（Ours vs w/o Lyapunov）
- Crypto消融：准确率对比 + 加密开销统计（Ours vs w/o Crypto）
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============= 配置参数 =============
ABLATION_DIR = '../save/ablation_20260315_235050'  # 修改为消融实验结果文件夹
OUTPUT_DIR = '../save'
# ===================================

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'Ours (Full)': '#2ca02c',
    'w/o SV':      '#ff7f0e',
    'w/o Lyapunov':'#1f77b4',
    'w/o Crypto':  '#9467bd',
}


def moving_average(data, window_size=10):
    result = np.array(data, dtype=float)
    for i in range(len(result)):
        start = max(0, i - window_size + 1)
        result[i] = np.mean(data[start:i + 1])
    return result


def load_ablation_data(save_dir=ABLATION_DIR):
    """加载消融实验所有pkl文件，按方法名分类"""
    pkl_files = list(Path(save_dir).glob('*.pkl'))
    data_dict = {}

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            if 'test_accuracy' not in data:
                continue

            args = data.get('args', {})
            filename = pkl_file.name.lower()

            use_sv       = args.get('use_shapley', False)
            use_lyapunov = args.get('use_lyapunov', False)
            use_crypto   = args.get('use_crypto', False)
            use_energy   = args.get('use_energy', False)

            if use_sv and use_lyapunov and use_crypto and use_energy:
                name = 'Ours (Full)'
            elif use_sv and use_lyapunov and use_energy and not use_crypto:
                name = 'w/o Crypto'
            elif use_sv and use_energy and use_crypto and not use_lyapunov:
                name = 'w/o Lyapunov'
            elif not use_sv and use_lyapunov and use_energy and use_crypto:
                name = 'w/o SV'
            else:
                name = filename.replace('.pkl', '')

            data_dict[name] = data
            print(f"加载: {name} <- {pkl_file.name}")

        except Exception as e:
            print(f"加载失败 {pkl_file.name}: {e}")

    return data_dict


# ============================================================
# 1. SV 消融：准确率曲线对比（Ours vs w/o SV）
# ============================================================
def plot_sv_ablation(save_dir=ABLATION_DIR, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'ablation_sv.png')

    print("\n" + "=" * 60)
    print("消融实验 1：Shapley Value 对准确率的贡献")
    print("=" * 60)

    data_dict = load_ablation_data(save_dir)
    targets = ['Ours (Full)', 'w/o SV']
    results = {k: data_dict[k] for k in targets if k in data_dict}

    if len(results) < 2:
        print(f"缺少数据，需要: {targets}，已有: {list(results.keys())}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：准确率收敛曲线
    ax1 = axes[0]
    for name, data in results.items():
        acc = data['test_accuracy']
        rounds = range(1, len(acc) + 1)
        smoothed = moving_average(acc, window_size=10)
        ax1.plot(rounds, acc, color=COLORS[name], alpha=0.25, linewidth=1)
        ax1.plot(rounds, smoothed, label=name, color=COLORS[name], linewidth=2.5)

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Accuracy: Ours vs w/o SV', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右图：收敛速度柱状图
    ax2 = axes[1]
    target_accs = [0.40, 0.45, 0.50]
    method_names = list(results.keys())
    x = np.arange(len(method_names))
    width = 0.25

    for i, target in enumerate(target_accs):
        rounds_list = []
        for name in method_names:
            acc = results[name]['test_accuracy']
            r = next((j+1 for j, a in enumerate(acc) if a >= target), len(acc))
            rounds_list.append(r)
        bars = ax2.bar(x + (i-1)*width, rounds_list, width,
                       label=f'{int(target*100)}% Acc', alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, str(int(h)),
                     ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Rounds to Target Accuracy', fontsize=12)
    ax2.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Ablation Study: Effect of Shapley Value Selection',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图已保存: {output_path}")
    plt.close()


# ============================================================
# 2. Energy + Lyapunov 消融：能耗 + Q(t) 曲线对比
# ============================================================
def plot_energy_lyapunov_ablation(save_dir=ABLATION_DIR, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'ablation_energy_lyapunov.png')

    print("\n" + "=" * 60)
    print("消融实验 2：Lyapunov 优化对能量约束的作用")
    print("=" * 60)

    data_dict = load_ablation_data(save_dir)
    targets = ['Ours (Full)', 'w/o Lyapunov']
    results = {k: data_dict[k] for k in targets if k in data_dict}

    if len(results) < 2:
        print(f"缺少数据，需要: {targets}，已有: {list(results.keys())}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：每轮平均能耗（consumption_history 的均值）
    ax1 = axes[0]
    for name, data in results.items():
        energy_stats = data.get('energy_statistics', {})
        consumption_history = energy_stats.get('consumption_history', [])
        if consumption_history:
            per_round_consumption = [np.sum(c) for c in consumption_history]
            rounds = range(1, len(per_round_consumption) + 1)
            smoothed = moving_average(per_round_consumption, window_size=5)
            ax1.plot(rounds, smoothed, label=name, color=COLORS[name], linewidth=2.5)
        else:
            print(f"  [{name}] 无能耗历史数据")

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Total Energy Consumption per Round', fontsize=12)
    ax1.set_title('Energy Consumption', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右图：Lyapunov 虚拟队列 L(Q) 历史
    ax2 = axes[1]
    for name, data in results.items():
        lyapunov_history = data.get('lyapunov_history', [])
        if lyapunov_history:
            rounds = range(1, len(lyapunov_history) + 1)
            ax2.plot(rounds, lyapunov_history, label=name,
                     color=COLORS[name], linewidth=2.5)
        else:
            print(f"  [{name}] 无Lyapunov历史数据")

    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Lyapunov Function L(Q)', fontsize=12)
    ax2.set_title('Virtual Queue Stability L(Q)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Ablation Study: Effect of Lyapunov Optimization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图已保存: {output_path}")
    plt.close()


# ============================================================
# 3. Crypto 消融：准确率不变 + 加密开销统计
# ============================================================
def plot_crypto_ablation(save_dir=ABLATION_DIR, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'ablation_crypto.png')

    print("\n" + "=" * 60)
    print("消融实验 3：AES-256-GCM 加密开销分析")
    print("=" * 60)

    data_dict = load_ablation_data(save_dir)
    targets = ['Ours (Full)', 'w/o Crypto']
    results = {k: data_dict[k] for k in targets if k in data_dict}

    if len(results) < 2:
        print(f"缺少数据，需要: {targets}，已有: {list(results.keys())}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：准确率曲线对比（应该基本相同）
    ax1 = axes[0]
    for name, data in results.items():
        acc = data['test_accuracy']
        rounds = range(1, len(acc) + 1)
        smoothed = moving_average(acc, window_size=10)
        ax1.plot(rounds, smoothed, label=name, color=COLORS[name], linewidth=2.5)

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Accuracy (Should Be Same)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 右图：加密统计柱状图
    ax2 = axes[1]
    ours_data = results.get('Ours (Full)', {})
    crypto_stats = ours_data.get('crypto_statistics', {})

    if crypto_stats:
        labels = ['Total Encrypt Ops', 'Plaintext (KB)', 'Ciphertext (KB)']
        values = [
            crypto_stats.get('total_encrypt_ops', 0),
            crypto_stats.get('total_plaintext_KB', 0),
            crypto_stats.get('total_ciphertext_KB', crypto_stats.get('total_plaintext_KB', 0) * 1.001),
        ]
        bars = ax2.bar(labels, values, color=['#2ca02c', '#1f77b4', '#ff7f0e'],
                       alpha=0.8, edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h,
                     f'{h:.1f}', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('AES-256-GCM Overhead Statistics', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        print(f"\n加密统计:")
        for k, v in crypto_stats.items():
            print(f"  {k}: {v}")
    else:
        ax2.text(0.5, 0.5, 'No crypto statistics available',
                 ha='center', va='center', transform=ax2.transAxes)

    plt.suptitle('Ablation Study: AES-256-GCM Encryption Overhead',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    print("消融实验可视化")
    print(f"数据目录: {ABLATION_DIR}")
    plot_sv_ablation()
    plot_energy_lyapunov_ablation()
    plot_crypto_ablation()
    print("\n所有消融图表生成完毕！")
