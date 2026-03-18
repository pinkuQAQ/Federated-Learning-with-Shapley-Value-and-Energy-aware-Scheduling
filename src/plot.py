#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
平滑收敛曲线对比脚本
使用移动平均平滑准确率曲线，更清晰地展示收敛趋势
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============= 配置参数 =============
SAVE_DIR = '../save/baseline_cmp_20260317'  # 修改此路径以指定要分析的文件夹
OUTPUT_DIR = '../save'  # 输出图表保存路径
# ===================================

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def moving_average(data, window_size=5):
    """
    计算移动平均

    Args:
        data: 原始数据列表
        window_size: 窗口大小

    Returns:
        平滑后的数据
    """
    if len(data) < window_size:
        return data

    smoothed = []
    for i in range(len(data)):
        if i < window_size - 1:
            # 前面不足窗口大小的点，使用从开始到当前的平均
            smoothed.append(np.mean(data[:i+1]))
        else:
            # 使用窗口平均
            smoothed.append(np.mean(data[i-window_size+1:i+1]))

    return smoothed


def exponential_moving_average(data, alpha=0.1):
    """
    计算指数移动平均（EMA）

    Args:
        data: 原始数据列表
        alpha: 平滑系数（0-1），越小越平滑

    Returns:
        平滑后的数据
    """
    if len(data) == 0:
        return data

    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])

    return ema


def load_experiment_result(save_dir, pattern):
    """加载实验结果"""
    pkl_files = list(Path(save_dir).glob(pattern))

    if not pkl_files:
        return None

    # 使用最新的文件
    pkl_file = sorted(pkl_files, key=lambda x: x.stat().st_mtime)[-1]

    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"加载: {pkl_file.name}")
        return data
    except Exception as e:
        print(f"加载失败 {pkl_file}: {e}")
        return None


def plot_comparison_with_smoothing(save_dir=SAVE_DIR, output_path=None):
    """
    绘制带平滑的对比曲线
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'smoothed_comparison.png')

    print("=" * 60)
    print("加载实验结果并生成平滑对比图")
    print("=" * 60)

    # 动态扫描所有 pkl 文件
    import glob
    pkl_files = glob.glob(os.path.join(save_dir, '*.pkl'))

    results = {}
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = {}

    for i, pkl_file in enumerate(pkl_files):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if 'test_accuracy' not in data:
                continue

            # 识别方法名
            filename = os.path.basename(pkl_file)
            args = data.get('args', {})

            if args.get('use_lyapunov') and args.get('use_shapley'):
                name = 'Ours'
            elif 'fedprox' in filename.lower() or args.get('use_fedprox'):
                name = 'FedProx'
            elif 'ucb' in filename.lower() or args.get('selection_method') == 'ucb':
                name = 'UCB1'
            elif 'poc' in filename.lower() or args.get('selection_method') == 'poc':
                candidate_size = args.get('poc_candidate_size', '')
                name = f'PoC (d={candidate_size})' if candidate_size else 'PoC'
            elif 'random' in filename.lower() or args.get('selection_method') == 'random':
                name = 'FedAvg'
            else:
                name = filename.replace('.pkl', '')

            results[name] = data['test_accuracy']
            colors[name] = colors_list[i % len(colors_list)]
            print(f"加载: {name}")

        except Exception as e:
            print(f"加载失败 {pkl_file}: {e}")

    if not results:
        print("错误: 未找到任何实验结果！")
        return

    print(f"\n成功加载 {len(results)} 个实验结果\n")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 原始曲线
    ax1 = axes[0, 0]
    for name, acc_history in results.items():
        rounds = list(range(1, len(acc_history) + 1))
        ax1.plot(rounds, acc_history, label=name, color=colors[name], alpha=0.3, linewidth=1)

    ax1.set_xlabel('Training Round', fontsize=11)
    ax1.set_ylabel('Test Accuracy', fontsize=11)
    ax1.set_title('Original Curves (Raw Data)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 移动平均平滑（窗口=10）
    ax2 = axes[0, 1]
    window_size = 10
    for name, acc_history in results.items():
        rounds = list(range(1, len(acc_history) + 1))
        smoothed = moving_average(acc_history, window_size=window_size)
        ax2.plot(rounds, smoothed, label=name, color=colors[name], linewidth=2)

    ax2.set_xlabel('Training Round', fontsize=11)
    ax2.set_ylabel('Test Accuracy', fontsize=11)
    ax2.set_title(f'Moving Average (Window={window_size})', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 指数移动平均（alpha=0.1）
    ax3 = axes[1, 0]
    alpha = 0.1
    for name, acc_history in results.items():
        rounds = list(range(1, len(acc_history) + 1))
        ema = exponential_moving_average(acc_history, alpha=alpha)
        ax3.plot(rounds, ema, label=name, color=colors[name], linewidth=2)

    ax3.set_xlabel('Training Round', fontsize=11)
    ax3.set_ylabel('Test Accuracy', fontsize=11)
    ax3.set_title(f'Exponential Moving Average (α={alpha})', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. 对比：原始 vs 平滑（显示所有实验）
    ax4 = axes[1, 1]

    for name in results.keys():
        acc_history = results[name]
        rounds = list(range(1, len(acc_history) + 1))

        # 原始曲线（半透明）
        ax4.plot(rounds, acc_history, color=colors[name], alpha=0.2,
                linewidth=1, linestyle='--')

        # 平滑曲线（实线）
        smoothed = moving_average(acc_history, window_size=10)
        ax4.plot(rounds, smoothed, label=f'{name} (Smoothed)',
                color=colors[name], linewidth=2.5)

    ax4.set_xlabel('Training Round', fontsize=11)
    ax4.set_ylabel('Test Accuracy', fontsize=11)
    ax4.set_title('Key Comparison: Raw vs Smoothed', fontsize=12, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Federated Learning Convergence Analysis with Smoothing',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # 保存图表
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n平滑对比图已保存到: {output_path}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)

    for name, acc_history in results.items():
        final_acc = acc_history[-1]
        max_acc = max(acc_history)
        mean_acc = np.mean(acc_history)
        std_acc = np.std(acc_history)

        # 计算平滑后的统计
        smoothed = moving_average(acc_history, window_size=10)
        smoothed_final = smoothed[-1]
        smoothed_std = np.std(smoothed)

        print(f"\n{name}:")
        print(f"  原始数据:")
        print(f"    最终准确率: {final_acc:.4f}")
        print(f"    最大准确率: {max_acc:.4f}")
        print(f"    平均准确率: {mean_acc:.4f}")
        print(f"    标准差: {std_acc:.4f}")
        print(f"  平滑后 (窗口=10):")
        print(f"    最终准确率: {smoothed_final:.4f}")
        print(f"    标准差: {smoothed_std:.4f} (降低 {(1-smoothed_std/std_acc)*100:.1f}%)")

    print("\n" + "=" * 60)
    plt.close()


def plot_comprehensive_metrics(save_dir=SAVE_DIR, output_path=None):
    """
    绘制综合指标：准确率和收敛速度对比（所有方法）
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'comprehensive_metrics.png')

    print("\n" + "=" * 60)
    print("生成综合指标可视化")
    print("=" * 60)

    # 加载所有实验结果
    pkl_files = list(Path(save_dir).glob('*.pkl'))
    results = {}

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if 'test_accuracy' not in data:
                continue

            filename = pkl_file.name
            args = data.get('args', {})

            if args.get('use_lyapunov') and args.get('use_shapley'):
                name = 'Ours'
            elif args.get('use_fedprox'):
                name = 'FedProx'
            elif 'ucb' in filename.lower() or args.get('selection_method') == 'ucb':
                name = 'UCB1'
            elif 'poc' in filename.lower() or args.get('selection_method') == 'poc':
                name = 'PoC'
            elif 'random' in filename.lower() and not args.get('use_fedprox'):
                name = 'FedAvg'
            else:
                continue

            results[name] = data
            print(f"加载: {name}")
        except Exception as e:
            print(f"加载失败 {pkl_file.name}: {e}")

    if not results:
        print("错误: 未找到实验结果")
        return

    # 颜色映射
    colors = {
        'Ours': '#2ca02c',
        'FedAvg': '#ff7f0e',
        'PoC': '#1f77b4',
        'UCB1': '#9467bd',
        'FedProx': '#d62728'
    }

    # 创建2个子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 收敛曲线对比
    ax1 = axes[0]
    for name, data in results.items():
        acc_history = data['test_accuracy']
        rounds = range(1, len(acc_history) + 1)
        smoothed = moving_average(acc_history, window_size=5)
        ax1.plot(rounds, smoothed, label=name, linewidth=2.5, color=colors.get(name))

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Convergence Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 2. 收敛速度（达到目标准确率的轮数）
    ax2 = axes[1]
    target_accs = [0.40, 0.45, 0.50]
    method_names = list(results.keys())
    rounds_to_target = {t: [] for t in target_accs}

    for name in method_names:
        acc_history = results[name]['test_accuracy']
        for target in target_accs:
            rounds = next((i+1 for i, acc in enumerate(acc_history) if acc >= target), len(acc_history))
            rounds_to_target[target].append(rounds)

    x = np.arange(len(method_names))
    width = 0.25
    for i, target in enumerate(target_accs):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, rounds_to_target[target], width,
               label=f'{int(target*100)}% Acc', alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                    ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('Rounds to Target', fontsize=12)
    ax2.set_title('Convergence Speed', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('Ours vs FedAvg: Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n综合指标图已保存到: {output_path}")

    # 打印统计表格
    print("\n" + "=" * 60)
    print("性能对比表")
    print("=" * 60)
    print(f"{'Method':<15} {'Final Acc':<15} {'Conv@40%':<15} {'Conv@50%':<15}")
    print("-" * 60)

    for name, data in results.items():
        final_acc = data['test_accuracy'][-1]
        acc_history = data['test_accuracy']
        conv_40 = next((i+1 for i, acc in enumerate(acc_history) if acc >= 0.40), len(acc_history))
        conv_50 = next((i+1 for i, acc in enumerate(acc_history) if acc >= 0.50), len(acc_history))

        print(f"{name:<15} {final_acc:<15.4f} {conv_40:<15} {conv_50:<15}")

    print("=" * 60)
    plt.close()


def plot_crypto_overhead(save_dir=SAVE_DIR, output_path=None):
    """
    绘制加密开销对比（消融实验用）
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'crypto_overhead.png')

    print("\n" + "=" * 60)
    print("生成加密开销可视化")
    print("=" * 60)

    pkl_files = list(Path(save_dir).glob('*.pkl'))
    crypto_data, no_crypto_data = None, None

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if 'test_accuracy' not in data:
                continue

            args = data.get('args', {})
            if args.get('use_crypto') and args.get('use_lyapunov'):
                crypto_data = data
                print(f"加载加密实验: {pkl_file.name}")
            elif not args.get('use_crypto') and args.get('use_lyapunov'):
                no_crypto_data = data
                print(f"加载无加密实验: {pkl_file.name}")
        except Exception as e:
            print(f"加载失败: {e}")

    if not crypto_data:
        print("未找到加密实验数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 1. 准确率对比
    ax1 = axes[0]
    if no_crypto_data:
        methods = ['No Crypto', 'With Crypto']
        accs = [no_crypto_data['test_accuracy'][-1], crypto_data['test_accuracy'][-1]]
        bars = ax1.bar(methods, accs, color=['#99ff99', '#ffcc99'], edgecolor='black', linewidth=1.2)
        ax1.set_ylabel('Final Accuracy', fontsize=11)
        ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=10)

    # 2. 收敛曲线对比
    ax2 = axes[1]
    if no_crypto_data:
        rounds = range(1, len(no_crypto_data['test_accuracy']) + 1)
        ax2.plot(rounds, no_crypto_data['test_accuracy'], label='No Crypto', linewidth=2, color='#2ca02c')
        ax2.plot(rounds, crypto_data['test_accuracy'], label='With Crypto', linewidth=2, color='#ff7f0e')
        ax2.set_xlabel('Training Round', fontsize=11)
        ax2.set_ylabel('Test Accuracy', fontsize=11)
        ax2.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.suptitle('Encryption (AES-256-GCM) Overhead Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n加密开销图已保存到: {output_path}")

    if 'crypto_statistics' in crypto_data:
        stats = crypto_data['crypto_statistics']
        print("\n" + "=" * 60)
        print("加密统计信息")
        print("=" * 60)
        print(f"算法: {stats.get('algorithm', 'AES-256-GCM')}")
        print(f"总加密操作: {stats.get('total_encrypt_ops', 0)}")
        print(f"数据膨胀率: {stats.get('overhead_ratio', 0):.3f}x")
        if no_crypto_data:
            acc_loss = (no_crypto_data['test_accuracy'][-1] - crypto_data['test_accuracy'][-1]) * 100
            print(f"准确率损失: {acc_loss:.2f}%")
        print("=" * 60)

    plt.close()


if __name__ == '__main__':
    plot_comparison_with_smoothing()
    plot_comprehensive_metrics(save_dir=SAVE_DIR, output_path=os.path.join(OUTPUT_DIR, 'comprehensive_metrics.png'))
