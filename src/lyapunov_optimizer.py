#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
lyapunov_optimizer.py - 基于李雅普诺夫优化的双重调度权重动态调整
实现动态权重优化，平衡 Shapley值和能量消耗两个目标
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class LyapunovTripleScheduler:
    """
    基于李雅普诺夫优化的双重调度器（论文标准实现）

    优化目标：最大化模型性能（Shapley值）
    约束条件：长期平均能量消耗 ≤ 预算

    方法：使用虚拟队列Q跟踪能量约束违反，在选择时用Q作为惩罚项
    """

    def __init__(self,
                 num_clients: int,
                 V: float = 10.0,
                 energy_budget: float = 2.0):
        """
        初始化李雅普诺夫优化器

        Args:
            num_clients: 客户端数量
            V: 控制参数，权衡性能和能量约束（越大越重视性能）
            energy_budget: 平均能量消耗预算
        """
        self.num_clients = num_clients
        self.V = V
        self.energy_budget = energy_budget

        # 虚拟队列（跟踪能量约束违反）
        self.energy_queue = np.zeros(num_clients)

        # 历史记录
        self.queue_history = []
        self.lyapunov_history = []

        # 记录初始李雅普诺夫函数值
        self.lyapunov_history.append(self.compute_lyapunov_function())

        print(f"\n[李雅普诺夫优化器] 初始化完成")
        print(f"  客户端数量: {num_clients}")
        print(f"  控制参数 V: {V}")
        print(f"  能量预算: {self.energy_budget}")

    def compute_lyapunov_function(self) -> float:
        """计算李雅普诺夫函数值 L(Q) = 1/2 * ||Q||^2"""
        return 0.5 * np.sum(self.energy_queue ** 2)

    def update_queue(self,
                     energy_consumed: np.ndarray,
                     selected_clients: List[int],
                     round_num: int):
        """
        更新虚拟队列（基于瞬时能量违反量）

        Q(t+1) = max(0, Q(t) + e(t) - budget)

        Args:
            energy_consumed: 本轮选中客户端的实际能量消耗
            selected_clients: 本轮选中的客户端
            round_num: 当前轮次
        """
        # 瞬时能量违反量
        energy_violation = energy_consumed - self.energy_budget
        self.energy_queue[selected_clients] = np.maximum(
            0, self.energy_queue[selected_clients] + energy_violation
        )

        # 记录历史
        self.queue_history.append({
            'round': round_num,
            'energy_queue': np.mean(self.energy_queue)
        })
        self.lyapunov_history.append(self.compute_lyapunov_function())

    def compute_scores(self, shapley_values: np.ndarray, energy_consumed: np.ndarray) -> np.ndarray:
        """
        计算客户端得分（用于选择）——标准 Lyapunov drift-plus-penalty

        Score_n = V · φ̂_n - Q_n · Ê_n

        注意：Q_n 不做归一化，让虚拟队列自然增长以平衡 V·SV 项。
        这是 Lyapunov 优化的核心机制：当能量约束持续违反时，Q 增大，
        最终超过 V·SV，自动将选择转向低能耗客户端。

        Args:
            shapley_values: Shapley值
            energy_consumed: 预期能量消耗

        Returns:
            scores: 每个客户端的得分（越高越好）
        """
        # 归一化Shapley值到[0,1]
        sv_range = shapley_values.max() - shapley_values.min()
        if sv_range > 1e-10:
            sv_norm = (shapley_values - shapley_values.min()) / sv_range
        else:
            sv_norm = np.ones_like(shapley_values)

        # 归一化能量消耗到[0,1]
        energy_max = energy_consumed.max()
        if energy_max > 1e-10:
            energy_norm = energy_consumed / energy_max
        else:
            energy_norm = np.zeros_like(energy_consumed)

        # 标准 Lyapunov 得分：V * SV_norm - Q * E_norm
        # Q 不归一化，使其能够随约束违反自然增长，最终压过 V·SV 项
        scores = self.V * sv_norm - self.energy_queue * energy_norm

        return scores

    @staticmethod
    def _minmax(values: np.ndarray, default: float = 1.0) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        v_min = np.nanmin(values)
        v_max = np.nanmax(values)
        if not np.isfinite(v_min) or not np.isfinite(v_max):
            return np.ones_like(values) * default
        if v_max - v_min > 1e-10:
            return (values - v_min) / (v_max - v_min)
        return np.ones_like(values) * default

    def compute_scores(self, shapley_values: np.ndarray, energy_consumed: np.ndarray,
                       battery_scores: np.ndarray = None,
                       channel_gains: np.ndarray = None,
                       sv_weight: float = 0.7,
                       battery_weight: float = 0.15,
                       channel_weight: float = 0.15) -> np.ndarray:
        """
        Compute the deployed Lyapunov scheduling score.

        Instant utility rewards contribution, residual battery, and channel quality:
            U_n = w_sv * SV_n + w_b * B_n + w_c * H_n

        The virtual queue keeps the long-term energy penalty:
            Score_n = V * U_n - Q_n * E_n
        """
        sv_norm = self._minmax(shapley_values, default=1.0)

        energy_consumed = np.asarray(energy_consumed, dtype=np.float64)
        energy_max = np.nanmax(energy_consumed)
        if np.isfinite(energy_max) and energy_max > 1e-10:
            energy_norm = energy_consumed / energy_max
        else:
            energy_norm = np.zeros_like(energy_consumed)

        if battery_scores is None:
            battery_norm = np.zeros_like(sv_norm)
        else:
            battery_norm = self._minmax(battery_scores, default=1.0)

        if channel_gains is None:
            channel_norm = np.zeros_like(sv_norm)
        else:
            channel_norm = self._minmax(channel_gains, default=1.0)

        total_weight = max(sv_weight + battery_weight + channel_weight, 1e-12)
        sv_weight = sv_weight / total_weight
        battery_weight = battery_weight / total_weight
        channel_weight = channel_weight / total_weight

        utility = (
            sv_weight * sv_norm
            + battery_weight * battery_norm
            + channel_weight * channel_norm
        )
        scores = self.V * utility - self.energy_queue * energy_norm

        return scores

    def get_statistics(self) -> Dict:
        """获取优化统计信息"""
        return {
            'queue_mean': np.mean(self.energy_queue),
            'queue_max': np.max(self.energy_queue),
            'lyapunov_value': self.lyapunov_history[-1] if self.lyapunov_history else 0
        }

    def print_statistics(self, round_num: int):
        """打印当前统计信息"""
        print(f"\n[李雅普诺夫优化 - 轮次 {round_num}]")
        print(f"  能量队列均值: {np.mean(self.energy_queue):.2f}")
        print(f"  能量队列最大值: {np.max(self.energy_queue):.2f}")
        if len(self.lyapunov_history) > 0:
            print(f"  李雅普诺夫函数: {self.lyapunov_history[-1]:.2f}")

    def visualize_optimization(self, save_path: str = None):
        """可视化优化过程"""
        if not self.queue_history:
            print("没有历史数据可供可视化")
            return

        rounds = [q['round'] for q in self.queue_history]
        queue_values = [q['energy_queue'] for q in self.queue_history]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # 李雅普诺夫函数
        ax.plot(rounds, self.lyapunov_history[1:], 'purple', linewidth=2)
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Lyapunov Function L(Q)')
        ax.set_title('Lyapunov Function Over Time')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Lyapunov Optimization for Dual Scheduling', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

