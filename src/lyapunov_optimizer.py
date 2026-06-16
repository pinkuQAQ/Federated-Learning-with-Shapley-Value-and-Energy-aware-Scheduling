#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
lyapunov_optimizer.py - Lyapunov-based client scheduling.

This module maintains virtual energy queues and computes the deployed
utility-minus-queue score used by the Shapley, battery, and channel-aware
client scheduler.
"""

import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# Plot style.
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class LyapunovTripleScheduler:
    """
    Lyapunov scheduler used by the paper implementation.

    Objective: favor high-contribution clients while respecting long-term
    average energy budgets. The virtual queue Q tracks cumulative budget
    pressure and enters the selection score as an energy penalty.
    """

    def __init__(self,
                 num_clients: int,
                 V: float = 10.0,
                 energy_budget: float = 2.0):
        """
        Initialize the Lyapunov scheduler.

        Args:
            num_clients: Number of clients.
            V: Control parameter; larger values emphasize utility.
            energy_budget: Per-round average energy budget.
        """
        self.num_clients = num_clients
        self.V = V
        self.energy_budget = energy_budget

        # Virtual queues track per-client energy budget pressure.
        self.energy_queue = np.zeros(num_clients)

        # History records for reporting and visualization.
        self.queue_history = []
        self.lyapunov_history = []

        # Initial Lyapunov value.
        self.lyapunov_history.append(self.compute_lyapunov_function())

        print("\n[Lyapunov Scheduler] Initialized")
        print(f"  Clients: {num_clients}")
        print(f"  Control parameter V: {V}")
        print(f"  Energy budget: {self.energy_budget}")

    def compute_lyapunov_function(self) -> float:
        """Compute L(Q) = 1/2 * ||Q||^2."""
        return 0.5 * np.sum(self.energy_queue ** 2)

    def update_queue(self,
                     energy_consumed: np.ndarray,
                     selected_clients: List[int],
                     round_num: int):
        """
        Update the virtual energy queues.

        Q(t+1) = max(0, Q(t) + e(t) - budget)

        Args:
            energy_consumed: Actual energy consumed by selected clients.
            selected_clients: Selected client ids.
            round_num: Current communication round.
        """
        # Positive values mean this round exceeded the per-client budget.
        energy_violation = energy_consumed - self.energy_budget
        self.energy_queue[selected_clients] = np.maximum(
            0, self.energy_queue[selected_clients] + energy_violation
        )

        # Record history.
        self.queue_history.append({
            'round': round_num,
            'energy_queue': np.mean(self.energy_queue)
        })
        self.lyapunov_history.append(self.compute_lyapunov_function())

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
        """Return scheduler statistics."""
        return {
            'queue_mean': np.mean(self.energy_queue),
            'queue_max': np.max(self.energy_queue),
            'lyapunov_value': self.lyapunov_history[-1] if self.lyapunov_history else 0
        }

    def print_statistics(self, round_num: int):
        """Print current scheduler statistics."""
        print(f"\n[Lyapunov Scheduler - Round {round_num}]")
        print(f"  Mean energy queue: {np.mean(self.energy_queue):.2f}")
        print(f"  Max energy queue: {np.max(self.energy_queue):.2f}")
        if len(self.lyapunov_history) > 0:
            print(f"  Lyapunov value: {self.lyapunov_history[-1]:.2f}")

    def visualize_optimization(self, save_path: str = None):
        """Visualize the Lyapunov value over time."""
        if not self.queue_history:
            print("No history data available for visualization.")
            return

        rounds = [q['round'] for q in self.queue_history]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        ax.plot(rounds, self.lyapunov_history[1:], 'purple', linewidth=2)
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Lyapunov Function L(Q)')
        ax.set_title('Lyapunov Function Over Time')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Lyapunov Optimization for Dual Scheduling', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()
