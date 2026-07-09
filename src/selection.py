#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
selection.py - 客户端选择策略模块
包含基于Shapley值的贪心选择、轮询选择等策略
"""

import numpy as np
import random
from typing import Dict, List


def round_robin_selection(num_clients: int, num_selected: int,
                          current_round: int, participation_counts: np.ndarray = None) -> List[int]:
    """
    轮询选择策略（用于初始阶段）

    Args:
        num_clients: 总客户端数
        num_selected: 每轮选择数
        current_round: 当前轮次
        participation_counts: 客户端历史参与次数（可选）

    Returns:
        selected_clients: 选择的客户端索引列表
    """
    if participation_counts is not None:
        # 优先选择参与次数少的客户端
        sorted_indices = np.argsort(participation_counts)
        return sorted_indices[:num_selected].tolist()
    else:
        # 简单的轮询
        start_idx = (current_round * num_selected) % num_clients
        indices = []

        for i in range(num_selected):
            idx = (start_idx + i) % num_clients
            indices.append(idx)

        return indices


def greedy_shapley_selection(shapley_values: np.ndarray, num_selected: int) -> List[int]:
    """
    最简单的基于Shapley值的贪心客户端选择
    只选择Shapley值最高的客户端

    Args:
        shapley_values: 每个客户端的Shapley值数组
        num_selected: 要选择的客户端数量

    Returns:
        selected_clients: 选择的客户端索引列表
    """
    # 直接返回SV值最高的num_selected个客户端
    return np.argsort(shapley_values)[-num_selected:][::-1].tolist()


def hybrid_selection(shapley_values: np.ndarray, num_selected: int,
                     participation_counts: np.ndarray,
                     current_round: int,
                     initial_rounds: int = 10,) -> List[int]:

    if current_round < initial_rounds:
        # 初始阶段：轮询，优先选择参与次数少的客户端
        return round_robin_selection(len(shapley_values), num_selected,
                                     current_round, participation_counts)
    else:
        # 贪心阶段：基于Shapley值选择
        return greedy_shapley_selection(shapley_values, num_selected)


def random_selection(num_clients: int, num_selected: int) -> List[int]:
    """
    随机选择策略（基线方法）
    """
    return np.random.choice(num_clients, min(num_selected, num_clients),
                            replace=False).tolist()


def energy_aware_selection(shapley_values: np.ndarray,
                           energy_scores: np.ndarray,
                           num_selected: int,
                           shapley_weight: float = 0.5,
                           energy_weight: float = 0.5,
                           available_clients: List[int] = None) -> List[int]:
    """
    能量感知的客户端选择策略
    综合考虑 Shapley 值和能量状态

    Args:
        shapley_values: 每个客户端的 Shapley 值数组
        energy_scores: 每个客户端的能量得分数组（0-1之间，越高越好）
        num_selected: 要选择的客户端数量
        shapley_weight: Shapley 值的权重（默认 0.5）
        energy_weight: 能量得分的权重（默认 0.5）
        available_clients: 可用客户端列表（能量充足的客户端）

    Returns:
        selected_clients: 选择的客户端索引列表
    """
    num_clients = len(shapley_values)

    # 如果指定了可用客户端，只在这些客户端中选择
    if available_clients is not None and len(available_clients) > 0:
        candidate_mask = np.zeros(num_clients, dtype=bool)
        candidate_mask[available_clients] = True
    else:
        candidate_mask = np.ones(num_clients, dtype=bool)

    # 归一化 Shapley 值到 [0, 1]
    sv_min = np.min(shapley_values[candidate_mask])
    sv_max = np.max(shapley_values[candidate_mask])

    if sv_max > sv_min:
        normalized_sv = (shapley_values - sv_min) / (sv_max - sv_min)
    else:
        # 所有SV相等 → 所有客户端贡献相同 → 选择由能量决定
        normalized_sv = np.ones_like(shapley_values)

    # 计算综合得分
    composite_scores = (shapley_weight * normalized_sv +
                       energy_weight * energy_scores)

    # 只考虑可用客户端
    composite_scores[~candidate_mask] = -np.inf

    # 选择得分最高的客户端
    if np.sum(candidate_mask) < num_selected:
        # 可用客户端不足，选择所有可用的
        selected = np.where(candidate_mask)[0].tolist()
        print(f"  [警告] 可用客户端不足: {len(selected)}/{num_selected}")
    else:
        selected_indices = np.argsort(composite_scores)[-num_selected:][::-1]
        selected = selected_indices.tolist()

    return selected


def hybrid_energy_aware_selection(shapley_values: np.ndarray,
                                  energy_scores: np.ndarray,
                                  num_selected: int,
                                  participation_counts: np.ndarray,
                                  current_round: int,
                                  initial_rounds: int = 10,
                                  shapley_weight: float = 0.5,
                                  energy_weight: float = 0.5,
                                  available_clients: List[int] = None) -> List[int]:
    """
    混合能量感知选择策略
    初始阶段：轮询选择（考虑能量）
    后续阶段：基于 Shapley 值和能量的综合选择

    Args:
        shapley_values: Shapley 值数组
        energy_scores: 能量得分数组
        num_selected: 选择数量
        participation_counts: 参与次数统计
        current_round: 当前轮次
        initial_rounds: 初始轮询轮数
        shapley_weight: Shapley 权重
        energy_weight: 能量权重
        available_clients: 可用客户端列表

    Returns:
        selected_clients: 选择的客户端索引列表
    """
    if current_round < initial_rounds:
        # 初始阶段：普通轮询选择（和hybrid_selection保持一致）
        num_clients = len(shapley_values)
        return round_robin_selection(num_clients, num_selected,
                                     current_round, participation_counts)
    else:
        # 贪心阶段：基于 Shapley 值和能量的综合选择
        return energy_aware_selection(
            shapley_values=shapley_values,
            energy_scores=energy_scores,
            num_selected=num_selected,
            shapley_weight=shapley_weight,
            energy_weight=energy_weight,
            available_clients=available_clients
        )


def ucb_selection(num_clients: int, num_selected: int,
                  ucb_rewards: np.ndarray,
                  ucb_counts: np.ndarray,
                  current_round: int,
                  c: float = 1.0,
                  available_clients: List[int] = None) -> List[int]:
    """
    UCB1 客户端选择策略 (Auer et al., 2002)
    Score_i = reward_i + c * sqrt(2 * ln(t) / n_i)

    Args:
        num_clients: 总客户端数
        num_selected: 每轮选择数
        ucb_rewards: 每个客户端的奖励估计（本地损失均值，越高越优先）
        ucb_counts: 每个客户端的历史参与次数
        current_round: 当前轮次（从1开始）
        c: 探索系数，控制探索与利用的平衡
        available_clients: 可用客户端列表

    Returns:
        selected_clients: 选择的客户端索引列表
    """
    t = max(current_round, 1)
    ucb_scores = np.zeros(num_clients)

    for i in range(num_clients):
        n_i = max(ucb_counts[i], 1)
        ucb_scores[i] = ucb_rewards[i] + c * np.sqrt(2 * np.log(t) / n_i)

    if available_clients is not None and len(available_clients) > 0:
        masked = np.full(num_clients, -np.inf)
        masked[available_clients] = ucb_scores[available_clients]
        ucb_scores = masked

    return np.argsort(ucb_scores)[-num_selected:][::-1].tolist()


def power_of_choice_selection(client_losses: np.ndarray,
                              num_selected: int,
                              candidate_size: int = None,
                              available_clients: List[int] = None) -> List[int]:
    """
    Power of Choice 客户端选择策略 (Cho et al., 2020)
    从 candidate_size 个随机候选中选择损失最高的 num_selected 个客户端

    Args:
        client_losses: 每个客户端的本地损失数组（上一轮的损失作为代理）
        num_selected: 最终选择的客户端数量 (K)
        candidate_size: 候选池大小 (d)，默认为 num_selected * 2
        available_clients: 可用客户端列表，None 表示全部可用

    Returns:
        selected_clients: 选择的客户端索引列表
    """
    num_clients = len(client_losses)

    if available_clients is None:
        available_clients = list(range(num_clients))

    if candidate_size is None:
        candidate_size = min(num_selected * 2, len(available_clients))

    candidate_size = min(candidate_size, len(available_clients))
    num_selected = min(num_selected, candidate_size)

    # 从可用客户端中随机采样候选集
    candidates = np.random.choice(available_clients, candidate_size, replace=False).tolist()

    # 从候选集中选损失最高的 K 个
    candidate_losses = [(c, client_losses[c]) for c in candidates]
    candidate_losses.sort(key=lambda x: x[1], reverse=True)

    return [c for c, _ in candidate_losses[:num_selected]]

class OortSelector:
    """Training participant selector following Oort's Algorithm 1.

    The selector keeps Oort's observed statistical utility U(i), duration D(i),
    last selected round L(i), pacer target T, epsilon exploration schedule, and
    robust exploitation controls. Feedback is supplied after each local round.
    """

    def __init__(self,
                 num_clients: int,
                 sample_size: int,
                 epsilon: float = 0.9,
                 epsilon_decay: float = 0.98,
                 epsilon_min: float = 0.2,
                 pacer_step: float = 0.0,
                 pacer_window: int = 20,
                 straggler_penalty: float = 2.0,
                 cutoff_util: float = 0.95,
                 clip_percentile: float = 95.0,
                 blacklist_rounds: int = 0,
                 seed: int = 42):
        self.num_clients = int(num_clients)
        self.sample_size = int(sample_size)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.pacer_step = float(pacer_step)
        self.pacer_window = int(pacer_window)
        self.straggler_penalty = float(straggler_penalty)
        self.cutoff_util = float(cutoff_util)
        self.clip_percentile = float(clip_percentile)
        self.blacklist_rounds = int(blacklist_rounds)
        self.rng = np.random.RandomState(seed)

        self.round = 0
        self.target_duration = max(self.pacer_step, 1e-6)
        self.statistical_utility = np.zeros(self.num_clients, dtype=np.float64)
        self.durations = np.zeros(self.num_clients, dtype=np.float64)
        self.last_selected_round = -np.ones(self.num_clients, dtype=np.int64)
        self.participation_counts = np.zeros(self.num_clients, dtype=np.int64)
        self.explored = np.zeros(self.num_clients, dtype=bool)
        self.blacklist_until = np.zeros(self.num_clients, dtype=np.int64)
        self.round_utility_history = []
        self.selection_history = []

    def select(self, available_clients: List[int] = None) -> List[int]:
        self.round += 1
        if available_clients is None or len(available_clients) == 0:
            available_clients = list(range(self.num_clients))
        candidates = np.asarray(available_clients, dtype=np.int64)
        if candidates.size == 0:
            return []

        candidates = candidates[
            (candidates >= 0)
            & (candidates < self.num_clients)
            & (self.blacklist_until[candidates] <= self.round)
        ]
        if candidates.size == 0:
            candidates = np.asarray(available_clients, dtype=np.int64)

        k = min(self.sample_size, int(candidates.size))
        explored = [int(c) for c in candidates if self.explored[c]]
        unexplored = [int(c) for c in candidates if not self.explored[c]]

        explore_k = min(int(np.ceil(self.epsilon * k)), len(unexplored))
        exploit_k = max(k - explore_k, 0)
        if exploit_k > len(explored):
            explore_k = min(k - len(explored), len(unexplored))
            exploit_k = k - explore_k

        selected = []
        if exploit_k > 0 and explored:
            selected.extend(self._sample_exploited(explored, exploit_k))

        if explore_k > 0 and unexplored:
            selected.extend(self._sample_unexplored_by_speed(unexplored, explore_k))

        if len(selected) < k:
            remaining = [int(c) for c in candidates if int(c) not in set(selected)]
            if remaining:
                selected.extend(self.rng.choice(
                    remaining, size=min(k - len(selected), len(remaining)), replace=False
                ).astype(int).tolist())

        self.selection_history.append({
            'round': self.round,
            'selected': list(selected),
            'epsilon': float(self.epsilon),
            'target_duration': float(self.target_duration),
        })
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return selected[:k]

    def update_feedback(self, feedback: Dict[int, Dict[str, float]]) -> None:
        if not feedback:
            self.round_utility_history.append(0.0)
            self._update_pacer()
            return

        raw_utils = []
        for client_id, item in feedback.items():
            loss_square_mean = max(float(item.get('loss_square_mean', item.get('loss', 0.0) ** 2)), 0.0)
            samples = max(float(item.get('num_samples', 1.0)), 1.0)
            duration = max(float(item.get('duration', 0.0)), 1e-6)
            utility = samples * np.sqrt(loss_square_mean)
            if np.isfinite(utility):
                raw_utils.append(utility)

        clip_value = np.inf
        if raw_utils and self.clip_percentile < 100.0:
            clip_value = float(np.percentile(raw_utils, self.clip_percentile))

        achieved_utility = 0.0
        for client_id, item in feedback.items():
            cid = int(client_id)
            if cid < 0 or cid >= self.num_clients:
                continue
            loss_square_mean = max(float(item.get('loss_square_mean', item.get('loss', 0.0) ** 2)), 0.0)
            samples = max(float(item.get('num_samples', 1.0)), 1.0)
            duration = max(float(item.get('duration', 0.0)), 1e-6)
            utility = samples * np.sqrt(loss_square_mean)
            if np.isfinite(clip_value):
                utility = min(utility, clip_value)
            if not np.isfinite(utility):
                utility = 0.0

            self.statistical_utility[cid] = utility
            self.durations[cid] = duration
            self.last_selected_round[cid] = self.round
            self.participation_counts[cid] += 1
            self.explored[cid] = True
            achieved_utility += utility

            if self.blacklist_rounds > 0 and self.participation_counts[cid] >= self.blacklist_rounds:
                self.blacklist_until[cid] = self.round + self.blacklist_rounds
                self.participation_counts[cid] = 0

        self.round_utility_history.append(float(achieved_utility))
        self._update_pacer()

    def get_state(self) -> Dict[str, object]:
        return {
            'epsilon': float(self.epsilon),
            'target_duration': float(self.target_duration),
            'statistical_utility': self.statistical_utility.copy(),
            'durations': self.durations.copy(),
            'last_selected_round': self.last_selected_round.copy(),
            'explored': self.explored.copy(),
            'participation_counts': self.participation_counts.copy(),
            'round_utility_history': list(self.round_utility_history),
            'selection_history': list(self.selection_history),
        }

    def _client_utilities(self, clients: List[int]) -> np.ndarray:
        clients_arr = np.asarray(clients, dtype=np.int64)
        utility = self.statistical_utility[clients_arr].copy()
        stale = np.maximum(self.round - self.last_selected_round[clients_arr], 1)
        utility += 0.1 * np.sqrt(np.log(max(self.round, 2)) / stale)

        durations = self.durations[clients_arr]
        slow = durations > self.target_duration
        if np.any(slow):
            penalty = (self.target_duration / np.maximum(durations[slow], 1e-6)) ** self.straggler_penalty
            utility[slow] *= penalty
        utility[~np.isfinite(utility)] = 0.0
        return np.maximum(utility, 0.0)

    def _sample_exploited(self, clients: List[int], k: int) -> List[int]:
        utilities = self._client_utilities(clients)
        if utilities.size == 0:
            return []

        sorted_utils = np.sort(utilities)[::-1]
        cutoff_index = min(max(k - 1, 0), sorted_utils.size - 1)
        threshold = self.cutoff_util * sorted_utils[cutoff_index]
        pool_mask = utilities >= threshold
        pool = np.asarray(clients, dtype=np.int64)[pool_mask]
        pool_utils = utilities[pool_mask]
        if pool.size == 0:
            pool = np.asarray(clients, dtype=np.int64)
            pool_utils = utilities

        probs = pool_utils / pool_utils.sum() if pool_utils.sum() > 1e-12 else None
        return self.rng.choice(pool, size=min(k, pool.size), replace=False, p=probs).astype(int).tolist()

    def _sample_unexplored_by_speed(self, clients: List[int], k: int) -> List[int]:
        speed_proxy = 1.0 / np.maximum(self.durations[np.asarray(clients, dtype=np.int64)], 1e-6)
        if np.allclose(speed_proxy, speed_proxy[0]):
            probs = None
        else:
            probs = speed_proxy / speed_proxy.sum()
        return self.rng.choice(clients, size=min(k, len(clients)), replace=False, p=probs).astype(int).tolist()

    def _update_pacer(self) -> None:
        w = self.pacer_window
        if self.pacer_step <= 0.0 or w <= 0 or len(self.round_utility_history) < 2 * w:
            return
        previous = np.sum(self.round_utility_history[-2 * w:-w])
        recent = np.sum(self.round_utility_history[-w:])
        if previous > recent:
            self.target_duration += self.pacer_step


def gradient_channel_aware_selection(learning_signals: np.ndarray,
                                     channel_gains: np.ndarray,
                                     energy_costs: np.ndarray,
                                     num_selected: int,
                                     learning_weight: float = 0.5,
                                     channel_weight: float = 0.3,
                                     energy_weight: float = 0.2,
                                     available_clients: List[int] = None) -> List[int]:
    """Gradient/channel/energy-aware scheduler adapted from AirComp FEEL.

    The original AirComp scheduler ranks devices by update importance, channel
    quality, and energy cost. In the digital-FL baseline we use stale local loss
    as the available learning-importance proxy and keep standard FedAvg
    aggregation unchanged.
    """
    num_clients = len(learning_signals)
    if available_clients is None or len(available_clients) == 0:
        available_clients = list(range(num_clients))

    def _minmax(x, default=1.0):
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        if not finite.any():
            return np.ones_like(x) * default
        lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
        if hi - lo <= 1e-10:
            return np.ones_like(x) * default
        return (x - lo) / (hi - lo)

    learning = _minmax(learning_signals, default=1.0)
    channel = _minmax(np.abs(channel_gains), default=1.0)
    energy = _minmax(energy_costs, default=0.0)

    score = (
        learning_weight * learning
        + channel_weight * channel
        - energy_weight * energy
    )

    mask = np.zeros(num_clients, dtype=bool)
    mask[available_clients] = True
    score[~mask] = -np.inf

    selected_count = min(num_selected, len(available_clients))
    return np.argsort(score)[-selected_count:][::-1].tolist()
