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


def softmax_score_selection(scores: np.ndarray,
                            num_selected: int,
                            temperature: float = 1.0,
                            available_clients: List[int] = None,
                            rng=None) -> List[int]:
    """Sample clients without replacement from a score-induced softmax."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite value")

    if available_clients is None:
        remaining = np.arange(scores.size, dtype=np.int64)
    else:
        remaining = np.asarray(list(dict.fromkeys(available_clients)), dtype=np.int64)
        if np.any(remaining < 0) or np.any(remaining >= scores.size):
            raise ValueError("available_clients contains an invalid client index")

    target = min(max(int(num_selected), 0), remaining.size)
    if target == 0:
        return []

    rng = np.random if rng is None else rng
    selected = []
    for _ in range(target):
        candidate_scores = scores[remaining]
        finite_mask = np.isfinite(candidate_scores)

        if np.any(finite_mask):
            max_score = np.max(candidate_scores[finite_mask])
            shifted = np.clip(
                (candidate_scores[finite_mask] - max_score) / temperature,
                -745.0,
                0.0,
            )
            weights = np.zeros(remaining.size, dtype=np.float64)
            weights[finite_mask] = np.exp(shifted)
            probabilities = weights / weights.sum()
        else:
            probabilities = np.full(remaining.size, 1.0 / remaining.size)

        selected_position = int(rng.choice(remaining.size, p=probabilities))
        selected.append(int(remaining[selected_position]))
        remaining = np.delete(remaining, selected_position)

    return selected


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
    """Source-faithful Oort training selector under a matched FL protocol.

    Clients are registered with initial size rewards and duration estimates.
    Rewards are normalized and clipped before temporal uncertainty is added.
    """

    def __init__(self,
                 num_clients: int,
                 sample_size: int,
                 epsilon: float = 0.9,
                 epsilon_decay: float = 0.98,
                 epsilon_min: float = 0.2,
                 pacer_step: float = 0.0,
                 pacer_window: int = 20,
                 pacer_delta: float = 5.0,
                 round_threshold: float = 10.0,
                 straggler_penalty: float = 2.0,
                 cutoff_util: float = 0.95,
                 clip_percentile: float = 95.0,
                 blacklist_rounds: int = 10,
                 blacklist_max_fraction: float = 0.3,
                 sample_window: float = 5.0,
                 initial_rewards: np.ndarray = None,
                 initial_durations: np.ndarray = None,
                 seed: int = 42):
        self.num_clients = int(num_clients)
        self.sample_size = int(sample_size)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.pacer_delta = float(pacer_step) if float(pacer_step) > 0.0 else float(pacer_delta)
        self.pacer_window = int(pacer_window)
        self.round_threshold = float(np.clip(round_threshold, 0.0, 100.0))
        self.straggler_penalty = float(straggler_penalty)
        self.cutoff_util = float(cutoff_util)
        self.clip_percentile = float(clip_percentile)
        self.blacklist_rounds = int(blacklist_rounds)
        self.blacklist_max_fraction = float(np.clip(blacklist_max_fraction, 0.0, 1.0))
        self.sample_window = max(float(sample_window), 1.0)
        self.rng = np.random.RandomState(seed)

        if initial_rewards is None:
            initial_rewards = np.ones(self.num_clients, dtype=np.float64)
        if initial_durations is None:
            initial_durations = np.ones(self.num_clients, dtype=np.float64)
        initial_rewards = np.asarray(initial_rewards, dtype=np.float64)
        initial_durations = np.asarray(initial_durations, dtype=np.float64)
        if initial_rewards.shape != (self.num_clients,):
            raise ValueError('initial_rewards must contain one value per client')
        if initial_durations.shape != (self.num_clients,):
            raise ValueError('initial_durations must contain one value per client')
        initial_rewards = np.nan_to_num(initial_rewards, nan=1.0, posinf=1.0, neginf=1.0)
        initial_durations = np.nan_to_num(initial_durations, nan=1.0, posinf=1.0, neginf=1.0)
        self.initial_rewards = np.maximum(initial_rewards, 1e-4)
        self.initial_durations = np.maximum(initial_durations, 1e-4)

        self.round = 0
        self.target_duration = float('inf')
        self.statistical_utility = self.initial_rewards.copy()
        self.durations = self.initial_durations.copy()
        self.last_selected_round = np.zeros(self.num_clients, dtype=np.int64)
        self.participation_counts = np.zeros(self.num_clients, dtype=np.int64)
        self.explored = np.zeros(self.num_clients, dtype=bool)
        self.blacklisted = np.zeros(self.num_clients, dtype=bool)
        self.exploit_utility_history = []
        self.explore_utility_history = []
        self.last_exploit_clients = []
        self.last_explore_clients = []
        self.selection_history = []

    def select(self, available_clients: List[int] = None) -> List[int]:
        self.round += 1
        if available_clients is None:
            available_clients = list(range(self.num_clients))
        candidates = np.asarray(available_clients, dtype=np.int64)
        if candidates.size == 0:
            return []

        candidates = np.unique(candidates[(candidates >= 0) & (candidates < self.num_clients)])
        self._refresh_blacklist()
        candidates = candidates[~self.blacklisted[candidates]]
        if candidates.size == 0:
            return []

        k = min(self.sample_size, int(candidates.size))
        self._update_pacer()
        self._update_target_duration()
        explored = [int(c) for c in candidates if self.explored[c]]
        unexplored = [int(c) for c in candidates if not self.explored[c]]

        epsilon_used = float(self.epsilon)
        exploit_k = min(int(k * (1.0 - epsilon_used)), len(explored))
        explore_k = min(k - exploit_k, len(unexplored))

        exploited = []
        if exploit_k > 0 and explored:
            exploited = self._sample_exploited(explored, exploit_k)

        explored_now = []
        if explore_k > 0 and unexplored:
            explored_now = self._sample_unexplored(unexplored, explore_k)

        selected = exploited + explored_now

        if len(selected) < k:
            remaining = [int(c) for c in candidates if int(c) not in set(selected)]
            if remaining:
                selected.extend(self.rng.choice(
                    remaining, size=min(k - len(selected), len(remaining)), replace=False
                ).astype(int).tolist())

        self.selection_history.append({
            'round': self.round,
            'selected': list(selected),
            'exploited': list(exploited),
            'explored': list(explored_now),
            'epsilon': epsilon_used,
            'round_threshold': float(self.round_threshold),
            'target_duration': float(self.target_duration),
        })
        self.last_exploit_clients = list(exploited)
        self.last_explore_clients = list(explored_now)
        if unexplored:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            # The public Oort selector disables exploration once every client
            # has been tried; subsequent rounds fully exploit observed scores.
            self.epsilon = 0.0
        return selected[:k]

    def update_feedback(self, feedback: Dict[int, Dict[str, float]]) -> None:
        feedback = feedback or {}
        for client_id, item in feedback.items():
            cid = int(client_id)
            if cid < 0 or cid >= self.num_clients:
                continue
            loss_square_mean = max(float(item.get('loss_square_mean', item.get('loss', 0.0) ** 2)), 0.0)
            samples = max(float(item.get('num_samples', 1.0)), 1.0)
            duration = max(float(item.get('duration', self.durations[cid])), 1e-4)
            utility = samples * np.sqrt(loss_square_mean)
            if not np.isfinite(utility):
                utility = 0.0

            self.statistical_utility[cid] = utility
            self.durations[cid] = duration
            self.last_selected_round[cid] = self.round
            self.participation_counts[cid] += 1
            self.explored[cid] = True

        self.exploit_utility_history.append(
            self._mean_feedback_utility(self.last_exploit_clients, feedback)
        )
        self.explore_utility_history.append(
            self._mean_feedback_utility(self.last_explore_clients, feedback)
        )

    def get_state(self) -> Dict[str, object]:
        return {
            'implementation': 'source_faithful_matched_protocol',
            'duration_proxy': 'local_training_sample_count',
            'epsilon': float(self.epsilon),
            'pacer_delta': float(self.pacer_delta),
            'round_threshold': float(self.round_threshold),
            'target_duration': float(self.target_duration),
            'statistical_utility': self.statistical_utility.copy(),
            'durations': self.durations.copy(),
            'initial_rewards': self.initial_rewards.copy(),
            'initial_durations': self.initial_durations.copy(),
            'last_selected_round': self.last_selected_round.copy(),
            'explored': self.explored.copy(),
            'blacklisted': self.blacklisted.copy(),
            'participation_counts': self.participation_counts.copy(),
            'exploit_utility_history': list(self.exploit_utility_history),
            'explore_utility_history': list(self.explore_utility_history),
            'selection_history': list(self.selection_history),
        }

    def _client_utilities(self, clients: List[int]) -> np.ndarray:
        clients_arr = np.asarray(clients, dtype=np.int64)
        rewards = self.statistical_utility[clients_arr].copy()
        finite_rewards = rewards[np.isfinite(rewards) & (rewards > 0.0)]
        if finite_rewards.size == 0:
            normalized = np.zeros_like(rewards)
        else:
            clip_index = min(
                int(finite_rewards.size * np.clip(self.clip_percentile / 100.0, 0.0, 1.0)),
                finite_rewards.size - 1,
            )
            clip_value = float(np.sort(finite_rewards)[clip_index])
            reward_min = float(np.min(finite_rewards)) * 0.999
            reward_max = float(np.max(finite_rewards))
            reward_range = max(reward_max - reward_min, 1e-4)
            normalized = (np.minimum(rewards, clip_value) - reward_min) / reward_range

        # Match Oort's public implementation: L(i) is the last-involved round,
        # not the elapsed value R-L(i), and 0.1 stays inside the square root.
        last_round = np.maximum(self.last_selected_round[clients_arr], 1)
        uncertainty = np.sqrt(0.1 * np.log(max(self.round, 2)) / last_round)
        utility = normalized + uncertainty

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
        cutoff_index = min(max(k, 0), sorted_utils.size - 1)
        threshold = self.cutoff_util * sorted_utils[cutoff_index]
        pool_mask = utilities >= threshold
        pool = np.asarray(clients, dtype=np.int64)[pool_mask]
        pool_utils = utilities[pool_mask]
        if pool.size == 0:
            pool = np.asarray(clients, dtype=np.int64)
            pool_utils = utilities

        probs = pool_utils / pool_utils.sum() if pool_utils.sum() > 1e-12 else None
        return self.rng.choice(pool, size=min(k, pool.size), replace=False, p=probs).astype(int).tolist()

    def _sample_unexplored(self, clients: List[int], k: int) -> List[int]:
        clients_arr = np.asarray(clients, dtype=np.int64)
        rewards = self.initial_rewards[clients_arr].copy()
        durations = self.durations[clients_arr]
        slow = durations > self.target_duration
        if np.any(slow):
            rewards[slow] *= (
                self.target_duration / np.maximum(durations[slow], 1e-4)
            ) ** self.straggler_penalty
        rewards = np.maximum(np.nan_to_num(rewards, nan=0.0), 0.0)

        window_size = min(max(int(np.ceil(self.sample_window * k)), k), len(clients_arr))
        order = np.argsort(rewards)[::-1][:window_size]
        pool = clients_arr[order]
        pool_rewards = rewards[order]
        probs = pool_rewards / pool_rewards.sum() if pool_rewards.sum() > 1e-12 else None
        return self.rng.choice(
            pool, size=min(k, len(pool)), replace=False, p=probs
        ).astype(int).tolist()

    def _update_pacer(self) -> None:
        w = self.pacer_window
        if w <= 0 or self.round < 2 * w or self.round % w != 0:
            return
        if len(self.exploit_utility_history) < 2 * w:
            return
        previous = float(np.sum(self.exploit_utility_history[-2 * w:-w]))
        recent = float(np.sum(self.exploit_utility_history[-w:]))
        change = abs(recent - previous)
        if change <= previous * 0.1:
            self.round_threshold = min(100.0, self.round_threshold + self.pacer_delta)
        elif change >= previous * 5.0:
            self.round_threshold = max(self.pacer_delta, self.round_threshold - self.pacer_delta)

    def _update_target_duration(self) -> None:
        if self.round_threshold >= 100.0:
            self.target_duration = float('inf')
            return
        index = min(
            int(len(self.durations) * self.round_threshold / 100.0),
            len(self.durations) - 1,
        )
        self.target_duration = float(np.sort(self.durations)[index])

    def _refresh_blacklist(self) -> None:
        self.blacklisted[:] = False
        if self.blacklist_rounds < 0:
            return
        ordered = np.argsort(self.participation_counts)[::-1]
        eligible = [
            int(cid) for cid in ordered
            if self.participation_counts[cid] > self.blacklist_rounds
        ]
        max_len = int(self.blacklist_max_fraction * self.num_clients)
        if max_len <= 0:
            return
        self.blacklisted[eligible[:max_len]] = True

    def _mean_feedback_utility(self, clients: List[int],
                               feedback: Dict[int, Dict[str, float]]) -> float:
        utilities = []
        for client_id in clients:
            item = feedback.get(int(client_id))
            if item is None:
                continue
            loss_square_mean = max(
                float(item.get('loss_square_mean', item.get('loss', 0.0) ** 2)),
                0.0,
            )
            samples = max(float(item.get('num_samples', 1.0)), 1.0)
            utility = samples * np.sqrt(loss_square_mean)
            if np.isfinite(utility):
                utilities.append(float(utility))
        return float(np.mean(utilities)) if utilities else 0.0


def gradient_channel_aware_selection(learning_signals: np.ndarray,
                                     channel_gains: np.ndarray,
                                     energy_costs: np.ndarray,
                                     num_selected: int,
                                     learning_weight: float = 0.5,
                                     channel_weight: float = 0.3,
                                     energy_weight: float = 0.2,
                                     mode: str = 'paper',
                                     rho_dsi: float = 0.5,
                                     rho_csi: float = 0.5,
                                     lambda_energy: float = 0.5,
                                     available_clients: List[int] = None) -> List[int]:
    """Gradient/channel/energy-aware scheduler adapted from AirComp FEEL.

    ``mode='paper'`` follows the source paper's hierarchical indicator:

        V_n,t = rho_dsi * v_DSI + rho_csi * v_CSI
        I_n,t = (1 - lambda_energy) * V_n,t - lambda_energy * E_n,t

    The source normalizes DSI and CSI by their respective round maxima. The
    current digital-FL experiment retains fixed-K client selection and standard
    FedAvg aggregation; consequently ``learning_signals`` is the latest
    available proxy for the source paper's local gradient norm.

    ``mode='legacy'`` preserves the former direct three-weight adaptation for
    backwards compatibility with archived runs.
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

    def _max_normalize(x, default=1.0):
        x = np.asarray(x, dtype=np.float64)
        x = np.abs(x)
        finite = np.isfinite(x)
        if not finite.any():
            return np.ones_like(x) * default
        scale = float(np.max(x[finite]))
        if scale <= 1e-12:
            return np.ones_like(x) * default
        result = x / scale
        result[~np.isfinite(result)] = 0.0
        return result

    if mode == 'legacy':
        learning = _minmax(learning_signals, default=1.0)
        channel = _minmax(np.abs(channel_gains), default=1.0)
        energy = _minmax(energy_costs, default=0.0)
        score = (
            learning_weight * learning
            + channel_weight * channel
            - energy_weight * energy
        )
    elif mode == 'paper':
        rho_sum = float(rho_dsi) + float(rho_csi)
        if rho_sum <= 0.0:
            raise ValueError('GCA paper rho weights must have a positive sum')
        rho_dsi = float(rho_dsi) / rho_sum
        rho_csi = float(rho_csi) / rho_sum
        lambda_energy = float(lambda_energy)
        if not 0.0 <= lambda_energy <= 1.0:
            raise ValueError('GCA paper lambda_energy must lie in [0, 1]')
        dsi = _max_normalize(learning_signals, default=1.0)
        csi = _max_normalize(channel_gains, default=1.0)
        # Equation (21) followed by Equation (23) in Du et al. (JSAC 2023).
        device_quality = rho_dsi * dsi + rho_csi * csi
        score = (1.0 - lambda_energy) * device_quality - lambda_energy * np.asarray(
            energy_costs, dtype=np.float64
        )
        score[~np.isfinite(score)] = -np.inf
    else:
        raise ValueError(f'Unknown GCA mode: {mode}')

    mask = np.zeros(num_clients, dtype=bool)
    mask[available_clients] = True
    score[~mask] = -np.inf

    selected_count = min(num_selected, len(available_clients))
    return np.argsort(score)[-selected_count:][::-1].tolist()
