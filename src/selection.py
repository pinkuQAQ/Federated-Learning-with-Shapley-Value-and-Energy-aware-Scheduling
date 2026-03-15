#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
selection.py - 客户端选择策略模块
包含基于Shapley值的贪心选择、轮询选择等策略
"""

import numpy as np
import random
from typing import List


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
        normalized_sv = np.ones_like(shapley_values) * 0.5

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
