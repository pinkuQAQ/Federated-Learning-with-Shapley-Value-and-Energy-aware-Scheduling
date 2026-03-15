#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # 联邦学习基本参数
    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (L2 regularization) for optimizer')
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")

    # 模型参数
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', action='store_true',default=False, help='whether i.i.d or not')
    parser.add_argument('--unequal', action='store_true',
                        help='whether to use unequal data splits for non-i.i.d setting (use with --iid False)')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                        help='Dirichlet parameter for non-iid data partitioning')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Number of test samples to use (None = use all)')

    # 其他参数
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Output folder name (default: auto-generate timestamp)')

    # ============= 新增：Shapley相关参数 =============
    parser.add_argument('--no_shapley', action='store_false', default=True, dest='use_shapley',
                        help='disable Shapley-based client selection (default: use Shapley)')
    parser.add_argument('--selection_method', type=str, default='hybrid',
                        choices=['random', 'round_robin', 'greedy', 'hybrid', 'poc', 'ucb'],
                        help='client selection method')
    parser.add_argument('--poc_candidate_size', type=int, default=None,
                        help='Power of Choice candidate pool size (d), default: num_selected * 2')
    parser.add_argument('--poc_decay_rate', type=float, default=0.9,
                        help='Power of Choice loss decay rate (default: 0.9)')

    # Shapley计算参数
    parser.add_argument('--shapley_epsilon', type=float, default=0.01,
                        help='threshold for truncation in GTG-Shapley')
    parser.add_argument('--shapley_max_iter', type=int, default=5,
                        help='maximum Monte Carlo iterations for Shapley')
    parser.add_argument('--shapley_fast', action='store_true',default=False,
                        help='use fast approximation for Shapley calculation')

    # Shapley值更新参数
    parser.add_argument('--shapley_update_method', type=str, default='exponential',
                        choices=['mean', 'exponential', 'recent'],
                        help='method to update global Shapley values')
    parser.add_argument('--shapley_alpha', type=float, default=0.5,
                        help='exponential smoothing parameter for Shapley values (0-1)')

    # 客户端选择参数
    parser.add_argument('--initial_rounds', type=int, default=10,
                        help='number of initial rounds for round-robin selection')
    parser.add_argument('--num_selected', type=int, default=10,
                        help='number of clients selected per round')

    # ============= 新增：能量相关参数 =============
    parser.add_argument('--use_energy', action='store_true', default=False,
                        help='enable energy-aware client selection')
    parser.add_argument('--sigma_squared', type=float, default=1.0,
                        help='noise power for energy calculation (σ²)')
    parser.add_argument('--channel_model', type=str, default='rayleigh',
                        choices=['rayleigh', 'path_loss', 'combined'],
                        help='channel model for energy calculation')
    parser.add_argument('--initial_energy', type=float, default=500.0,
                        help='initial energy for each client')
    parser.add_argument('--energy_threshold', type=float, default=50.0,
                        help='minimum energy threshold for client participation')
    parser.add_argument('--shapley_weight', type=float, default=0.5,
                        help='weight for Shapley value in composite score (0-1)')
    parser.add_argument('--energy_weight', type=float, default=0.5,
                        help='weight for energy score in composite score (0-1)')
    # ================================================

    # ============= 新增：李雅普诺夫优化参数 =============
    parser.add_argument('--use_lyapunov', action='store_true', default=False,
                        help='enable Lyapunov-based dynamic weight optimization')
    parser.add_argument('--lyapunov_V', type=float, default=10.0,
                        help='Lyapunov control parameter V (trade-off performance vs stability)')
    parser.add_argument('--lyapunov_lr', type=float, default=0.01,
                        help='learning rate for weight updates in Lyapunov optimization')
    parser.add_argument('--energy_budget', type=float, default=2.0,
                        help='average energy consumption budget per client per round for Lyapunov optimization')
    # ================================================

    # ============= UCB 客户端选择参数 =============
    parser.add_argument('--ucb_c', type=float, default=1.0,
                        help='UCB1 exploration coefficient c (default: 1.0)')
    # ================================================

    # ============= FedProx 参数 =============
    parser.add_argument('--use_fedprox', action='store_true', default=False,
                        help='enable FedProx proximal term in local training')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='FedProx proximal term coefficient μ (default: 0.01)')
    # ================================================

    # ============= 对称加密参数（方案2：AES-256-GCM + 即时明文销毁）=============
    parser.add_argument('--use_crypto', action='store_true', default=False,
                        help='enable AES-256-GCM encryption for gradient transmission. '
                             'Threat model: honest-but-curious server. '
                             'Plaintext is immediately destroyed after each use.')
    # ================================================

    # ============= 计算能量参数 =============
    parser.add_argument('--kappa', type=float, default=1e-28,
                        help='CPU effective switched capacitance coefficient')
    parser.add_argument('--cpu_freq', type=float, default=1e9,
                        help='CPU frequency in Hz')
    parser.add_argument('--cycles_per_sample', type=float, default=20.0,
                        help='CPU cycles per training sample')
    # ================================================

    args = parser.parse_args()
    return args