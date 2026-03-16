#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import sys
import io

# Fix Windows console encoding for Chinese characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import copy
import time
import pickle
import shutil
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, LocalUpdateFedProx, test_inference, DatasetSplit
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

from shapley import MCShapley
from selection import (hybrid_selection, random_selection, round_robin_selection,
                       greedy_shapley_selection, energy_aware_selection,
                       hybrid_energy_aware_selection, power_of_choice_selection,
                       ucb_selection)
from torch.utils.data import DataLoader
from energy import EnergyAwareClientManager
from crypto_utils import CryptoManager


def build_model(args, train_dataset):
    """构建模型并返回模型实例和模型类"""
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            model = CNNMnist(args=args)
            model_class = CNNMnist
        elif args.dataset == 'fmnist':
            model = CNNFashion_Mnist(args=args)
            model_class = CNNFashion_Mnist
        elif args.dataset == 'cifar':
            model = CNNCifar(args=args)
            model_class = CNNCifar
        else:
            exit(f'Error: unrecognized dataset {args.dataset}')
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
        model_class = MLP
    else:
        exit('Error: unrecognized model')

    return model, model_class


def evaluate_poc_candidates(args, global_model, train_dataset, user_groups,
                            device, criterion):
    """高效地评估Power of Choice候选池的损失（无需创建LocalUpdate实例）"""
    num_selected = args.num_selected
    candidate_size = args.poc_candidate_size if args.poc_candidate_size else num_selected * 3
    candidate_size = min(candidate_size, args.num_users)
    candidates = np.random.choice(args.num_users, candidate_size, replace=False)

    candidate_losses = np.zeros(args.num_users)

    global_model.eval()
    with torch.no_grad():
        for idx in candidates:
            idxs = list(user_groups[idx])
            idxs_train = idxs[:int(0.8 * len(idxs))]
            if len(idxs_train) == 0:
                candidate_losses[idx] = 1.0
                continue

            loader = DataLoader(DatasetSplit(train_dataset, idxs_train),
                                batch_size=args.local_bs, shuffle=False)

            loss_sum = 0
            count = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                count += 1

            candidate_losses[idx] = loss_sum / max(count, 1)

    return candidates, candidate_losses


def _get_client_data_sizes(user_groups, num_users):
    """获取所有客户端的训练数据量"""
    sizes = np.zeros(num_users)
    for i in range(num_users):
        sizes[i] = int(len(user_groups[i]) * 0.8)
    return sizes


def select_clients(args, epoch, num_selected, initial_rounds,
                   shapley_values, client_participation_counts,
                   energy_scores, available_clients,
                   energy_manager, lyapunov_optimizer,
                   client_local_losses, user_groups=None,
                   ucb_rewards=None, ucb_counts=None):
    """统一的客户端选择逻辑"""

    # ---- 非Shapley路径 ----
    if not args.use_shapley:
        if args.selection_method == 'ucb':
            return ucb_selection(
                num_clients=args.num_users,
                num_selected=num_selected,
                ucb_rewards=ucb_rewards,
                ucb_counts=ucb_counts,
                current_round=epoch + 1,
                c=args.ucb_c,
            )
        if args.selection_method == 'poc':
            return power_of_choice_selection(
                client_losses=client_local_losses,
                num_selected=num_selected,
                candidate_size=args.poc_candidate_size,
                available_clients=available_clients,
            )
        if args.use_energy and available_clients and len(available_clients) >= num_selected:
            return np.random.choice(available_clients, num_selected, replace=False).tolist()
        return np.random.choice(range(args.num_users), num_selected, replace=False).tolist()

    # ---- Shapley + Energy + Lyapunov 路径 ----
    if args.use_energy and energy_manager is not None:
        if args.use_lyapunov and lyapunov_optimizer is not None:
            return _select_lyapunov(args, epoch, num_selected, shapley_values,
                                    client_participation_counts, available_clients,
                                    energy_manager, lyapunov_optimizer, user_groups)

        return _select_energy_aware(args, epoch, num_selected, initial_rounds,
                                     shapley_values, client_participation_counts,
                                     energy_scores, available_clients)

    # ---- Shapley only 路径（无Energy）----
    return _select_shapley_only(args, epoch, num_selected, initial_rounds,
                                 shapley_values, client_participation_counts,
                                 client_local_losses)


def _select_lyapunov(args, epoch, num_selected, shapley_values,
                     client_participation_counts, available_clients,
                     energy_manager, lyapunov_optimizer, user_groups):
    """Lyapunov路径：前initial_rounds轮轮询初始化SV，之后Lyapunov动态选择"""
    if epoch < args.initial_rounds:
        return round_robin_selection(
            args.num_users, num_selected, epoch, client_participation_counts
        )

    all_data_sizes = _get_client_data_sizes(user_groups, args.num_users) if user_groups is not None else None
    energy_consumed_estimate = energy_manager.compute_energy_consumption(
        energy_manager.channel_gains, client_data_sizes=all_data_sizes
    )
    scores = lyapunov_optimizer.compute_scores(shapley_values, energy_consumed_estimate)

    # 在可用客户端（含Poisson子采样）内选择
    if available_clients and len(available_clients) >= num_selected:
        scores_masked = np.full(len(scores), -np.inf)
        for c in available_clients:
            scores_masked[c] = scores[c]
    else:
        scores_masked = scores

    return np.argsort(scores_masked)[-num_selected:].tolist()


def _select_energy_aware(args, epoch, num_selected, initial_rounds,
                          shapley_values, client_participation_counts,
                          energy_scores, available_clients):
    """Shapley + Energy 双重调度（无Lyapunov）"""
    if args.selection_method == 'hybrid':
        return hybrid_energy_aware_selection(
            shapley_values=shapley_values,
            energy_scores=energy_scores,
            num_selected=num_selected,
            participation_counts=client_participation_counts,
            current_round=epoch,
            initial_rounds=initial_rounds,
            shapley_weight=args.shapley_weight,
            energy_weight=args.energy_weight,
            available_clients=available_clients
        )
    if args.selection_method == 'greedy':
        return energy_aware_selection(
            shapley_values=shapley_values,
            energy_scores=energy_scores,
            num_selected=num_selected,
            shapley_weight=args.shapley_weight,
            energy_weight=args.energy_weight,
            available_clients=available_clients
        )
    if args.selection_method == 'random':
        if available_clients and len(available_clients) >= num_selected:
            return np.random.choice(available_clients, num_selected, replace=False).tolist()
        return random_selection(args.num_users, num_selected)
    # round_robin fallback
    return round_robin_selection(
        args.num_users, num_selected, epoch, client_participation_counts
    )


def _select_shapley_only(args, epoch, num_selected, initial_rounds,
                           shapley_values, client_participation_counts,
                           client_local_losses):
    """纯Shapley选择（无Energy）"""
    method = args.selection_method
    if method == 'random':
        return random_selection(args.num_users, num_selected)
    if method == 'round_robin':
        return round_robin_selection(
            args.num_users, num_selected, epoch, client_participation_counts
        )
    if method == 'greedy':
        return greedy_shapley_selection(
            shapley_values=shapley_values,
            num_selected=num_selected)
    if method == 'poc':
        return power_of_choice_selection(
            client_losses=client_local_losses,
            num_selected=num_selected,
            candidate_size=args.poc_candidate_size,
        )
    # hybrid (default)
    return hybrid_selection(
        shapley_values=shapley_values,
        num_selected=num_selected,
        participation_counts=client_participation_counts,
        current_round=epoch,
        initial_rounds=initial_rounds,
    )


def update_shapley_values(args, epoch, shapley_values, shapley_calculator,
                          round_client_models, global_model, val_data_loader,
                          user_groups, client_participation_counts):
    """计算并更新Shapley值，同时清理旧数据"""
    if not args.use_shapley or epoch < 1:
        return

    prev_round_data = round_client_models.get(epoch - 1)

    if not prev_round_data or len(prev_round_data['selected_clients']) == 0:
        if args.verbose:
            print(f"  [Shapley] 轮次 {epoch - 1} 无数据，跳过Shapley计算")
        return

    prev_selected = prev_round_data['selected_clients']
    prev_models = prev_round_data['client_models']

    client_model_list = []
    client_id_list = []
    client_data_size_list = []

    for client_id in prev_selected:
        if client_id in prev_models:
            client_model_list.append(prev_models[client_id])
            client_id_list.append(client_id)
            data_size = len(user_groups[client_id]) * 0.8
            client_data_size_list.append(int(data_size))

    if len(client_model_list) == 0:
        return

    print(f"  [Shapley] 计算轮次 {epoch} 的Shapley值（{len(client_id_list)}个客户端）...")

    try:
        round_shapley = shapley_calculator.compute_with_history(
            previous_model=prev_round_data['previous_global'],
            client_models=client_model_list,
            current_global_model=global_model.state_dict(),
            val_data_loader=val_data_loader,
            client_ids=client_id_list,
            client_data_sizes=client_data_size_list
        )

        updated_count = 0
        for i, client_id in enumerate(client_id_list):
            new_sv = round_shapley[i]

            if args.shapley_update_method == 'mean':
                old_count = max(0, client_participation_counts[client_id] - 1)
                if old_count > 0:
                    shapley_values[client_id] = (
                        shapley_values[client_id] * old_count + new_sv
                    ) / (old_count + 1)
                else:
                    shapley_values[client_id] = new_sv
            elif args.shapley_update_method == 'exponential':
                shapley_values[client_id] = (
                    args.shapley_alpha * shapley_values[client_id] +
                    (1 - args.shapley_alpha) * new_sv
                )
            elif args.shapley_update_method == 'recent':
                shapley_values[client_id] = new_sv

            updated_count += 1

        if args.verbose:
            print(f"  [Shapley] 更新了 {updated_count} 个客户端的Shapley值")
            if len(round_shapley) > 0:
                print(f"  [Shapley] 本轮Shapley值范围: [{min(round_shapley):.6f}, "
                      f"{max(round_shapley):.6f}]")

    except Exception as e:
        print(f"  [Shapley] 计算失败: {e}")
        for client_id in client_id_list:
            shapley_values[client_id] = 0.0

    # Fix 2: 清理旧的round_client_models防止内存泄漏
    if epoch >= 2 and (epoch - 2) in round_client_models:
        del round_client_models[epoch - 2]


def save_results(args, exp_folder, timestamp, num_selected, train_loss, train_accuracy,
                 test_accuracies, test_acc, shapley_values, client_participation_counts,
                 client_last_round, energy_manager, lyapunov_optimizer,
                 crypto_manager, start_time, sv_sample_size=0,
                 ucb_rewards=None, ucb_counts=None):
    """保存实验结果"""
    os.makedirs(exp_folder, exist_ok=True)

    # 生成方法后缀
    if args.use_shapley:
        method_suffix = f"_{args.selection_method}_SV"
        if args.use_energy:
            method_suffix += "_Energy"
    else:
        method_suffix = f"_{args.selection_method}"
        if args.use_energy:
            method_suffix += "_Energy"
    if args.use_fedprox:
        method_suffix += "_FedProx"

    file_name = f'{exp_folder}/{args.dataset}_{args.model}_{args.epochs}_' \
                f'C[{num_selected}]_iid[{1 if args.iid else 0}]_E[{args.local_ep}]_' \
                f'B[{args.local_bs}]{method_suffix}.pkl'

    print(f"[保存] 实验文件夹: {exp_folder}")
    print(f"[保存] 准备保存到: {file_name}")

    save_data = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracies,
        'args': vars(args)
    }

    if args.use_shapley:
        save_data.update({
            'shapley_values': shapley_values,
            'client_participation_counts': client_participation_counts,
            'client_last_round': client_last_round,
        })

    if args.use_energy and energy_manager is not None:
        energy_stats = energy_manager.get_statistics()
        save_data.update({
            'energy_statistics': energy_stats,
            'final_client_energy': energy_stats['current_energy'],
            'energy_history': energy_stats['energy_history'],
        })

    if args.use_lyapunov and lyapunov_optimizer is not None:
        lyap_stats = lyapunov_optimizer.get_statistics()
        save_data.update({
            'lyapunov_statistics': lyap_stats,
            'lyapunov_history': lyapunov_optimizer.lyapunov_history,
        })

    if args.use_crypto and crypto_manager is not None:
        save_data.update({
            'crypto_statistics': crypto_manager.get_statistics(),
        })

    if args.selection_method == 'ucb' and ucb_rewards is not None:
        save_data.update({
            'ucb_rewards': ucb_rewards,
            'ucb_counts': ucb_counts,
        })

    try:
        with open(file_name, 'wb') as f:
            pickle.dump(save_data, f)

        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            print(f'\n数据已保存到: {file_name}')
            print(f'文件大小: {file_size / 1024:.2f} KB')

            objects_dir = '../save/objects'
            os.makedirs(objects_dir, exist_ok=True)
            objects_file = f'{objects_dir}/result_{timestamp}.pkl'
            shutil.copy2(file_name, objects_file)
            print(f'副本已保存到: {objects_file}')
        else:
            print(f'\n[警告] 文件保存失败，文件不存在: {file_name}')

        # 保存实验参数到MD文件
        params_file = f'{exp_folder}/experiment_params.md'
        with open(params_file, 'a', encoding='utf-8') as f:
            f.write(f"# 实验参数记录\n\n")
            f.write(f"**实验时间**: {timestamp}\n\n")
            f.write(f"## 基本配置\n\n")
            f.write(f"- 数据集: {args.dataset}\n")
            f.write(f"- 模型: {args.model}\n")
            f.write(f"- 训练轮数: {args.epochs}\n")
            f.write(f"- 客户端总数: {args.num_users}\n")
            f.write(f"- 每轮选择: {num_selected}\n")
            f.write(f"- 本地epochs: {args.local_ep}\n")
            f.write(f"- 本地batch size: {args.local_bs}\n")
            f.write(f"- 学习率: {args.lr}\n")
            f.write(f"- 优化器: {args.optimizer}\n\n")
            f.write(f"## 数据分布\n\n")
            f.write(f"- IID: {args.iid}\n")
            f.write(f"- Dirichlet alpha: {args.dirichlet_alpha}\n\n")
            f.write(f"## 客户端选择\n\n")
            f.write(f"- 使用Shapley: {args.use_shapley}\n")
            if args.use_shapley:
                f.write(f"- 选择方法: {args.selection_method}\n")
                f.write(f"- Shapley更新方法: {args.shapley_update_method}\n")
                f.write(f"- Shapley alpha: {args.shapley_alpha}\n")
                f.write(f"- 快速模式: {args.shapley_fast}\n")
                f.write(f"- 初始轮数: {args.initial_rounds}\n")
                f.write(f"- SV验证集样本数: {sv_sample_size}\n")
                f.write(f"- SV测试集采样种子: 42\n")
            f.write(f"\n## 能量感知\n\n")
            f.write(f"- 使用能量感知: {args.use_energy}\n")
            if args.use_energy:
                f.write(f"- 初始能量: {args.initial_energy}\n")
                f.write(f"- 能量阈值: {args.energy_threshold}\n")
                if args.use_lyapunov:
                    f.write(f"- 调度方式: Lyapunov动态权重 (V={args.lyapunov_V})\n")
                    f.write(f"- 能量预算: {args.energy_budget}\n")
                else:
                    f.write(f"- Shapley权重: {args.shapley_weight}\n")
                    f.write(f"- 能量权重: {args.energy_weight}\n")
            f.write(f"\n## 梯度加密\n\n")
            f.write(f"- 使用AES加密: {args.use_crypto}\n")
            if args.use_crypto:
                f.write(f"- 算法: AES-256-GCM\n")
                f.write(f"- 威胁模型: 诚实但好奇服务器\n")
            f.write(f"\n## 实验结果\n\n")
            f.write(f"- 最终测试准确率: {test_acc * 100:.2f}%\n")
            f.write(f"- 总运行时间: {time.time() - start_time:.2f}秒\n")
        print(f'参数已保存到: {params_file}')

    except Exception as e:
        print(f'\n[错误] 保存数据时出错: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    start_time = time.time()

    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Fix 5: 全局可复现性种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    num_selected = args.num_selected
    initial_rounds = max(1, args.num_users // num_selected)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"每轮选择的客户端数: {num_selected}")
    print(f"初始轮询轮数: {initial_rounds}")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL (Fix 6: 提取为函数)
    global_model, model_class = build_model(args, train_dataset)
    global_model.to(device)
    global_model.train()
    print(global_model)

    global_weights = global_model.state_dict()

    # ============= Shapley相关初始化 =============
    if args.use_shapley:
        print("\n" + "=" * 60)
        print("启用基于Shapley值的客户端选择")
        print(f"选择方法: {args.selection_method}")
        print(f"Shapley更新方法: {args.shapley_update_method}")
        print("=" * 60 + "\n")

        shapley_calculator = MCShapley(
            model_class=model_class, args=args,
            epsilon=args.shapley_epsilon,
            max_iterations=args.shapley_max_iter,
            device=device, verbose=args.verbose
        )

        print(f"[设备检查] 主程序设备: {device}")
        print(f"[设备检查] Shapley计算器设备: {shapley_calculator.device}")

        shapley_values = np.zeros(args.num_users)
        client_participation_counts = np.zeros(args.num_users)
        client_last_round = -np.ones(args.num_users)

        round_client_models = {}

        # 从训练集中为每个客户端预留一部分数据，汇总为全局验证集（避免使用测试集导致数据泄露）
        sv_val_indices = []
        sv_rng = np.random.RandomState(42)
        for uid in range(args.num_users):
            client_idxs = list(user_groups[uid])
            # 从每个客户端取最后10%作为SV验证数据（与train_val_test的划分一致）
            val_start = int(0.8 * len(client_idxs))
            val_end = int(0.9 * len(client_idxs))
            sv_val_indices.extend(client_idxs[val_start:val_end])
        # 随机采样最多1000个样本，加快SV计算
        sv_sample_size = min(1000, len(sv_val_indices))
        if len(sv_val_indices) > sv_sample_size:
            sv_val_indices = list(sv_rng.choice(sv_val_indices, sv_sample_size, replace=False))
        sv_subset = torch.utils.data.Subset(train_dataset, sv_val_indices)
        val_batch_size = min(128, sv_sample_size)
        val_data_loader = DataLoader(sv_subset, batch_size=val_batch_size, shuffle=False)
        print(f"[Shapley] SV验证集大小: {sv_sample_size}（从训练集验证部分采样，避免测试集泄露）")

    else:
        print("\n" + "=" * 60)
        if args.selection_method == 'poc':
            print("使用Power of Choice客户端选择")
            print(f"候选池大小: {args.poc_candidate_size}")
            print(f"损失衰减率: {args.poc_decay_rate}")
            client_loss_history = np.zeros(args.num_users)
        elif args.selection_method == 'ucb':
            print("使用UCB1客户端选择")
            print(f"探索系数 c: {args.ucb_c}")
        elif args.selection_method == 'random':
            print("使用随机客户端选择")
        else:
            print(f"使用{args.selection_method}客户端选择")
        print("=" * 60 + "\n")
        shapley_calculator = None
        shapley_values = None
        client_participation_counts = None
        client_last_round = None
        round_client_models = None
        val_data_loader = None

    # ============= 能量管理器初始化 =============
    if args.use_energy:
        print("\n" + "=" * 60)
        print("启用能量感知的客户端选择")
        print(f"信道模型: {args.channel_model}")
        print("=" * 60 + "\n")

        energy_manager = EnergyAwareClientManager(
            num_clients=args.num_users,
            sigma_squared=args.sigma_squared,
            channel_model=args.channel_model,
            initial_energy=args.initial_energy,
            energy_threshold=args.energy_threshold,
            seed=args.seed,
            kappa=args.kappa,
            cpu_freq=args.cpu_freq,
            cycles_per_sample=args.cycles_per_sample
        )
    else:
        energy_manager = None

    # ============= 李雅普诺夫优化器初始化 =============
    if args.use_lyapunov and args.use_energy:
        print("\n" + "=" * 60)
        print("启用李雅普诺夫动态权重优化")
        print(f"控制参数 V: {args.lyapunov_V}")
        print(f"学习率: {args.lyapunov_lr}")
        print("=" * 60 + "\n")

        from lyapunov_optimizer import LyapunovTripleScheduler

        lyapunov_optimizer = LyapunovTripleScheduler(
            num_clients=args.num_users,
            V=args.lyapunov_V,
            energy_budget=args.energy_budget
        )

        prev_selected_clients = []
        prev_energy_consumed = None
    else:
        lyapunov_optimizer = None
        prev_selected_clients = []
        prev_energy_consumed = None

    # ============= 对称加密初始化（方案2：AES-256-GCM）=============
    if args.use_crypto:
        crypto_manager = CryptoManager(num_clients=args.num_users)
        print("\n" + "=" * 60)
        print("启用 AES-256-GCM 梯度加密")
        print("  威胁模型: 诚实但好奇服务器（honest-but-curious）")
        print("  密钥管理: 每客户端独立 256-bit 会话密钥")
        print("  隐私保证: 梯度传输密文化，服务器计算后立即销毁明文")
        print("=" * 60 + "\n")
    else:
        crypto_manager = None

    # ============= UCB 状态初始化 =============
    if args.selection_method == 'ucb' and not args.use_shapley:
        ucb_rewards = np.zeros(args.num_users)
        ucb_counts = np.zeros(args.num_users, dtype=int)
    else:
        ucb_rewards = None
        ucb_counts = None

    # ============= FedProx 初始化提示 =============
    if args.use_fedprox:
        print("\n" + "=" * 60)
        print(f"启用 FedProx 近端项 (μ={args.fedprox_mu})")
        print("=" * 60 + "\n")

    # ============= 隐私会计初始化 已移除（DP模块已删除）=============

    # 记录每个客户端的本地损失
    client_local_losses = np.ones(args.num_users)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 2
    test_accuracies = []

    # PoC评估用的criterion
    poc_criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        # ============= 李雅普诺夫权重更新 =============
        if args.use_lyapunov and lyapunov_optimizer is not None and epoch >= args.initial_rounds:
            if len(prev_selected_clients) > 0 and prev_energy_consumed is not None:
                lyapunov_optimizer.update_queue(
                    energy_consumed=prev_energy_consumed,
                    selected_clients=prev_selected_clients,
                    round_num=epoch
                )
                if args.verbose and epoch % print_every == 0:
                    lyapunov_optimizer.print_statistics(epoch)

        # ============= Power of Choice: 候选池损失评估 (Fix 11: 高效实现) =============
        if args.selection_method == 'poc' and epoch > 0:
            candidates, candidate_losses = evaluate_poc_candidates(
                args, global_model, train_dataset, user_groups, device, poc_criterion
            )
            for idx in candidates:
                client_local_losses[idx] = candidate_losses[idx]

            # 用衰减率更新历史损失
            if not args.use_shapley:
                for idx in candidates:
                    client_loss_history[idx] = (args.poc_decay_rate * client_loss_history[idx] +
                                               (1 - args.poc_decay_rate) * client_local_losses[idx])
                client_local_losses = client_loss_history.copy()

        # ============= 客户端选择策略 (Fix 6: 提取为函数) =============
        if args.use_energy and energy_manager is not None:
            channel_gains = energy_manager.generate_channel_gains(epoch)
            energy_scores = energy_manager.get_energy_scores(normalize=True)
            available_clients = energy_manager.get_available_clients()

            if args.verbose and epoch % print_every == 0:
                energy_manager.print_energy_status(epoch)
        else:
            energy_scores = None
            available_clients = None

        idxs_users = select_clients(
            args, epoch, num_selected, initial_rounds,
            shapley_values, client_participation_counts,
            energy_scores, available_clients,
            energy_manager, lyapunov_optimizer,
            client_local_losses, user_groups=user_groups,
            ucb_rewards=ucb_rewards, ucb_counts=ucb_counts
        )

        if args.use_shapley and args.verbose and epoch % print_every == 0:
            print(f"选择的客户端: {sorted(idxs_users)}")
            print(f"客户端参与次数统计: 平均={np.mean(client_participation_counts):.1f}, "
                  f"最小={np.min(client_participation_counts)}, 最大={np.max(client_participation_counts)}")

        # 保存本轮选择的客户端（用于下一轮李雅普诺夫更新）
        if args.use_lyapunov and lyapunov_optimizer is not None:
            prev_selected_clients = list(idxs_users) if isinstance(idxs_users, np.ndarray) else idxs_users.copy()

        # 保存当前全局模型（用于Shapley计算）
        if args.use_shapley:
            previous_global_model = copy.deepcopy(global_model.state_dict())
            current_round_models = {}

        # ============= 本地训练 =============
        for idx in idxs_users:
            if args.use_shapley:
                client_participation_counts[idx] += 1
                client_last_round[idx] = epoch

            if args.use_fedprox:
                local_model = LocalUpdateFedProx(args=args, dataset=train_dataset,
                                                 idxs=user_groups[idx], logger=logger,
                                                 device=device, mu=args.fedprox_mu)
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger, device=device)
            w, loss, actual_samples = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)

            # ============= AES-256-GCM 加密传输（方案2）=============
            if args.use_crypto and crypto_manager is not None:
                # Step 1: 客户端加密权重（模拟加密上传）
                encrypted_pkg = crypto_manager.encrypt(idx, w)
                # Step 2: 服务器解密，获取短暂明文用于 SV 和聚合
                w_plain = crypto_manager.decrypt_and_destroy(encrypted_pkg)
                del encrypted_pkg          # 销毁密文传输包
                # 只做一次deepcopy，local_weights和current_round_models共享引用
                w_copy = copy.deepcopy(w_plain)
                local_weights.append(w_copy)
                if args.use_shapley:
                    current_round_models[idx] = w_copy
                del w_plain                # Step 3: 立即销毁明文，不持久化
            else:
                # 只做一次deepcopy，共享引用以减少内存开销
                w_copy = copy.deepcopy(w)
                local_weights.append(w_copy)
                if args.use_shapley:
                    current_round_models[idx] = w_copy
            # ======================================================

            local_losses.append(copy.deepcopy(loss))
            client_local_losses[idx] = loss

        # 保存本轮数据（用于下一轮计算Shapley值）
        if args.use_shapley:
            round_client_models[epoch] = {
                'previous_global': previous_global_model,
                'client_models': current_round_models,
                'selected_clients': list(idxs_users) if isinstance(idxs_users, np.ndarray) else idxs_users.copy()
            }

        # ============= 更新能量消耗 =============
        if args.use_energy and energy_manager is not None:
            all_data_sizes = _get_client_data_sizes(user_groups, args.num_users)
            energy_consumed = energy_manager.compute_energy_consumption(
                channel_gains, selected_clients=idxs_users,
                client_data_sizes=all_data_sizes
            )
            energy_manager.update_client_energy(idxs_users, energy_consumed)
            prev_energy_consumed = energy_consumed.copy()

            if args.verbose and epoch % print_every == 0:
                print(f"  [能量] 本轮平均消耗: {np.mean(energy_consumed):.2f}")
                print(f"  [能量] 消耗范围: [{np.min(energy_consumed):.2f}, {np.max(energy_consumed):.2f}]")

        # ============= 聚合全局权重 =============
        client_data_sizes = []
        for idx in idxs_users:
            data_size = len(user_groups[idx]) * 0.8
            client_data_sizes.append(int(data_size))

        global_weights = average_weights(local_weights, client_data_sizes)
        global_model.load_state_dict(global_weights)

        # ============= UCB 奖励更新 =============
        if args.selection_method == 'ucb' and ucb_rewards is not None:
            for idx in idxs_users:
                ucb_counts[idx] += 1
                # 增量均值更新：reward = 本地损失（越高说明该客户端数据越有价值）
                ucb_rewards[idx] += (client_local_losses[idx] - ucb_rewards[idx]) / ucb_counts[idx]

        # ============= 计算Shapley值 =============
        if args.use_shapley:
            update_shapley_values(
                args, epoch, shapley_values, shapley_calculator,
                round_client_models, global_model, val_data_loader,
                user_groups, client_participation_counts
            )

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Fix 3: 只在选中客户端上评估训练准确率（而非所有客户端）
        list_acc, list_loss = [], []
        global_model.eval()
        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger, device=device)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        # 每轮都记录测试集准确率
        test_acc, test_loss = test_inference(args, global_model, test_dataset, device=device)
        test_accuracies.append(test_acc)

        if (epoch + 1) % print_every == 0:
            print('Test Accuracy: {:.2f}%'.format(100 * test_acc))

            if args.use_shapley and shapley_values is not None:
                non_zero_sv = shapley_values[shapley_values != 0]
                if len(non_zero_sv) > 0:
                    print(f'Shapley Values - 非零客户端数: {len(non_zero_sv)}/{args.num_users}')
                    print(f'  Mean: {np.mean(non_zero_sv):.6f}, Std: {np.std(non_zero_sv):.6f}')
                    print(f'  Min: {np.min(non_zero_sv):.6f}, Max: {np.max(non_zero_sv):.6f}')

                    top_k = min(5, args.num_users)
                    top_indices = np.argsort(shapley_values)[-top_k:][::-1]
                    print(f'  Top {top_k} clients by SV:')
                    for rank, idx in enumerate(top_indices):
                        part_count = int(client_participation_counts[idx]) if client_participation_counts is not None else 0
                        print(f'    {rank + 1}. Client {idx}: SV={shapley_values[idx]:.6f}, '
                              f'Participation={part_count}')

                if client_participation_counts is not None:
                    participation_stats = {
                        'min': int(np.min(client_participation_counts)),
                        'max': int(np.max(client_participation_counts)),
                        'mean': float(np.mean(client_participation_counts)),
                        'std': float(np.std(client_participation_counts))
                    }
                    print(f'客户端参与统计: {participation_stats}')

    # ============= 训练结束 =============
    test_acc, test_loss = test_inference(args, global_model, test_dataset, device=device)

    print(f' \n Results after {args.epochs} global rounds of training:')
    if train_accuracy:
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    else:
        print("|---- Avg Train Accuracy: N/A (early stop before evaluation)")
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # 打印最终的Shapley值分析
    if args.use_shapley and shapley_values is not None:
        print(f'\n{"=" * 60}')
        print("最终Shapley值分析:")
        print(f"{'=' * 60}")

        non_zero_mask = shapley_values != 0
        non_zero_count = np.sum(non_zero_mask)
        non_zero_values = shapley_values[non_zero_mask]

        if non_zero_count > 0:
            print(f"有Shapley值的客户端数: {non_zero_count}/{args.num_users} ({non_zero_count / args.num_users * 100:.1f}%)")
            print(f"Shapley值统计:")
            print(f"  均值: {np.mean(non_zero_values):.6f}")
            print(f"  标准差: {np.std(non_zero_values):.6f}")
            print(f"  最小值: {np.min(non_zero_values):.6f}")
            print(f"  最大值: {np.max(non_zero_values):.6f}")
            print(f"  中位数: {np.median(non_zero_values):.6f}")

            quantiles = np.percentile(non_zero_values, [25, 50, 75, 90, 95])
            print(f"  25分位数: {quantiles[0]:.6f}")
            print(f"  75分位数: {quantiles[2]:.6f}")
            print(f"  90分位数: {quantiles[3]:.6f}")
            print(f"  95分位数: {quantiles[4]:.6f}")

            if client_participation_counts is not None:
                print(f"\n客户端参与统计:")
                print(f"  平均参与次数: {np.mean(client_participation_counts):.1f}")
                print(f"  最少参与: {int(np.min(client_participation_counts))}")
                print(f"  最多参与: {int(np.max(client_participation_counts))}")

                corr = np.corrcoef(shapley_values, client_participation_counts)[0, 1]
                print(f"  SV与参与次数的相关性: {corr:.4f}")

                sv_sorted = np.sort(shapley_values)
                if len(sv_sorted) >= 10:
                    top_10_mean = np.mean(sv_sorted[-len(sv_sorted) // 10:])
                    bottom_10_mean = np.mean(sv_sorted[:len(sv_sorted) // 10])
                    print(f"  前10%客户端平均SV: {top_10_mean:.6f}")
                    print(f"  后10%客户端平均SV: {bottom_10_mean:.6f}")
                    print(f"  比值 (前10%/后10%): {top_10_mean / max(bottom_10_mean, 1e-10):.2f}")
        else:
            print("没有客户端获得非零Shapley值")

        print(f"{'=' * 60}")

    # ============= 保存结果 (Fix 6: 提取为函数) =============
    if args.output_folder:
        exp_folder = f'../save/{args.output_folder}'
        timestamp = args.output_folder
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_folder = f'../save/{timestamp}'

    save_results(
        args, exp_folder, timestamp, num_selected,
        train_loss, train_accuracy, test_accuracies, test_acc,
        shapley_values, client_participation_counts, client_last_round,
        energy_manager, lyapunov_optimizer,
        crypto_manager, start_time,
        sv_sample_size=sv_sample_size if args.use_shapley else 0,
        ucb_rewards=ucb_rewards, ucb_counts=ucb_counts
    )

    print('\n Total Run Time: {0:0.4f} seconds'.format(time.time() - start_time))

    # 打印总结
    print(f"\n{'=' * 60}")
    print("训练总结:")
    print(f"{'=' * 60}")
    print(f"数据集: {args.dataset}")
    print(f"模型: {args.model}")
    print(f"客户端数: {args.num_users}")
    print(f"每轮选择客户端数: {num_selected}")
    print(f"总轮数: {args.epochs}")
    print(f"最终测试准确率: {test_acc * 100:.2f}%")

    if args.use_shapley:
        print(f"客户端选择方法: {args.selection_method}")
        print(f"Shapley更新方法: {args.shapley_update_method}")
        if shapley_values is not None and np.any(shapley_values != 0):
            print(f"Shapley值范围: [{np.min(shapley_values[shapley_values != 0]):.6f}, "
                  f"{np.max(shapley_values):.6f}]")
    else:
        print(f"客户端选择方法: 随机")

    if args.use_energy and energy_manager is not None:
        print(f"\n能量管理统计:")
        final_energy = energy_manager.client_energy
        print(f"  最终平均剩余能量: {np.mean(final_energy):.2f}")
        print(f"  最终能量范围: [{np.min(final_energy):.2f}, {np.max(final_energy):.2f}]")
        depleted = np.sum(final_energy < args.energy_threshold)
        print(f"  能量耗尽的客户端: {depleted}/{args.num_users}")

    if args.use_lyapunov and lyapunov_optimizer is not None:
        print(f"\n李雅普诺夫优化统计:")
        lyap_stats = lyapunov_optimizer.get_statistics()
        print(f"  能量队列均值: {lyap_stats['queue_mean']:.4f}")
        print(f"  能量队列最大值: {lyap_stats['queue_max']:.4f}")
        print(f"  李雅普诺夫函数值: {lyap_stats['lyapunov_value']:.2f}")

        lyap_viz_path = f'{exp_folder}/lyapunov_{args.dataset}_{timestamp}.png'
        lyapunov_optimizer.visualize_optimization(lyap_viz_path)

    if args.use_crypto and crypto_manager is not None:
        print(f"\n梯度加密统计:")
        crypto_manager.print_statistics()

    print(f"总运行时间: {time.time() - start_time:.2f}秒")
    print(f"{'=' * 60}")
