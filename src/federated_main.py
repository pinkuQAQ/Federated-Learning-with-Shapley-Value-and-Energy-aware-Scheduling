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
import math
from tqdm import tqdm

import torch
try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:
        """Fallback logger used when tensorboardX is not installed."""

        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


class NullSummaryWriter:
    """No-op logger used when TensorBoard cannot start in the local environment."""

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def make_summary_writer(logdir):
    try:
        return SummaryWriter(logdir)
    except Exception as exc:
        print(f"[warning] TensorBoard logger disabled: {exc}")
        return NullSummaryWriter()

from options import args_parser
from update import LocalUpdate, LocalUpdateFedProx, test_inference, DatasetSplit
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LeNet5Cifar
from utils import get_dataset, average_weights, exp_details

from shapley import MCShapley
from selection import (hybrid_selection, random_selection, round_robin_selection,
                       greedy_shapley_selection, energy_aware_selection,
                       hybrid_energy_aware_selection, power_of_choice_selection,
                       softmax_score_selection, ucb_selection, OortSelector,
                       gradient_channel_aware_selection)
from torch.utils.data import DataLoader
from energy import EnergyAwareClientManager
from fedmsv import (
    FedMSVSelector,
    LabelOverrideDataset,
    ModelAccuracyUtility,
    add_gaussian_model_noise,
    build_label_overrides,
    choose_low_quality_clients,
    random_free_rider_state,
)

# 训练/验证/测试数据划分比例（每个客户端的本地数据中，前80%训练，中间10%验证，后10%测试）
TRAIN_SPLIT_RATIO = 0.8


def build_model(args, train_dataset):
    """构建模型并返回模型实例和模型类"""
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            model = CNNMnist(args=args)
            model_class = CNNMnist
        elif args.dataset == 'fmnist':
            model = CNNFashion_Mnist(args=args)
            model_class = CNNFashion_Mnist
        elif args.dataset in ('cifar', 'cifar100'):
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
    elif args.model == 'lenet5' and args.dataset == 'cifar':
        model = LeNet5Cifar(args=args)
        model_class = LeNet5Cifar
    else:
        exit('Error: unrecognized model')

    return model, model_class


def build_fedmsv_utility_loader(args, train_dataset, test_dataset, user_groups):
    """Build the utility loader for either fair-baseline or paper mode."""
    max_samples = max(int(getattr(args, 'fedmsv_utility_samples', 1000)), 0)
    if args.fedmsv_utility_source == 'test':
        source = test_dataset
        source_size = len(source)
        if max_samples and source_size > max_samples:
            rng = np.random.RandomState(9999)
            indices = rng.choice(source_size, max_samples, replace=False).tolist()
            source = torch.utils.data.Subset(source, indices)
        sample_size = len(source)
    else:
        validation_indices = []
        for client_id in range(args.num_users):
            client_indices = list(user_groups[client_id])
            val_start = int(TRAIN_SPLIT_RATIO * len(client_indices))
            val_end = int(0.9 * len(client_indices))
            validation_indices.extend(client_indices[val_start:val_end])
        rng = np.random.RandomState(42)
        if max_samples and len(validation_indices) > max_samples:
            validation_indices = rng.choice(
                validation_indices, max_samples, replace=False
            ).tolist()
        source = torch.utils.data.Subset(train_dataset, validation_indices)
        sample_size = len(validation_indices)

    batch_size = min(128, max(sample_size, 1))
    return DataLoader(source, batch_size=batch_size, shuffle=False), sample_size


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
            idxs_train = idxs[:int(TRAIN_SPLIT_RATIO * len(idxs))]
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
        sizes[i] = int(len(user_groups[i]) * TRAIN_SPLIT_RATIO)
    return sizes


def _get_trainable_clients(user_groups, num_users):
    """Return clients that own at least one local sample."""
    if user_groups is None:
        return list(range(num_users))
    return [i for i in range(num_users) if len(user_groups[i]) > 0]


def _resolve_initial_rounds(args):
    if args.initial_rounds is None:
        return max(1, math.ceil(args.num_users / args.num_selected))
    return max(1, int(args.initial_rounds))


def _filter_available_clients(available_clients, trainable_clients):
    """Keep only trainable clients in an availability list."""
    trainable_set = set(trainable_clients)
    if available_clients is None:
        return None
    return [int(c) for c in available_clients if int(c) in trainable_set]


def _ensure_trainable_selection(selected_clients, trainable_clients, num_selected,
                                participation_counts=None):
    """Drop empty-client selections and refill from trainable clients."""
    trainable_set = set(trainable_clients)
    selected = []
    seen = set()
    for client_id in selected_clients:
        client_id = int(client_id)
        if client_id in trainable_set and client_id not in seen:
            selected.append(client_id)
            seen.add(client_id)
        if len(selected) >= num_selected:
            return selected

    remaining = [c for c in trainable_clients if c not in seen]
    if participation_counts is not None:
        remaining = sorted(remaining, key=lambda c: (participation_counts[c], c))
    else:
        remaining = list(remaining)
        np.random.shuffle(remaining)

    selected.extend(remaining[:max(0, num_selected - len(selected))])
    if len(selected) < num_selected:
        raise RuntimeError(
            f"Only {len(selected)} trainable clients are available, "
            f"but num_selected={num_selected}."
        )
    return selected


def _client_update_norm(global_state, local_state, args):
    """Return the L2 norm of one client's model update."""
    total_sq_norm = 0.0
    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        if torch.is_floating_point(local_tensor):
            delta = (local_tensor - global_tensor).float()
            total_sq_norm += torch.sum(delta * delta).item()
    return total_sq_norm ** 0.5


def _client_update_layer_norms(global_state, local_state, args):
    """Return per-tensor L2 norms for one client's model update."""
    norms = {}
    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        if torch.is_floating_point(local_tensor):
            delta = (local_tensor - global_tensor).float()
            norms[key] = torch.norm(delta).item()
    return norms


def _adaptive_clip_norm(update_norms, previous_clip_norm, args):
    """Estimate the round clipping norm from selected-client update norms."""
    if not getattr(args, 'dp_adaptive_clip', False):
        return max(float(args.dp_clip_norm), 1e-12)

    valid_norms = np.asarray([n for n in update_norms if np.isfinite(n) and n > 0], dtype=np.float64)
    if valid_norms.size == 0:
        target = max(float(args.dp_clip_norm), 1e-12)
    else:
        pct = min(max(float(args.dp_clip_percentile), 0.0), 100.0)
        target = float(np.percentile(valid_norms, pct))

    rho = min(max(float(args.dp_clip_ema), 0.0), 1.0)
    if previous_clip_norm is None:
        smoothed = target
    else:
        smoothed = rho * float(previous_clip_norm) + (1.0 - rho) * target
        growth = max(float(getattr(args, 'dp_clip_growth', 1.2)), 1.0)
        smoothed = min(smoothed, float(previous_clip_norm) * growth)

    lo = max(float(args.dp_min_clip_norm), 1e-12)
    hi = max(float(args.dp_max_clip_norm), lo)
    return float(np.clip(smoothed, lo, hi))


def _adaptive_layer_clip_norms(layer_norm_records, previous_layer_clip_norms, args):
    """Estimate layer-wise clipping norms from selected-client update norms."""
    if not layer_norm_records:
        return previous_layer_clip_norms or {}

    pct = min(max(float(args.dp_clip_percentile), 0.0), 100.0)
    rho = min(max(float(args.dp_clip_ema), 0.0), 1.0)
    growth = max(float(getattr(args, 'dp_clip_growth', 1.2)), 1.0)
    lo = max(float(args.dp_min_clip_norm), 1e-12)
    hi = max(float(args.dp_max_clip_norm), lo)
    previous_layer_clip_norms = previous_layer_clip_norms or {}

    all_keys = sorted({key for record in layer_norm_records for key in record.keys()})
    clip_norms = {}
    for key in all_keys:
        values = np.asarray(
            [record[key] for record in layer_norm_records if key in record and np.isfinite(record[key]) and record[key] > 0],
            dtype=np.float64,
        )
        if values.size == 0:
            target = float(previous_layer_clip_norms.get(key, args.dp_clip_norm))
        else:
            target = float(np.percentile(values, pct))

        if key in previous_layer_clip_norms:
            prev = float(previous_layer_clip_norms[key])
            smoothed = rho * prev + (1.0 - rho) * target
            smoothed = min(smoothed, prev * growth)
        else:
            smoothed = target
        clip_norms[key] = float(np.clip(smoothed, lo, hi))
    return clip_norms


def _clip_client_update(global_state, local_state, clip_norm, args):
    """Clip one client update and return the clipped local model."""
    clip_norm = max(float(clip_norm), 1e-12)
    total_sq_norm = 0.0
    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        if torch.is_floating_point(local_tensor):
            delta = (local_tensor - global_tensor).float()
            total_sq_norm += torch.sum(delta * delta).item()

    pre_clip_norm = total_sq_norm ** 0.5
    clip_factor = min(1.0, clip_norm / (pre_clip_norm + 1e-12)) if pre_clip_norm > 0 else 1.0

    clipped_state = copy.deepcopy(local_state)
    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        if not torch.is_floating_point(local_tensor):
            clipped_state[key] = local_tensor.clone()
            continue
        delta = (local_tensor - global_tensor).float() * clip_factor
        clipped_state[key] = (global_tensor.float() + delta).to(local_tensor.dtype)

    return clipped_state, {
        'pre_clip_norm': pre_clip_norm,
        'post_clip_norm': pre_clip_norm * clip_factor,
        'clip_factor': clip_factor,
        'noise_std': 0.0,
    }


def _clip_client_update_layerwise(global_state, local_state, layer_clip_norms, args):
    """Clip one client update independently for each floating tensor."""
    clipped_state = copy.deepcopy(local_state)
    layer_stats = {}
    total_pre_sq = 0.0
    total_post_sq = 0.0

    for key, local_tensor in local_state.items():
        global_tensor = global_state[key].to(local_tensor.device)
        if not torch.is_floating_point(local_tensor):
            clipped_state[key] = local_tensor.clone()
            continue

        delta = (local_tensor - global_tensor).float()
        pre_norm = torch.norm(delta).item()
        clip_norm = max(float(layer_clip_norms.get(key, 1e-12)), 1e-12)
        clip_factor = min(1.0, clip_norm / (pre_norm + 1e-12)) if pre_norm > 0 else 1.0
        clipped_delta = delta * clip_factor
        clipped_state[key] = (global_tensor.float() + clipped_delta).to(local_tensor.dtype)

        post_norm = pre_norm * clip_factor
        layer_stats[key] = {
            'pre_clip_norm': pre_norm,
            'post_clip_norm': post_norm,
            'clip_factor': clip_factor,
            'clip_norm': clip_norm,
        }
        total_pre_sq += pre_norm * pre_norm
        total_post_sq += post_norm * post_norm

    total_pre = total_pre_sq ** 0.5
    total_post = total_post_sq ** 0.5
    return clipped_state, {
        'pre_clip_norm': total_pre,
        'post_clip_norm': total_post,
        'clip_factor': total_post / (total_pre + 1e-12) if total_pre > 0 else 1.0,
        'noise_std': 0.0,
        'clip_norm': float(np.mean(list(layer_clip_norms.values()))) if layer_clip_norms else 0.0,
        'layer_clip_norms': layer_clip_norms,
        'layer_stats': layer_stats,
    }


def _channel_noise_multiplier_for_round(args):
    """Return the equivalent channel-noise multiplier for this round."""
    if not getattr(args, 'dp_channel_assisted', False):
        return 0.0
    base = max(float(getattr(args, 'dp_channel_noise_multiplier', 0.0)), 0.0)
    if base <= 0:
        return 0.0
    gains = getattr(args, '_current_selected_channel_gains', None)
    if gains is None or len(gains) == 0:
        factor = 1.0
    else:
        gains = np.asarray(gains, dtype=float)
        factor = float(np.sqrt(np.mean(1.0 / (gains * gains + 1e-12))))
    cap = max(float(getattr(args, 'dp_channel_gain_cap', 2.0)), 1e-12)
    return base * min(factor, cap)


def _effective_noise_multipliers(args, target_multiplier):
    """Use channel noise as the only perturbation in the active privacy route."""
    channel = _channel_noise_multiplier_for_round(args)
    return 0.0, channel, channel


def _add_central_dp_noise(global_state, averaged_state, args, selected_count,
                          aggregation_max_weight=None):
    """Add aggregate Gaussian noise with fixed client-count normalization."""
    selected_count = max(int(selected_count), 1)
    layer_mode = getattr(args, 'dp_clip_scope', 'global') == 'layer'
    layer_clip_norms = getattr(args, '_current_dp_layer_clip_norms', {}) if layer_mode else {}
    clip_norm = max(float(getattr(args, '_current_dp_clip_norm', args.dp_clip_norm)), 1e-12)
    target_multiplier = float(getattr(args, '_current_dp_noise_multiplier', args.dp_noise_multiplier))
    alg_multiplier, channel_multiplier, effective_multiplier = _effective_noise_multipliers(args, target_multiplier)
    if aggregation_max_weight is None:
        aggregation_max_weight = 1.0 / selected_count
    aggregation_max_weight = min(max(float(aggregation_max_weight), 0.0), 1.0)
    noise_scale_weight = 1.0 / selected_count
    default_noise_std = effective_multiplier * clip_norm * noise_scale_weight

    noised_state = copy.deepcopy(averaged_state)
    noise_sq_norm = 0.0
    alg_noise_sq_norm = 0.0
    channel_noise_sq_norm = 0.0
    noise_stds = []
    alg_noise_stds = []
    channel_noise_stds = []
    for key, avg_tensor in averaged_state.items():
        if not torch.is_floating_point(avg_tensor):
            noised_state[key] = avg_tensor.clone()
            continue
        key_clip_norm = max(float(layer_clip_norms.get(key, clip_norm)), 1e-12) if layer_mode else clip_norm
        alg_std = alg_multiplier * key_clip_norm * noise_scale_weight
        ch_std = channel_multiplier * key_clip_norm * noise_scale_weight
        noise_std = math.sqrt(alg_std * alg_std + ch_std * ch_std)
        noise_stds.append(noise_std)
        alg_noise_stds.append(alg_std)
        channel_noise_stds.append(ch_std)
        avg_float = avg_tensor.float()
        alg_noise = torch.randn_like(avg_float) * alg_std if alg_std > 0 else torch.zeros_like(avg_float)
        ch_noise = torch.randn_like(avg_float) * ch_std if ch_std > 0 else torch.zeros_like(avg_float)
        noise = alg_noise + ch_noise
        noised_state[key] = (avg_float + noise).to(avg_tensor.dtype)
        noise_sq_norm += torch.sum(noise * noise).item()
        alg_noise_sq_norm += torch.sum(alg_noise * alg_noise).item()
        channel_noise_sq_norm += torch.sum(ch_noise * ch_noise).item()

    return noised_state, {
        'clip_norm': clip_norm,
        'noise_multiplier': effective_multiplier,
        'target_noise_multiplier': target_multiplier,
        'algorithmic_noise_multiplier': alg_multiplier,
        'channel_noise_multiplier': channel_multiplier,
        'channel_assisted': bool(getattr(args, 'dp_channel_assisted', False)),
        'channel_mode': getattr(args, 'dp_channel_mode', 'channel_only'),
        'noise_std': float(np.mean(noise_stds)) if noise_stds else default_noise_std,
        'algorithmic_noise_std': float(np.mean(alg_noise_stds)) if alg_noise_stds else 0.0,
        'channel_noise_std': float(np.mean(channel_noise_stds)) if channel_noise_stds else 0.0,
        'noise_norm': noise_sq_norm ** 0.5,
        'algorithmic_noise_norm': alg_noise_sq_norm ** 0.5,
        'channel_noise_norm': channel_noise_sq_norm ** 0.5,
        'clip_scope': getattr(args, 'dp_clip_scope', 'global'),
        'layer_clip_norms': layer_clip_norms,
        'aggregation_max_weight': aggregation_max_weight,
        'noise_scale_weight': noise_scale_weight,
        'noise_scaling': 'selected_count',
    }


def _compute_dp_epsilon(noise_multiplier, sample_rate, rounds, delta):
    """RDP accountant for the subsampled Gaussian mechanism."""
    sigma = float(noise_multiplier)
    if sigma <= 0:
        return float('inf')
    q = min(max(float(sample_rate), 1e-12), 1.0)
    orders = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    best_eps = float('inf')
    for alpha in orders:
        log_terms = []
        for k in range(alpha + 1):
            log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
            log_terms.append(
                log_binom
                + k * math.log(max(q, 1e-300))
                + (alpha - k) * math.log(max(1.0 - q, 1e-300))
                + (k * k - k) / (2.0 * sigma * sigma)
            )
        max_log = max(log_terms)
        rdp_round = (max_log + math.log(sum(math.exp(v - max_log) for v in log_terms))) / (alpha - 1)
        eps = rounds * rdp_round + math.log(1.0 / float(delta)) / (alpha - 1)
        best_eps = min(best_eps, eps)
    return best_eps


def _noise_multiplier_for_round(args, epoch, total_rounds):
    """Return sigma_t for the configured DP noise schedule."""
    base = float(args.dp_noise_multiplier)
    schedule = getattr(args, 'dp_noise_schedule', 'constant')
    if not getattr(args, 'dp_advanced', False) or schedule == 'constant' or total_rounds <= 1:
        return base

    start = min(max(float(getattr(args, 'dp_noise_start_multiplier', 0.7)), 1e-6), 1.0)
    progress = min(max(epoch / float(max(total_rounds - 1, 1)), 0.0), 1.0)
    if schedule == 'linear_increase':
        factor = start + (1.0 - start) * progress
    elif schedule == 'cosine_increase':
        factor = start + (1.0 - start) * (1.0 - math.cos(math.pi * progress)) / 2.0
    else:
        factor = 1.0
    return base * factor


def _compute_dp_epsilon_schedule(noise_multipliers, sample_rate, delta):
    """RDP accountant for a variable per-round sigma schedule."""
    q = min(max(float(sample_rate), 1e-12), 1.0)
    orders = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    total_rdp = {alpha: 0.0 for alpha in orders}
    for sigma in noise_multipliers:
        sigma = float(sigma)
        if sigma <= 0:
            return float('inf')
        for alpha in orders:
            log_terms = []
            for k in range(alpha + 1):
                log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
                log_terms.append(
                    log_binom
                    + k * math.log(max(q, 1e-300))
                    + (alpha - k) * math.log(max(1.0 - q, 1e-300))
                    + (k * k - k) / (2.0 * sigma * sigma)
                )
            max_log = max(log_terms)
            total_rdp[alpha] += (max_log + math.log(sum(math.exp(v - max_log) for v in log_terms))) / (alpha - 1)

    best_eps = float('inf')
    for alpha, rdp in total_rdp.items():
        eps = rdp + math.log(1.0 / float(delta)) / (alpha - 1)
        best_eps = min(best_eps, eps)
    return best_eps


def _compute_configured_dp_epsilon(args, sample_rate):
    if getattr(args, 'dp_advanced', False) and getattr(args, 'dp_noise_schedule', 'constant') != 'constant':
        sigmas = [_noise_multiplier_for_round(args, t, args.epochs) for t in range(args.epochs)]
        return _compute_dp_epsilon_schedule(sigmas, sample_rate, args.dp_delta)
    return _compute_dp_epsilon(args.dp_noise_multiplier, sample_rate, args.epochs, args.dp_delta)


def _compute_observed_dp_epsilon(dp_round_history, sample_rate, delta):
    sigmas = [
        float(item.get('noise_multiplier', 0.0))
        for item in (dp_round_history or [])
        if int(item.get('client', -1)) < 0 and item.get('mode') == 'central_aggregate'
    ]
    if not sigmas:
        return float('inf')
    return _compute_dp_epsilon_schedule(sigmas, sample_rate, delta)


def _build_oort_selector(args, num_selected, user_groups):
    train_sizes = np.asarray([
        max(int(len(user_groups[i]) * TRAIN_SPLIT_RATIO), 1)
        for i in range(args.num_users)
    ], dtype=np.float64)
    if getattr(args, 'oort_pacer_step', 0.0) > 0.0:
        pacer_step = float(args.oort_pacer_step)
    else:
        pacer_step = max(float(np.percentile(train_sizes, 25)) / max(float(np.median(train_sizes)), 1.0), 1e-3)
    return OortSelector(
        num_clients=args.num_users,
        sample_size=num_selected,
        epsilon=args.oort_epsilon,
        epsilon_decay=args.oort_epsilon_decay,
        epsilon_min=args.oort_epsilon_min,
        pacer_step=pacer_step,
        pacer_window=args.oort_pacer_window,
        straggler_penalty=args.oort_straggler_penalty,
        cutoff_util=args.oort_cutoff_util,
        clip_percentile=args.oort_clip_percentile,
        blacklist_rounds=args.oort_blacklist_rounds,
        seed=args.seed,
    )


def select_clients(args, epoch, num_selected, initial_rounds,
                   shapley_values, client_participation_counts,
                   energy_scores, available_clients,
                   energy_manager, lyapunov_optimizer,
                   client_local_losses, user_groups=None,
                   ucb_rewards=None, ucb_counts=None, oort_selector=None,
                   fedmsv_selector=None, gca_dsi_signals=None):
    """统一的客户端选择逻辑"""
    if shapley_values is not None:
        np.nan_to_num(shapley_values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- 非Shapley路径 ----
    if not args.use_shapley:
        if (args.selection_method == 'random' and args.use_energy and args.use_lyapunov and
                energy_manager is not None and lyapunov_optimizer is not None):
            return _select_lyapunov_without_shapley(
                args, epoch, num_selected, available_clients,
                energy_manager, lyapunov_optimizer, user_groups,
                client_participation_counts
            )
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
        if args.selection_method == 'oort':
            pool = available_clients if available_clients is not None else list(range(args.num_users))
            if oort_selector is None:
                return np.random.choice(pool, min(num_selected, len(pool)), replace=False).tolist()
            return oort_selector.select(available_clients=pool)
        if args.selection_method == 'fedmsv':
            pool = available_clients if available_clients is not None else list(range(args.num_users))
            if fedmsv_selector is None:
                return np.random.choice(pool, min(num_selected, len(pool)), replace=False).tolist()
            return fedmsv_selector.select(available_clients=pool)
        if args.selection_method == 'gca':
            client_data_sizes = _get_client_data_sizes(user_groups, args.num_users) if user_groups is not None else None
            if energy_manager is not None and energy_manager.channel_gains is not None:
                channel_gains = energy_manager.channel_gains
                if getattr(args, 'gca_mode', 'paper') == 'paper':
                    # Equation (5) in the source paper. Do not use the common
                    # simulator's numerical cap or compute-energy extension in
                    # the GCA ranking indicator.
                    per_round_energy = (
                        float(energy_manager.sigma_sq)
                        / (np.square(np.abs(channel_gains)) + 1e-10)
                    )
                else:
                    per_round_energy = energy_manager.compute_energy_consumption(
                        channel_gains,
                        selected_clients=None,
                        client_data_sizes=client_data_sizes,
                    )
            else:
                channel_gains = np.ones(args.num_users, dtype=np.float64)
                per_round_energy = np.ones(args.num_users, dtype=np.float64)
            pool = available_clients if available_clients is not None else list(range(args.num_users))
            return gradient_channel_aware_selection(
                learning_signals=(gca_dsi_signals
                                  if gca_dsi_signals is not None
                                  else client_local_losses),
                channel_gains=channel_gains,
                energy_costs=per_round_energy,
                num_selected=num_selected,
                learning_weight=args.gca_learning_weight,
                channel_weight=args.gca_channel_weight,
                energy_weight=args.gca_energy_weight,
                mode=getattr(args, 'gca_mode', 'paper'),
                rho_dsi=getattr(args, 'gca_rho_dsi', 0.5),
                rho_csi=getattr(args, 'gca_rho_csi', 0.5),
                lambda_energy=getattr(args, 'gca_lambda_energy', 0.5),
                available_clients=pool,
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
        candidates = (
            list(range(args.num_users))
            if available_clients is None
            else [int(client_id) for client_id in available_clients]
        )
        if client_participation_counts is None:
            if candidates:
                offset = (epoch * num_selected) % len(candidates)
                candidates = candidates[offset:] + candidates[:offset]
        else:
            candidates.sort(key=lambda client_id: (client_participation_counts[client_id], client_id))
        return candidates[:min(num_selected, len(candidates))]

    all_data_sizes = _get_client_data_sizes(user_groups, args.num_users) if user_groups is not None else None
    energy_consumed_estimate = energy_manager.compute_energy_consumption(
        energy_manager.channel_gains, client_data_sizes=all_data_sizes
    )
    battery_scores = energy_manager.get_energy_scores(normalize=True)
    scores = lyapunov_optimizer.compute_scores(
        shapley_values,
        energy_consumed_estimate,
        battery_scores=battery_scores,
        channel_gains=energy_manager.channel_gains,
        sv_weight=args.sv_weight,
        battery_weight=args.battery_weight,
        channel_weight=args.channel_weight,
        disable_queue_penalty=getattr(args, 'disable_queue_penalty', False),
    )

    return softmax_score_selection(
        scores=scores,
        num_selected=num_selected,
        temperature=args.selection_beta,
        available_clients=available_clients,
    )


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


def _select_lyapunov_without_shapley(args, epoch, num_selected, available_clients,
                                     energy_manager, lyapunov_optimizer,
                                     user_groups, client_participation_counts):
    """Lyapunov-only path used by the w/o SV ablation."""
    zero_shapley = np.zeros(args.num_users)
    return _select_lyapunov(
        args, epoch, num_selected, zero_shapley,
        client_participation_counts, available_clients,
        energy_manager, lyapunov_optimizer, user_groups
    )


def update_shapley_values(args, epoch, shapley_values, shapley_calculator,
                          round_client_models, val_data_loader,
                          user_groups, client_participation_counts,
                          shapley_observation_counts=None,
                          shapley_time_history=None):
    """计算并更新Shapley值，同时清理旧数据"""
    if not args.use_shapley:
        return

    round_data = round_client_models.get(epoch)

    if not round_data or len(round_data['selected_clients']) == 0:
        if args.verbose:
            print(f"  [Shapley] 轮次 {epoch + 1} 无数据，跳过Shapley计算")
        return

    selected_clients = round_data['selected_clients']
    client_models = round_data['client_models']

    client_model_list = []
    client_id_list = []
    client_data_size_list = []

    for client_id in selected_clients:
        if client_id in client_models:
            client_model_list.append(client_models[client_id])
            client_id_list.append(client_id)
            data_size = len(user_groups[client_id]) * TRAIN_SPLIT_RATIO
            client_data_size_list.append(int(data_size))

    if len(client_model_list) == 0:
        return

    print(f"  [Shapley] 计算轮次 {epoch + 1} 的Shapley值（{len(client_id_list)}个客户端, estimator={args.shapley_estimator}）...")

    try:
        _t0 = time.time()
        round_shapley = shapley_calculator.compute_with_history(
            previous_model=round_data['previous_global'],
            client_models=client_model_list,
            current_global_model=round_data['current_global'],
            val_data_loader=val_data_loader,
            client_ids=client_id_list,
            client_data_sizes=client_data_size_list
        )
        if len(round_shapley) != len(client_id_list):
            raise ValueError("Shapley result length does not match selected clients")
        _shapley_elapsed = time.time() - _t0
        if shapley_time_history is not None:
            shapley_time_history.append({
                'round': epoch + 1,
                'time_s': _shapley_elapsed,
                'estimator': args.shapley_estimator,
                'allocation': getattr(args, 'shapley_allocation', 'uniform'),
            })

        updated_count = 0
        for i, client_id in enumerate(client_id_list):
            raw_sv = round_shapley[i]
            if not np.isfinite(float(raw_sv)):
                continue
            new_sv = float(raw_sv)

            if args.shapley_update_method == 'mean':
                if shapley_observation_counts is None:
                    old_count = max(0, int(client_participation_counts[client_id]) - 1)
                else:
                    old_count = int(shapley_observation_counts[client_id])
                if old_count > 0:
                    shapley_values[client_id] = (
                        shapley_values[client_id] * old_count + new_sv
                    ) / (old_count + 1)
                else:
                    shapley_values[client_id] = new_sv
            elif args.shapley_update_method == 'exponential':
                alpha = args.shapley_alpha
                shapley_values[client_id] = (
                    alpha * shapley_values[client_id] +
                    (1 - alpha) * new_sv
                )
            elif args.shapley_update_method == 'recent':
                shapley_values[client_id] = new_sv

            if not np.isfinite(float(shapley_values[client_id])):
                shapley_values[client_id] = 0.0
                continue

            if shapley_observation_counts is not None:
                shapley_observation_counts[client_id] += 1

            updated_count += 1

        if shapley_time_history is not None:
            raw_scores = np.asarray(round_shapley, dtype=np.float64)
            stored_scores = np.array([shapley_values[c] for c in client_id_list], dtype=np.float64)
            shapley_time_history[-1].update({
                'raw_score_mean': float(raw_scores.mean()) if raw_scores.size else 0.0,
                'raw_score_std': float(raw_scores.std()) if raw_scores.size else 0.0,
                'stored_score_mean': float(stored_scores.mean()) if stored_scores.size else 0.0,
                'stored_score_std': float(stored_scores.std()) if stored_scores.size else 0.0,
            })

        if args.verbose:
            print(f"  [Shapley] 更新了 {updated_count} 个客户端的Shapley值")
            if len(round_shapley) > 0:
                print(f"  [Shapley] 本轮Shapley值范围: [{min(round_shapley):.6f}, "
                      f"{max(round_shapley):.6f}]")

    except Exception as e:
        print(f"  [Shapley] 计算失败: {e}")

    # Fix 2: 清理旧的round_client_models防止内存泄漏
    if epoch >= 1 and (epoch - 1) in round_client_models:
        del round_client_models[epoch - 1]


def save_results(args, exp_folder, timestamp, num_selected, train_loss, train_accuracy,
                 test_accuracies, test_acc, shapley_values, client_participation_counts,
                 client_last_round, shapley_observation_counts,
                 energy_manager, lyapunov_optimizer,
                  start_time, sv_sample_size=0, ucb_rewards=None, ucb_counts=None,
                  dp_round_history=None, shapley_time_history=None,
                  oort_selector=None, fedmsv_selector=None,
                  fedmsv_low_quality_clients=None):
    """保存实验结果"""
    os.makedirs(exp_folder, exist_ok=True)
    privacy_mode = getattr(args, 'privacy_mode', 'none')

    # 生成方法后缀
    if args.use_shapley:
        method_suffix = f"_{args.selection_method}_SV"
        if args.use_energy:
            method_suffix += "_Energy"
    else:
        method_suffix = f"_{args.selection_method}"
        if args.use_energy:
            method_suffix += "_Energy"
    if args.use_lyapunov:
        if getattr(args, 'disable_queue_penalty', False):
            method_suffix += "_NoQueue"
        else:
            method_suffix += "_Lyapunov"
    if args.use_fedprox:
        method_suffix += "_FedProx"
    if privacy_mode == 'central':
        method_suffix += "_CDP"

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
            'shapley_observation_counts': shapley_observation_counts,
            'shapley_time_history': shapley_time_history or [],
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
        time_average_energy = np.asarray([], dtype=np.float64)
        energy_violation = np.asarray([], dtype=np.float64)
        if args.use_energy and energy_manager is not None:
            time_average_energy = (
                energy_manager.initial_energy - energy_manager.client_energy
            ) / max(int(args.epochs), 1)
            energy_violation = np.maximum(time_average_energy - args.energy_budget, 0.0)
        save_data.update({
            'lyapunov_statistics': lyap_stats,
            'lyapunov_history': lyapunov_optimizer.lyapunov_history,
            'queue_history': lyapunov_optimizer.queue_history,
            'energy_constraint_statistics': {
                'time_average_energy_per_client': time_average_energy,
                'max_time_average_energy': float(np.max(time_average_energy)) if time_average_energy.size else 0.0,
                'mean_time_average_energy': float(np.mean(time_average_energy)) if time_average_energy.size else 0.0,
                'max_budget_violation': float(np.max(energy_violation)) if energy_violation.size else 0.0,
                'mean_budget_violation': float(np.mean(energy_violation)) if energy_violation.size else 0.0,
                'constraint_satisfied_fraction': float(np.mean(energy_violation <= 1e-12)) if energy_violation.size else 1.0,
                'queue_over_horizon': np.asarray(lyap_stats['energy_queue']) / max(int(args.epochs), 1),
            },
        })

    if args.selection_method == 'ucb' and ucb_rewards is not None:
        save_data.update({
            'ucb_rewards': ucb_rewards,
            'ucb_counts': ucb_counts,
        })

    if args.selection_method == 'oort' and oort_selector is not None:
        save_data.update({
            'oort_state': oort_selector.get_state(),
            'client_participation_counts': oort_selector.participation_counts.copy(),
        })

    if args.selection_method == 'fedmsv' and fedmsv_selector is not None:
        fedmsv_state = fedmsv_selector.get_state()
        save_data.update({
            'fedmsv_state': fedmsv_state,
            'fedmsv_history': fedmsv_state['history'],
            'fedmsv_values': fedmsv_state['cumulative_msv'],
            'fedmsv_sampling_weights': fedmsv_state['normalized_sampling_weights'],
            'client_participation_counts': fedmsv_state['participation_counts'],
            'fedmsv_low_quality_clients': list(fedmsv_low_quality_clients or []),
        })

    if privacy_mode == 'central':
        sampling_rate = num_selected / float(args.num_users)
        configured_update_epsilon = _compute_configured_dp_epsilon(args, sampling_rate)
        observed_update_epsilon = _compute_observed_dp_epsilon(dp_round_history, sampling_rate, args.dp_delta)
        save_data.update({
            'dp_statistics': {
                'privacy_mode': privacy_mode,
                'clip_norm': args.dp_clip_norm,
                'adaptive_clip': getattr(args, 'dp_adaptive_clip', False),
                'clip_scope': getattr(args, 'dp_clip_scope', 'global'),
                'clip_percentile': getattr(args, 'dp_clip_percentile', None),
                'clip_ema': getattr(args, 'dp_clip_ema', None),
                'min_clip_norm': getattr(args, 'dp_min_clip_norm', None),
                'max_clip_norm': getattr(args, 'dp_max_clip_norm', None),
                'advanced': getattr(args, 'dp_advanced', False),
                'noise_schedule': getattr(args, 'dp_noise_schedule', 'constant'),
                'noise_start_multiplier': getattr(args, 'dp_noise_start_multiplier', 1.0),
                'noise_multiplier': args.dp_noise_multiplier,
                'channel_assisted': getattr(args, 'dp_channel_assisted', False),
                'channel_noise_multiplier': getattr(args, 'dp_channel_noise_multiplier', 0.0),
                'channel_gain_cap': getattr(args, 'dp_channel_gain_cap', None),
                'channel_mode': getattr(args, 'dp_channel_mode', 'channel_only'),
                'configured_update_epsilon': configured_update_epsilon,
                'observed_update_epsilon': observed_update_epsilon,
                'update_epsilon': observed_update_epsilon if np.isfinite(observed_update_epsilon) else configured_update_epsilon,
                'delta': args.dp_delta,
                'sampling_rate': sampling_rate,
            },
            'dp_round_history': dp_round_history or [],
        })

    try:
        with open(file_name, 'wb') as f:
            pickle.dump(save_data, f)

        if os.path.exists(file_name):
            file_size = os.path.getsize(file_name)
            print(f'\n数据已保存到: {file_name}')
            print(f'文件大小: {file_size / 1024:.2f} KB')

            safe_timestamp = str(timestamp).replace('\\', '_').replace('/', '_')
            objects_dir = '../save/objects'
            os.makedirs(objects_dir, exist_ok=True)
            objects_file = f'{objects_dir}/result_{safe_timestamp}.pkl'
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
            f.write(f"- 选择方法: {args.selection_method}\n")
            if args.use_shapley:
                f.write(f"- Shapley估计器: {args.shapley_estimator}\n")
                if args.shapley_estimator == 'complementary':
                    f.write(f"- 互补贡献分配: {args.shapley_allocation}\n")
                    f.write(f"- pilot samples per stratum: {args.shapley_pilot_samples}\n")
                f.write(f"- Shapley更新方法: {args.shapley_update_method}\n")
                f.write(f"- Shapley alpha: {args.shapley_alpha}\n")
                f.write(f"- 快速模式: {args.shapley_fast}\n")
                f.write(f"- 初始轮数: {args.initial_rounds}\n")
                f.write(f"- SV验证集样本数: {sv_sample_size}\n")
                f.write(f"- SV测试集采样种子: 42\n")
            if args.selection_method == 'oort':
                f.write(f"- Oort epsilon: {args.oort_epsilon}\n")
                f.write(f"- Oort epsilon decay/min: {args.oort_epsilon_decay}/{args.oort_epsilon_min}\n")
                f.write(f"- Oort pacer window: {args.oort_pacer_window}\n")
                f.write(f"- Oort pacer step: {args.oort_pacer_step}\n")
                f.write(f"- Oort straggler penalty alpha: {args.oort_straggler_penalty}\n")
                f.write(f"- Oort cutoff utility ratio: {args.oort_cutoff_util}\n")
                f.write(f"- Oort clip percentile: {args.oort_clip_percentile}\n")
                f.write(f"- Oort blacklist rounds: {args.oort_blacklist_rounds}\n")
            if args.selection_method == 'fedmsv':
                f.write(f"- Fed-MSV guided prefix m: {args.fedmsv_guided_prefix}\n")
                f.write(f"- Fed-MSV epsilon_a/b/c: {args.fedmsv_epsilon_a}/{args.fedmsv_epsilon_b}/{args.fedmsv_epsilon_c}\n")
                f.write(f"- Fed-MSV max permutations: {args.fedmsv_max_permutations} (0 = paper enumeration)\n")
                f.write(f"- Fed-MSV utility source: {args.fedmsv_utility_source}\n")
                f.write(f"- Fed-MSV utility sample cap: {args.fedmsv_utility_samples}\n")
                f.write(f"- Fed-MSV low-quality scenario: {args.fedmsv_low_quality_type}\n")
                f.write(f"- Fed-MSV low-quality fraction: {args.fedmsv_low_quality_fraction}\n")
                f.write(f"- Fed-MSV low-quality clients: {list(fedmsv_low_quality_clients or [])}\n")
            if args.use_fedprox:
                f.write(f"- FedProx proximal term: enabled\n")
                f.write(f"- FedProx mu: {args.fedprox_mu}\n")
                f.write(f"- FedProx baseline selection: {'random' if args.selection_method == 'random' else 'combined with ' + args.selection_method}\n")
            f.write(f"\n## 能量感知\n\n")
            f.write(f"- 使用能量感知: {args.use_energy}\n")
            if args.use_energy:
                f.write(f"- 初始能量: {args.initial_energy}\n")
                f.write(f"- 能量阈值: {args.energy_threshold}\n")
                if args.use_lyapunov:
                    f.write(f"- 调度方式: Lyapunov动态权重 (V={args.lyapunov_V})\n")
                    f.write(f"- 能量预算: {args.energy_budget}\n")
                    f.write(f"- 分数Softmax温度 beta: {args.selection_beta}\n")
                else:
                    f.write(f"- Shapley权重: {args.shapley_weight}\n")
                    f.write(f"- 能量权重: {args.energy_weight}\n")
            f.write(f"\n## Privacy Module\n\n")
            f.write(f"- Privacy mode: {privacy_mode}\n")
            if privacy_mode == 'central':
                f.write(f"- 裁剪阈值 C: {args.dp_clip_norm}\n")
                if getattr(args, 'dp_adaptive_clip', False):
                    f.write(f"- Adaptive clipping: percentile={args.dp_clip_percentile}, "
                            f"EMA={args.dp_clip_ema}, scope={args.dp_clip_scope}, "
                            f"range=[{args.dp_min_clip_norm}, {args.dp_max_clip_norm}]\n")
                f.write(f"- 噪声乘子 sigma_dp: {args.dp_noise_multiplier}\n")
                f.write(f"- delta: {args.dp_delta}\n")
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
    logger = make_summary_writer('../logs')

    args = args_parser()
    if args.selection_method == 'fedmsv' and args.use_shapley:
        print("[Fed-MSV] Disabling the repository's separate Shapley scheduler; Fed-MSV maintains its own MSV state.")
        args.use_shapley = False
    exp_details(args)

    # Fix 5: 全局可复现性种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    num_selected = args.num_selected
    initial_rounds = _resolve_initial_rounds(args)
    args.initial_rounds = initial_rounds
    if not np.isfinite(args.selection_beta) or args.selection_beta <= 0:
        raise ValueError("selection_beta must be a positive finite value")

    if torch.cuda.is_available() and int(getattr(args, 'gpu', 0)) >= 0:
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device('cpu')
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
        print(f"Shapley估计器: {args.shapley_estimator}")
        if args.shapley_estimator == 'complementary':
            print(f"互补贡献分配: {args.shapley_allocation}")
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
        shapley_observation_counts = np.zeros(args.num_users, dtype=np.int64)
        client_last_round = -np.ones(args.num_users)

        round_client_models = {}

        # 从训练集中为每个客户端预留一部分数据，汇总为全局验证集（避免使用测试集导致数据泄露）
        sv_val_indices = []
        sv_rng = np.random.RandomState(42)
        for uid in range(args.num_users):
            client_idxs = list(user_groups[uid])
            # 从每个客户端取最后10%作为SV验证数据（与train_val_test的划分一致）
            val_start = int(TRAIN_SPLIT_RATIO * len(client_idxs))
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
        client_participation_counts = (
            np.zeros(args.num_users)
            if args.selection_method in ('oort', 'fedmsv')
            else None
        )
        client_last_round = None
        shapley_observation_counts = None
        round_client_models = None
        val_data_loader = None

    if args.selection_method == 'oort' and not args.use_shapley:
        oort_selector = _build_oort_selector(args, num_selected, user_groups)
        client_participation_counts = oort_selector.participation_counts
        print("\n" + "=" * 60)
        print("启用 Oort 训练端客户端选择 (Algorithm 1)")
        print(f"epsilon: {args.oort_epsilon} -> min {args.oort_epsilon_min}, decay={args.oort_epsilon_decay}")
        print(f"pacer: W={args.oort_pacer_window}, step={oort_selector.pacer_step:.6f}, alpha={args.oort_straggler_penalty}")
        print(f"utility clip percentile: {args.oort_clip_percentile}")
        print("=" * 60 + "\n")
    else:
        oort_selector = None

    training_dataset = train_dataset
    fedmsv_low_quality_clients = []
    fedmsv_low_quality_set = set()
    if args.selection_method == 'fedmsv' and not args.use_shapley:
        fedmsv_selector = FedMSVSelector(
            num_clients=args.num_users,
            sample_size=num_selected,
            guided_prefix=args.fedmsv_guided_prefix,
            epsilon_a=args.fedmsv_epsilon_a,
            epsilon_b=args.fedmsv_epsilon_b,
            epsilon_c=args.fedmsv_epsilon_c,
            max_permutations=args.fedmsv_max_permutations,
            seed=args.seed,
        )
        client_participation_counts = fedmsv_selector.participation_counts
        fedmsv_loader, fedmsv_sample_size = build_fedmsv_utility_loader(
            args, train_dataset, test_dataset, user_groups
        )
        fedmsv_utility = ModelAccuracyUtility(global_model, fedmsv_loader, device)
        fedmsv_low_quality_clients = choose_low_quality_clients(
            args.num_users, args.fedmsv_low_quality_fraction, args.seed + 1701
        )
        fedmsv_low_quality_set = set(fedmsv_low_quality_clients)
        if args.fedmsv_low_quality_type == 'label_flip' and fedmsv_low_quality_clients:
            label_overrides = build_label_overrides(
                train_dataset,
                user_groups,
                fedmsv_low_quality_clients,
                args.fedmsv_label_flip_fraction,
                args.num_classes,
                args.seed + 1702,
            )
            training_dataset = LabelOverrideDataset(train_dataset, label_overrides)

        print("\n" + "=" * 60)
        print("启用 Fed-MSV 客户端选择 (Algorithm 2)")
        print(f"m={args.fedmsv_guided_prefix}, epsilon_a={args.fedmsv_epsilon_a}, "
              f"epsilon_b={args.fedmsv_epsilon_b}, epsilon_c={args.fedmsv_epsilon_c}")
        print(f"排列上限: {args.fedmsv_max_permutations} (0 表示枚举 K!/(K-m)!)")
        print(f"效用数据: {args.fedmsv_utility_source}, 样本数: {fedmsv_sample_size}")
        if args.fedmsv_utility_source == 'test':
            print("[Fed-MSV] 论文复现模式使用测试集效用；该模式不用于公平主表比较。")
        print(f"低质量客户端场景: {args.fedmsv_low_quality_type}, "
              f"客户端数: {len(fedmsv_low_quality_clients)}")
        print("=" * 60 + "\n")
    else:
        fedmsv_selector = None
        fedmsv_utility = None
        fedmsv_sample_size = 0

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
            cycles_per_sample=args.cycles_per_sample,
            local_epochs=args.local_ep
        )
    else:
        energy_manager = None

    # ============= 李雅普诺夫优化器初始化 =============
    if args.use_lyapunov and args.use_energy:
        print("\n" + "=" * 60)
        print("启用李雅普诺夫动态权重优化")
        print(f"控制参数 V: {args.lyapunov_V}")
        print(f"分数Softmax温度 beta: {args.selection_beta}")
        print(f"学习率: {args.lyapunov_lr}")
        print("=" * 60 + "\n")

        from lyapunov_optimizer import LyapunovTripleScheduler

        lyapunov_optimizer = LyapunovTripleScheduler(
            num_clients=args.num_users,
            V=args.lyapunov_V,
            energy_budget=args.energy_budget
        )

    else:
        lyapunov_optimizer = None

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
        if args.selection_method != 'random':
            print(f"[提示] 当前 FedProx 与 {args.selection_method} 选择器组合使用；纯 FedProx baseline 建议使用 --selection_method random")
        print("=" * 60 + "\n")

    # ============= Privacy module 初始化 =============
    privacy_mode = getattr(args, 'privacy_mode', 'none')
    if privacy_mode == 'central':
        print("\n" + "=" * 60)
        print(f"启用隐私保护模块: {privacy_mode}")
        print(f"裁剪阈值 C: {args.dp_clip_norm}")
        if privacy_mode == 'central' and getattr(args, 'dp_adaptive_clip', False):
            print(f"自适应裁剪: percentile={args.dp_clip_percentile}, EMA={args.dp_clip_ema}, "
                  f"scope={args.dp_clip_scope}, range=[{args.dp_min_clip_norm}, {args.dp_max_clip_norm}]")
        print(f"噪声乘子 sigma_dp: {args.dp_noise_multiplier}")
        if privacy_mode == 'central' and getattr(args, 'dp_channel_assisted', False):
            print(f"Channel-assisted DP: channel_sigma={args.dp_channel_noise_multiplier}, "
                  f"mode={args.dp_channel_mode}, gain_cap={args.dp_channel_gain_cap}")
        if getattr(args, 'dp_advanced', False):
            print(f"Advanced CDP: schedule={args.dp_noise_schedule}, start={args.dp_noise_start_multiplier}")
        print(f"DP delta: {args.dp_delta}")
        print("=" * 60 + "\n")

    # 记录每个客户端的本地损失
    client_local_losses = np.ones(args.num_users)
    # GCA's source DSI is the squared local-update norm. Since the current
    # digital-FL protocol selects clients before local training, retain the
    # latest observed norm as a stale, source-aligned proxy for unselected
    # clients.
    gca_dsi_signals = np.ones(args.num_users, dtype=np.float64)

    # Training
    train_loss, train_accuracy = [], []
    print_every = 2
    test_accuracies = []
    shapley_time_history = []   # 每轮 Shapley 计算耗时
    dp_round_history = []
    adaptive_clip_norm = max(float(args.dp_clip_norm), 1e-12)
    adaptive_layer_clip_norms = None

    # PoC评估用的criterion
    poc_criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

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

        trainable_clients = _get_trainable_clients(user_groups, args.num_users)
        if len(trainable_clients) < num_selected:
            raise RuntimeError(
                f"Only {len(trainable_clients)} clients have local samples, "
                f"but num_selected={num_selected}."
            )
        available_clients = _filter_available_clients(available_clients, trainable_clients)

        idxs_users = select_clients(
            args, epoch, num_selected, initial_rounds,
            shapley_values, client_participation_counts,
            energy_scores, available_clients,
            energy_manager, lyapunov_optimizer,
            client_local_losses, user_groups=user_groups,
            ucb_rewards=ucb_rewards, ucb_counts=ucb_counts,
            oort_selector=oort_selector,
            fedmsv_selector=fedmsv_selector,
            gca_dsi_signals=gca_dsi_signals,
        )
        if available_clients is None:
            selection_candidates = trainable_clients
            target_selection_count = num_selected
        else:
            selection_candidates = available_clients
            target_selection_count = min(num_selected, len(selection_candidates))
        if target_selection_count == 0:
            raise RuntimeError("No energy-feasible trainable clients are available for this round.")
        idxs_users = _ensure_trainable_selection(
            idxs_users, selection_candidates, target_selection_count,
            client_participation_counts if client_participation_counts is not None else None
        )
        if fedmsv_selector is not None:
            fedmsv_selector.record_selection(idxs_users)

        if args.use_shapley and args.verbose and epoch % print_every == 0:
            print(f"选择的客户端: {sorted(idxs_users)}")
            print(f"客户端参与次数统计: 平均={np.mean(client_participation_counts):.1f}, "
                  f"最小={np.min(client_participation_counts)}, 最大={np.max(client_participation_counts)}")

        # 保存当前全局模型（用于Shapley计算）
        if args.use_shapley or fedmsv_selector is not None:
            previous_global_model = copy.deepcopy(global_model.state_dict())
            current_round_models = {}

        # ============= 本地训练 =============
        raw_client_records = []
        oort_feedback = {}
        for idx in idxs_users:
            if args.use_shapley:
                client_participation_counts[idx] += 1
                client_last_round[idx] = epoch
            elif args.selection_method == 'gca' and client_participation_counts is not None:
                client_participation_counts[idx] += 1

            local_start_time = time.time()
            is_low_quality = int(idx) in fedmsv_low_quality_set
            corruption_seed = int(args.seed) * 1000003 + int(epoch) * 1009 + int(idx)
            if (fedmsv_selector is not None and is_low_quality and
                    args.fedmsv_low_quality_type == 'free_rider'):
                w = random_free_rider_state(
                    global_model.state_dict(), args.fedmsv_free_rider_std, corruption_seed
                )
                loss = 0.0
                actual_samples = 0
                local_model = None
            else:
                if args.use_fedprox:
                    local_model = LocalUpdateFedProx(args=args, dataset=training_dataset,
                                                     idxs=user_groups[idx], logger=logger,
                                                     device=device, mu=args.fedprox_mu)
                else:
                    local_model = LocalUpdate(args=args, dataset=training_dataset,
                                              idxs=user_groups[idx], logger=logger, device=device)
                w, loss, actual_samples = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                if (fedmsv_selector is not None and is_low_quality and
                        args.fedmsv_low_quality_type == 'gaussian_noise'):
                    w = add_gaussian_model_noise(
                        w, args.fedmsv_noise_variance, corruption_seed
                    )
            local_duration = time.time() - local_start_time

            if args.selection_method == 'oort' and oort_selector is not None:
                oort_feedback[int(idx)] = {
                    'loss': float(loss),
                    'loss_square_mean': float(getattr(local_model, 'loss_square_mean', float(loss) ** 2)),
                    'num_samples': float(actual_samples),
                    'duration': float(local_duration),
                }

            if getattr(args, 'privacy_mode', 'none') == 'central':
                raw_client_records.append((idx, w, loss))
                continue

            else:
                dp_stats = {
                    'pre_clip_norm': 0.0,
                    'post_clip_norm': 0.0,
                    'clip_factor': 1.0,
                    'noise_std': 0.0,
                }

            if args.selection_method == 'gca':
                gca_dsi_signals[idx] = max(
                    _client_update_norm(global_weights, w, args) ** 2,
                    0.0,
                )

            # 只做一次deepcopy，共享引用以减少内存开销
            w_copy = copy.deepcopy(w)
            local_weights.append(w_copy)
            if args.use_shapley or fedmsv_selector is not None:
                current_round_models[idx] = w_copy

            local_losses.append(copy.deepcopy(loss))
            client_local_losses[idx] = loss

        if getattr(args, 'privacy_mode', 'none') == 'central':
            layer_mode = getattr(args, 'dp_clip_scope', 'global') == 'layer'
            update_norms = [_client_update_norm(global_weights, w, args) for _, w, _ in raw_client_records]
            layer_norm_records = [_client_update_layer_norms(global_weights, w, args) for _, w, _ in raw_client_records] if layer_mode else []

            if layer_mode:
                adaptive_layer_clip_norms = _adaptive_layer_clip_norms(
                    layer_norm_records, adaptive_layer_clip_norms, args
                )
                args._current_dp_layer_clip_norms = adaptive_layer_clip_norms
                adaptive_clip_norm = float(np.mean(list(adaptive_layer_clip_norms.values()))) if adaptive_layer_clip_norms else max(float(args.dp_clip_norm), 1e-12)
                args._current_dp_clip_norm = adaptive_clip_norm
            else:
                adaptive_clip_norm = _adaptive_clip_norm(update_norms, adaptive_clip_norm, args)
                args._current_dp_clip_norm = adaptive_clip_norm

            for pos, ((idx, raw_w, loss), update_norm) in enumerate(zip(raw_client_records, update_norms)):
                if args.selection_method == 'gca':
                    gca_dsi_signals[idx] = max(float(update_norm) ** 2, 0.0)
                if layer_mode:
                    w, dp_stats = _clip_client_update_layerwise(global_weights, raw_w, adaptive_layer_clip_norms, args)
                else:
                    w, dp_stats = _clip_client_update(global_weights, raw_w, adaptive_clip_norm, args)

                w_copy = copy.deepcopy(w)
                local_weights.append(w_copy)
                if args.use_shapley or fedmsv_selector is not None:
                    current_round_models[idx] = w_copy

                local_losses.append(copy.deepcopy(loss))
                client_local_losses[idx] = loss

                dp_round_history.append({
                    'round': epoch,
                    'client': int(idx),
                    'mode': 'central',
                    'pre_clip_norm': dp_stats['pre_clip_norm'],
                    'post_clip_norm': dp_stats['post_clip_norm'],
                    'clip_factor': dp_stats['clip_factor'],
                    'noise_std': dp_stats['noise_std'],
                    'clip_norm': dp_stats.get('clip_norm', adaptive_clip_norm),
                    'clip_scope': getattr(args, 'dp_clip_scope', 'global'),
                    'adaptive_clip': bool(getattr(args, 'dp_adaptive_clip', False)),
                    'round_norm_percentile': float(np.percentile(update_norms, min(max(float(args.dp_clip_percentile), 0.0), 100.0))) if len(update_norms) > 0 else adaptive_clip_norm,
                    'raw_update_norm': float(update_norm),
                    'layer_clip_norm_mean': float(np.mean(list(adaptive_layer_clip_norms.values()))) if layer_mode and adaptive_layer_clip_norms else 0.0,
                })

        if args.selection_method == 'oort' and oort_selector is not None:
            oort_selector.update_feedback(oort_feedback)
            client_participation_counts = oort_selector.participation_counts

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

            if args.use_lyapunov and lyapunov_optimizer is not None:
                lyapunov_optimizer.update_queue(
                    energy_consumed=energy_consumed,
                    selected_clients=idxs_users,
                    round_num=epoch + 1,
                )
                if args.verbose and epoch % print_every == 0:
                    lyapunov_optimizer.print_statistics(epoch + 1)

            if args.verbose and epoch % print_every == 0:
                print(f"  [能量] 本轮平均消耗: {np.mean(energy_consumed):.2f}")
                print(f"  [能量] 消耗范围: [{np.min(energy_consumed):.2f}, {np.max(energy_consumed):.2f}]")

        # ============= 聚合全局权重 =============
        client_data_sizes = []
        for idx in idxs_users:
            data_size = len(user_groups[idx]) * TRAIN_SPLIT_RATIO
            client_data_sizes.append(int(data_size))

        global_weights = average_weights(local_weights, client_data_sizes)
        if args.use_shapley:
            round_client_models[epoch]['current_global'] = copy.deepcopy(global_weights)
        if fedmsv_selector is not None:
            fedmsv_sizes = {
                int(client_id): int(data_size)
                for client_id, data_size in zip(idxs_users, client_data_sizes)
            }
            fedmsv_record = fedmsv_selector.update_from_round(
                selected_clients=idxs_users,
                previous_global=previous_global_model,
                current_global=global_weights,
                client_models=current_round_models,
                client_data_sizes=fedmsv_sizes,
                utility_fn=fedmsv_utility,
                round_index=epoch,
            )
            client_participation_counts = fedmsv_selector.participation_counts
            if args.verbose and fedmsv_record:
                selected_msv = fedmsv_record['round_msv'][list(idxs_users)]
                print(f"  [Fed-MSV] skipped={fedmsv_record['round_skipped']}, "
                      f"permutations={fedmsv_record['permutation_count']}, "
                      f"evaluations={fedmsv_record['utility_evaluations']}, "
                      f"MSV=[{selected_msv.min():.3f}, {selected_msv.max():.3f}]")

        total_selected_data = max(sum(client_data_sizes), 1)
        aggregation_max_weight = max(client_data_sizes) / total_selected_data
        if getattr(args, 'privacy_mode', 'none') == 'central':
            args._current_dp_noise_multiplier = _noise_multiplier_for_round(args, epoch, args.epochs)
            if getattr(args, 'dp_channel_assisted', False) and 'channel_gains' in locals() and channel_gains is not None:
                args._current_selected_channel_gains = np.asarray(channel_gains)[list(idxs_users)]
            else:
                args._current_selected_channel_gains = None
            global_weights, central_dp_stats = _add_central_dp_noise(
                previous_global_model if args.use_shapley else global_model.state_dict(),
                global_weights,
                args,
                len(idxs_users),
                aggregation_max_weight=aggregation_max_weight,
            )
            dp_round_history.append({
                'round': epoch,
                'client': -1,
                'mode': 'central_aggregate',
                'pre_clip_norm': 0.0,
                'post_clip_norm': 0.0,
                'clip_factor': 1.0,
                'noise_std': central_dp_stats['noise_std'],
                'noise_multiplier': central_dp_stats.get('noise_multiplier', args.dp_noise_multiplier),
                'target_noise_multiplier': central_dp_stats.get('target_noise_multiplier', args.dp_noise_multiplier),
                'algorithmic_noise_multiplier': central_dp_stats.get('algorithmic_noise_multiplier', args.dp_noise_multiplier),
                'channel_noise_multiplier': central_dp_stats.get('channel_noise_multiplier', 0.0),
                'channel_assisted': central_dp_stats.get('channel_assisted', False),
                'algorithmic_noise_std': central_dp_stats.get('algorithmic_noise_std', 0.0),
                'channel_noise_std': central_dp_stats.get('channel_noise_std', 0.0),
                'noise_norm': central_dp_stats['noise_norm'],
                'algorithmic_noise_norm': central_dp_stats.get('algorithmic_noise_norm', 0.0),
                'channel_noise_norm': central_dp_stats.get('channel_noise_norm', 0.0),
                'clip_norm': central_dp_stats['clip_norm'],
                'clip_scope': central_dp_stats.get('clip_scope', getattr(args, 'dp_clip_scope', 'global')),
                'adaptive_clip': bool(getattr(args, 'dp_adaptive_clip', False)),
                'aggregation_max_weight': central_dp_stats.get('aggregation_max_weight', aggregation_max_weight),
                'noise_scale_weight': central_dp_stats.get('noise_scale_weight', 1.0 / max(len(idxs_users), 1)),
                'noise_scaling': central_dp_stats.get('noise_scaling', 'selected_count'),
            })
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
                round_client_models, val_data_loader,
                user_groups, client_participation_counts,
                shapley_observation_counts=shapley_observation_counts,
                shapley_time_history=shapley_time_history
            )

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Fix 3: 只在选中客户端上评估训练准确率（而非所有客户端）
        list_acc, list_loss = [], []
        global_model.eval()
        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=training_dataset,
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
        shapley_observation_counts,
        energy_manager, lyapunov_optimizer,
        start_time,
        sv_sample_size=sv_sample_size if args.use_shapley else 0,
        ucb_rewards=ucb_rewards, ucb_counts=ucb_counts,
        dp_round_history=dp_round_history,
        shapley_time_history=shapley_time_history if args.use_shapley else None,
        oort_selector=oort_selector,
        fedmsv_selector=fedmsv_selector,
        fedmsv_low_quality_clients=fedmsv_low_quality_clients,
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
        print(f"客户端选择方法: {args.selection_method}")
        if fedmsv_selector is not None:
            fedmsv_state = fedmsv_selector.get_state()
            print(f"Fed-MSV 累计值范围: [{fedmsv_state['cumulative_msv'].min():.3f}, "
                  f"{fedmsv_state['cumulative_msv'].max():.3f}]")
            print(f"Fed-MSV 效用评估总数: "
                  f"{sum(item['utility_evaluations'] for item in fedmsv_state['history'])}")

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

        safe_timestamp = str(timestamp).replace('\\', '_').replace('/', '_')
        lyap_viz_path = f'{exp_folder}/lyapunov_{args.dataset}_{safe_timestamp}.png'
        lyapunov_optimizer.visualize_optimization(lyap_viz_path)

    if privacy_mode == 'central' and dp_round_history:
        client_dp_items = [item for item in dp_round_history if item.get('client', -1) >= 0]
        clip_factors = [item['clip_factor'] for item in client_dp_items]
        pre_clip_norms = [item['pre_clip_norm'] for item in client_dp_items]
        print(f"\nPrivacy module统计:")
        if pre_clip_norms:
            print(f"  平均更新范数(裁剪前): {np.mean(pre_clip_norms):.4f}")
            print(f"  平均裁剪系数: {np.mean(clip_factors):.4f}")
        print(f"  噪声乘子 sigma_dp: {args.dp_noise_multiplier:.4f}")
        aggregate_dp_items = [item for item in dp_round_history if item.get('client', -1) < 0]
        if aggregate_dp_items and getattr(args, 'dp_channel_assisted', False):
            eff_sigmas = [item.get('noise_multiplier', 0.0) for item in aggregate_dp_items]
            alg_sigmas = [item.get('algorithmic_noise_multiplier', 0.0) for item in aggregate_dp_items]
            ch_sigmas = [item.get('channel_noise_multiplier', 0.0) for item in aggregate_dp_items]
            obs_eps = _compute_observed_dp_epsilon(dp_round_history, num_selected / float(args.num_users), args.dp_delta)
            print(f"  Channel-assisted DP: eff_sigma={np.mean(eff_sigmas):.4f}, "
                  f"alg_sigma={np.mean(alg_sigmas):.4f}, ch_sigma={np.mean(ch_sigmas):.4f}, "
                  f"epsilon_h={obs_eps:.4f}")
    print(f"总运行时间: {time.time() - start_time:.2f}秒")
    print(f"{'=' * 60}")
