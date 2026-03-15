#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
shapley.py - GTG-Shapley值计算模块
"""

import numpy as np
import torch
import torch.nn as nn
import copy
import random
import time
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader


class MCShapley:
    """
    Monte Carlo Permutation Sampling for Shapley Value Approximation
    with Truncation Strategies for efficiency.
    """

    def __init__(self, model_class, args, epsilon=None, max_iterations=None,
                 device=None, verbose=False):
        """
        初始化GTG-Shapley计算器

        Args:
            model_class: 模型类（如CNNMnist, CNNCifar等）
            args: 训练参数
            epsilon: 截断阈值
            max_iterations: 蒙特卡洛最大迭代次数
            device: 计算设备
            verbose: 是否打印详细信息
        """
        self.model_class = model_class
        self.args = args
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.device = device
        self.verbose = verbose

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # 效用值缓存
        self.utility_cache = {}

        # 历史记录存储
        self.client_history = {}  # client_id -> list of shapley values

    def _create_model(self):
        """创建模型实例"""
        if self.model_class.__name__ in ['MLP']:
            # 根据数据集确定输入维度，避免硬编码
            dataset = self.args.dataset
            if dataset == 'cifar':
                img_size = (3, 32, 32)
            else:  # mnist, fmnist
                img_size = (1, 28, 28)
            len_in = 1
            for x in img_size:
                len_in *= x
            return self.model_class(dim_in=len_in, dim_hidden=64,
                                    dim_out=self.args.num_classes)
        else:
            return self.model_class(self.args)

    def _normalize_to_device(self, model_state_dict, target_device):
        """将模型状态字典标准化到指定设备"""
        normalized = {}
        for key, value in model_state_dict.items():
            if isinstance(value, torch.Tensor):
                normalized[key] = value.to(target_device)
            else:
                normalized[key] = value
        return normalized

    def compute_utility(self, model_state_dict: Dict, data_loader: DataLoader,
                        max_batches: int = None) -> float:
        """
        计算模型在验证集上的效用（负损失）

        Args:
            model_state_dict: 模型权重
            data_loader: 验证数据加载器
            max_batches: 最多使用的batch数量（None表示使用全部）
        """
        model = self._create_model()
        model.to(self.device)

        model_state_on_device = self._normalize_to_device(model_state_dict, self.device)
        model.load_state_dict(model_state_on_device)
        model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)

                total_loss += batch_loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        utility = -avg_loss

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return utility

    def aggregate_models(self, models: List[Dict], weights: List[float] = None) -> Dict:
        """
        聚合多个客户端模型
        """
        if len(models) == 0:
            raise ValueError("模型列表不能为空")

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        # 检查第一个模型的设备
        first_model = models[0]
        sample_key = list(first_model.keys())[0]
        sample_tensor = first_model[sample_key]

        if not isinstance(sample_tensor, torch.Tensor):
            target_device = torch.device('cpu')
        else:
            target_device = sample_tensor.device

        # 确保所有模型在相同设备
        normalized_models = []
        for model in models:
            normalized_models.append(self._normalize_to_device(model, target_device))

        # 初始化为零
        aggregated_model = {}
        for key in normalized_models[0].keys():
            tensor = normalized_models[0][key]
            if isinstance(tensor, torch.Tensor):
                if tensor.is_floating_point():
                    aggregated_model[key] = torch.zeros_like(tensor, device=target_device)
                else:
                    aggregated_model[key] = torch.zeros_like(tensor.float(),
                                                             device=target_device).long()
            else:
                aggregated_model[key] = tensor

        # 加权平均
        for i, model in enumerate(normalized_models):
            weight = weights[i]
            for key in aggregated_model.keys():
                if key in model:
                    tensor = model[key]
                    if isinstance(tensor, torch.Tensor) and isinstance(aggregated_model[key], torch.Tensor):
                        if aggregated_model[key].is_floating_point():
                            aggregated_model[key] += weight * tensor
                        else:
                            aggregated_model[key] = (aggregated_model[key].float() +
                                                     weight * tensor.float()).long()

        return aggregated_model

    def compute(self, previous_model, client_models, current_global_model,
                val_data_loader, client_ids=None, client_data_sizes=None):
        """
        计算Shapley值（蒙特卡洛排列方法）
        """
        M = len(client_models)

        if M == 0:
            return []

        if client_ids is None:
            client_ids = list(range(M))

        if len(client_ids) != M:
            raise ValueError(f"client_ids长度({len(client_ids)})与client_models长度({M})不匹配")

        print(f"\n[Shapley] 开始计算 {M} 个客户端的Shapley值")
        print(f"[Shapley] 客户端ID列表: {client_ids}")

        id_to_idx = {client_id: idx for idx, client_id in enumerate(client_ids)}

        # 每轮计算开始时清空缓存（因为模型权重已变化）
        self.utility_cache = {}

        v0 = self.compute_utility(previous_model, val_data_loader)
        vM = self.compute_utility(current_global_model, val_data_loader)

        print(f"[Shapley] v0={v0:.6f}, vM={vM:.6f}, Δ={vM - v0:.6f}")

        if abs(vM - v0) < self.epsilon:
            print(f"[Shapley] 轮间截断触发")
            return [0.0] * M

        shapley_dict = {cid: 0.0 for cid in client_ids}
        count_dict = {cid: 0 for cid in client_ids}

        start_time = time.time()

        for tau in range(self.max_iterations):
            print(f"\n=== 第{tau + 1}/{self.max_iterations}次迭代 ===")

            permutation = client_ids.copy()
            random.shuffle(permutation)

            print(f"客户端ID排列: {permutation}")

            v_prev = v0

            for j in range(1, M + 1):
                current_client_id = permutation[j - 1]

                print(f"  j={j}: 处理客户端 {current_client_id}")

                subset_ids = permutation[:j]
                subset_indices = [id_to_idx[cid] for cid in subset_ids]

                if client_data_sizes is not None:
                    subset_sizes = [client_data_sizes[idx] for idx in subset_indices]
                    total_data = sum(subset_sizes)
                    weights = [size / total_data for size in subset_sizes]
                else:
                    weights = None

                if j == M:
                    v_current = vM
                else:
                    # 使用frozenset作为缓存key
                    cache_key = frozenset(subset_ids)
                    if cache_key in self.utility_cache:
                        v_current = self.utility_cache[cache_key]
                    else:
                        subset_models = [client_models[idx] for idx in subset_indices]
                        subset_model = self.aggregate_models(subset_models, weights)
                        v_current = self.compute_utility(subset_model, val_data_loader)
                        self.utility_cache[cache_key] = v_current

                marginal = v_current - v_prev

                print(f"    边际贡献: {marginal:.6f} 属于客户端 {current_client_id}")

                old_value = shapley_dict[current_client_id]
                old_count = count_dict[current_client_id]

                if old_count == 0:
                    shapley_dict[current_client_id] = marginal
                else:
                    shapley_dict[current_client_id] = (old_value * old_count + marginal) / (old_count + 1)

                count_dict[current_client_id] += 1

                print(f"    客户端{current_client_id}更新: {shapley_dict[current_client_id]:.6f}")

                v_prev = v_current

                if abs(vM - v_current) < self.epsilon:
                    print(f"    轮内截断触发")
                    for remaining_id in permutation[j:]:
                        count_dict[remaining_id] += 1
                    break

            elapsed = time.time() - start_time
            print(f"\n迭代{tau + 1}完成，耗时{elapsed:.2f}秒")
            print(f"当前Shapley值:")
            for cid in client_ids:
                print(f"  客户端{cid}: {shapley_dict[cid]:.6f}")

        result = [shapley_dict[cid] for cid in client_ids]

        print(f"\n Shapley计算完成！")
        print(f"最终结果:")
        for idx, cid in enumerate(client_ids):
            print(f"  客户端{cid}: {result[idx]:.6f}")

        return result

    def compute_with_history(self, previous_model, client_models, current_global_model,
                             val_data_loader, client_ids=None, client_data_sizes=None):
        """
        计算Shapley值并记录历史（不做平均，平滑逻辑由调用方控制）
        """
        current_shapley = self.compute(
            previous_model, client_models, current_global_model,
            val_data_loader, client_ids, client_data_sizes
        )

        # 只记录历史，不做平均，避免与调用方的平滑逻辑重复
        for client_id, current_val in zip(client_ids, current_shapley):
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            self.client_history[client_id].append(current_val)

            if self.verbose:
                print(
                    f"客户端 {client_id}: 当前值={current_val:.6f}, 历史轮数={len(self.client_history[client_id])}")

        return current_shapley


