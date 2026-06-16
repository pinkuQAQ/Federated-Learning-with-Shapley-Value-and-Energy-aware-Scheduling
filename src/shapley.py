#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shapley value approximation utilities."""

import copy
import logging
import math
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _n_choose_k(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    if k == 0:
        return 1
    numer = 1
    denom = 1
    for i in range(1, k + 1):
        numer *= n - (k - i)
        denom *= i
    return numer // denom


class MCShapley:
    """
    Shapley value approximation for one selected coalition.

    Supported estimators:
    - permutation: classic Monte Carlo permutation sampling
    - complementary: complementary-contribution stratified sampling
    """

    def __init__(self, model_class, args, epsilon=None, max_iterations=None,
                 device=None, verbose=False, rng=None):
        self.model_class = model_class
        self.args = args
        self.epsilon = epsilon if epsilon is not None else 1e-5
        self.max_iterations = max_iterations if max_iterations is not None else 20
        self.device = device
        self.verbose = verbose
        if rng is not None:
            self.rng = rng
        else:
            seed = getattr(args, 'seed', 0)
            self.rng = np.random.default_rng(seed * 9973 + 7)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.utility_cache = {}
        self.client_history = {}

    def _create_model(self):
        if self.model_class.__name__ in ['MLP']:
            dataset = self.args.dataset
            if dataset == 'cifar':
                img_size = (3, 32, 32)
            else:
                img_size = (1, 28, 28)
            dim_in = 1
            for x in img_size:
                dim_in *= x
            return self.model_class(dim_in=dim_in, dim_hidden=64,
                                    dim_out=self.args.num_classes)
        return self.model_class(self.args)

    @staticmethod
    def _normalize_to_device(model_state_dict, target_device):
        normalized = {}
        for key, value in model_state_dict.items():
            if isinstance(value, torch.Tensor):
                normalized[key] = value.to(target_device)
            else:
                normalized[key] = value
        return normalized

    def compute_utility(self, model_state_dict: Dict, data_loader: DataLoader,
                        max_batches: int = None) -> float:
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
                if (not torch.isfinite(outputs).all()) or (not torch.isfinite(batch_loss)):
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return float('-inf')

                total_loss += batch_loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        utility = -avg_loss
        if not np.isfinite(utility):
            utility = float('-inf')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return utility

    def aggregate_models(self, models: Sequence[Dict], weights: Sequence[float] = None) -> Dict:
        if len(models) == 0:
            raise ValueError("models cannot be empty")

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        first_model = models[0]
        sample_key = list(first_model.keys())[0]
        sample_tensor = first_model[sample_key]
        target_device = sample_tensor.device if isinstance(sample_tensor, torch.Tensor) else torch.device('cpu')

        normalized_models = [
            self._normalize_to_device(model, target_device)
            for model in models
        ]

        aggregated_model = {}
        for key in normalized_models[0].keys():
            tensor = normalized_models[0][key]
            if isinstance(tensor, torch.Tensor):
                if tensor.is_floating_point():
                    aggregated_model[key] = torch.zeros_like(tensor, device=target_device)
                else:
                    aggregated_model[key] = torch.zeros_like(tensor.float(), device=target_device).long()
            else:
                aggregated_model[key] = tensor

        for weight, model in zip(weights, normalized_models):
            for key in aggregated_model.keys():
                if key not in model:
                    continue
                tensor = model[key]
                if isinstance(tensor, torch.Tensor) and isinstance(aggregated_model[key], torch.Tensor):
                    if aggregated_model[key].is_floating_point():
                        aggregated_model[key] += weight * tensor
                    else:
                        aggregated_model[key] = (
                            aggregated_model[key].float() + weight * tensor.float()
                        ).long()

        return aggregated_model

    def _subset_weights(self, subset_indices: List[int], client_data_sizes=None):
        if client_data_sizes is None:
            return None
        subset_sizes = [client_data_sizes[idx] for idx in subset_indices]
        total_size = float(sum(subset_sizes))
        if total_size <= 0:
            return None
        return [size / total_size for size in subset_sizes]

    def _subset_utility(self, subset_ids: Sequence[int], client_models: Sequence[Dict],
                        id_to_idx: Dict[int, int], client_data_sizes,
                        val_data_loader: DataLoader, v_empty: float, v_full: float) -> float:
        subset_ids = list(subset_ids)
        num_clients = len(client_models)
        if len(subset_ids) == 0:
            return v_empty
        if len(subset_ids) == num_clients:
            return v_full

        cache_key = frozenset(subset_ids)
        if cache_key in self.utility_cache:
            return self.utility_cache[cache_key]

        subset_indices = [id_to_idx[cid] for cid in subset_ids]
        subset_models = [client_models[idx] for idx in subset_indices]
        weights = self._subset_weights(subset_indices, client_data_sizes)
        subset_model = self.aggregate_models(subset_models, weights)
        utility = self.compute_utility(subset_model, val_data_loader)
        self.utility_cache[cache_key] = utility
        return utility

    def _compute_permutation(self, previous_model, client_models, current_global_model,
                             val_data_loader, client_ids=None, client_data_sizes=None):
        num_clients = len(client_models)
        if num_clients == 0:
            return []

        if client_ids is None:
            client_ids = list(range(num_clients))
        if len(client_ids) != num_clients:
            raise ValueError("client_ids length does not match client_models length")

        id_to_idx = {client_id: idx for idx, client_id in enumerate(client_ids)}
        self.utility_cache = {}

        v_empty = self.compute_utility(previous_model, val_data_loader)
        v_full = self.compute_utility(current_global_model, val_data_loader)
        if not (np.isfinite(v_empty) and np.isfinite(v_full)):
            logger.warning("Non-finite Shapley endpoint utility detected; returning zero values.")
            return [0.0] * num_clients
        if abs(v_full - v_empty) < self.epsilon:
            return [0.0] * num_clients

        shapley_dict = {cid: 0.0 for cid in client_ids}
        count_dict = {cid: 0 for cid in client_ids}

        for _ in range(self.max_iterations):
            permutation = client_ids.copy()
            self.rng.shuffle(permutation)
            v_prev = v_empty

            for j in range(1, num_clients + 1):
                current_client_id = permutation[j - 1]
                subset_ids = permutation[:j]
                v_current = self._subset_utility(
                    subset_ids, client_models, id_to_idx, client_data_sizes,
                    val_data_loader, v_empty, v_full
                )
                if not np.isfinite(v_current):
                    v_current = v_prev

                marginal = v_current - v_prev
                if not np.isfinite(marginal):
                    marginal = 0.0

                old_value = shapley_dict[current_client_id]
                old_count = count_dict[current_client_id]
                if old_count == 0:
                    shapley_dict[current_client_id] = marginal
                else:
                    shapley_dict[current_client_id] = (
                        old_value * old_count + marginal
                    ) / (old_count + 1)
                count_dict[current_client_id] += 1
                v_prev = v_current

                if abs(v_full - v_current) < self.epsilon:
                    break

        return [
            float(shapley_dict[cid]) if np.isfinite(shapley_dict[cid]) else 0.0
            for cid in client_ids
        ]

    def _primary_sizes(self, num_clients: int) -> List[int]:
        upper = max(1, num_clients // 2)
        budget = max(1, int(self.max_iterations))
        num_strata = min(max(2, budget), upper)
        raw = np.linspace(1, upper, num=num_strata)
        sizes = sorted({int(round(x)) for x in raw})
        return [s for s in sizes if 1 <= s <= upper]

    def _balanced_coalitions_for_size(self, client_ids: Sequence[int], size: int) -> List[List[int]]:
        if size <= 0 or size > len(client_ids):
            return []
        if size == len(client_ids):
            return [list(client_ids)]

        perm = list(self.rng.permutation(np.asarray(client_ids, dtype=np.int64)).tolist())
        doubled = perm + perm
        coalitions = []
        for start in range(len(perm)):
            coalitions.append(doubled[start:start + size])
        return coalitions

    def _random_coalition_of_size(self, client_ids: Sequence[int], size: int) -> List[int]:
        chosen = self.rng.choice(np.asarray(client_ids, dtype=np.int64), size=size, replace=False)
        return chosen.tolist()

    def _record_complementary_sample(self, subset_ids: Sequence[int], all_client_ids: Sequence[int],
                                     cc_value: float, stratum_sums: Dict[int, np.ndarray],
                                     stratum_counts: Dict[int, np.ndarray]):
        subset = list(subset_ids)
        subset_set = set(subset)
        subset_size = len(subset)
        complement = [cid for cid in all_client_ids if cid not in subset_set]
        complement_size = len(complement)

        if subset_size >= 1:
            for cid in subset:
                stratum_sums[cid][subset_size] += cc_value
                stratum_counts[cid][subset_size] += 1

        if complement_size >= 1:
            neg_value = -cc_value
            for cid in complement:
                stratum_sums[cid][complement_size] += neg_value
                stratum_counts[cid][complement_size] += 1

    def _evaluate_complementary_coalition(self, subset_ids: Sequence[int], client_ids: Sequence[int],
                                          client_models: Sequence[Dict], id_to_idx: Dict[int, int],
                                          client_data_sizes, val_data_loader: DataLoader,
                                          v_empty: float, v_full: float) -> float:
        subset = list(subset_ids)
        subset_set = set(subset)
        complement = [cid for cid in client_ids if cid not in subset_set]

        utility_subset = self._subset_utility(
            subset, client_models, id_to_idx, client_data_sizes, val_data_loader, v_empty, v_full
        )
        utility_complement = self._subset_utility(
            complement, client_models, id_to_idx, client_data_sizes, val_data_loader, v_empty, v_full
        )

        if not np.isfinite(utility_subset):
            utility_subset = v_empty
        if not np.isfinite(utility_complement):
            utility_complement = v_empty

        cc_value = utility_subset - utility_complement
        if not np.isfinite(cc_value):
            cc_value = 0.0
        return float(cc_value)

    def _allocate_extra_by_neyman(self, primary_sizes: Sequence[int], observed_std: Dict[int, float],
                                  extra_budget: int) -> Dict[int, int]:
        if extra_budget <= 0:
            return {size: 0 for size in primary_sizes}

        weights = np.asarray(
            [max(float(observed_std.get(size, 0.0)), 1e-8) for size in primary_sizes],
            dtype=np.float64,
        )
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = np.ones(len(primary_sizes), dtype=np.float64) / max(len(primary_sizes), 1)
        else:
            weights = weights / weight_sum

        raw = weights * extra_budget
        counts = np.floor(raw).astype(int)
        remainder = extra_budget - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:remainder]:
                counts[idx] += 1

        return {size: int(counts[idx]) for idx, size in enumerate(primary_sizes)}

    def _compute_complementary(self, previous_model, client_models, current_global_model,
                               val_data_loader, client_ids=None, client_data_sizes=None):
        num_clients = len(client_models)
        if num_clients == 0:
            return []

        if client_ids is None:
            client_ids = list(range(num_clients))
        if len(client_ids) != num_clients:
            raise ValueError("client_ids length does not match client_models length")

        id_to_idx = {client_id: idx for idx, client_id in enumerate(client_ids)}
        self.utility_cache = {}

        v_empty = self.compute_utility(previous_model, val_data_loader)
        v_full = self.compute_utility(current_global_model, val_data_loader)
        if not (np.isfinite(v_empty) and np.isfinite(v_full)):
            logger.warning("Non-finite Shapley endpoint utility detected; returning zero values.")
            return [0.0] * num_clients
        if abs(v_full - v_empty) < self.epsilon:
            return [0.0] * num_clients

        stratum_sums = {cid: np.zeros(num_clients + 1, dtype=np.float64) for cid in client_ids}
        stratum_counts = {cid: np.zeros(num_clients + 1, dtype=np.int64) for cid in client_ids}

        primary_sizes = self._primary_sizes(num_clients)
        total_samples = max(1, int(self.max_iterations))
        pilot_per_stratum = max(1, int(getattr(self.args, 'shapley_pilot_samples', 1)))
        size_observations = {size: [] for size in primary_sizes}

        used_samples = 0
        for size in primary_sizes:
            if used_samples >= total_samples:
                break
            subset_ids = self._random_coalition_of_size(client_ids, size)
            cc_value = self._evaluate_complementary_coalition(
                subset_ids, client_ids, client_models, id_to_idx,
                client_data_sizes, val_data_loader, v_empty, v_full
            )
            self._record_complementary_sample(
                subset_ids, client_ids, cc_value, stratum_sums, stratum_counts
            )
            size_observations[size].append(cc_value)
            used_samples += 1
            if pilot_per_stratum > 1:
                extra_pilot = min(pilot_per_stratum - 1, total_samples - used_samples)
                for _ in range(extra_pilot):
                    if used_samples >= total_samples:
                        break
                    subset_ids = self._random_coalition_of_size(client_ids, size)
                    cc_value = self._evaluate_complementary_coalition(
                        subset_ids, client_ids, client_models, id_to_idx,
                        client_data_sizes, val_data_loader, v_empty, v_full
                    )
                    self._record_complementary_sample(
                        subset_ids, client_ids, cc_value, stratum_sums, stratum_counts
                    )
                    size_observations[size].append(cc_value)
                    used_samples += 1

        # The |S| = n stratum is deterministic once v(full) and v(empty) are known.
        full_cc_value = float(v_full - v_empty) if np.isfinite(v_full - v_empty) else 0.0
        self._record_complementary_sample(
            client_ids, client_ids, full_cc_value, stratum_sums, stratum_counts
        )

        extra_budget = max(total_samples - used_samples, 0)
        allocation = getattr(self.args, 'shapley_allocation', 'uniform')
        if allocation == 'neyman':
            observed_std = {}
            for size, values in size_observations.items():
                arr = np.asarray(values, dtype=np.float64)
                if arr.size >= 2:
                    observed_std[size] = float(arr.std(ddof=1))
                elif arr.size == 1:
                    observed_std[size] = max(abs(float(arr[0])), 1e-8)
                else:
                    observed_std[size] = 1.0
            extra_per_size = self._allocate_extra_by_neyman(primary_sizes, observed_std, extra_budget)
        else:
            extra_per_size = {size: 0 for size in primary_sizes}
            if extra_budget > 0 and len(primary_sizes) > 0:
                order = list(self.rng.permutation(np.asarray(primary_sizes, dtype=np.int64)).tolist())
                for step in range(extra_budget):
                    extra_per_size[order[step % len(order)]] += 1

        for size in primary_sizes:
            for _ in range(extra_per_size[size]):
                subset_ids = self._random_coalition_of_size(client_ids, size)
                cc_value = self._evaluate_complementary_coalition(
                    subset_ids, client_ids, client_models, id_to_idx,
                    client_data_sizes, val_data_loader, v_empty, v_full
                )
                self._record_complementary_sample(
                    subset_ids, client_ids, cc_value, stratum_sums, stratum_counts
                )

        observed_sizes = sorted(
            {size for size in primary_sizes if size > 0 and size < num_clients}
            | {num_clients - size for size in primary_sizes if 0 < num_clients - size < num_clients}
            | {num_clients}
        )

        result = []
        for cid in client_ids:
            per_size_means = []
            for coalition_size in observed_sizes:
                count = int(stratum_counts[cid][coalition_size])
                if count <= 0:
                    continue
                else:
                    per_size_means.append(float(stratum_sums[cid][coalition_size] / count))
            value = float(np.mean(per_size_means)) if per_size_means else 0.0
            if not np.isfinite(value):
                value = 0.0
            result.append(value)

        return result

    def compute(self, previous_model, client_models, current_global_model,
                val_data_loader, client_ids=None, client_data_sizes=None):
        estimator = getattr(self.args, 'shapley_estimator', 'permutation')
        start_time = time.time()

        if estimator == 'complementary':
            result = self._compute_complementary(
                previous_model, client_models, current_global_model,
                val_data_loader, client_ids, client_data_sizes
            )
        else:
            result = self._compute_permutation(
                previous_model, client_models, current_global_model,
                val_data_loader, client_ids, client_data_sizes
            )

        logger.info(
            "Shapley computation finished by %s estimator in %.2fs, range=[%.6f, %.6f]",
            estimator,
            time.time() - start_time,
            min(result) if result else 0.0,
            max(result) if result else 0.0,
        )
        return result

    def compute_with_history(self, previous_model, client_models, current_global_model,
                             val_data_loader, client_ids=None, client_data_sizes=None):
        current_shapley = self.compute(
            previous_model, client_models, current_global_model,
            val_data_loader, client_ids, client_data_sizes
        )

        for client_id, current_val in zip(client_ids, current_shapley):
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            self.client_history[client_id].append(current_val)

        return current_shapley
