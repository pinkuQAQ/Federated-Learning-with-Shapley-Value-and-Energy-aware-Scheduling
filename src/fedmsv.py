# -*- coding: utf-8 -*-

"""Fed-MSV client sampling and paper-reproduction helpers."""

import copy
import itertools
import math
import time
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _unique_clients(client_ids: Iterable[int], num_clients: int) -> List[int]:
    result = []
    seen = set()
    for client_id in client_ids:
        client_id = int(client_id)
        if 0 <= client_id < num_clients and client_id not in seen:
            result.append(client_id)
            seen.add(client_id)
    return result


def aggregate_client_models(client_models: Mapping[int, Dict[str, torch.Tensor]],
                            coalition: Sequence[int],
                            client_data_sizes: Mapping[int, float]) -> Dict[str, torch.Tensor]:
    """Aggregate one coalition exactly as data-size-weighted FedAvg."""
    coalition = [int(client_id) for client_id in coalition]
    if not coalition:
        raise ValueError("coalition cannot be empty")

    sizes = np.asarray(
        [max(float(client_data_sizes.get(client_id, 0.0)), 0.0) for client_id in coalition],
        dtype=np.float64,
    )
    if not np.isfinite(sizes).all() or float(sizes.sum()) <= 0.0:
        sizes = np.ones(len(coalition), dtype=np.float64)
    weights = sizes / sizes.sum()

    first_state = client_models[coalition[0]]
    aggregated = {}
    for key, first_tensor in first_state.items():
        if not isinstance(first_tensor, torch.Tensor):
            aggregated[key] = copy.deepcopy(first_tensor)
            continue

        first_tensor = first_tensor.detach().cpu()
        if not torch.is_floating_point(first_tensor):
            aggregated[key] = first_tensor.clone()
            continue

        value = torch.zeros_like(first_tensor)
        for weight, client_id in zip(weights, coalition):
            tensor = client_models[client_id][key].detach().cpu().to(dtype=first_tensor.dtype)
            value.add_(tensor, alpha=float(weight))
        aggregated[key] = value
    return aggregated


class ModelAccuracyUtility:
    """Reusable model-accuracy evaluator for Fed-MSV coalition utilities."""

    def __init__(self, model, data_loader, device):
        self.model = copy.deepcopy(model).to(device)
        self.data_loader = data_loader
        self.device = device
        self.calls = 0

    def __call__(self, state_dict: Dict[str, torch.Tensor]) -> float:
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in self.data_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(images)
                if not torch.isfinite(outputs).all():
                    return float("-inf")
                predictions = outputs.argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
        self.calls += 1
        return float(correct) / float(total) if total else 0.0


class FedMSVSelector:
    """Fed-MSV Algorithm 2 with guided permutations and two-level truncation.

    The paper's pseudocode reverses the round-truncation condition. This
    implementation follows the accompanying text and skips MSV evaluation when
    the full-round accuracy change is at most ``epsilon_a``.
    """

    def __init__(self, num_clients: int, sample_size: int,
                 guided_prefix: int = 4,
                 epsilon_a: float = 0.01,
                 epsilon_b: float = 0.01,
                 epsilon_c: float = 0.1,
                 max_permutations: int = 0,
                 seed: int = 42):
        self.num_clients = int(num_clients)
        self.sample_size = int(sample_size)
        self.guided_prefix = max(int(guided_prefix), 0)
        self.epsilon_a = max(float(epsilon_a), 0.0)
        self.epsilon_b = max(float(epsilon_b), 0.0)
        self.epsilon_c = float(epsilon_c)
        self.max_permutations = max(int(max_permutations), 0)
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not 0.0 < self.epsilon_c < 1.0:
            raise ValueError("epsilon_c must be in (0, 1)")

        self.rng = np.random.RandomState(seed)
        self.cumulative_msv = np.zeros(self.num_clients, dtype=np.float64)
        self.raw_sampling_weights = np.full(
            self.num_clients, 1.0 / self.num_clients, dtype=np.float64
        )
        self.participation_counts = np.zeros(self.num_clients, dtype=np.int64)
        self.history = []

    def select(self, available_clients: Sequence[int] = None) -> List[int]:
        if available_clients is None:
            remaining = list(range(self.num_clients))
        else:
            remaining = _unique_clients(available_clients, self.num_clients)
        target = min(self.sample_size, len(remaining))
        selected = []
        for _ in range(target):
            weights = np.asarray(
                [self.raw_sampling_weights[client_id] for client_id in remaining],
                dtype=np.float64,
            )
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
                probabilities = np.full(len(remaining), 1.0 / len(remaining))
            else:
                probabilities = weights / weights.sum()
            position = int(self.rng.choice(len(remaining), p=probabilities))
            selected.append(int(remaining.pop(position)))
        return selected

    def record_selection(self, selected_clients: Sequence[int]) -> None:
        for client_id in _unique_clients(selected_clients, self.num_clients):
            self.participation_counts[client_id] += 1

    def _guided_permutations(self, selected_clients: Sequence[int]) -> List[List[int]]:
        selected = _unique_clients(selected_clients, self.num_clients)
        if not selected:
            return []
        prefix_size = min(self.guided_prefix, len(selected))
        prefixes = list(itertools.permutations(selected, prefix_size))
        if self.max_permutations and len(prefixes) > self.max_permutations:
            chosen = self.rng.choice(len(prefixes), self.max_permutations, replace=False)
            prefixes = [prefixes[int(index)] for index in chosen]

        guided = []
        for prefix in prefixes:
            prefix_set = set(prefix)
            suffix = [client_id for client_id in selected if client_id not in prefix_set]
            self.rng.shuffle(suffix)
            guided.append(list(prefix) + suffix)
        return guided

    def _refresh_sampling_weights(self) -> None:
        base_weight = 1.0 / self.num_clients
        weights = np.full(self.num_clients, base_weight, dtype=np.float64)
        negative = self.cumulative_msv < 0.0
        if np.any(negative):
            exponents = np.clip(-self.cumulative_msv[negative], 0.0, 700.0)
            weights[negative] = base_weight * np.power(1.0 - self.epsilon_c, exponents)
        self.raw_sampling_weights = weights

    def update_from_round(self, selected_clients: Sequence[int],
                          previous_global: Dict[str, torch.Tensor],
                          current_global: Dict[str, torch.Tensor],
                          client_models: Mapping[int, Dict[str, torch.Tensor]],
                          client_data_sizes: Mapping[int, float],
                          utility_fn: Callable[[Dict[str, torch.Tensor]], float],
                          round_index: int = None) -> Dict:
        started = time.time()
        selected = _unique_clients(selected_clients, self.num_clients)
        if not selected:
            return {}
        if any(client_id not in client_models for client_id in selected):
            raise ValueError("client_models is missing a selected client")

        utility_evaluations = 0
        v_empty = float(utility_fn(previous_global))
        utility_evaluations += 1
        v_full = float(utility_fn(current_global))
        utility_evaluations += 1
        if not (np.isfinite(v_empty) and np.isfinite(v_full)):
            round_skipped = True
            round_msv = np.zeros(self.num_clients, dtype=np.float64)
            permutations = []
        elif abs(v_full - v_empty) <= self.epsilon_a:
            round_skipped = True
            round_msv = np.zeros(self.num_clients, dtype=np.float64)
            permutations = []
        else:
            round_skipped = False
            permutations = self._guided_permutations(selected)
            if not permutations:
                permutations = [selected]
            contribution_sums = np.zeros(self.num_clients, dtype=np.float64)
            cache = {}

            for permutation in permutations:
                v_previous = v_empty
                truncated = False
                for position, client_id in enumerate(permutation):
                    if truncated or abs(v_full - v_previous) < self.epsilon_b:
                        v_current = v_previous
                        truncated = True
                    else:
                        coalition = permutation[:position + 1]
                        if len(coalition) == len(selected):
                            v_current = v_full
                        else:
                            cache_key = frozenset(coalition)
                            if cache_key not in cache:
                                coalition_model = aggregate_client_models(
                                    client_models, coalition, client_data_sizes
                                )
                                cache[cache_key] = float(utility_fn(coalition_model))
                                utility_evaluations += 1
                            v_current = cache[cache_key]
                        if not np.isfinite(v_current):
                            v_current = v_previous
                    contribution_sums[client_id] += float(np.sign(v_current - v_previous))
                    v_previous = v_current

            round_msv = contribution_sums / float(len(permutations))
            self.cumulative_msv[selected] += round_msv[selected]
            self._refresh_sampling_weights()

        record = {
            "round": int(round_index) if round_index is not None else len(self.history),
            "selected_clients": list(selected),
            "round_skipped": bool(round_skipped),
            "v_empty": v_empty,
            "v_full": v_full,
            "permutation_count": len(permutations),
            "utility_evaluations": int(utility_evaluations),
            "round_msv": round_msv.copy(),
            "cumulative_msv": self.cumulative_msv.copy(),
            "raw_sampling_weights": self.raw_sampling_weights.copy(),
            "time_s": float(time.time() - started),
        }
        self.history.append(record)
        return record

    def get_state(self) -> Dict:
        normalized = self.raw_sampling_weights / max(float(self.raw_sampling_weights.sum()), 1e-300)
        return {
            "cumulative_msv": self.cumulative_msv.copy(),
            "raw_sampling_weights": self.raw_sampling_weights.copy(),
            "normalized_sampling_weights": normalized,
            "participation_counts": self.participation_counts.copy(),
            "history": list(self.history),
            "guided_prefix": self.guided_prefix,
            "epsilon_a": self.epsilon_a,
            "epsilon_b": self.epsilon_b,
            "epsilon_c": self.epsilon_c,
            "max_permutations": self.max_permutations,
        }


class LabelOverrideDataset(Dataset):
    """Dataset view that replaces labels at selected global indices."""

    def __init__(self, dataset, label_overrides: Mapping[int, int]):
        self.dataset = dataset
        self.label_overrides = {int(key): int(value) for key, value in label_overrides.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        replacement = self.label_overrides.get(int(index))
        if replacement is None:
            return image, label
        if isinstance(label, torch.Tensor):
            label = label.new_tensor(replacement)
        else:
            label = replacement
        return image, label


def choose_low_quality_clients(num_clients: int, fraction: float, seed: int) -> List[int]:
    count = int(round(max(min(float(fraction), 1.0), 0.0) * int(num_clients)))
    if count <= 0:
        return []
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(int(num_clients), count, replace=False).astype(int).tolist())


def build_label_overrides(dataset, user_groups: Mapping[int, Sequence[int]],
                          low_quality_clients: Sequence[int],
                          flip_fraction: float, num_classes: int,
                          seed: int) -> Dict[int, int]:
    rng = np.random.RandomState(seed)
    targets = getattr(dataset, "targets", None)
    overrides = {}
    for client_id in low_quality_clients:
        indices = np.asarray(list(user_groups[int(client_id)]), dtype=np.int64)
        count = int(round(len(indices) * max(min(float(flip_fraction), 1.0), 0.0)))
        if count <= 0:
            continue
        chosen = rng.choice(indices, count, replace=False)
        for index in chosen:
            if targets is not None:
                label = int(targets[int(index)])
            else:
                _, label = dataset[int(index)]
                label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            overrides[int(index)] = int(num_classes) - label - 1
    return overrides


def random_free_rider_state(reference_state: Mapping[str, torch.Tensor],
                            std: float, seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = {}
    for key, tensor in reference_state.items():
        if not isinstance(tensor, torch.Tensor):
            result[key] = copy.deepcopy(tensor)
            continue
        tensor = tensor.detach().cpu()
        if torch.is_floating_point(tensor):
            result[key] = torch.randn(
                tensor.shape, generator=generator, dtype=tensor.dtype
            ) * float(std)
        else:
            result[key] = torch.zeros_like(tensor)
    return result


def add_gaussian_model_noise(state_dict: Mapping[str, torch.Tensor],
                             variance: float, seed: int) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    std = math.sqrt(max(float(variance), 0.0))
    result = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            result[key] = copy.deepcopy(tensor)
            continue
        tensor = tensor.detach().cpu()
        if torch.is_floating_point(tensor):
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            result[key] = tensor + std * noise
        else:
            result[key] = tensor.clone()
    return result
