#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def _noniid_dirichlet(dataset, num_users, alpha=0.5, seed=42, dataset_name='dataset'):
    """
    使用Dirichlet分布从数据集中采样非独立同分布客户端数据（通用实现）

    Args:
        dataset: PyTorch数据集
        num_users: 客户端数量
        alpha: Dirichlet分布参数，控制非独立同分布程度
        seed: 随机种子
        dataset_name: 数据集名称（用于打印）
    Returns:
        dict: 客户端ID -> 数据索引集合
    """
    rng = np.random.RandomState(seed)

    num_classes = 10
    num_samples = len(dataset)

    # 获取标签
    if hasattr(dataset, 'targets'):
        labels = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy() if isinstance(dataset.train_labels, torch.Tensor) else np.array(dataset.train_labels)
    else:
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        labels = np.array(labels)

    # 按类别组织数据
    class_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    # 为每个类别生成Dirichlet分布
    client_proportions = rng.dirichlet(
        alpha=[alpha] * num_users,
        size=num_classes
    )  # shape: (num_classes, num_users)

    dict_users = {i: [] for i in range(num_users)}

    for class_id in range(num_classes):
        indices = class_indices[class_id]
        class_size = len(indices)

        proportions = client_proportions[class_id]
        client_counts = (proportions * class_size).astype(int)

        total_allocated = sum(client_counts)
        remaining = class_size - total_allocated

        if remaining > 0:
            sorted_clients = np.argsort(proportions)[::-1]
            for i in range(remaining):
                client_counts[sorted_clients[i]] += 1

        start_idx = 0
        for client_id in range(num_users):
            count = client_counts[client_id]
            if count > 0:
                end_idx = start_idx + count
                dict_users[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx

    for client_id in range(num_users):
        indices = dict_users[client_id]
        rng.shuffle(indices)
        dict_users[client_id] = set(indices)

    # 打印统计信息
    print(f"{dataset_name} Dirichlet Non-IID Distribution (alpha={alpha}):")
    print(f"{'Client':<10} {'Samples':<10} {'Class Distribution':<60}")
    print("-" * 85)

    for client_id in range(min(5, num_users)):
        indices = list(dict_users[client_id])
        if len(indices) > 0:
            client_labels = [labels[i] for i in indices]
            label_counts = np.bincount(client_labels, minlength=num_classes)
            total = len(indices)
            distribution = [f"{c}:{label_counts[c]}({label_counts[c] / total * 100:.1f}%)"
                            for c in range(num_classes) if label_counts[c] > 0]
            print(f"{client_id:<10} {len(indices):<10} {', '.join(distribution[:5])}...")

    total_samples = sum(len(indices) for indices in dict_users.values())
    print(f"\nTotal samples allocated: {total_samples}/{num_samples}")

    # 计算non-iid程度
    client_distributions = []
    for client_id in range(num_users):
        indices = list(dict_users[client_id])
        if len(indices) > 0:
            client_labels = [labels[i] for i in indices]
            label_counts = np.bincount(client_labels, minlength=num_classes)
            distribution = label_counts / len(indices)
            client_distributions.append(distribution)

    if len(client_distributions) > 0:
        client_distributions = np.array(client_distributions)
        uniform = np.ones(num_classes) / num_classes
        avg_kl = np.mean([kl_divergence(dist, uniform) for dist in client_distributions])
        print(f"Average KL divergence from uniform: {avg_kl:.4f}")

    # 打印样本分布统计
    samples_per_client = [len(dict_users[i]) for i in range(num_users)]
    if samples_per_client:
        print(f"Samples per client: min={min(samples_per_client)}, max={max(samples_per_client)}, "
              f"std={np.std(samples_per_client):.1f}")
        insufficient_clients = sum(1 for s in samples_per_client if s < 100)
        if insufficient_clients > 0:
            print(f"Warning: {insufficient_clients}/{num_users} clients have < 100 samples")

    return dict_users


def mnist_noniid_dirichlet(dataset, num_users, alpha=0.5, seed=42):
    """Sample non-I.I.D client data from MNIST using Dirichlet distribution"""
    return _noniid_dirichlet(dataset, num_users, alpha, seed, dataset_name='MNIST')


def cifar_noniid_dirichlet(dataset, num_users, alpha=0.5, seed=42):
    """Sample non-I.I.D client data from CIFAR10 using Dirichlet distribution"""
    return _noniid_dirichlet(dataset, num_users, alpha, seed, dataset_name='CIFAR10')


def kl_divergence(p, q):
    """Calculate KL divergence between two distributions"""
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# 为了兼容性，保留原来的cifar_noniid函数
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    # Convert to sets for consistency
    for i in range(num_users):
        dict_users[i] = set(dict_users[i])

    return dict_users

