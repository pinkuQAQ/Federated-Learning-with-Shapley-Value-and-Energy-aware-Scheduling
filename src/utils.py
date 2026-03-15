#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid_dirichlet, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid,cifar_noniid_dirichlet

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'

        # 训练集transform（加上数据增强）
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 重要：数据增强
            transforms.RandomHorizontalFlip(),  # 重要：数据增强
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])

        # 测试集transform（不加增强）
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])

        train_dataset = datasets.CIFAR10(
            data_dir,
            train=True,
            download=True,
            transform=transform_train  # 使用训练transform
        )

        test_dataset = datasets.CIFAR10(
            data_dir,
            train=False,
            download=True,
            transform=transform_test  # 使用测试transform
        )

        # 如果指定了test_size，只使用部分测试集
        if hasattr(args, 'test_size') and args.test_size is not None:
            from torch.utils.data import Subset
            test_rng = np.random.RandomState(9999)
            indices = test_rng.choice(len(test_dataset), args.test_size, replace=False)
            test_dataset = Subset(test_dataset, indices)
            print(f"使用测试集子集: {args.test_size}/{10000} 张图片")

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from CIFAR10
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from CIFAR10
            if args.unequal:
                # Chose unequal splits for every user
                raise NotImplementedError()
            else:
                # 使用Dirichlet分布划分数据
                if hasattr(args, 'dirichlet_alpha'):
                    alpha = args.dirichlet_alpha
                else:
                    alpha = 0.5  # 默认值，中度Non-IID

                if hasattr(args, 'seed'):
                    seed = args.seed
                else:
                    seed = 42

                print(f"使用Dirichlet分布划分CIFAR10数据 (alpha={alpha}, seed={seed})")
                user_groups = cifar_noniid_dirichlet(
                    train_dataset,
                    args.num_users,
                    alpha=alpha,
                    seed=seed
                )

    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from dataset
            if args.dataset == 'mnist':
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # FashionMNIST使用同样的IID划分函数
                user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from dataset
            if args.unequal:
                # Chose unequal splits for every user
                if args.dataset == 'mnist':
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # FashionMNIST使用同样的unequal划分函数
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # 使用Dirichlet分布划分数据
                if hasattr(args, 'dirichlet_alpha'):
                    alpha = args.dirichlet_alpha
                else:
                    alpha = 0.5  # 默认值，中度Non-IID

                if hasattr(args, 'seed'):
                    seed = args.seed
                else:
                    seed = 42

                print(f"使用Dirichlet分布划分{args.dataset.upper()}数据 (alpha={alpha}, seed={seed})")

                if args.dataset == 'mnist':
                    user_groups = mnist_noniid_dirichlet(
                        train_dataset,
                        args.num_users,
                        alpha=alpha,
                        seed=seed
                    )
                else:
                    # FashionMNIST也使用同样的Dirichlet函数
                    user_groups = mnist_noniid_dirichlet(
                        train_dataset,
                        args.num_users,
                        alpha=alpha,
                        seed=seed
                    )

    return train_dataset, test_dataset, user_groups

def average_weights(w, client_data_sizes=None):
    """
    Returns the average of the weights.

    Args:
        w: 客户端模型权重列表
        client_data_sizes: 客户端数据量列表（None时均匀平均）
    """
    if client_data_sizes is None or len(client_data_sizes) != len(w):
        # 均匀平均
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
    else:
        # 加权平均
        total_size = sum(client_data_sizes)
        w_avg = copy.deepcopy(w[0])

        for key in w_avg.keys():
            w_avg[key] = w_avg[key] * client_data_sizes[0]

        for i in range(1, len(w)):
            for key in w_avg.keys():
                w_avg[key] += w[i][key] * client_data_sizes[i]

        for key in w_avg.keys():
            w_avg[key] = torch.div(w_avg[key], total_size)

        return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}')
    print(f'    DataSet   : {args.dataset}\n')


    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Selection of users  : {args.num_selected}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
