#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]

        if isinstance(image, torch.Tensor):
            image_tensor = image.clone().detach()
        else:
            # 假设image是numpy数组或类似结构
            image_tensor = torch.tensor(image, dtype=torch.float32)

        # 处理标签：如果是张量，克隆它；如果是整数，转换为张量
        if isinstance(label, torch.Tensor):
            label_tensor = label.clone().detach()
        else:
            # 标签通常是整数，转换为LongTensor
            label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, device=None):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))

        # 确保device被正确设置
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 调试信息
        # if args.verbose:
        #     print(f"LocalUpdate设备: {self.device}")

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # 新增：记录训练统计
        self.total_samples_trained = 0
        self.epochs_completed = 0

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # 先打乱索引，避免Non-IID下按类别顺序切分导致验证/测试集类别失衡
        import numpy as np
        idxs = list(idxs)
        np.random.shuffle(idxs)

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)):int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=self.args.local_bs, shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_bs, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # 重置计数器
        self.total_samples_trained = 0
        self.epochs_completed = 0

        # 确保模型在正确的设备上
        model.to(self.device)

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay
                                        )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        for iter in range(self.args.local_ep):
            self.epochs_completed += 1
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # 确保数据在正确的设备上
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()

                optimizer.step()

                # 记录实际使用的样本数
                batch_size = images.size(0)
                self.total_samples_trained += batch_size

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        '| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} | Samples: {}'.format(
                            global_round, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                100. * batch_idx / len(self.trainloader),
                            loss.item(),
                            self.total_samples_trained))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 返回模型前，确保它在CPU上（为了序列化和传递）
        model.to('cpu')

        # 返回实际训练数据量作为第三个返回值
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.total_samples_trained

    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'total_samples': self.total_samples_trained,
            'local_epochs': self.epochs_completed,
            'assigned_samples': len(self.trainloader.dataset),  # 分配的数据量
            'utilization_ratio': self.total_samples_trained / max(len(self.trainloader.dataset), 1)
        }

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.to(self.device)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, loss

class LocalUpdateFedProx(LocalUpdate):
    """
    FedProx 本地训练 (Li et al., 2020)
    在损失函数中加入近端项: μ/2 * ||w - w_global||²
    防止 Non-IID 下本地更新过度偏离全局模型
    """

    def __init__(self, args, dataset, idxs, logger, device=None, mu=0.01):
        super().__init__(args, dataset, idxs, logger, device)
        self.mu = mu

    def update_weights(self, model, global_round):
        self.total_samples_trained = 0
        self.epochs_completed = 0

        model.to(self.device)
        model.train()
        epoch_loss = []

        # 保存全局模型参数作为近端项参考，不参与梯度更新
        global_params = [p.clone().detach().to(self.device) for p in model.parameters()]

        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        for iter in range(self.args.local_ep):
            self.epochs_completed += 1
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)

                # 近端项: μ/2 * Σ||w_i - w_global||²
                proximal_term = sum(
                    torch.norm(p - gp) ** 2
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss += (self.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

                self.total_samples_trained += images.size(0)

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        model.to('cpu')
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), self.total_samples_trained


def test_inference(args, model, test_dataset, device=None):
    """ Returns the test accuracy and loss.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')

    model.to(device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, loss