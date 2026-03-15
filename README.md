# Federated Learning with Shapley Value and Energy-aware Scheduling

## 项目简介

本项目实现了基于 **Shapley值 + 能量感知 + 李雅普诺夫优化** 的联邦学习客户端双重调度方法，并与4种基线方法进行对比实验。

## 方法概览

| 方法 | 说明 |
|------|------|
| **Dual Scheduling（本文方法）** | Shapley值 + 能量感知 + 李雅普诺夫优化 + AES-256-GCM加密 |
| **FedAvg** | 随机客户端选择（基线） |
| **PoC** | Power of Choice，基于本地损失的两阶段选择 |
| **UCB** | UCB1 Bandit客户端选择 |
| **FedProx** | 近端项正则化，缓解Non-IID漂移 |

## 项目结构

```
FLSV/
├── src/
│   ├── federated_main.py       # 主程序
│   ├── options.py              # 命令行参数
│   ├── models.py               # 神经网络模型（CNN / MLP）
│   ├── update.py               # 本地训练（LocalUpdate / LocalUpdateFedProx）
│   ├── utils.py                # 工具函数
│   ├── sampling.py             # Non-IID数据采样（Dirichlet分布）
│   ├── selection.py            # 客户端选择策略
│   ├── shapley.py              # Shapley值计算（GTG-Shapley）
│   ├── energy.py               # 无线信道能量模型
│   ├── lyapunov_optimizer.py   # 李雅普诺夫优化器
│   ├── crypto_utils.py         # AES-256-GCM梯度加密
│   └── plot.py                 # 实验结果可视化
├── run_baseline_comparison.bat  # 主对比实验（5种方法）
├── run_channel_robustness.bat   # 信道鲁棒性实验（σ²=0.5/1.0/3.0）
├── run_vs_fedavg_crypto.bat     # 加密消融实验
├── requirements.txt
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行主对比实验（5种方法）

```bash
run_baseline_comparison.bat
```

### 运行信道鲁棒性实验

```bash
run_channel_robustness.bat
```

### 可视化结果

```bash
cd src
# 修改 plot.py 中的 SAVE_DIR 为对应实验文件夹路径
python plot.py
```

## 实验配置

| 参数 | 值 |
|------|----|
| 数据集 | CIFAR-10 |
| 模型 | CNN |
| 客户端总数 | 100 |
| 每轮选择 | 10 |
| 训练轮次 | 100 |
| 本地训练 epochs | 2 |
| 批大小 | 32 |
| 学习率 | 0.01 |
| 数据分布 | Non-IID（Dirichlet α=0.1） |
| 初始能量 | 500.0 |
| 能量阈值 | 50.0 |
| Lyapunov V | 10.0 |
| 能量预算 | 5.0 |

## 主要特性

- **Shapley值评估**：GTG-Shapley 算法评估客户端对全局模型的边际贡献
- **能量感知调度**：基于无线信道模型（Rayleigh衰落）建模客户端能耗
- **李雅普诺夫优化**：动态平衡模型性能与长期能量约束
- **AES-256-GCM加密**：梯度传输加密，防御诚实但好奇的服务器
- **信道鲁棒性**：支持不同噪声方差（σ²）下的性能评估

## 依赖环境

```
Python 3.8+
torch
torchvision
numpy
matplotlib
pycryptodome
tensorboard
tqdm
```
