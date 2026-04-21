# Federated Learning with Shapley Value and Energy-aware Scheduling

## 项目简介

本项目实现了基于 **Shapley值 + 能量感知 + 李雅普诺夫优化** 的联邦学习客户端双重调度方法，并与4种基线方法进行对比实验。

## 方法概览

| 方法 | 说明 |
|------|------|
| **Dual Scheduling（本文方法）** | Shapley值 + 能量感知 + 李雅普诺夫优化 |
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
│   └── plot.py                 # 实验结果可视化
├── run_baseline_comparison.sh      # 主对比实验（5种方法）
├── run_ablation_study.sh           # 消融实验
├── run_sensitivity_dp.sh           # Local DP 噪声敏感性
├── run_sensitivity_V.sh            # Lyapunov V 敏感性
├── run_sensitivity_M.sh            # MC-Shapley 迭代次数敏感性
├── run_noniid_comparison.sh        # CIFAR-10 不同 Dirichlet α 对比
├── run_multidataset_comparison.sh  # MNIST / FMNIST 补充对比
├── run_cifar_multiseed_main.sh     # CIFAR-10 多随机种子主对比
├── run_cifar_multiseed_ablation.sh # CIFAR-10 多随机种子消融
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
sbatch run_baseline_comparison.sh
```

### 运行消融实验

```bash
sbatch run_ablation_study.sh
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

## 当前推荐脚本

- `run_baseline_comparison.sh`：主对比实验
- `run_ablation_study.sh`：单种子消融
- `run_sensitivity_dp.sh`：Local DP 噪声系数扫描
- `run_sensitivity_V.sh`：Lyapunov 参数 $V$ 扫描
- `run_sensitivity_M.sh`：Shapley 采样次数 $M$ 扫描
- `run_noniid_comparison.sh`：CIFAR-10 下不同 $\alpha$ 单种子对比
- `run_multidataset_comparison.sh`：MNIST / FMNIST 补充实验
- `run_cifar_multiseed_main.sh`：CIFAR-10 多种子主实验
- `run_cifar_multiseed_ablation.sh`：CIFAR-10 多种子消融

## 主要特性

- **Shapley值评估**：GTG-Shapley 算法评估客户端对全局模型的边际贡献
- **能量感知调度**：基于无线信道模型（Rayleigh衰落）建模客户端能耗
- **李雅普诺夫优化**：动态平衡模型性能与长期能量约束
- **隐私增强上传**：支持客户端本地差分隐私（LDP）上传机制

## 依赖环境

```
Python 3.8+
torch
torchvision
numpy
matplotlib
tensorboard
tqdm
```
