# 联邦学习客户端选择对比实验

## 项目简介

本项目实现了三种联邦学习客户端选择方法的对比实验：
1. **Random Selection（随机选择）** - 基线方法
2. **Power of Choice (PoC)** - 两阶段选择方法
3. **Dual Scheduling（双重调度）** - Shapley值 + 能量感知 + 李雅普诺夫优化 + 差分隐私

## 项目结构

```
FLSV/
├── src/
│   ├── federated_main.py          # 主程序
│   ├── options.py                 # 命令行参数
│   ├── models.py                  # 神经网络模型
│   ├── update.py                  # 本地训练
│   ├── utils.py                   # 工具函数
│   ├── sampling.py                # Non-IID数据采样
│   ├── selection.py               # 客户端选择方法
│   ├── shapley.py                 # Shapley值计算
│   ├── energy.py                  # 能量模型
│   ├── lyapunov_optimizer.py      # 李雅普诺夫优化器
│   └── plot_smoothed_comparison.py # 对比图绘制
├── run_poc_vs_triple.bat          # 对比实验批处理
└── README.md
```

## 快速开始

### 运行对比实验

```bash
run_poc_vs_triple.bat
```

该脚本依次运行三个实验：
1. Random Selection
2. Power of Choice (d=30)
3. Dual Scheduling

### 实验参数

- 数据集: CIFAR-10
- 客户端总数: 100
- 每轮选择: 10
- 训练轮次: 100
- 本地训练: 5 epochs
- 批大小: 32
- 学习率: 0.01
- Non-IID: Dirichlet α=0.1

### 查看结果

结果保存在 `save/objects/`：
- `compare_random.pkl`
- `compare_poc.pkl`
- `compare_dual.pkl`

绘制对比图：
```bash
cd src
python plot_smoothed_comparison.py --pkl_dir ../save/objects
```

## 三种方法说明

### 1. Random Selection
每轮随机选择K个客户端，作为基线。

### 2. Power of Choice (PoC)
两阶段选择：
- 随机采样 d=30 个候选
- 选择损失最大的 K=10 个

### 3. Dual Scheduling
综合两个维度的客户端选择方法：

**Shapley值** (权重0.7)：评估客户端对模型的边际贡献
**能量感知** (权重0.3)：考虑客户端剩余能量（初始800，阈值80）
**李雅普诺夫优化**：动态调整Shapley和能量权重，平衡性能与资源约束
**差分隐私（CDP）**：梯度裁剪(1.0) + 噪声添加(0.1)，保护训练数据隐私

## 差分隐私

中心化差分隐私（CDP）：
- 梯度裁剪: max_norm=1.0
- 噪声添加: σ=0.1

## 依赖环境

```
Python 3.6+
PyTorch
torchvision
numpy
matplotlib
```
