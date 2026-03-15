# 论文大纲：基于Shapley值和能量感知的联邦学习双重调度机制
## IEEE 期刊论文标准版（10-12页）

## 基本信息
- **标题（英文）**：Shapley Value and Energy-Aware Dual Scheduling for Federated Learning with Differential Privacy
- **标题（中文）**：基于Shapley值和能量感知的联邦学习双重调度机制
- **目标期刊**：IEEE Journal on Selected Areas in Communications (JSAC) / IEEE Transactions on Mobile Computing (TMC)
- **参考论文**：Gradient and Channel Aware Dynamic Scheduling for Over-the-Air Computation in Federated Edge Learning Systems (IEEE JSAC 2023)

## 核心创新点
1. **Shapley值评估客户端贡献** - 替代梯度范数，更准确量化边际价值，采用指数平滑（α=0.5）实现动态适应
2. **能量可持续性建模** - 累积剩余能量而非瞬时消耗，长期优化资源分配
3. **李雅普诺夫动态权重调整** - 自适应平衡Shapley和能量两个目标，无需离线参数调优
4. **中心化差分隐私保护** - 梯度裁剪+高斯噪声，提供形式化隐私保证
5. **Non-IID数据场景验证** - Dirichlet(α=0.1)极端异构数据分布

## 与参考论文的核心区别

| 维度 | 参考论文 (JSAC 2023) | 本文方法 |
|------|---------------------|---------|
| **应用场景** | Over-the-Air Computation | 传统点对点通信 |
| **聚合方式** | 物理层模拟聚合 | 服务器端数字聚合（FedAvg） |
| **贡献评估** | 梯度范数 ‖g‖² (DSI) | Shapley值（边际贡献） |
| **信道建模** | 直接作为CSI指标 | 通过能量消耗间接影响 |
| **能量建模** | 瞬时能量消耗 | 累积剩余能量+阈值约束 |
| **优化框架** | 李雅普诺夫漂移最小化 | 李雅普诺夫动态权重调整 |
| **隐私保护** | 未涉及 | CDP差分隐私 (ε,δ)-DP |
| **数据分布** | i.i.d. 和 non-i.i.d. | 极端 Non-IID (α=0.1) |
| **残差机制** | 累积未选中设备梯度 | 无（传统点对点通信） |

---

## 论文完整结构

### Abstract (摘要) - 0.3页
**结构：背景 → 问题 → 方法 → 结果**

Federated learning (FL) enables collaborative model training across distributed edge devices without sharing raw data. However, existing client selection mechanisms face challenges in heterogeneous environments: non-IID data distribution, limited energy resources, and insufficient quantification of individual client contributions. While recent works explore gradient-based and channel-aware scheduling for over-the-air computation, they focus on instantaneous metrics and short-term optimization, overlooking long-term sustainability and accurate marginal contribution assessment.

This paper proposes a Shapley value and energy-aware dual scheduling mechanism for federated learning with differential privacy guarantees. We employ Shapley values to quantify each client's marginal contribution using a fast GTG-Shapley approximation with exponential smoothing (α=0.5) for dynamic adaptation in non-IID scenarios. Energy consumption is modeled through wireless channel conditions (E = σ²/|h|²) with cumulative residual energy tracking. A Lyapunov optimization framework dynamically adjusts weights between Shapley values and energy scores. Centralized differential privacy (CDP) is integrated through gradient clipping and Gaussian noise addition.

Extensive simulations on CIFAR-10 with 100 clients demonstrate that our mechanism achieves [X%] higher test accuracy and [Y%] faster convergence compared to random selection and Power of Choice baselines under highly non-IID data distribution (Dirichlet α=0.1), while maintaining energy efficiency and (ε,δ)-DP guarantees.

**Index Terms**: Federated learning, client selection, Shapley value, energy-aware scheduling, differential privacy, non-IID data, Lyapunov optimization

---

### I. INTRODUCTION - 2页

#### A. Background and Motivation (0.5页)
- 边缘计算和IoT设备的快速增长
- 传统集中式学习的局限：通信成本、隐私风险、计算瓶颈
- 联邦学习的优势：数据本地化、隐私保护、分布式计算
- 实际部署面临的挑战：
  - 客户端异构性（数据分布、计算能力、网络条件、能量资源）
  - 能量约束（电池供电设备的长期可持续性）
  - 通信效率（大规模客户端的选择问题）
  - 隐私保护（梯度泄露风险）

#### B. Limitations of Existing Approaches (0.7页)
1. **Random Selection (FedAvg)**
   - 忽视客户端异构性
   - 可能选择低质量数据或不良条件的客户端

2. **Power of Choice (PoC)**
   - 基于本地损失值选择客户端
   - 局限：高损失可能仅反映数据分布偏差而非有价值信息
   - 缺乏时间平滑处理噪声损失估计

3. **Gradient-Based Methods**
   - 使用梯度范数作为数据重要性指标
   - 局限：梯度范数不能准确反映边际贡献
   - 只考虑短期贡献，忽视长期价值

4. **Channel-Aware Scheduling (参考论文)**
   - 针对空中计算（AirComp）场景设计
   - 关注瞬时能量消耗而非长期可持续性
   - 不适用于传统点对点通信系统

5. **Shapley Value-Based Methods**
   - 现有工作未集成能量感知或动态优化框架
   - 缺乏隐私保护机制

#### C. Our Contributions (0.5页)
本文提出基于Shapley值和能量感知的双重调度机制，主要贡献包括：

1. **Shapley值贡献评估**：采用GTG-Shapley近似算法量化客户端边际贡献，使用指数平滑（α=0.5）实现动态适应，特别适用于极端Non-IID场景

2. **能量可持续性建模**：通过无线信道条件建模能量消耗（E = σ²/|h|²），跟踪累积剩余能量，实现长期可持续性优化

3. **李雅普诺夫动态优化**：构建李雅普诺夫漂移加惩罚优化问题，动态调整Shapley值和能量得分的权重，无需离线参数调优

4. **中心化差分隐私**：集成梯度裁剪和高斯噪声机制，提供形式化(ε,δ)-DP隐私保证

5. **全面实验验证**：在CIFAR-10数据集上进行100客户端实验，验证在极端Non-IID（α=0.1）场景下优于随机选择和PoC基线

#### D. Paper Organization (0.3页)
论文组织如下：第II节综述相关工作；第III节介绍系统模型；第IV节建立优化问题；第V节详述双重调度机制；第VI节提供理论分析；第VII节展示仿真结果；第VIII节总结全文。

---

### II. RELATED WORK - 1.5-2页

#### A. Federated Learning and Client Selection (0.4页)
- **FedAvg及其变体**
  - McMahan et al. (2017): 原始FedAvg算法
  - FedProx, FedNova: 处理系统异构性
  - FedOpt: 自适应优化器

- **客户端选择策略**
  - 随机选择 vs. 偏向选择
  - 通信效率优化
  - 公平性考虑

#### B. Contribution-Based Client Selection (0.4页)
- **梯度范数方法**
  - 参考论文：使用‖g‖²作为数据状态指标（DSI）
  - 局限：不能准确反映边际贡献

- **损失值方法**
  - Power of Choice: 基于本地损失选择
  - 局限：高损失可能仅反映分布偏差

- **Shapley值方法**
  - Wang et al. (2019): 联邦学习中的贡献度量
  - Song et al. (2019): 利润分配
  - 本文区别：集成能量感知和动态优化

#### C. Resource-Aware Federated Learning (0.4页)
- **能量约束联邦学习**
  - 电池供电设备的能量管理
  - 能量收集场景

- **信道感知调度**
  - 参考论文：梯度和信道感知动态调度
  - Over-the-Air Computation
  - 本文区别：传统点对点通信，累积能量建模

#### D. Privacy-Preserving Federated Learning (0.3-0.4页)
- **差分隐私**
  - Local DP (LDP): 客户端添加噪声
  - Centralized DP (CDP): 服务器端添加噪声
  - 本文采用CDP：更好的隐私-效用权衡

- **安全聚合**
  - 同态加密
  - 安全多方计算
  - 与DP的互补性

---

### III. SYSTEM MODEL - 2-2.5页

#### A. Federated Learning Framework (0.5页)
**网络架构**
- N个边缘设备（clients）
- 1个中心服务器（parameter server）
- 每轮选择K个客户端（K << N）

**训练协议**
1. 服务器广播全局模型 w_t
2. 选中客户端本地训练 E 个epoch
3. 客户端上传模型更新 Δw_n
4. 服务器聚合更新全局模型

**FedAvg聚合规则**
```
w_{t+1} = w_t + (1/K) Σ_{n∈S_t} Δw_n
```

**优化目标**
```
min F(w) = Σ_{n=1}^N (p_n/N) F_n(w)
其中 F_n(w) = E_{(x,y)~D_n}[ℓ(w; x, y)]
```

#### B. Non-IID Data Distribution (0.5页)
**Dirichlet分布建模**
- 每个客户端的类别分布：p_n ~ Dir(α)
- α控制异构程度：
  - α → ∞: i.i.d.
  - α → 0: 极端non-IID
  - 本文实验：α = 0.1（极端异构）

**异构性量化**
- 变异系数（Coefficient of Variation）
- 类别分布熵
- 对收敛的影响

#### C. Wireless Channel and Energy Model (0.7页)
**信道模型**
- Rayleigh衰落：h_n ~ CN(0, 1)
- 信道增益：|h_n|²

**能量消耗模型**
```
E_n(t) = σ² / |h_n(t)|²
```
- σ²: 噪声功率
- 信道条件差 → 能量消耗高

**剩余能量跟踪**
```
E_remain,n(t+1) = E_remain,n(t) - a_n(t) × E_n(t)
```
- E_remain,n(0) = E_init（初始能量）
- a_n(t) ∈ {0,1}（选择指示变量）

**能量约束**
```
E_remain,n(t) ≥ E_threshold
```
- 低于阈值的客户端不可选

#### D. Differential Privacy Model (0.5-0.7页)
**梯度裁剪**
```
Δw_n^clip = Δw_n / max(1, ‖Δw_n‖₂ / C)
```
- C: 裁剪阈值

**高斯机制**
```
w_{t+1} = w_t + (1/K) Σ_{n∈S_t} Δw_n^clip + N(0, σ_DP² C² I / K²)
```
- σ_DP: 噪声标准差

**(ε, δ)-差分隐私**
- 隐私预算：ε
- 失败概率：δ
- 组合定理：T轮训练的总隐私预算

**隐私-效用权衡**
- 噪声增加 → 隐私增强 → 效用下降
- 参数选择：平衡隐私和准确率

---

### IV. PROBLEM FORMULATION - 1-1.5页

#### A. Optimization Objective (0.3页)
**多目标优化问题**
```
(P1) min  F(w_T)                    (最小化训练损失)
     w,S

     s.t. E_remain,n(t) ≥ E_threshold, ∀n,t
          Σ_{n=1}^N a_n(t) = K, ∀t
          a_n(t) ∈ {0,1}, ∀n,t
          (ε,δ)-DP constraint
```

**目标冲突**
- 最大化模型准确率 ↔ 最小化能量消耗
- 选择高贡献客户端 ↔ 保护低能量客户端
- 隐私保护 ↔ 模型效用

#### B. Problem Transformation (0.4页)
**引入Shapley值**
- 客户端n的Shapley值：φ_n(t)
- 量化边际贡献：v(S∪{n}) - v(S)

**能量得分**
```
s_energy,n(t) = E_remain,n(t) / E_init
```

**综合得分函数**
```
Score_n(t) = w_shapley(t) × φ_n(t) + w_energy(t) × s_energy,n(t)
```

**转化后的问题**
```
(P2) max  Σ_{n∈S_t} Score_n(t)
     S_t

     s.t. |S_t| = K
          E_remain,n(t) ≥ E_threshold, ∀n∈S_t
```

#### C. Lyapunov Optimization Framework (0.5-0.7页)
**虚拟队列定义**
```
Q_n(t+1) = max{Q_n(t) + E_target - E_remain,n(t), 0}
```
- E_target: 目标能量水平
- Q_n: 能量虚拟队列

**李雅普诺夫函数**
```
L(Q(t)) = (1/2) Σ_{n=1}^N Q_n²(t)
```

**漂移加惩罚**
```
Δ(Q(t)) = E[L(Q(t+1)) - L(Q(t)) | Q(t)]

min: Δ(Q(t)) + V × (-Utility(t))
```
- V: 权衡参数（控制准确率 vs. 能量）
- Utility(t): 当前轮次的模型改进

**动态权重更新**
```
w_shapley(t) = V / (V + ‖Q(t)‖)
w_energy(t) = ‖Q(t)‖ / (V + ‖Q(t)‖)
```
- 能量充足时：w_shapley ↑（关注准确率）
- 能量紧张时：w_energy ↑（保护能量）

---
