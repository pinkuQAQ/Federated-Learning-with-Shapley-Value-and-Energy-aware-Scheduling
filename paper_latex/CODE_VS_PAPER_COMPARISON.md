# 论文公式 vs 代码实现对比分析

## ✅ 完全对应的部分

### 1. 能量消耗模型
**论文公式** (section_system.tex:61):
```
E_n(t) = β · σ²/|h_n(t)|²
```

**代码实现** (energy.py:413):
```python
def compute(self, channel_gains):
    return self.sigma_sq / (channel_gains ** 2 + 1e-10)
```

**状态**: ✅ 对应，但代码中β=1（隐含）
**建议**: 论文中提到β是缩放因子，代码实现时β=1，需要在论文实验部分说明

---

### 2. 信道模型
**论文公式** (section_system.tex:55):
```
h_n(t) ~ CN(0, 1)  (Rayleigh fading)
```

**代码实现** (energy.py:44):
```python
if channel_model == 'rayleigh':
    channel_gains = np.random.rayleigh(scale=1, size=self.num_devices)
```

**状态**: ✅ 完全对应

---

### 3. 剩余能量更新
**论文公式** (section_system.tex:67):
```
E_remain,n(t+1) = E_remain,n(t) - a_n(t)·E_n(t)
```

**代码实现** (energy.py:538):
```python
def update_client_energy(self, selected_clients, energy_consumed):
    for idx, client_id in enumerate(selected_clients):
        self.client_energy[client_id] -= energy_consumed[idx]
```

**状态**: ✅ 完全对应（a_n(t)=1 for selected clients）

---

## ⚠️ 部分不一致的部分

### 4. Lyapunov虚拟队列更新
**论文公式** (section_problem.tex:60):
```
Q_n(t+1) = max{Q_n(t) + (E_target - E_remain,n(t+1)), 0}
```

**代码实现** (lyapunov_optimizer.py:110):
```python
energy_deficit = self.energy_target - current_energy
self.energy_queue = np.maximum(0, self.energy_queue + energy_deficit)
```

**状态**: ⚠️ 基本对应，但代码使用的是 current_energy（当前能量），而论文公式使用 E_remain,n(t+1)（更新后的能量）
**影响**: 逻辑一致，只是时间步的表述差异

---

### 5. Lyapunov权重更新
**论文公式** (section_problem.tex:77):
```
w_shapley(t) = V/(V + ||Q(t)||_1)
w_energy(t) = ||Q(t)||_1/(V + ||Q(t)||_1)
```

**代码实现** (lyapunov_optimizer.py:118-119):
```python
self.shapley_weight += self.learning_rate * self.V * selected_shapley
self.energy_weight -= self.learning_rate * selected_energy_queue
# ... then normalize
```

**状态**: ❌ **不一致！**
**问题**:
- 论文给出的是**闭式解**（closed-form solution）
- 代码使用的是**梯度下降**方法

**这是最大的不一致点！**

---

## ❌ 主要问题：Lyapunov权重计算方法不同

### 论文方法（闭式解）
```python
# 论文中的公式
Q_sum = np.sum(self.energy_queue)  # ||Q(t)||_1
w_shapley = self.V / (self.V + Q_sum)
w_energy = Q_sum / (self.V + Q_sum)
```

### 代码方法（梯度下降）
```python
# 当前代码实现
self.shapley_weight += lr * V * selected_shapley
self.energy_weight -= lr * selected_energy_queue
# normalize to sum=1
```

### 为什么不一致？
1. **论文推导**: 从drift-plus-penalty最小化推导出闭式解
2. **代码实现**: 使用梯度下降迭代优化

### 哪个更好？
- **论文方法**: 理论严谨，直接从Lyapunov优化推导
- **代码方法**: 更灵活，可以加入学习率和边界约束

---

## 建议修改方案

### 方案1: 修改代码以匹配论文（推荐）
```python
def update_weights_closed_form(self, current_energy):
    """使用论文中的闭式解"""
    # 更新虚拟队列
    energy_deficit = self.energy_target - current_energy
    self.energy_queue = np.maximum(0, self.energy_queue + energy_deficit)

    # 计算权重（闭式解）
    Q_sum = np.sum(self.energy_queue)
    self.shapley_weight = self.V / (self.V + Q_sum)
    self.energy_weight = Q_sum / (self.V + Q_sum)

    return (self.shapley_weight, self.energy_weight)
```

### 方案2: 修改论文以匹配代码
在论文中说明：
"We use a gradient-based approach to update the weights dynamically, which provides more flexibility than the closed-form solution."

---

## 其他需要说明的参数

### 1. β参数（能量缩放因子）
- **论文**: E_n(t) = β·σ²/|h|²
- **代码**: β = 1（隐含）
- **建议**: 在实验部分说明 "For simplicity, we set β=1"

### 2. E_target参数
- **论文**: 定义为"desired energy level"，建议 E_target = 0.5·E_init
- **代码**: energy_target=500.0（如果initial_energy=1000，则正好是50%）
- **状态**: ✅ 一致

### 3. V参数（Lyapunov控制参数）
- **论文**: V > 0，权衡性能和稳定性
- **代码**: V=1.0（默认值）
- **建议**: 在实验部分说明V的取值及其影响

---

## 总结

| 组件 | 论文 | 代码 | 状态 |
|------|------|------|------|
| 能量模型 | E=β·σ²/\|h\|² | E=σ²/\|h\|² | ⚠️ β=1 |
| 信道模型 | Rayleigh | Rayleigh | ✅ |
| 能量更新 | E_remain -= E_n | client_energy -= energy | ✅ |
| 虚拟队列 | Q=max{Q+(E_t-E_r),0} | Q=max{Q+deficit,0} | ✅ |
| **权重更新** | **闭式解** | **梯度下降** | ❌ |
| Shapley计算 | GTG-Shapley | GTG-Shapley | ✅ |
| DP机制 | Clip+Noise | Clip+Noise | ✅ |

**关键问题**: Lyapunov权重更新方法不一致，需要统一。
