import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# 过滤掉Matplotlib的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 设置字体 - 只使用系统通用字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class EnergyConsumptionSimulatorEN:
    """
    Independent Energy Consumption Simulator
    Based on formula: E = σ²/|h|²
    """

    def __init__(self, num_devices=10, sigma_squared=1.0, seed=42):
        self.num_devices = num_devices
        self.sigma_sq = sigma_squared
        self.rng = np.random.RandomState(seed)
        self.devices_positions = None
        self.channel_history = []
        self.energy_history = []

    def setup_devices(self, area_size=1000):
        self.devices_positions = self.rng.rand(self.num_devices, 2) * area_size
        self.server_position = np.array([area_size / 2, area_size / 2])
        self.distances = np.linalg.norm(
            self.devices_positions - self.server_position, axis=1
        )
        return self.devices_positions

    def generate_channel_gains(self, time_slot, channel_model='rayleigh'):
        rng = np.random.RandomState(time_slot * 42)

        if channel_model == 'rayleigh':
            channel_gains = rng.rayleigh(scale=1, size=self.num_devices)
        elif channel_model == 'path_loss':
            ref_distance = 50
            channel_gains = ref_distance / (self.distances + 1e-10)
            channel_gains *= (0.9 + 0.2 * rng.rand(self.num_devices))
        elif channel_model == 'combined':
            d0 = 50
            alpha = 2.7
            pl = (d0 / (self.distances + 1e-10)) ** alpha
            shadowing_std = 8
            shadowing = 10 ** (rng.randn(self.num_devices) * shadowing_std / 20)
            rayleigh = rng.rayleigh(scale=1, size=self.num_devices)
            channel_gains = pl * shadowing * rayleigh
        else:
            raise ValueError(f"Unknown channel model: {channel_model}")

        self.channel_history.append(channel_gains.copy())
        return channel_gains

    def calculate_energy(self, channel_gains):
        epsilon = 1e-10
        energy = self.sigma_sq / (channel_gains ** 2 + epsilon)
        self.energy_history.append(energy.copy())
        return energy

    def simulate_time_slots(self, num_slots=50, channel_model='rayleigh'):
        energies_over_time = []
        channels_over_time = []

        print(f"\n{'=' * 50}")
        print(f"Simulation: {num_slots} time slots, Channel model: {channel_model}")
        print(f"Number of devices: {self.num_devices}, σ² = {self.sigma_sq}")
        print('=' * 50)

        for t in range(num_slots):
            channels = self.generate_channel_gains(t, channel_model)
            energies = self.calculate_energy(channels)
            energies_over_time.append(energies)
            channels_over_time.append(channels)

            if t % 10 == 0:
                avg_energy = np.mean(energies)
                avg_channel = np.mean(channels)
                print(f"Slot {t:3d} | Avg channel: {avg_channel:.3f} | Avg energy: {avg_energy:.3e}")

        return np.array(energies_over_time), np.array(channels_over_time)


def plot_analysis_en(simulator, energies, channels):
    """Plot comprehensive analysis with fixed issues"""
    num_devices = simulator.num_devices
    num_slots = len(energies)

    # 创建更大的画布，避免tight_layout问题
    fig = plt.figure(figsize=(22, 16))

    # 1. Device Positions
    ax1 = plt.subplot(3, 4, 1)
    positions = simulator.devices_positions
    server_pos = simulator.server_position

    ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, alpha=0.7, label='Devices')
    ax1.scatter(server_pos[0], server_pos[1], c='red', s=300, marker='*', label='Server')

    for i, pos in enumerate(positions):
        ax1.plot([pos[0], server_pos[0]], [pos[1], server_pos[1]],
                 'gray', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel('X Coordinate (m)')
    ax1.set_ylabel('Y Coordinate (m)')
    ax1.set_title('Device and Server Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 2. Channel Gain Distribution
    ax2 = plt.subplot(3, 4, 2)
    all_channels = channels.flatten()
    ax2.hist(all_channels, bins=30, density=True, alpha=0.7, color='skyblue')

    if len(all_channels) > 1:
        rayleigh_fit = stats.rayleigh.fit(all_channels)
        x = np.linspace(0, max(all_channels) * 1.1, 100)
        pdf = stats.rayleigh.pdf(x, *rayleigh_fit)
        ax2.plot(x, pdf, 'r-', linewidth=2, label='Rayleigh Fit')

    ax2.set_xlabel('Channel Gain |h|')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Channel Gain Distribution')
    ax2.legend()

    # 3. Energy Distribution
    ax3 = plt.subplot(3, 4, 3)
    all_energies = energies.flatten()
    # 使用自然对数避免下标字体问题
    log_energies = np.log(all_energies + 1e-10)
    ax3.hist(log_energies, bins=30, alpha=0.7, color='salmon')
    ax3.set_xlabel('ln(Energy Consumption)')  # 改为ln而不是log₁₀
    ax3.set_ylabel('Frequency')
    ax3.set_title('Energy Distribution (Log Scale)')

    # 4. Energy vs Channel
    ax4 = plt.subplot(3, 4, 4)
    sample_indices = np.random.choice(len(all_channels), min(1000, len(all_channels)), replace=False)
    sampled_channels = all_channels[sample_indices]
    sampled_energies = all_energies[sample_indices]

    ax4.scatter(sampled_channels, sampled_energies, alpha=0.6, s=10, c='green')

    h_theory = np.linspace(0.1, max(sampled_channels), 100)
    E_theory = simulator.sigma_sq / (h_theory ** 2)
    ax4.plot(h_theory, E_theory, 'r-', linewidth=2, label='Theory: E=σ²/|h|²')

    ax4.set_xlabel('Channel Gain |h|')
    ax4.set_ylabel('Energy Consumption E')
    ax4.set_title('Energy vs Channel Gain')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Channel Over Time (First 5 devices)
    ax5 = plt.subplot(3, 4, 5)
    time_slots = np.arange(num_slots)
    for i in range(min(5, num_devices)):
        ax5.plot(time_slots, channels[:, i], alpha=0.7, label=f'Device {i + 1}')

    ax5.set_xlabel('Time Slot')
    ax5.set_ylabel('Channel Gain |h|')
    ax5.set_title('Channel Gain Over Time (First 5 devices)')
    ax5.legend(loc='upper right', fontsize='small')
    ax5.grid(True, alpha=0.3)

    # 6. Energy Over Time (First 5 devices)
    ax6 = plt.subplot(3, 4, 6)
    for i in range(min(5, num_devices)):
        ax6.plot(time_slots, energies[:, i], alpha=0.7, label=f'Device {i + 1}')

    ax6.set_xlabel('Time Slot')
    ax6.set_ylabel('Energy Consumption E')
    ax6.set_title('Energy Consumption Over Time')
    ax6.set_yscale('log')
    ax6.legend(loc='upper right', fontsize='small')
    ax6.grid(True, alpha=0.3)

    # 7. Energy Heatmap - 使用普通标签避免下标
    ax7 = plt.subplot(3, 4, 7)
    im = ax7.imshow(np.log(energies.T + 1e-10), aspect='auto',
                    cmap='hot_r', interpolation='nearest')
    ax7.set_xlabel('Time Slot')
    ax7.set_ylabel('Device ID')
    ax7.set_title('Energy Consumption Heatmap (log scale)')
    cbar = plt.colorbar(im, ax=ax7)
    cbar.set_label('ln(E)')  # 改为ln

    # 8. Cumulative Energy
    ax8 = plt.subplot(3, 4, 8)
    cumulative_energy = np.cumsum(energies, axis=0)
    for i in range(min(6, num_devices)):
        ax8.plot(time_slots, cumulative_energy[:, i], alpha=0.8, label=f'Device {i + 1}')

    ax8.set_xlabel('Time Slot')
    ax8.set_ylabel('Cumulative Energy')
    ax8.set_title('Cumulative Energy Consumption')
    ax8.set_yscale('log')
    ax8.legend(loc='upper left', fontsize='small')
    ax8.grid(True, alpha=0.3)

    # 9. Boxplot - 修复labels参数问题
    ax9 = plt.subplot(3, 4, 9)
    box_data = []
    labels = []
    for i in range(min(8, num_devices)):
        box_data.append(energies[:, i])
        labels.append(f'D{i + 1}')

    # 检查Matplotlib版本
    import matplotlib
    if matplotlib.__version__ >= '3.9':
        # 新版本使用tick_labels
        bp = ax9.boxplot(box_data, tick_labels=labels, patch_artist=True)
    else:
        # 旧版本使用labels
        bp = ax9.boxplot(box_data, labels=labels, patch_artist=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax9.set_ylabel('Energy Consumption E')
    ax9.set_title('Energy Distribution by Device')
    ax9.set_yscale('log')
    ax9.grid(True, alpha=0.3)

    # 10. Energy vs Distance
    ax10 = plt.subplot(3, 4, 10)
    avg_energy_per_device = np.mean(energies, axis=0)
    scatter10 = ax10.scatter(simulator.distances, avg_energy_per_device,
                             c=avg_energy_per_device, cmap='viridis', s=100, alpha=0.7)

    for i, (dist, energy) in enumerate(zip(simulator.distances[:5], avg_energy_per_device[:5])):
        ax10.annotate(f'D{i + 1}', xy=(dist, energy), xytext=(5, 5),
                      textcoords='offset points', fontsize=8)

    ax10.set_xlabel('Distance from Server (m)')
    ax10.set_ylabel('Avg Energy Consumption')
    ax10.set_title('Energy vs Distance')
    ax10.set_yscale('log')
    plt.colorbar(scatter10, ax=ax10, label='Average Energy')
    ax10.grid(True, alpha=0.3)

    # 11. Statistics Table
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('tight')
    ax11.axis('off')

    stats_text = []
    stats_text.append("ENERGY CONSUMPTION STATISTICS:")
    stats_text.append(f"Total Devices: {num_devices}")
    stats_text.append(f"Total Time Slots: {num_slots}")
    stats_text.append(f"Total Energy: {np.sum(energies):.3e} J")
    stats_text.append(f"Avg Energy per Slot per Device: {np.mean(energies):.3e} J")
    stats_text.append(f"Energy Std Dev: {np.std(energies):.3e} J")
    stats_text.append(f"Max Energy: {np.max(energies):.3e} J")
    stats_text.append(f"Min Energy: {np.min(energies):.3e} J")
    stats_text.append(f"Avg Channel Gain: {np.mean(channels):.3f}")

    ax11.text(0, 0.95, "\n".join(stats_text),
              fontsize=9, family='monospace',
              verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 12. 简化3D View - 避免复杂字符
    ax12 = plt.subplot(3, 4, 12, projection='3d')

    sample_t = np.linspace(0, num_slots - 1, 20).astype(int)
    sample_d = np.linspace(0, num_devices - 1, 10).astype(int)

    T, D = np.meshgrid(sample_t, sample_d)
    H_sample = channels[T, D]
    E_sample = energies[T, D]

    scatter12 = ax12.scatter(T.flatten(), D.flatten(),
                             np.log(E_sample.flatten() + 1e-10),  # 使用ln
                             c=H_sample.flatten(), cmap='plasma',
                             alpha=0.7, s=30)

    ax12.set_xlabel('Time Slot')
    ax12.set_ylabel('Device ID')
    ax12.set_zlabel('ln(Energy)')  # 改为ln
    ax12.set_title('3D View: Energy-Channel-Time')

    # 添加总标题，但不使用tight_layout
    plt.suptitle(f'Federated Learning Energy Consumption Simulation (sigma²={simulator.sigma_sq})',
                 fontsize=14, fontweight='bold', y=0.98)

    # 使用constrained_layout代替tight_layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05,
                        wspace=0.3, hspace=0.4)

    plt.show()

    return fig


def run_simulation_simple():
    """简化版本的模拟，只生成关键图表"""
    print("Federated Learning Energy Consumption Simulator")
    print("=" * 60)

    # 创建模拟器
    simulator = EnergyConsumptionSimulatorEN(num_devices=10, sigma_squared=1.0)
    simulator.setup_devices(area_size=500)

    # 模拟
    energies, channels = simulator.simulate_time_slots(
        num_slots=50, channel_model='rayleigh'
    )

    # 只生成关键图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 能量vs信道
    ax1 = axes[0, 0]
    all_channels = channels.flatten()
    all_energies = energies.flatten()
    sample_idx = np.random.choice(len(all_channels), 500, replace=False)
    ax1.scatter(all_channels[sample_idx], all_energies[sample_idx],
                alpha=0.5, s=10, c='blue')
    h_range = np.linspace(0.1, max(all_channels), 100)
    E_theory = simulator.sigma_sq / (h_range ** 2)
    ax1.plot(h_range, E_theory, 'r-', linewidth=2, label='E=σ²/|h|²')
    ax1.set_xlabel('Channel Gain |h|')
    ax1.set_ylabel('Energy E')
    ax1.set_title('Energy vs Channel Gain')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 信道分布
    ax2 = axes[0, 1]
    ax2.hist(all_channels, bins=30, alpha=0.7, color='skyblue', density=True)
    ax2.set_xlabel('Channel Gain |h|')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Channel Gain Distribution')

    # 3. 能量分布
    ax3 = axes[0, 2]
    ax3.hist(np.log(all_energies + 1e-10), bins=30, alpha=0.7, color='salmon')
    ax3.set_xlabel('ln(Energy)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Energy Distribution (log scale)')

    # 4. 随时间变化的能量（前3个设备）
    ax4 = axes[1, 0]
    time_slots = np.arange(len(energies))
    for i in range(min(3, simulator.num_devices)):
        ax4.plot(time_slots, energies[:, i], alpha=0.8, label=f'Device {i + 1}')
    ax4.set_xlabel('Time Slot')
    ax4.set_ylabel('Energy')
    ax4.set_title('Energy Over Time')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 累计能量
    ax5 = axes[1, 1]
    cumulative = np.cumsum(energies, axis=0)
    for i in range(min(3, simulator.num_devices)):
        ax5.plot(time_slots, cumulative[:, i], alpha=0.8, label=f'Device {i + 1}')
    ax5.set_xlabel('Time Slot')
    ax5.set_ylabel('Cumulative Energy')
    ax5.set_title('Cumulative Energy')
    ax5.set_yscale('log')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 统计摘要
    ax6 = axes[1, 2]
    ax6.axis('off')
    stats_text = [
        "STATISTICS:",
        f"Avg Channel: {np.mean(channels):.3f}",
        f"Avg Energy: {np.mean(energies):.3e}",
        f"σ² = {simulator.sigma_sq}",
        f"Devices: {simulator.num_devices}",
        f"Time Slots: {len(energies)}"
    ]
    ax6.text(0.1, 0.5, "\n".join(stats_text), fontsize=12,
             family='monospace', verticalalignment='center')

    plt.suptitle('Energy Consumption Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('energy_analysis_simple.png', dpi=150, bbox_inches='tight')
    plt.show()

    return simulator, energies, channels


# 最简单的能量计算器
class MinimalEnergyCalculator:
    """
    Minimal energy calculator for integration with your SV scheduler
    """

    def __init__(self, sigma_squared=1.0):
        self.sigma_sq = sigma_squared

    def compute(self, channel_gains):
        """Compute energy: E = σ²/|h|² with upper bound"""
        energy = self.sigma_sq / (channel_gains ** 2 + 1e-10)
        # 限制最大能量消耗，避免极端值
        return np.minimum(energy, 50.0)  # 上限50

    def test_formula(self):
        """Test the energy formula"""
        print("Testing Energy Formula: E = σ²/|h|²")
        print(f"σ² = {self.sigma_sq}")
        print("\n|h|   |   E   |   |h|²×E (should be σ²)")
        print("-" * 40)

        test_h = [0.1, 0.5, 1.0, 2.0, 5.0]
        for h in test_h:
            E = self.compute(np.array([h]))[0]
            check = h ** 2 * E
            print(f"{h:4.1f}   | {E:7.4f} | {check:7.4f}")


class EnergyAwareClientManager:
    """
    能量感知的客户端管理器
    用于联邦学习中的客户端选择，综合考虑 Shapley 值和能量消耗
    """

    def __init__(self, num_clients, sigma_squared=1.0, channel_model='rayleigh',
                 initial_energy=1000.0, energy_threshold=100.0, seed=42,
                 kappa=1e-28, cpu_freq=1e9, cycles_per_sample=20.0):
        """
        初始化能量感知客户端管理器

        Args:
            num_clients: 客户端数量
            sigma_squared: 噪声功率
            channel_model: 信道模型 ('rayleigh', 'path_loss', 'combined')
            initial_energy: 每个客户端的初始能量
            energy_threshold: 能量阈值（低于此值的客户端不参与训练）
            seed: 随机种子
            kappa: CPU有效开关电容系数
            cpu_freq: CPU频率 (Hz)
            cycles_per_sample: 每个训练样本的CPU周期数
        """
        self.num_clients = num_clients
        self.sigma_sq = sigma_squared
        self.channel_model = channel_model
        self.energy_threshold = energy_threshold
        self.seed = seed

        # 计算能量参数
        self.kappa = kappa
        self.cpu_freq = cpu_freq
        self.cycles_per_sample = cycles_per_sample

        # 初始化客户端能量（每个客户端的剩余能量）
        self.client_energy = np.ones(num_clients) * initial_energy
        self.initial_energy = initial_energy

        # 能量历史记录
        self.energy_history = []
        self.consumption_history = []

        # 使用局部随机数生成器
        self.rng = np.random.RandomState(seed)

        # 设置设备位置（用于路径损耗模型）
        area_size = 1000
        self.devices_positions = self.rng.rand(num_clients, 2) * area_size
        self.server_position = np.array([area_size / 2, area_size / 2])
        self.distances = np.linalg.norm(
            self.devices_positions - self.server_position, axis=1
        )

        # 能量计算器
        self.energy_calculator = MinimalEnergyCalculator(sigma_squared)

        print(f"\n[能量管理器] 初始化完成")
        print(f"  客户端数量: {num_clients}")
        print(f"  初始能量: {initial_energy}")
        print(f"  能量阈值: {energy_threshold}")
        print(f"  信道模型: {channel_model}")

    def generate_channel_gains(self, round_num):
        """
        生成当前轮次的信道增益

        Args:
            round_num: 当前训练轮次

        Returns:
            channel_gains: 每个客户端的信道增益数组
        """
        rng = np.random.RandomState(round_num * self.seed)

        if self.channel_model == 'rayleigh':
            channel_gains = rng.rayleigh(scale=1, size=self.num_clients)
        elif self.channel_model == 'path_loss':
            ref_distance = 50
            channel_gains = ref_distance / (self.distances + 1e-10)
            channel_gains *= (0.9 + 0.2 * rng.rand(self.num_clients))
        elif self.channel_model == 'combined':
            d0 = 50
            alpha = 2.7
            pl = (d0 / (self.distances + 1e-10)) ** alpha
            shadowing_std = 8
            shadowing = 10 ** (rng.randn(self.num_clients) * shadowing_std / 20)
            rayleigh = rng.rayleigh(scale=1, size=self.num_clients)
            channel_gains = pl * shadowing * rayleigh
        else:
            raise ValueError(f"Unknown channel model: {self.channel_model}")

        self.channel_gains = channel_gains
        return channel_gains

    def compute_energy_consumption(self, channel_gains, selected_clients=None,
                                    client_data_sizes=None):
        """
        计算能量消耗（传输能量 + 计算能量）

        E_total = E_trans + E_comp
        E_trans = σ²/|h|²
        E_comp  = κ · f² · C · D_i

        Args:
            channel_gains: 信道增益数组
            selected_clients: 选中的客户端索引列表（如果为None，计算所有客户端）
            client_data_sizes: 客户端训练数据量数组（如果为None，不计算计算能量）

        Returns:
            energy_consumption: 能量消耗数组
        """
        # 传输能量
        e_trans = self.energy_calculator.compute(channel_gains)

        # 计算能量: E_comp = κ · f² · C · D
        if client_data_sizes is not None:
            data_sizes = np.array(client_data_sizes, dtype=np.float64)
            e_comp = self.kappa * (self.cpu_freq ** 2) * self.cycles_per_sample * data_sizes
        else:
            e_comp = 0.0

        if selected_clients is not None:
            return e_trans[selected_clients] + (e_comp[selected_clients] if isinstance(e_comp, np.ndarray) else e_comp)

        return e_trans + e_comp

    def update_client_energy(self, selected_clients, energy_consumed):
        """
        更新客户端的剩余能量

        Args:
            selected_clients: 参与训练的客户端索引列表
            energy_consumed: 每个客户端消耗的能量数组
        """
        for idx, client_id in enumerate(selected_clients):
            self.client_energy[client_id] -= energy_consumed[idx]
            # 确保能量不为负
            self.client_energy[client_id] = max(0, self.client_energy[client_id])

        # 记录历史
        self.energy_history.append(self.client_energy.copy())
        self.consumption_history.append(energy_consumed.copy())

    def get_available_clients(self):
        """
        获取能量充足的可用客户端

        Returns:
            available_clients: 能量高于阈值的客户端索引列表
        """
        available = np.where(self.client_energy >= self.energy_threshold)[0]
        return available.tolist()

    def get_energy_scores(self, normalize=True):
        """
        计算能量得分（剩余能量越多，得分越高）

        Args:
            normalize: 是否归一化到 [0, 1]

        Returns:
            energy_scores: 每个客户端的能量得分
        """
        # 能量得分 = 剩余能量 / 初始能量（已经在0-1之间）
        scores = self.client_energy / self.initial_energy

        # 不需要二次归一化，因为已经在[0,1]区间了
        # 二次归一化会过度放大能量差异，导致选择偏向能量而忽略Shapley值

        return scores

    def print_energy_status(self, round_num):
        """打印能量状态"""
        avg_energy = np.mean(self.client_energy)
        min_energy = np.min(self.client_energy)
        max_energy = np.max(self.client_energy)
        available = len(self.get_available_clients())

        print(f"\n[能量状态 - 轮次 {round_num}]")
        print(f"  平均剩余能量: {avg_energy:.2f}")
        print(f"  最小剩余能量: {min_energy:.2f}")
        print(f"  最大剩余能量: {max_energy:.2f}")
        print(f"  可用客户端数: {available}/{self.num_clients}")

    def get_statistics(self):
        """获取能量统计信息"""
        return {
            'current_energy': self.client_energy.copy(),
            'energy_history': self.energy_history.copy() if self.energy_history else [],
            'consumption_history': self.consumption_history.copy() if self.consumption_history else [],
            'available_clients': self.get_available_clients(),
            'energy_scores': self.get_energy_scores()
        }


# 主程序 - 运行最简单的版本
if __name__ == "__main__":
    print("=" * 60)
    print("ENERGY CONSUMPTION SIMULATOR")
    print("=" * 60)

    # 运行最简单的版本
    simulator, energies, channels = run_simulation_simple()

    # 测试公式
    print("\n" + "=" * 60)
    calc = MinimalEnergyCalculator(sigma_squared=1.0)
    calc.test_formula()

    print("\n" + "=" * 60)
    print("Simulation complete! Check 'energy_analysis_simple.png'")
    print("=" * 60)