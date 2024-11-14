# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimSun', 'Arial']

# 定义基础数值
ρ = 1.45 * 10**3  # 密度 kg/m3
d = 0.11  # 管柱直径 m
ε_d = 0.001  # 相对粗糙度
ns = 100 * 10**3  # 泵功率 W
g = 9.8  # 重力加速度 m/s
C = 0.5  # 与喷嘴阻力系数有关的无量纲系数，总是小于 1
A0 = 1.75 * 10**(-4)  # 所有喷嘴的截面积之和 m2

Q = None  # 流量 m3/s
n3 = 0  # 钻头水功率 W

H = 2000  # 井深 m
u = 45 * 10**(-3)  # 流体动力粘度 Pa.s


#  计算钻头水功率
def water_power_of_drilling_bit(u, ε_d, ns, ρ, g, H, Q, d, C, A0):
    f = 1 / (1.8 * np.log10((6.9 * np.pi * u * d) / (4 * ρ * Q) +
                            (ε_d / 3.7)**1.11))**2
    n3 = ns + ρ * g * H * Q - ((f * 8 * ρ * H) / (np.pi**2 * d**5) +
                               (0.005 * ρ) / (C * A0)**2) * Q**3
    return n3 * 0.001


# 生成Q
Q_values = np.linspace(0.0000001, 0.08, 1000)  # 0到0.1之间的1000个点

# 计算N3
n3_values = water_power_of_drilling_bit(u, ε_d, ns, ρ, g, H, Q_values, d, C,
                                        A0)

# 找出钻头水功率的最大值及其对应的Q值
n3_max_value = np.max(n3_values)  # 找出最大的钻头水功率
n3_max_index = np.argmax(n3_values)  # 找出最大的钻头水功率对应的索引值
Q_max_x = Q_values[n3_max_index]  # 根据最大的钻头水功率对应的索引值对应相应的流量

print(f"最大钻头水功率为: {n3_max_value:.2f} W, 对应的流量为Q={Q_max_x * 1000:.2f} L/s")

# 绘图
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(Q_values, n3_values, label='钻头水功率')
plt.scatter(
    [Q_max_x], [n3_max_value],
    color='red',
    zorder=5,
    label=f'最大钻头水功率: {n3_max_value:.2f} kw 在 Q={Q_max_x * 1000:.2f} m3/s处')
plt.xlabel('流量 m3/s')
plt.ylabel('钻头水功率 kw')
plt.title('钻头水功率随流量的变化')
plt.legend()
plt.grid(False)
plt.show()