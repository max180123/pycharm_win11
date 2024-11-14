# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['SimSun', 'Arial']

# 定义基础数值
n = 3  # 喷嘴数量
kl = 0.08  # 钻头损耗系数
ρ = 1.45 * 10**3  # 密度 kg/m3
d = 0.11  # 管柱直径 m
ε_d = 0.001  # 相对粗糙度
ns = 100 * 10**3  # 泵功率 W
g = 9.8  # 重力加速度 m/s
C = 0.5  # 与喷嘴阻力系数有关的无量纲系数，总是小于 1
dne = 1.5*0.01  # 喷嘴当量直径 m
# A0 = 1.75 * 10**(-4)  # 所有喷嘴的截面积之和 m2

Q = 0.0001  # 流量 m3/s
j = 0  # 喷嘴处机械能 W

H = 2000  # 井深 m
u = 45 * 10**(-3)  # 流体动力粘度 Pa.s


#  计算喷嘴处机械能功率
def drilling_Hydraulic_Optimization(u, ε_d, ns, ρ, g, H, Q, d, dne, n, kl):
    f = 1 / (1.8 * np.log10((6.9 * np.pi * u * d) / (4 * ρ * Q) +
                            (ε_d / 3.7)**1.11))**2
    j = ns + ρ*g*H*Q - 8*ρ*Q**3/(np.pi**2*d**4)*(H*f/d + kl*d**4/(n**2*dne**4) - 1)
    return j * 0.001


# 生成Q
Q_values = np.linspace(0.0000001, 0.08, 1000)  # 0到0.1之间的1000个点

# 计算j
j_values = np.array([drilling_Hydraulic_Optimization(u, ε_d, ns, ρ, g, H, Q, d, dne, n, kl) for Q in Q_values])

# 找出喷嘴处机械能功率的最大值及其对应的Q值
j_max_value = np.max(j_values)  # 找出最大的喷嘴处机械能功率
j_max_index = np.argmax(j_values)  # 找出最大的喷嘴处机械能功率对应的索引值
Q_max_x = Q_values[j_max_index]  # 根据最大的喷嘴处机械能功率对应的索引值对应相应的流量

print(f"最大喷嘴处机械能功率为: {j_max_value:.2f} kW, 对应的流量为Q={Q_max_x * 1000:.2f} L/s")

# 绘图
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(Q_values, j_values, label='喷嘴处机械能功率')
plt.scatter(
    [Q_max_x], [j_max_value],
    color='red',
    zorder=5,
    label=f'最大的喷嘴处机械能功率: {j_max_value:.2f} kw 在 Q={Q_max_x * 1000:.2f} m3/s处')
plt.xlabel('流量 m3/s')
plt.ylabel('喷嘴处机械能功率 kw')
plt.title('喷嘴处机械能功率随流量的变化')
plt.legend()
plt.grid(False)
plt.show()