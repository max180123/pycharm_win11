import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 定义计算钻头水功率的函数
def water_power_of_drilling_bit(u, ε_d, ns, ρ, g, H, Q, d, C, A0):
    f = 1 / (1.8 * np.log10((6.9 * np.pi * u * d) / (4 * ρ * Q) +
                            (ε_d / 3.7)**1.11))**2
    n3 = ns + ρ * g * H * Q - ((f * 8 * ρ * H) / (np.pi**2 * d**5) +
                               (0.005 * ρ) / (C * A0)**2) * Q**3
    return n3 * 0.001

# 定义按钮点击事件
def on_calculate():
    ρ = float(ρ_entry.get())
    d = float(d_entry.get())
    ε_d = float(ε_d_entry.get())
    ns = float(ns_entry.get())
    g = float(g_entry.get())
    C = float(C_entry.get())
    A0 = float(A0_entry.get())
    H = float(H_entry.get())
    u = float(u_entry.get())

    Q_values = np.linspace(0.0000001, 0.08, 1000)
    n3_values = water_power_of_drilling_bit(u, ε_d, ns, ρ, g, H, Q_values, d, C, A0)

    n3_max_value = np.max(n3_values)
    n3_max_index = np.argmax(n3_values)
    Q_max_x = Q_values[n3_max_index]

    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(Q_values, n3_values, label='钻头水功率')
    plt.scatter(
        [Q_max_x], [n3_max_value],
        color='red',
        zorder=5,
        label=f'最大钻头水功率: {n3_max_value:.2f} kW 在 Q={Q_max_x * 1000:.2f} L/s处')
    plt.xlabel('流量 L/s')
    plt.ylabel('钻头水功率 kW')
    plt.title('钻头水功率随流量的变化')
    plt.legend()
    plt.grid(False)
    plt.show()

# 创建GUI窗口
root = Tk()
root.title("钻头水功率计算")

# 创建并布置标签和输入框
Label(root, text="密度 ρ (kg/m³):").grid(row=0, column=0)
ρ_entry = Entry(root)
ρ_entry.grid(row=0, column=1)
ρ_entry.insert(0, "1450")

Label(root, text="管柱直径 d (m):").grid(row=1, column=0)
d_entry = Entry(root)
d_entry.grid(row=1, column=1)
d_entry.insert(0, "0.11")

Label(root, text="相对粗糙度 ε_d:").grid(row=2, column=0)
ε_d_entry = Entry(root)
ε_d_entry.grid(row=2, column=1)
ε_d_entry.insert(0, "0.001")

Label(root, text="泵功率 ns (W):").grid(row=3, column=0)
ns_entry = Entry(root)
ns_entry.grid(row=3, column=1)
ns_entry.insert(0, "100000")

Label(root, text="重力加速度 g (m/s²):").grid(row=4, column=0)
g_entry = Entry(root)
g_entry.grid(row=4, column=1)
g_entry.insert(0, "9.8")

Label(root, text="无量纲系数 C:").grid(row=5, column=0)
C_entry = Entry(root)
C_entry.grid(row=5, column=1)
C_entry.insert(0, "0.5")

Label(root, text="喷嘴截面积 A0 (m²):").grid(row=6, column=0)
A0_entry = Entry(root)
A0_entry.grid(row=6, column=1)
A0_entry.insert(0, "1.75e-4")

Label(root, text="井深 H (m):").grid(row=7, column=0)
H_entry = Entry(root)
H_entry.grid(row=7, column=1)
H_entry.insert(0, "2000")

Label(root, text="流体动力粘度 u (Pa.s):").grid(row=8, column=0)
u_entry = Entry(root)
u_entry.grid(row=8, column=1)
u_entry.insert(0, "0.045")

# 计算按钮
Button(root, text="计算", command=on_calculate).grid(row=9, column=0, columnspan=2)

# 运行主循环
root.mainloop()