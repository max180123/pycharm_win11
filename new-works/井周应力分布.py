# 首先，我们需要导入必要的库和设置魔法命令

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["font.family"] = ["Simsun", "Arial"]
# 设置极坐标系参数
r = np.linspace(0.3, 3, 50)  # 半径范围,按行重复
theta = np.linspace(0, 2 * np.pi, 50)  # 角度范围，按列重复
r, theta = np.meshgrid(r, theta)  # 大小为（100,50）

# 将极坐标转换为笛卡尔坐标
x = r * np.cos(theta)
y = r * np.sin(theta)

# 设置z轴数据
a = 0.3  # 设置井筒半径
r2 = (a / r) ** 2
r4 = (a / r) ** 4
segema_H = 10
segema_r = segema_H / 2 * (1 - r2) + segema_H / 2 * (1 + 3 * r4 - 4 * r2) * np.cos(
    2 * theta
)


# 创建图形和3D坐标系
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# 绘制3D表面
surf_r = ax.plot_surface(x, y, segema_r, cmap="viridis")
# 添加 y=0 的平面
y_plane = 0  # y=0
z_plane = np.linspace(
    np.min(segema_r), np.max(segema_r), x.shape[0]
)  # 根据 segema_r 的范围设置 z 的值
x_plane = x  # 使用 x 的值
# test 1+1
# 创建一个网格以绘制平面
X_plane, Z_plane = np.meshgrid(x_plane, z_plane)

# 绘制平面
ax.plot_surface(
    X_plane, y_plane * np.ones_like(X_plane), Z_plane, color="red", alpha=0.5
)  # 绘制 y=0 的平面

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("应力")

# 添加颜色条
cabr_r = fig.colorbar(surf_r, shrink=0.5, aspect=5)
cabr_r.set_label("径向应力")
# 设置标题
plt.title("最大主应力引起的应力分布")

# 显示图形
plt.show()