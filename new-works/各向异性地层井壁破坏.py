# 导入库
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# 获取原始的Greens cmap
cmap = plt.get_cmap('Greens')

# 通过颜色映射函数来截取cmap的某一部分，生成一个深绿色版本
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    'deep_greens',
    cmap(np.linspace(0.45, 1, 256))  # 0.45开始，去除较浅的颜色部分
)
plt.rcParams['font.family'] = ['SimSun', 'Arial']

# 定义所有的基础参数及其范围
gird_size = 200
a = 0.3  # 'min': 0, 'max': 1, 'description': '井筒半径/m'},
w_p = 2438.4  # 'min': 0, 'max': 12000, 'description': '井深/m'},
σ_v = 44.13 # , 'min': 0, 'max': 200, 'description': '上覆地层压力/MPa'},
σ_H = 49.64  # , 'min': 0, 'max': 200, 'description': '最大主应力/MPa'},
σ_h = 38.61  # , 'min': 0, 'max': 200, 'description': '最小主应力/MPa'},
ν = 0.35  # , 'min': -1, 'max': 0.5, 'description': '岩石泊松比'},
p_o = 25.37  # , 'min': 0, 'max': 200, 'description': '孔隙压力/MPa'},
t_o = 4.41  # , 'min': 0, 'max': 200, 'description': '岩石的抗拉强度/MPa'},
s_o = 7  # , 'min': 0, 'max': 200, 'description': '岩石的剪切强度/MPa'},
μ_o = 0.58  # , 'min': 0, 'max': 2, 'description': '岩石的内摩擦系数'},
t_w = 0.0  # , 'min': 0, 'max': 200, 'description': '弱面的拉伸强度/MPa'},
s_w = 2.98  # , 'min': 0, 'max': 200, 'description': '弱面的剪切强度/MPa'},
μ_w = 0.47  # , 'min': 0, 'max': 2, 'description': '弱面的内摩擦系数'},
α_w = 225  # , 'min': 0, 'max': 360, 'description': '弱面的倾向/°'},
β_W = 60  # , 'min': 0, 'max': 360, 'description': '弱面的倾角/°'},
γ = 0.8  # , 'min': 0, 'max': 1, 'description': '渗透系数'},


# 计算无裂缝情况下钻井液密度窗口    p_min, p_max 单位为MPa
def calculete_drilling_fluid_window(w_p, σ_H, σ_h, p_o, t_o, s_o, μ_o):
    φ = np.arctan(μ_o)
    p_min = (p_o + (3 * σ_H - σ_h - 2 * p_o - 2 * s_o * (np.cos(φ) / (1 - np.sin(φ)))) / (
            1 + (np.cos(φ) / (1 - np.sin(φ))) ** 2)) / (0.00981 * w_p)
    p_max = (3 * σ_h - σ_H - p_o + t_o) / (0.00981 * w_p)
    # 保留两位小数
    return np.round(p_min, 2), np.round(p_max, 2)


# 计算沿弱面的剪切拉伸破坏指数
def calculete_weak_plane(grid_size, a, σ_v, σ_H, σ_h, ν, p_o, t_w, s_w, μ_w, α_w, β_W, γ, pm):
    # 将角度转换为弧度制
    α_w = np.deg2rad(α_w)
    β_W = np.deg2rad(β_W)
    w_r = 1.3 * a
    # 生成网格
    x_store = np.linspace(-w_r, w_r, int(grid_size + 1))
    y_store = np.linspace(-w_r, w_r, int(grid_size + 1))
    X, Y = np.meshgrid(x_store, y_store)

    # 计算半径和角度
    R = np.sqrt(X ** 2 + Y ** 2)
    valid_mask = R >= a
    R = R[valid_mask]
    cos_t = X[valid_mask] / R
    sin_t = Y[valid_mask] / R

    cos_2t = cos_t ** 2 - sin_t ** 2
    sin_2t = 2 * cos_t * sin_t

    # 计算极坐标应力分量
    one_over_r2 = (a / R) ** 2
    one_over_r4 = one_over_r2 ** 2

    σ_r = (σ_H + σ_h) / 2 * (1 - one_over_r2) + (σ_H - σ_h) / 2 * (
            1 - 4 * one_over_r2 + 3 * one_over_r4) * cos_2t + pm * one_over_r2
    σ_t = (σ_H + σ_h) / 2 * (1 + one_over_r2) - (σ_H - σ_h) / 2 * (1 + 3 * one_over_r4) * cos_2t - pm * one_over_r2
    τ_r = -(σ_H - σ_h) / 2 * (1 + 2 * one_over_r2 - 3 * one_over_r4) * sin_2t
    σ_z = σ_v - 2 * ν * (σ_H - σ_h) * one_over_r2 * cos_2t

    # 逐元素构造旋转矩阵
    polar_xyz_list = np.array([np.array([[c, -s, 0],
                                         [s, c, 0],
                                         [0, 0, 1]]) for c, s in zip(cos_t, sin_t)])

    xyz_W = np.array([[np.cos(α_w) * np.cos(β_W), np.sin(α_w) * np.cos(β_W), -np.sin(β_W)],
                      [-np.sin(α_w), np.cos(α_w), 0],
                      [np.cos(α_w) * np.sin(β_W), np.sin(α_w) * np.sin(β_W), np.cos(β_W)]])

    # 将 xyz_W 扩充到 (n, 3, 3) 的数组，其中 n 是 polar_xyz_list 的长度
    n = polar_xyz_list.shape[0]
    xyz_W_expanded = np.tile(xyz_W[np.newaxis, :, :], (n, 1, 1))

    # 计算应力在直角坐标系的分量
    s_polar_list = np.array([np.array([[a, d, 0],
                                       [d, b, 0],
                                       [0, 0, c]]) for a, b, c, d in zip(σ_r, σ_t, σ_z, τ_r)])

    # 计算应力在直角坐标系的分量
    s_xyz = np.einsum('nij,njk,nkl->nil', polar_xyz_list, s_polar_list, np.transpose(polar_xyz_list, (0, 2, 1)))
    s_W = np.einsum('nij,njk,nkl->nil', xyz_W_expanded, s_xyz, xyz_W_expanded.transpose(0, 2, 1))

    # 计算弱面坐标系的应力分量
    σz_w = s_W[:, 2, 2]
    τzx_w = s_W[:, 2, 0]
    τzy_w = s_W[:, 2, 1]

    # 计算剪应力
    delta_sz = σz_w - p_o + γ * (p_o - pm)  # 有效应力
    l = np.sqrt(τzx_w ** 2 + τzy_w ** 2) - s_w
    l[delta_sz > 0] -= μ_w * delta_sz[delta_sz > 0]

    # 筛选剪切失效点
    failure_mask = l > 0
    xw_plot = X[valid_mask][failure_mask]
    yw_plot = Y[valid_mask][failure_mask]
    l_store = l[failure_mask]

    # 计算拉伸应力
    lt = -(σz_w - p_o + γ * (p_o - pm) + t_w)

    # 筛选拉伸失效点
    tension_failure_mask = lt > 0
    xt_plot = X[valid_mask][tension_failure_mask]
    yt_plot = Y[valid_mask][tension_failure_mask]
    lt_store = lt[tension_failure_mask]

    R = np.size(R)

    return xw_plot, yw_plot, l_store, xt_plot, yt_plot, lt_store, R


def plot_shear_tension_curve(grid_size, a, w_p, σ_v, σ_H, σ_h, ν, p_o, t_o, s_o, μ_o, t_w, s_w, μ_w, α_w, β_W, γ):
    # 计算钻井液密度窗口
    p_min, p_max = calculete_drilling_fluid_window(w_p, σ_H, σ_h, p_o, t_o, s_o, μ_o)
    pm = np.arange(p_min, p_max, 0.01) * w_p * 0.00981
    # 记录数据
    n = len(pm)
    xw_plots = np.empty(n, dtype=object)
    yw_plots = np.empty(n, dtype=object)
    lmax_stores = np.zeros(n)
    l_stores = np.empty(n, dtype=object)

    xt_plots = np.empty(n, dtype=object)
    yt_plots = np.empty(n, dtype=object)
    ltmax_stores = np.zeros(n)
    lt_stores = np.empty(n, dtype=object)
    for i, p in enumerate(pm):
        xw_plot, yw_plot, l_store, xt_plot, yt_plot, lt_store, R = calculete_weak_plane(grid_size, a, σ_v, σ_H, σ_h,
                                                                                        ν, p_o, t_w, s_w, μ_w, α_w, β_W,
                                                                                        γ, p)
        # 将结果追加到对应的列表中
        xw_plots[i] = xw_plot
        yw_plots[i] = yw_plot
        # 防止出现未发生剪切破坏
        lmax_stores[i] = np.nanmean(l_store) * np.size(l_store) / R if l_store.size > 0 else 0.0
        l_stores[i] = l_store

        xt_plots[i] = xt_plot
        yt_plots[i] = yt_plot
        # 防止出现未发生拉伸破坏
        ltmax_stores[i] = np.nanmean(lt_store) * np.size(lt_store) / R if lt_store.size > 0 else 0.0
        lt_stores[i] = lt_store

    # 找出最小的剪切、拉伸破坏指数及其对应的钻井液密度
    pm = pm / (w_p * 0.00981)
    if lmax_stores.size > 0:
        lmin_value = np.min(lmax_stores)
        lmin_value_index = np.argmin(lmax_stores)
        pm_min_x = pm[lmin_value_index]

        ltmin_value = ltmax_stores[lmin_value_index]
        ltmin_value_index = lmin_value_index
        pmt_min_x = pm_min_x
    else:
        lmin_value = 0
        lmin_value_index = 0
        pm_min_x = pm[lmin_value_index]

        ltmin_value = np.min(ltmax_stores)
        ltmin_value_index = np.argmin(ltmax_stores)
        pmt_min_x = pm[ltmin_value_index]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=500)

    # 最大剪切应力随钻井液密度的变化
    axs[0, 0].plot(pm, lmax_stores, label='最大剪应力', color='black')
    axs[0, 0].scatter(pm_min_x, lmin_value, color='red', zorder=5, s=10,
                      label=f'最小剪应力:{lmin_value: .2f} MPa 在 pm={pm_min_x:.2f}g/cm3处')
    axs[0, 0].set_title('最大剪应力随钻井液密度的变化')
    axs[0, 0].set_xlabel('钻井液密度 g/cm3')
    axs[0, 0].set_ylabel('最大剪应力 MPa')
    axs[0, 0].legend(loc='upper left', fontsize='medium', frameon=True)  # 显示图例

    if l_stores[lmin_value_index].size > 0:  # 检查数组是否为空
        vmax_value = np.max(l_stores[lmin_value_index])
    else:
        vmax_value = 0.001  # 或者设定一个合理的默认值

    # 最小剪应力对应的井周剪应力分布
    scatter = axs[0, 1].scatter(xw_plots[lmin_value_index], yw_plots[lmin_value_index],
                                c=l_stores[lmin_value_index],
                                marker=',',  # 像素点
                                s=10,
                                vmin=0.1 * vmax_value, vmax=0.9 * vmax_value,  # 颜色映射的值
                                cmap=new_cmap,  # 颜色映射表
                                zorder=1)  # 图层
    fig.colorbar(scatter, ax=axs[0, 1], label='')
    axs[0, 1].set_title('井周剪应力分布')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')

    # 在图二上画一个半径为1的圆
    circle1 = Circle((0, 0), a, color='black', fill=False, linestyle='-')
    axs[0, 1].add_patch(circle1)
    # 设置图二的轴比例为相同
    axs[0, 1].set_aspect('equal')

    # 最大拉应力随钻井液密度的变化
    axs[1, 0].plot(pm, ltmax_stores, label='最大拉伸应力', color='black')
    axs[1, 0].scatter(pmt_min_x, ltmin_value, color='red', zorder=5, s=10,
                      label=f'最小剪应力对应的拉伸应力:{ltmin_value: .2f} MPa 在 pm={pmt_min_x:.2f}g/cm3处')
    axs[1, 0].set_title('最大拉伸应力随钻井液密度的变化')
    axs[1, 0].set_xlabel('钻井液密度 g/cm3')
    axs[1, 0].set_ylabel('最大拉伸应力 MPa')
    axs[1, 0].legend(loc='upper left', fontsize='medium', frameon=True)  # 显示图例

    if lt_stores[ltmin_value_index].size > 0:  # 检查数组是否为空
        vmax_value = np.max(lt_stores[ltmin_value_index])
    else:
        vmax_value = 0.001  # 或者设定一个合理的默认值

    # 最小拉应力对应的井周拉应力分布
    scatter = axs[1, 1].scatter(xt_plots[ltmin_value_index], yt_plots[ltmin_value_index],
                                c=lt_stores[ltmin_value_index], marker=',',
                                s=10, vmin=0.1 * vmax_value, vmax=0.9 * vmax_value, cmap=new_cmap, zorder=1)
    fig.colorbar(scatter, ax=axs[1, 1], label='')
    axs[1, 1].set_title('井周拉应力分布')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')

    # 在图四上画一个半径为1的圆
    circle2 = Circle((0, 0), a, color='black', fill=False, linestyle='-')
    axs[1, 1].add_patch(circle2)
    # 设置图四的轴比例为相同
    axs[1, 1].set_aspect('equal')

    # 调整布局，使子图之间没有重叠
    plt.tight_layout()

    # 显示图像
    plt.show()


plot_shear_tension_curve(gird_size, a, w_p, σ_v, σ_H, σ_h, ν, p_o, t_o, s_o, μ_o, t_w, s_w, μ_w, α_w, β_W, γ)