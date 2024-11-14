from pyecharts.charts import Timeline, Bar
from pyecharts.options import *
from pyecharts.globals import ThemeType
# 读取文件
file = open("D:/1960-2019全球GDP数据.csv", "r", encoding="GB2312")
# 读取文件  按行读取，返回列表
data_list = file.readlines()
# 关闭文件
file.close()
# 删除首行
del data_list[0]
# 创建字典接受数据
data_dict = dict()
# 遍历数据
for line in data_list:
    line = line.split(",")
    year = int(line[0])  # 取得年份
    country = line[1]    # 取得国家
    gdp = float(line[2])  # 取得GDP   float可将科学计算法直接转换为数字
    # 向字典中添加数据
    try:
        data_dict[year].append([country, gdp])
    except KeyError:
        data_dict[year] = []
        data_dict[year].append([country, gdp])
# 排序年份  在字典中存入内容随机无需
data_dict_year = sorted(data_dict.keys())
for key in data_dict_year:
    data_dict[key].sort(
        key=lambda element: element[1],
        reverse=True
    )
# 创建时间线对象
time_line = Timeline(
    {"theme": ThemeType.LIGHT}
)
for key in data_dict:
    bar = Bar()  # 创建柱状图
    data_x = []  # 创建x轴列表数据
    data_y = []  # 创建x轴列表数据
    for li in data_dict[key][:10]:
        data_x.append(li[0])        # 添加前十的国家
        data_y.append(round(li[1]/10**9, 2))  # 添加前十的国家的gdp
    data_y.reverse()  # 数据反转
    data_x.reverse()  # 数据反转
    bar.add_xaxis(data_x)
    bar.add_yaxis("GDP亿元", data_y, label_opts=LabelOpts(
        position="right"
    ))
    bar.reversal_axis()  # 反转X、Y轴
    # 设置全局选项
    bar.set_global_opts(
        title_opts=TitleOpts(title=f"{key}年全球GDP前十国家", pos_right='center', pos_top="5%"),
        toolbox_opts=ToolboxOpts(is_show=True)
    )
    time_line.add(bar, "key")   # 向时间线中添加柱状图
time_line.add_schema(
    play_interval=1000,     # 设置时间间隔1000毫秒
    is_timeline_show=True,  # 显示时间线
    is_loop_play=True,      # 循环播放
    is_auto_play=True       # 自动播放
)
# 绘制图像
time_line.render("1960-2019全球前十GDP.html")