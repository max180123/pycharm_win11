from pyecharts.charts import Bar, Timeline
from pyecharts.options import *
from pyecharts.globals import ThemeType
# 创建柱状图对象1
bar1 = Bar()
bar1.add_xaxis(["中国", "日本", "美国"])
bar1.add_yaxis("GDP", [100, 90, 110], label_opts=LabelOpts(
    position="right"
))
bar1.reversal_axis()

# 创建柱状图对象2
bar2 = Bar()
bar2.add_xaxis(["中国", "日本", "美国"])
bar2.add_yaxis("GDP", [120, 80, 150], label_opts=LabelOpts(
    position="right"
))
bar2.reversal_axis()
# 创建柱状图对象3
bar3 = Bar()
bar3.add_xaxis(["中国", "日本", "美国"])
bar3.add_yaxis("GDP", [150, 100, 200], label_opts=LabelOpts(
    position="right"
))
bar3.reversal_axis()

# 添加时间线对象
time_line = Timeline(
    {"theme": ThemeType.LIGHT}  # 设置主题
)
# time_line添加bar柱状图
time_line.add(bar1, "2018年各国GDP")
time_line.add(bar2, "2019年各国GDP")
time_line.add(bar3, "2020年各国GDP")
# 设置自动播放
time_line.add_schema(
    play_interval=1000,      # 自动播放的时间间隔 毫秒
    is_timeline_show=True,  # 在播放时显示时间线
    is_auto_play=True,      # 自动播放
    is_loop_play=True       # 循环播放
)
# 绘制图像
time_line.render("各国GDP变化简.html")
