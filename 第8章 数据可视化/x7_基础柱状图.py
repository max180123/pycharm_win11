# 通过bar构建柱状图
from pyecharts.charts import Bar
from pyecharts.options import *
# 构建柱状图对象
bar = Bar()
# 添加x轴数据
bar.add_xaxis(["中国", "美国", "英国", "刚果"])
# 添加Y轴数据
bar.add_yaxis("GDP总量", [100, 80, 70, 10], label_opts=LabelOpts(
    position="right"  # 将数值标签设置在右侧
))
# 反转xy轴
bar.reversal_axis()
# 设置全局选项
bar.set_global_opts(
    title_opts=TitleOpts(title="各国GDP总量", pos_right="center", pos_top="10%"),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(is_show=True),
    legend_opts=LegendOpts(is_show=True)
)
# 绘制图像
bar.render("基础柱状图.html")