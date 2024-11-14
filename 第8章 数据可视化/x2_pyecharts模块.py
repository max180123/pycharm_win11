from pyecharts.charts import Line  # 导入折线图功能
from pyecharts.options import TitleOpts, LegendOpts, ToolboxOpts, VisualMapOpts
line = Line()  # 得到折线图对象
line.add_xaxis(["中国", "美国", "瑞士"])  # 添加x轴数据
line.add_yaxis("GDP", [30, 70, 20])  # 添加y轴数据
# 全局配置：标题、图例、工具箱  line.set_global_opts()
line.set_global_opts(
    title_opts=TitleOpts(title="GDP展示", pos_right="center", pos_top="10%"),  # 标题配置
    legend_opts=LegendOpts(is_show=True),  # 图例展示
    toolbox_opts=ToolboxOpts(is_show=True),  # 工具箱
    visualmap_opts=VisualMapOpts(is_show=True)  # 视觉映射
)
line.render()  # 生成图表
