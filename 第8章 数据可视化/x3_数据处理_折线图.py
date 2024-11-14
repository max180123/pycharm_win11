import json
from pyecharts.charts import Line  # 导入折线图功能
from pyecharts.options import TitleOpts, ToolboxOpts, VisualMapOpts, LegendOpts, LabelOpts
# 处理数据
f_us = open("D:/美国.txt", "r", encoding="UTF-8")  # 读取美国的json数据
data_us = f_us.read()
f_jp = open("D:/日本.txt", "r", encoding="UTF-8")  # 读取日本的json数据
data_jp = f_jp.read()
f_in = open("D:/印度.txt", "r", encoding="UTF-8")  # 读取印度的json数据
data_in = f_in.read()
# 去除开头不符合json格式的内容
data_us = data_us.replace("jsonp_1629344292311_69436(", "")
data_jp = data_jp.replace("jsonp_1629350871167_29498(", "")
data_in = data_in.replace("jsonp_1629350745930_63180(", "")
# 去除结尾不符合json格式的内容
data_us = data_us[:-2]
data_jp = data_jp[:-2]
data_in = data_in[:-2]
# 将json文件格式转化为python
dict_us = json.loads(data_us)
dict_jp = json.loads(data_jp)
dict_in = json.loads(data_in)
# 提取trend_key 方便后续提取数据
trend_key_us = dict_us["data"][0]["trend"]
trend_key_jp = dict_jp["data"][0]["trend"]
trend_key_in = dict_in["data"][0]["trend"]
# 获取x轴坐标第一年日期
x_data_us = trend_key_us["updateDate"][:314]
x_data_jp = trend_key_jp["updateDate"][:314]
x_data_in = trend_key_in["updateDate"][:314]
# 获取Y轴坐标确诊数据
y_data_us = trend_key_us["list"][0]["data"][:314]
y_data_jp = trend_key_jp["list"][0]["data"][:314]
y_data_in = trend_key_in["list"][0]["data"][:314]
# 得到折线图对象
line = Line()
# 为X轴添加数据
line.add_xaxis(x_data_us)
# 为Y轴添加数据
line.add_yaxis("美国确诊人数", y_data_us, label_opts=LabelOpts(is_show=False))  # 不显示数据
line.add_yaxis("日本确诊人数", y_data_jp, label_opts=LabelOpts(is_show=False))
line.add_yaxis("印度确诊人数", y_data_in, label_opts=LabelOpts(is_show=False))
line.set_global_opts(
    title_opts=TitleOpts(title="全年新冠确证人数", pos_right="center", pos_top="5%"),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(is_show=True),
    legend_opts=LegendOpts(is_show=True)
)
line.render()  # 生成图表
# 关闭文件
f_us.close()
f_jp.close()
f_in.close()