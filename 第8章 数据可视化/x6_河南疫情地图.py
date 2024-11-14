from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts, TitleOpts, ToolboxOpts
import json

# 打开文件
file = open("D:/疫情.txt", "r", encoding="UTF-8")
# 读取文件
file_json = file.read()
# 关闭文件
file.close()
# 将json转换为python文件
data = json.loads(file_json)
# 提取河南省的数据
data_henan = data["areaTree"][0]["children"][3]["children"]
# 提取河南各城市名称和确诊人数
data_list = list()
for city in data_henan:
    city_name = city["name"] + "市"
    city_confirm = city["total"]["confirm"]
    data_list.append((city_name, city_confirm))
# 创建地图对象
map = Map()
# 添加数据
map.add("河南疫情地图", data_list, "河南")
# 全局选项设置
map.set_global_opts(
    title_opts=TitleOpts(title="河南疫情地图", pos_right="center", pos_top="10%"),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(
        is_show=True,
        is_piecewise=True,
        pieces=(
            {"min": 1, "max": 10, "label": "1-10人", "color": "#CCFFF"},
            {"min": 10, "max": 50, "label": "10-50人", "color": "#FFFF99"},
            {"min": 50, "max": 100, "label": "50-100人", "color": "#FF9966"},
            {"min": 100, "max": 500, "label": "100-500人", "color": "#FF6666"},
            {"min": 500, "max": 1000, "label": "500-1000人", "color": "#CC3333"},
            {"min": 1000, "label": "1000人以上", "color": "#990033"}
        )
    )
)
# 绘制图像
map.render("河南疫情地图.html")