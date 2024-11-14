from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts, ToolboxOpts, TitleOpts
import json
# 打开文件
file = open("D:/疫情.txt", "r", encoding="UTF-8")
# 读取文件
file1 = file.read()
# 将json文件转为python
file_p = json.loads(file1)
# 取到各省数据
data = file_p["areaTree"][0]["children"]
# 组装每个省份和确诊人数为元组封装入列表
data_list = list()
for state in data:  # 从列表数据中依次读取每个省
    tem_list = list()
    tem_list.append(state["name"]+"省")  # 提取省份
    tem_list.append(state["total"]["confirm"])  # 提取相应的省份的确诊人数
    data_list.append(tuple(tem_list))  # 将存有省份、确诊人数的数据转化为元组存入列表封装
data_list_append = [("天津市", 445), ("北京市", 1107), ("重庆市", 603), ("澳门特别行政区", 63), ("内蒙古自治区", 410),
                    ("宁夏回族自治区", 77), ("西藏自治区", 1), ("新疆维吾尔自治区", 980)]
for i in data_list_append:
    data_list.append(i)
print(data_list)
# 创建地图对象
map = Map()
# 添加数据（名称、数据、地图类型）
map.add("疫情地图", data_list, "china")
# 设置全局选项
map.set_global_opts(
    title_opts=TitleOpts(title="全国疫情地图", pos_right="center", pos_top="10%"),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(
        is_show=True,  # 打开视觉映射
        is_piecewise=True,  # 自定义范围
        pieces=(
            {"min": 1, "max": 10, "label": "1-10人", "color": "#CCFFF"},
            {"min": 10, "max": 100, "label": "10-100人", "color": "#FFFF99"},
            {"min": 100, "max": 500, "label": "100-500人", "color": "#FF9966"},
            {"min": 500, "max": 1000, "label": "500-1000人", "color": "#FF6666"},
            {"min": 1000, "max": 10000, "label": "1000-1000人", "color": "#CC3333"},
            {"min": 10000, "label": "10000人以上", "color": "#990033"}
        )
    )
)

# 绘制地图
map.render("全国疫情地图.html")
# 关闭文件
file.close()