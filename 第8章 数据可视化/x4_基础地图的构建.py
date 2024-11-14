from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts
# 准备地图对象
map = Map()
# 准备数据
data = [
    ("北京市", 99),
    ("上海市", 1235),
    ("湖南省", 144),
    ("陕西省", 4526),
    ("新疆维吾尔自治区", 31233),
    ("西藏自治区", 1055)
]
# 添加数据 地图名称 数据 地图类型
map.add("疫情地图", data, "china")
# 设置全局选项  RGB颜色对照表
map.set_global_opts(
    visualmap_opts=VisualMapOpts(
        is_show=True,  # 设置视觉映射为开
        is_piecewise=True,
        pieces=[
            {"min": 1, "max": 9, "label": "1-9人", "color": "#CCFFF"},
            {"min": 10, "max": 99, "label": "10-99人", "color": "#FFFF99"},
            {"min": 100, "max": 499, "label": "100-499人", "color": "#FF9966"},
            {"min": 500, "max": 999, "label": "500-999人", "color": "#FFF6666"},
            {"min": 1000, "max": 9999, "label": "1000-9999人", "color": "#CC3333"},
            {"min": 10000, "label": "10000人以上", "color": "#990033"}
        ]
    )

)
# 绘制图像
map.render()