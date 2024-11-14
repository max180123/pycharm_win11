# json在各种编程语言中流通的数据格式，负责不同编程语言的数据传递和交互
# 本质：带有特殊数据格式的字符串
# python中的字典、列表（内嵌套字典）转化为字符串

import json  # 导入json模块
data = {"name": "张三", "age": 15}  # 准备符合json格式要求的python数据
data1 = [{"name": "max", "age": 15}, {"name": "yue", "age": 17}]
data = json.dumps(data, ensure_ascii=False)  # 通过json.dumps（）方法将python数据转化为json数据
# ensure_ascii = False 字典中有中文时使用可显示中文
data1 = json.dumps(data1)
data = json.loads(data)  # 通过json.loads（）方法将json数据转化为python数据
data1 = json.loads(data1)