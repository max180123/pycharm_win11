# 模块就是python文件，里面有类、函数、变量等，可以直接使用
# 写入开头
# import 模块名
# 模块名.功能名（）
import time  # 导入python内置time模块（time.py）

# 按住ctrl☞time
print("开始")
time.sleep(1)  # 让程序睡眠一秒
print("结束")

# from 模块名 import 功能名
# 功能名（）
from time import sleep  # 将sleep更换为*导入全部功能
print("你好")
sleep(5)
print("大家好")
