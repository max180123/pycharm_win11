# 数据类型type（数据、变量）变量无类型 储存的数据有类型
a = "字符串"
a1 = type(a)
print(a1)

# 数据转换  整数int()  浮点数float（） 字符串str（）
# 整数转字符串
num_str = str(11)
print(type(num_str), num_str)
# 字符串转整数
num_int = int("11")
print(type(num_int), num_int)
# 向下取整
num_int1 = int(11.9)
print(num_int1)
