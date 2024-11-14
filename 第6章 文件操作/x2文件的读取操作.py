# 打开文件
# 语法 f = open(name, mode, encoding)   name:文件名（可包含路径）
# mode：只读r 写入w（原内容覆盖） 追加a
# encoding = "UTF-8"
f = open("/第6章 文件操作/test.txt", "r", encoding="UTF-8")
print(type(f))


# 读取文件
# 文件.read（num） num为字节  不设置读全文
#print(f.read())  # 下一次读从上一次的结尾开始


# 文件.readlines（）  按照行一次性读取全文  返回列表
# content = f.readlines()
# print(content)

# 文件.readline（）一次读取一行
# lin1 = f.readline()
# lin2 = f.readline()
# print(lin1)
# print(lin2)


# # for 循环读取
# for line in f:
#     print(f"每一行的数据是：{line}")
#
# # 文件关闭
# f.close()  # 结束文件占用
#
# #  with open 语法  操作完成后自动关闭
# with open("test.txt", "r", encoding="UTF-8") as f_:
#     f_.readlines()
#
#
# # 作业
# # word读取编码 encoding="ISO-8859-1"
# with open("test.txt", "r", encoding="ISO-8859-1") as file:
#     t = file.read().count("it")
#     print(f"文件中it出现的次数：{t}")
#
# with open("test.txt", "r", encoding="ISO-8859-1") as file:
#     count = 0
#     for line in file:
#         count += line.count("it")
#     print(f"文件中it出现的次数：{count}")
#
# with open("test.txt", "r", encoding="ISO-8859-1") as file:
#     count = 0
#     for line in file:
#         line = line.strip()  # 去除首位的换行符 空格
#         count += line.split(" ").count("it")
#     print(f"文件中it出现的次数：{count}")
#
# with open("test.txt", "r", encoding="ISO-8859-1") as file:
#     count = 0
#     for line in file:
#         line = line.replace(" ", "\n")  # 去除换行符
#         count += line.split().count("it")
#     print(f"文件中it出现的次数：{count}")

with open("test.txt", "r", encoding="UTF-8") as file:
    count = file.read().count("it")
    print(count)

with open("test.docx", "r", encoding="ISO-8859-1") as file:
    count = file.read().count("it")
    print(count)