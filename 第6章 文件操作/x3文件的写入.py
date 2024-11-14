# 删除原文件内容
# 打开文件 f = open("test.txt", "w")
# 写入文件 f.write("")
# 内容刷新 f.flush()  真正写入文件
file = open("test.txt", "w")
file.write("毛主席万岁")
file.flush()
file.close()  # 内置flush
file1 = open("test.txt", "r")
print(file1.read())

with open("tx.txt", "w") as file2:
    file2.write("你好!!!!")