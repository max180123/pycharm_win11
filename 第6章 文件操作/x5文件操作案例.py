# file1 = open("bill.txt", "r", encoding="UTF-8")
# file2 = open("bill.bak.txt", "a")
# for line in file1:
#     if line.count("测试") != 1:
#         file2.write(line)
# file1.close()
# file2.close()

file1 = open("bill.txt", "r", encoding="UTF-8")
file2 = open("bill.bak.txt", "w")
for line in file1:
    line = line.strip()  # 去除末尾的换行符
    if line.split(",")[4] == "测试":  # 将字符串转化为列表
        continue
    else:
        file2.write(line)
        file2.write("\n")
file1.close()
file2.close()