# task = [21, 25, 21, 23, 22, 20]
# task1 = [29, 33, 30]
# task.append(31)  # 追加一个数字31到尾部
# task.extend(task1)  # 追加一批元素
# print(task)
# print(task[0])  # 取出第一个元素
# print(task[-1])  # 取出最后一个元素
# print(task.index(31))   # 查找元素31的位置


#  取出列表中的偶数
task1 = list()
task2 = list()
for i in range(1, 11):
    task1.append(i)

for i in range(0, len(task1)):
    if task1[i] % 2 == 0:  # 取余
        task2.append(task1[i])
print(task2)

count = 0
while count < len(task1):
    if task1[count] % 2 == 0:  # 取余
        task2.append(task1[count])
    count += 1
print(task2)